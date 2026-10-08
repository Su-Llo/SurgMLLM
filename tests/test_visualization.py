from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from projects.surgmllm.evaluation.visualization import (
    VisualizationRenderError,
    _role_label_color,
    render_visualizations,
)


VALID_TEXT = (
    "<think>A grasper retracts tissue.</think>"
    "<answer>During the preparation phase, 1 surgical action triplet is identified: "
    "(1) the instrument is <p>grasper</p>[SEG], the target is "
    "<p>gallbladder</p>[SEG], based on the two components, the action is retract."
    "</answer>"
)


class VisualizationTests(unittest.TestCase):
    def _records(self, image_path: Path):
        instrument = np.array(
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool
        )
        target = np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool
        )
        prediction = {
            "video_id": "VID79",
            "frame_id": "28",
            "image_path": str(image_path),
            "generated_text": VALID_TEXT,
            "parse_error": None,
            "phase": {"label": "preparation", "confidence": 0.9},
            "triplets": [
                {
                    "instrument": "grasper",
                    "verb": "retract",
                    "target": "gallbladder",
                    "instrument_mask": {"array": instrument},
                    "target_mask": {"array": target},
                }
            ],
            "groundings": [],
        }
        annotation = {
            "video_id": "VID79",
            "frame_id": "28",
            "image_path": str(image_path),
            "phase": "preparation",
            "triplets": [
                {"instrument": "grasper", "verb": "retract", "target": "gallbladder"}
            ],
            "groundings": [
                {"role": "instrument", "label": "grasper", "mask": instrument},
                {"role": "target", "label": "gallbladder", "mask": target},
            ],
        }
        missing = {
            **annotation,
            "frame_id": "29",
        }
        report = {
            "overall": {
                "grounding": {
                    "frames": [{"video_id": "VID79", "frame_id": "28", "mIoU": 0.75}]
                }
            }
        }
        return prediction, annotation, missing, report

    def test_renders_triplet_masks_and_records_missing_without_fake_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.png"
            Image.new("RGB", (4, 4), (90, 100, 110)).save(image_path)
            prediction, annotation, missing, report = self._records(image_path)

            manifest = render_visualizations(
                [prediction], [annotation, missing], report, root / "visualizations", workers=2
            )

            output = root / "visualizations" / "VID79" / "000028.png"
            self.assertTrue(output.is_file())
            self.assertFalse((root / "visualizations" / "VID79" / "000029.png").exists())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["matched_frames"], 1)
            self.assertEqual(manifest["rendered_frames"], 1)
            self.assertEqual(manifest["missing_frame_ids"], ["VID79/29"])
            self.assertEqual(manifest["frames"][0]["frame_mIoU"], 0.75)
            self.assertEqual(manifest["frames"][0]["parse_status"], "OK")
            self.assertEqual(len(manifest["frames"][0]["pred_legend"]), 2)
            with Image.open(output) as rendered:
                self.assertGreater(rendered.width, rendered.height)
            persisted = json.loads((root / "visualizations" / "manifest.json").read_text())
            self.assertEqual(persisted["rendered_frames"], 1)
            self.assertNotIn(str(root), json.dumps(persisted))

    def test_role_label_colors_are_deterministic_and_role_aware(self):
        self.assertEqual(_role_label_color("instrument", "grasper"), _role_label_color("instrument", "grasper"))
        self.assertNotEqual(_role_label_color("instrument", "grasper"), _role_label_color("target", "grasper"))

    def test_aggregates_errors_writes_manifest_and_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_image = root / "does-not-exist.png"
            prediction, annotation, _missing, report = self._records(missing_image)

            with self.assertRaises(VisualizationRenderError) as caught:
                render_visualizations(
                    [prediction], [annotation], report, root / "visualizations", workers=1
                )

            manifest_path = root / "visualizations" / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["rendered_frames"], 0)
            self.assertEqual(manifest["error_count"], 1)
            self.assertEqual(manifest["errors"][0]["error_type"], "FileNotFoundError")
            self.assertNotIn(str(root), json.dumps(manifest))
            self.assertEqual(caught.exception.manifest_path, str(manifest_path))

    def test_rejects_non_positive_worker_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "at least 1"):
                render_visualizations([], [], {}, temporary, workers=0)


if __name__ == "__main__":
    unittest.main()
