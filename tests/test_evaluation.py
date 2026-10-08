from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from projects.surgmllm.evaluation import (
    evaluate,
    evaluate_grounding_frames,
    evaluate_phase_videos,
    evaluate_triplet_ap,
    match_grounding_instances,
    parse_generated_frame,
    parse_generated_window,
    sparse_average_precision,
    write_report,
)
from projects.surgmllm.evaluation.surgmllm_infer_gcg_fold1 import (
    SourceFrame,
    InferenceWindow,
    _portable_image_reference,
    _portable_model_reference,
    _resolve_image_path,
    _transition_probabilities,
    _window_records,
    binary_mask_to_rle,
    build_windows,
)
from projects.surgmllm.evaluation.io import load_prediction_records


VALID_TEXT = (
    "<think>A grasper retracts the gallbladder.</think>"
    "<answer>During the preparation phase, 1 surgical action triplet is identified: "
    "(1) the instrument is <p>grasper</p>[SEG], the target is "
    "<p>gallbladder</p>[SEG], based on the two components, the action is retract."
    "</answer>"
)

HEADER_GROUNDED_TEXT = (
    "<think>The hook dissects the gallbladder.</think>"
    "<answer>During the <p> gallbladder </p> [SEG] dissection phase, "
    "1 surgical action triplet is identified: (1) the instrument is "
    "<p> hook </p> [SEG], the target is gallbladder, based on the two "
    "components, the action is dissect.</answer>"
)

REPEATED_TARGET_HEADER_TEXT = (
    "<think>The grasper acts twice.</think>"
    "<answer>During the <p> gallbladder </p> [SEG] dissection phase, "
    "2 surgical action triplets are identified: (1) the instrument is "
    "grasper, the target is gallbladder, based on the two components, the "
    "action is retract; (2) the instrument is grasper, the target is "
    "gallbladder, based on the two components, the action is dissect.</answer>"
)


class EvaluationTests(unittest.TestCase):
    def test_per_video_prediction_path_supplies_missing_video_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "VID79" / "predictions.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps(
                    {
                        "frame_results": [
                            {
                                "frame_file": "000001.png",
                                "phase": "preparation",
                                "triplets": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = load_prediction_records(path)
            self.assertEqual(
                [(record["video_id"], record["frame_id"]) for record in records],
                [("VID79", "1")],
            )

    def test_prediction_directory_fails_closed_on_bad_json_or_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupt = root / "corrupt.json"
            corrupt.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_prediction_records(root)

            corrupt.unlink()
            conflicting = root / "conflicting.json"
            conflicting.write_text(
                json.dumps(
                    {
                        "video_id": "VID79",
                        "frame_id": "1",
                        "file_name": "VID80/000001.png",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Conflicting frame video identities"):
                load_prediction_records(root)

    def test_inference_rejects_cross_video_and_absolute_image_paths(self):
        absolute = SourceFrame(
            "VID79", "1", "/external/root/VID79/000001.png", "", 2, 2
        )
        with self.assertRaisesRegex(ValueError, "relative to data_root"):
            _resolve_image_path(Path("/unused"), absolute)

        cross_video = SourceFrame("VID79", "1", "VID02/000001.png", "", 2, 2)
        with self.assertRaisesRegex(ValueError, "disagrees with VID79"):
            _resolve_image_path(Path("/unused"), cross_video)

    def test_inference_accepts_a_confined_symlinked_videos_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            data_root = workspace / "CholecT45-Scene"
            external_videos = workspace / "shared-videos"
            image = external_videos / "VID79" / "000001.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"test image placeholder")
            data_root.mkdir()
            (data_root / "videos").symlink_to(external_videos, target_is_directory=True)
            frame = SourceFrame("VID79", "1", "VID79/000001.png", "", 2, 2)

            self.assertEqual(_resolve_image_path(data_root, frame), image.resolve())

    def test_generated_artifact_references_do_not_expose_local_paths(self):
        frame = SourceFrame(
            "VID79",
            "1",
            "/external/root/videos/VID79/000001.png",
            "/external/root/annotations",
            2,
            2,
        )
        self.assertEqual(_portable_image_reference(frame), "VID79/000001.png")
        self.assertEqual(
            _portable_model_reference("/external/root/HF_SurgMLLM_Fold1"),
            "HF_SurgMLLM_Fold1",
        )
        self.assertEqual(
            _portable_model_reference("organization/SurgMLLM"),
            "organization/SurgMLLM",
        )

    def test_inference_windows_never_cross_numeric_frame_gaps(self):
        frames = [
            SourceFrame(
                video_id="VID79",
                frame_id=str(frame_id),
                image_path=f"{frame_id}.png",
                base_dir="",
                height=None,
                width=None,
            )
            for frame_id in (1, 2, 4, 5, 6, 7, 8)
        ]
        windows = build_windows(frames, stride=1)
        self.assertEqual(
            [[frame.frame_id for frame in window.frames] for window in windows],
            [["4", "5", "6", "7", "8"]],
        )

    def test_inference_stride_restarts_inside_each_run_and_covers_tail(self):
        frames = [
            SourceFrame(
                video_id="VID79",
                frame_id=str(frame_id),
                image_path=f"{frame_id}.png",
                base_dir="",
                height=None,
                width=None,
            )
            for frame_id in (
                9,
                10,
                11,
                12,
                23,
                24,
                25,
                28,
                29,
                30,
                31,
                32,
                33,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
            )
        ]

        windows = build_windows(frames, stride=5)

        self.assertEqual(
            [[frame.frame_id for frame in window.frames] for window in windows],
            [
                ["28", "29", "30", "31", "32"],
                ["29", "30", "31", "32", "33"],
                ["44", "45", "46", "47", "48"],
                ["46", "47", "48", "49", "50"],
            ],
        )

    def test_structurally_invalid_window_does_not_propagate_frame_semantics(self):
        class FakeTokenizer:
            def convert_ids_to_tokens(self, values):
                return [str(value) for value in values]

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [], "offset_mapping": []}

        frames = tuple(
            SourceFrame("VID79", str(index), f"{index}.png", "", 2, 2)
            for index in range(1, 6)
        )
        window = InferenceWindow(0, "VID79", frames)
        six_frames = "\n\n".join(
            f"Frame {index}:\n{VALID_TEXT}" for index in range(1, 7)
        )
        predictions, _ = _window_records(
            window,
            {
                "generated_text": six_frames,
                "generated_ids": [],
                "token_probabilities": [],
                "role_keys_per_frame": [[] for _ in range(5)],
                "binary_masks": [[] for _ in range(5)],
            },
            {
                "original_sizes": [(2, 2)] * 5,
                "image_references": [f"VID79/{index}.png" for index in range(1, 6)],
                "tile_counts": [1] * 5,
            },
            FakeTokenizer(),
        )
        self.assertTrue(all(record["generated_text"] is None for record in predictions))
        self.assertTrue(all(not record["parser_result"]["ok"] for record in predictions))
        self.assertTrue(all(record["phase"]["label"] is None for record in predictions))
        self.assertTrue(all(not record["triplets"] for record in predictions))

    def test_text_fallback_confidence_uses_final_beam_transition_scores(self):
        beam_indices = torch.tensor([[0, 1]])

        class FakeLanguageModel:
            def compute_transition_scores(
                self, sequences, scores, *, beam_indices, normalize_logits
            ):
                self.received = (sequences, scores, beam_indices, normalize_logits)
                return torch.log(torch.tensor([[0.25, 0.75]]))

        language_model = FakeLanguageModel()
        generated = SimpleNamespace(
            sequences=torch.tensor([[7, 8]]),
            scores=[torch.zeros(2, 10), torch.zeros(2, 10)],
            beam_indices=beam_indices,
        )
        probabilities = _transition_probabilities(
            SimpleNamespace(language_model=language_model),
            generated,
            torch.tensor([[7, 8]]),
        )
        self.assertEqual(probabilities, [0.25, 0.75])
        self.assertIs(language_model.received[2], beam_indices)
        self.assertTrue(language_model.received[3])

    def test_strict_parser_accepts_one_frame_and_rejects_tag_boundaries(self):
        parsed = parse_generated_frame(VALID_TEXT)
        self.assertTrue(parsed.ok, parsed.parse_error)
        self.assertEqual(parsed.phase, "preparation")
        self.assertEqual(parsed.declared_triplets, 1)
        self.assertEqual(
            parsed.triplets[0].as_tuple(), ("grasper", "retract", "gallbladder")
        )

        duplicate = parse_generated_frame(VALID_TEXT + VALID_TEXT)
        self.assertFalse(duplicate.ok)
        self.assertEqual(duplicate.issues[0].code, "tag_count")

        missing_marker = parse_generated_frame(
            VALID_TEXT.replace("</p>[SEG]", "</p>", 1)
        )
        self.assertFalse(missing_marker.ok)
        self.assertTrue(
            any(issue.code == "triplet_grammar" for issue in missing_marker.issues)
        )

        marker_in_think = parse_generated_frame(
            VALID_TEXT.replace("</think>", " <p>grasper</p>[SEG]</think>")
        )
        self.assertFalse(marker_in_think.ok)
        self.assertTrue(
            any(
                issue.code == "grounding_marker_in_think"
                for issue in marker_in_think.issues
            )
        )

        spaced = parse_generated_frame(
            VALID_TEXT.replace("<think>", "<think>\n")
            .replace("</think>", "\n</think>")
            .replace("<answer>", "<answer>\n")
            .replace("</answer>", "\n</answer>")
            .replace("<p>", "<p> ")
            .replace("</p>[SEG]", " </p> [SEG]")
        )
        self.assertTrue(spaced.ok, spaced.parse_error)

    def test_strict_window_requires_five_ordered_frame_blocks(self):
        text = "\n\n".join(
            f"Frame {index}:\n{VALID_TEXT}" for index in range(1, 6)
        )
        parsed = parse_generated_window(text)
        self.assertTrue(parsed.ok, parsed.parse_error)
        self.assertEqual(parsed.frame_numbers, (1, 2, 3, 4, 5))

        missing = parse_generated_window(text.rsplit("\n\n", 1)[0])
        self.assertFalse(missing.ok)
        self.assertTrue(any(issue.code == "frame_count" for issue in missing.issues))

        reordered = parse_generated_window(text.replace("Frame 4:", "Frame 3:"))
        self.assertFalse(reordered.ok)
        self.assertTrue(any(issue.code == "frame_number" for issue in reordered.issues))

    def test_header_and_field_groundings_keep_source_order_and_schema_slots(self):
        parsed = parse_generated_frame(HEADER_GROUNDED_TEXT)
        self.assertTrue(parsed.ok, parsed.parse_error)
        self.assertEqual(parsed.phase, "gallbladder dissection")
        self.assertEqual(
            [grounding.role_key for grounding in parsed.groundings],
            ["target:gallbladder", "instrument:hook"],
        )
        self.assertEqual(
            [grounding.triplet_index for grounding in parsed.groundings], [0, 0]
        )
        self.assertTrue(parsed.triplets[0].instrument_grounded)
        self.assertTrue(parsed.triplets[0].target_grounded)

    def test_nonfield_grounding_rejects_ambiguity_unknown_and_excess_markers(self):
        ambiguous = HEADER_GROUNDED_TEXT.replace(
            "the instrument is <p> hook </p> [SEG]",
            "the instrument is gallbladder",
        )
        parsed = parse_generated_frame(ambiguous)
        self.assertFalse(parsed.ok)
        self.assertTrue(
            any(issue.code == "ambiguous_grounding_role" for issue in parsed.issues),
            parsed.parse_error,
        )

        unknown = HEADER_GROUNDED_TEXT.replace(
            "the target is gallbladder", "the target is liver"
        )
        parsed = parse_generated_frame(unknown)
        self.assertFalse(parsed.ok)
        self.assertTrue(
            any(issue.code == "unknown_grounding_label" for issue in parsed.issues),
            parsed.parse_error,
        )

        excess = HEADER_GROUNDED_TEXT.replace(
            "the target is gallbladder",
            "the target is <p> gallbladder </p> [SEG]",
        )
        parsed = parse_generated_frame(excess)
        self.assertFalse(parsed.ok)
        self.assertTrue(
            any(issue.code == "excess_grounding" for issue in parsed.issues),
            parsed.parse_error,
        )

        stray = VALID_TEXT.replace(
            "the action is retract", "the action is retract[SEG]"
        )
        parsed = parse_generated_frame(stray)
        self.assertFalse(parsed.ok)
        self.assertTrue(
            any(issue.code == "invalid_grounding_marker" for issue in parsed.issues),
            parsed.parse_error,
        )

    def test_nonfield_marker_uses_first_unassigned_matching_occurrence(self):
        parsed = parse_generated_frame(REPEATED_TARGET_HEADER_TEXT)
        self.assertTrue(parsed.ok, parsed.parse_error)
        self.assertEqual(
            [(value.role_key, value.triplet_index) for value in parsed.groundings],
            [("target:gallbladder", 0)],
        )

        second_field = REPEATED_TARGET_HEADER_TEXT.replace(
            "During the <p> gallbladder </p> [SEG] dissection phase",
            "During the gallbladder dissection phase",
        ).replace(
            "(2) the instrument is grasper, the target is gallbladder",
            "(2) the instrument is grasper, the target is <p> gallbladder </p> [SEG]",
        )
        parsed = parse_generated_frame(second_field)
        self.assertTrue(parsed.ok, parsed.parse_error)
        self.assertEqual(
            [(value.role_key, value.triplet_index) for value in parsed.groundings],
            [("target:gallbladder", 1)],
        )

    def test_header_marker_mask_is_bound_to_matching_triplet_field(self):
        class FakeTokenizer:
            def convert_ids_to_tokens(self, values):
                return [str(value) for value in values]

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [], "offset_mapping": []}

        frames = tuple(
            SourceFrame("VID79", str(index), f"{index}.png", "", 2, 2)
            for index in range(1, 6)
        )
        window = InferenceWindow(0, "VID79", frames)
        generated = "\n\n".join(
            f"Frame {index}:\n{HEADER_GROUNDED_TEXT}" for index in range(1, 6)
        )
        target_mask = [[1, 0], [0, 0]]
        instrument_mask = [[0, 0], [0, 1]]
        predictions, _ = _window_records(
            window,
            {
                "generated_text": generated,
                "generated_ids": [],
                "token_probabilities": [],
                "role_keys_per_frame": [
                    ["target:gallbladder", "instrument:hook"] for _ in range(5)
                ],
                "grounding_assignments_per_frame": [
                    [
                        {
                            "role": "target",
                            "label": "gallbladder",
                            "triplet_index": 0,
                        },
                        {
                            "role": "instrument",
                            "label": "hook",
                            "triplet_index": 0,
                        },
                    ]
                    for _ in range(5)
                ],
                "binary_masks": [
                    [target_mask, instrument_mask] for _ in range(5)
                ],
            },
            {
                "original_sizes": [(2, 2)] * 5,
                "image_references": [f"VID79/{index}.png" for index in range(1, 6)],
                "tile_counts": [1] * 5,
            },
            FakeTokenizer(),
        )
        first = predictions[0]
        self.assertIsNone(first["parse_error"])
        self.assertEqual(
            first["triplets"][0]["target_mask"]["rle"],
            binary_mask_to_rle(target_mask),
        )
        self.assertEqual(
            first["triplets"][0]["instrument_mask"]["rle"],
            binary_mask_to_rle(instrument_mask),
        )

        reversed_result = {
            "generated_text": generated,
            "generated_ids": [],
            "token_probabilities": [],
            "role_keys_per_frame": [
                ["instrument:hook", "target:gallbladder"] for _ in range(5)
            ],
            "binary_masks": [[instrument_mask, target_mask] for _ in range(5)],
        }
        reversed_predictions, _ = _window_records(
            window,
            reversed_result,
            {
                "original_sizes": [(2, 2)] * 5,
                "image_references": [f"VID79/{index}.png" for index in range(1, 6)],
                "tile_counts": [1] * 5,
            },
            FakeTokenizer(),
        )
        self.assertIn("expected ordered", reversed_predictions[0]["parse_error"])
        self.assertFalse(reversed_predictions[0]["parser_result"]["ok"])

    def test_null_target_is_semantic_not_a_segmentation_entity(self):
        text = (
            "<think>No spatial target is active.</think>"
            "<answer>During the preparation phase, 1 surgical action triplet "
            "is identified: (1) the instrument is <p> grasper </p> [SEG], "
            "the target is null target, based on the two components, the action "
            "is null verb.</answer>"
        )
        parsed = parse_generated_frame(text)
        self.assertTrue(parsed.ok, parsed.parse_error)
        self.assertEqual(
            parsed.triplets[0].as_tuple(),
            ("grasper", "null verb", "null target"),
        )
        wrongly_grounded = parse_generated_frame(
            text.replace("null target,", "<p>null target</p>[SEG],")
        )
        self.assertFalse(wrongly_grounded.ok)

    def test_unique_grounded_occurrences_do_not_require_duplicate_masks(self):
        text = (
            "<think>The grasper retracts while one null relation is also active.</think>"
            "<answer>During the preparation phase, 2 surgical action triplets "
            "are identified: (1) the instrument is <p> grasper </p> [SEG], "
            "the target is <p> gallbladder </p> [SEG], based on the two components, "
            "the action is retract; (2) the instrument is grasper, the target is "
            "null target, based on the two components, the action is null verb.</answer>"
        )
        parsed = parse_generated_frame(text)
        self.assertTrue(parsed.ok, parsed.parse_error)
        self.assertTrue(parsed.triplets[0].instrument_grounded)
        self.assertTrue(parsed.triplets[0].target_grounded)
        self.assertFalse(parsed.triplets[1].instrument_grounded)
        self.assertFalse(parsed.triplets[1].target_grounded)

        orphan_seg = parse_generated_frame(
            text.replace("the instrument is grasper,", "the instrument is grasper[SEG],")
        )
        self.assertFalse(orphan_seg.ok)

    def test_generated_grounding_masks_bind_to_tagged_occurrences(self):
        mask = [[1, 0], [0, 0]]
        annotation = {
            "video_id": "VID79",
            "frame_id": "1",
            "phase": "preparation",
            "triplets": [
                {
                    "instrument": "grasper",
                    "verb": "retract",
                    "target": "gallbladder",
                    "instrument_mask": {"array": mask},
                    "target_mask": {"array": mask},
                }
            ],
        }
        prediction = {
            "video_id": "VID79",
            "frame_id": "1",
            "generated_text": VALID_TEXT,
            "phase": {"label": "preparation", "confidence": 1.0},
            "triplets": [
                {
                    "instrument": "grasper",
                    "verb": "retract",
                    "target": "gallbladder",
                    "confidence": 1.0,
                    "instrument_mask": {"array": mask},
                }
            ],
        }
        report = evaluate([prediction], [annotation])
        self.assertEqual(report["overall"]["invalid"]["invalid_masks"], 1)
        self.assertEqual(
            report["overall"]["invalid"]["issue:missing_target_mask"], 1
        )
        self.assertEqual(report["overall"]["grounding"]["num_pred_masks"], 1)
        self.assertEqual(report["overall"]["grounding"]["num_unmatched_gt"], 1)

    def test_phase_metrics_keep_video_boundaries(self):
        metrics = evaluate_phase_videos(
            {
                "VID01": (["preparation"], ["preparation"]),
                "VID02": (
                    ["preparation"] * 9,
                    ["calot triangle dissection"] * 9,
                ),
            }
        )
        self.assertAlmostEqual(metrics["video_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["frame_accuracy"], 0.1)
        self.assertAlmostEqual(metrics["macro_recall"], 0.5)
        self.assertEqual(metrics["protocol"], "unrelaxed_video_independent")

    def test_phase_macro_counts_prediction_only_class_as_false_positive(self):
        metrics = evaluate_phase_videos(
            {
                "VID01": (
                    ["preparation", "preparation"],
                    ["preparation", "gallbladder extraction"],
                )
            }
        )
        extraction = metrics["per_video"]["VID01"]["per_class"][
            "gallbladder extraction"
        ]
        self.assertEqual(extraction["precision"], 0.0)
        self.assertEqual(extraction["jaccard"], 0.0)
        self.assertIsNone(extraction["recall"])
        self.assertAlmostEqual(metrics["macro_precision"], 0.5)
        self.assertAlmostEqual(metrics["macro_jaccard"], 0.25)

    def test_hallucinated_triplet_is_counted_and_penalizes_ap(self):
        result = evaluate_triplet_ap(
            {
                "VID01": [
                    {
                        "gt_triplets": [("grasper", "retract", "gallbladder")],
                        "pred_triplets": [
                            {
                                "instrument": "grasper",
                                "verb": "retract",
                                "target": "gallbladder",
                                "confidence": 0.9,
                            },
                            {
                                "instrument": "grasper",
                                "verb": "cut",
                                "target": "liver",
                                "confidence": 0.8,
                            },
                        ],
                    }
                ]
            }
        )
        self.assertEqual(result["invalid_prediction_count"], 1)
        self.assertAlmostEqual(result["AP_IVT"], 1.0)
        self.assertAlmostEqual(result["invalid_penalty_factor"], 0.5)
        self.assertAlmostEqual(result["AP_IVT_penalized"], 0.5)

    def test_duplicate_valid_triplet_is_an_invalid_penalized_prediction(self):
        triplet = {
            "instrument": "grasper",
            "verb": "retract",
            "target": "gallbladder",
            "confidence": 0.9,
        }
        result = evaluate_triplet_ap(
            {
                "VID01": [
                    {
                        "gt_triplets": [("grasper", "retract", "gallbladder")],
                        "pred_triplets": [triplet, {**triplet, "confidence": 0.8}],
                    }
                ]
            }
        )
        self.assertEqual(result["valid_prediction_count"], 1)
        self.assertEqual(result["duplicate_prediction_count"], 1)
        self.assertEqual(result["invalid_prediction_count"], 1)
        self.assertAlmostEqual(result["invalid_penalty_factor"], 0.5)

    def test_missing_triplet_confidence_is_invalid(self):
        result = evaluate_triplet_ap(
            {
                "VID01": [
                    {
                        "gt_triplets": [("grasper", "retract", "gallbladder")],
                        "pred_triplets": [
                            {
                                "instrument": "grasper",
                                "verb": "retract",
                                "target": "gallbladder",
                            }
                        ],
                    }
                ]
            }
        )
        self.assertEqual(result["valid_prediction_count"], 0)
        self.assertEqual(result["invalid_prediction_count"], 1)
        self.assertEqual(result["invalid_penalty_factor"], 0.0)

    def test_sparse_ap_matches_dense_zero_and_score_tie_protocol(self):
        self.assertAlmostEqual(
            sparse_average_precision(
                {0, 1}, [(1.0, 0)], total_frames=3
            ),
            5 / 6,
        )
        self.assertAlmostEqual(
            sparse_average_precision(
                {0, 2}, [(0.5, 0), (0.5, 1)], total_frames=4
            ),
            0.5,
        )

    def test_grounding_requires_role_and_label_and_unmatched_gt_is_zero(self):
        gt = [{"role": "instrument", "label": "grasper", "mask": [[1, 0]]}]
        wrong_role = [{"role": "target", "label": "grasper", "mask": [[1, 0]]}]
        result = match_grounding_instances(gt, wrong_role)
        self.assertEqual(result["mIoU"], 0.0)
        self.assertEqual(result["gt_ious"], [0.0])
        self.assertEqual(result["num_unmatched_gt"], 1)

    def test_bad_prediction_mask_shape_is_recorded_and_scores_zero(self):
        gt = [{"role": "instrument", "label": "grasper", "mask": [[1, 0]]}]
        malformed = [
            {"role": "instrument", "label": "grasper", "mask": [[1], [0]]}
        ]
        result = match_grounding_instances(gt, malformed)
        self.assertEqual(result["mIoU"], 0.0)
        self.assertEqual(result["num_unmatched_gt"], 1)
        self.assertEqual(result["invalid_prediction_mask_count"], 1)

    def test_unmatched_bad_shape_mask_is_recorded_and_pooled(self):
        frames = [
            {
                "video_id": "VID01",
                "frame_id": "1",
                "gt_groundings": [
                    {
                        "role": "instrument",
                        "label": "grasper",
                        "mask": [[1, 0], [0, 0]],
                    }
                ],
                "pred_groundings": [
                    {"role": "instrument", "label": "hook", "mask": [[1]]}
                ],
            }
        ]
        result = evaluate_grounding_frames(frames)
        self.assertEqual(result["invalid_prediction_mask_count"], 1)
        self.assertEqual(result["num_unmatched_pred"], 1)
        self.assertEqual(result["frames"][0]["num_unmatched_pred"], 1)
        self.assertEqual(
            result["frames"][0]["invalid_prediction_mask_count"], 1
        )

        generated_text = HEADER_GROUNDED_TEXT.replace(
            "During the <p> gallbladder </p> [SEG] dissection phase",
            "During the gallbladder dissection phase",
        )
        report = evaluate(
            [
                {
                    "video_id": "VID79",
                    "frame_id": "1",
                    "generated_text": generated_text,
                    "phase": {
                        "label": "gallbladder dissection",
                        "confidence": 1.0,
                    },
                    "triplets": [
                        {
                            "instrument": "hook",
                            "verb": "dissect",
                            "target": "gallbladder",
                            "confidence": 1.0,
                            "instrument_mask": {"array": [[1]]},
                        }
                    ],
                }
            ],
            [
                {
                    "video_id": "VID79",
                    "frame_id": "1",
                    "phase": "gallbladder dissection",
                    "triplets": [
                        {
                            "instrument": "grasper",
                            "verb": "retract",
                            "target": "gallbladder",
                            "instrument_mask": {
                                "array": [[1, 0], [0, 0]]
                            },
                        }
                    ],
                }
            ],
        )
        grounding = report["overall"]["grounding"]
        self.assertEqual(grounding["invalid_prediction_mask_count"], 1)
        self.assertEqual(grounding["num_unmatched_pred"], 1)
        self.assertEqual(
            report["overall"]["invalid"]["issue:incompatible_mask_shape"],
            1,
        )

    def test_phase_confidence_must_be_finite_and_bounded(self):
        zero_triplet_text = (
            "<think>No surgical action is present.</think>"
            "<answer>During the preparation phase, 0 surgical action triplets "
            "are identified: none.</answer>"
        )
        annotation = {
            "video_id": "VID79",
            "frame_id": "1",
            "phase": "preparation",
            "triplets": [],
        }
        for confidence in ("bad", float("nan"), -1.0, 2.0, None):
            with self.subTest(confidence=confidence):
                prediction = {
                    "video_id": "VID79",
                    "frame_id": "1",
                    "generated_text": zero_triplet_text,
                    "phase": {
                        "label": "preparation",
                        "confidence": confidence,
                    },
                    "triplets": [],
                }
                invalid = evaluate([prediction], [annotation])["overall"]["invalid"]
                self.assertEqual(invalid["issue:invalid_phase_confidence"], 1)

        for confidence in (0.0, 1.0):
            with self.subTest(valid_confidence=confidence):
                prediction = {
                    "video_id": "VID79",
                    "frame_id": "1",
                    "generated_text": zero_triplet_text,
                    "phase": {
                        "label": "preparation",
                        "confidence": confidence,
                    },
                    "triplets": [],
                }
                invalid = evaluate([prediction], [annotation])["overall"]["invalid"]
                self.assertNotIn("issue:invalid_phase_confidence", invalid)

    def test_structured_fields_cannot_replace_missing_generated_text(self):
        mask = [[1]]
        triplet = {
            "instrument": "grasper",
            "verb": "retract",
            "target": "gallbladder",
            "instrument_mask": {"array": mask},
            "target_mask": {"array": mask},
        }
        annotation = {
            "video_id": "VID79",
            "frame_id": "1",
            "phase": "preparation",
            "triplets": [triplet],
        }
        prediction = {
            "video_id": "VID79",
            "frame_id": "1",
            "phase": {"label": "preparation", "confidence": 1.0},
            "triplets": [triplet],
        }
        result = evaluate([prediction], [annotation])["overall"]
        self.assertEqual(result["phase"]["video_accuracy"], 0.0)
        self.assertEqual(result["grounding"]["mIoU"], 0.0)
        self.assertEqual(result["invalid"]["issue:missing_generated_text"], 1)

    def test_grounding_miou_is_pooled_over_all_gt_masks(self):
        frames = [
            {
                "video_id": "VID01",
                "frame_id": "1",
                "gt_groundings": [
                    {"role": "instrument", "label": "grasper", "mask": [[1]]}
                ],
                "pred_groundings": [
                    {"role": "instrument", "label": "grasper", "mask": [[1]]}
                ],
            },
            {
                "video_id": "VID01",
                "frame_id": "2",
                "gt_groundings": [
                    {"role": "instrument", "label": "grasper", "mask": [[1]]},
                    {"role": "instrument", "label": "hook", "mask": [[1]]},
                    {"role": "target", "label": "gallbladder", "mask": [[1]]},
                ],
                "pred_groundings": [],
            },
        ]
        result = evaluate_grounding_frames(frames)
        self.assertAlmostEqual(result["mIoU"], 0.25)
        self.assertAlmostEqual(result["IoU_I"], 1 / 3)
        self.assertEqual(result["IoU_T"], 0.0)
        self.assertEqual(result["num_gt_masks"], 4)
        self.assertEqual(result["num_unmatched_gt"], 3)

    def test_end_to_end_coverage_missing_frame_and_json_outputs(self):
        mask = [[1, 0], [0, 0]]
        predictions = [
            {
                "video_id": "VID79",
                "frame_id": "000001",
                "generated_text": VALID_TEXT,
                "phase": {"label": "preparation", "confidence": 0.9},
                "triplets": [
                    {
                        "instrument": "grasper",
                        "verb": "retract",
                        "target": "gallbladder",
                        "confidence": 0.8,
                        "instrument_mask": {"array": mask},
                        "target_mask": {"array": mask},
                    }
                ],
            }
        ]
        annotation_triplet = {
            "instrument": "grasper",
            "verb": "retract",
            "target": "gallbladder",
            "instrument_mask": {"array": mask},
            "target_mask": {"array": mask},
        }
        annotations = [
            {
                "video_id": "VID79",
                "frame_id": "1",
                "phase": "preparation",
                "triplets": [annotation_triplet],
            },
            {
                "video_id": "VID79",
                "frame_id": "2",
                "phase": "preparation",
                "triplets": [annotation_triplet],
            },
        ]

        report = evaluate(predictions, annotations)
        self.assertAlmostEqual(report["overall"]["coverage"]["coverage"], 0.5)
        self.assertEqual(report["overall"]["coverage"]["missing_frames"], 1)
        self.assertAlmostEqual(report["overall"]["phase"]["video_accuracy"], 0.5)
        self.assertAlmostEqual(report["overall"]["grounding"]["mIoU"], 0.5)

        with tempfile.TemporaryDirectory() as directory:
            paths = write_report(report, directory)
            self.assertEqual(set(paths), {"combined", "overall", "per_video"})
            payload = json.loads(
                (Path(directory) / "overall.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["overall"]["grounding"]["num_gt_masks"], 4)


if __name__ == "__main__":
    unittest.main()
