from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from projects.surgmllm.datasets import (
    ENTITY_ROLES,
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    InternVLDynamicProcessor,
    ProtocolError,
    SurgMLLMGCGVideoDataset,
    build_canonical_frame,
    normalize_label,
    surgmllm_collate_fn,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_JSON = REPO_ROOT / "examples" / "synthetic_5frame.json"


class FakeTokenizer:
    """Greedy special-token tokenizer with exact character offsets."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        visual_tokens = ["<IMG_CONTEXT>", "<img>", "</img>"]
        self._ids = {token: 100 + index for index, token in enumerate(visual_tokens)}
        self._tokens = {value: key for key, value in self._ids.items()}

    def add_tokens(self, tokens, special_tokens=False):
        del special_tokens
        added = 0
        for token in tokens:
            if token not in self._ids:
                token_id = 100 + len(self._ids)
                self._ids[token] = token_id
                self._tokens[token_id] = token
                added += 1
        return added

    def _tokenize(self, text: str):
        ids, offsets = [], []
        cursor = 0
        special_tokens = sorted(self._ids, key=len, reverse=True)
        while cursor < len(text):
            token = next(
                (value for value in special_tokens if text.startswith(value, cursor)),
                None,
            )
            if token is None:
                ids.append(1000 + ord(text[cursor]))
                offsets.append((cursor, cursor + 1))
                cursor += 1
            else:
                ids.append(self._ids[token])
                offsets.append((cursor, cursor + len(token)))
                cursor += len(token)
        return ids, offsets

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._tokenize(text)[0]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        ids, offsets = self._tokenize(text)
        output = {"input_ids": ids}
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(
            self._tokens.get(int(token_id), chr(int(token_id) - 1000))
            for token_id in token_ids
        )


@pytest.fixture(scope="module")
def tokenizer():
    return FakeTokenizer()


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return SurgMLLMGCGVideoDataset(
        annotation_file=SYNTHETIC_JSON,
        tokenizer=tokenizer,
        image_loader=lambda _: Image.new("RGB", (4, 4), color=(20, 40, 60)),
        max_length=4096,
    )


@pytest.fixture(scope="module")
def sample(dataset):
    return dataset[0]


def test_exactly_seven_structure_tokens():
    assert NUM_SPECIAL_TOKENS == 7
    assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)) == 7
    assert SPECIAL_TOKENS == (
        "[SEG]",
        "<p>",
        "</p>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    )


def test_original_caption_offsets_are_tagged_before_frame_wrapping():
    annotation = json.loads(SYNTHETIC_JSON.read_text(encoding="utf-8"))[0]
    frame = build_canonical_frame(
        annotation["caption"],
        annotation["groundings"],
        annotation["labels"],
    )
    assert frame.text == annotation["expected_canonical_text"]
    assert {
        grounding["mask_path"] for grounding in annotation["groundings"].values()
    } == {
        "synthetic_masks/frame_000_instrument.json",
        "synthetic_masks/frame_000_target.json",
    }
    think = frame.text.split("<think>\n", 1)[1].split("\n</think>", 1)[0]
    answer = frame.text.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]
    assert "<p>" not in think and "[SEG]" not in think
    assert think.count("grasper") == think.count("gallbladder") == 1
    assert answer.count("<p>") == answer.count("[SEG]") == 2
    assert "<p> grasper </p> [SEG]" in answer
    assert "<p> gallbladder </p> [SEG]" in answer


def test_grounding_inside_think_is_rejected():
    annotation = json.loads(SYNTHETIC_JSON.read_text(encoding="utf-8"))[0]
    invalid = copy.deepcopy(annotation["groundings"])
    invalid["instrument:grasper"]["token_positives"] = [11, 18]
    with pytest.raises(ProtocolError, match="not wholly inside"):
        build_canonical_frame(annotation["caption"], invalid, annotation["labels"])


@pytest.mark.parametrize(
    ("phase_label", "verb_label"),
    (
        ("gallbladder_dissection", "dissect"),
        ("gallbladder_packaging", "pack"),
    ),
)
def test_phase_label_remains_strict_when_grounding_tags_split_its_words(
    phase_label,
    verb_label,
):
    phase_text = phase_label.replace("_", " ")
    caption = (
        "<think> The grasper acts on the gallbladder. </think> "
        f"<answer> During the {phase_text} phase, 1 surgical action triplet is "
        "identified: (1) the instrument is grasper, the target is gallbladder, "
        f"based on the two components, the action is {verb_label}. </answer>"
    )
    answer_start = caption.index("<answer>")
    target_start = caption.index("gallbladder", answer_start)
    instrument_start = caption.index("grasper", answer_start)
    frame = build_canonical_frame(
        caption,
        {
            "target:gallbladder": {
                "token_positives": [target_start, target_start + len("gallbladder")]
            },
            "instrument:grasper": {
                "token_positives": [
                    instrument_start,
                    instrument_start + len("grasper"),
                ]
            },
        },
        {
            "instrument": ["grasper"],
            "verb": [verb_label],
            "target": ["gallbladder"],
            "phase": [phase_label],
        },
    )

    # The source grounding offset is authoritative and is not silently moved to
    # the later action-triplet mention.
    target = next(
        item
        for item in frame.groundings
        if item.role_key == "target:gallbladder"
    )
    assert (target.start, target.end) == (
        target_start,
        target_start + len("gallbladder"),
    )
    assert "<p> gallbladder </p> [SEG]" in frame.text

    phase_spans = frame.entity_char_spans["phase"]
    phase_words = [frame.text[span.start : span.end] for span in phase_spans]
    assert phase_words == ["gallbladder", phase_label.rsplit("_", 1)[1]]
    assert "_".join(normalize_label(word) for word in phase_words) == phase_label
    assert all(
        token not in frame.text[span.start : span.end]
        for span in phase_spans
        for token in ("<p>", "</p>", "[SEG]")
    )


def test_tag_aware_phase_matching_does_not_accept_an_incomplete_phase_name():
    caption = (
        "<think> The grasper acts. </think> "
        "<answer> During the dissection phase, 1 surgical action triplet is "
        "identified: (1) the instrument is grasper, the target is gallbladder, "
        "based on the two components, the action is dissect. </answer>"
    )
    answer_start = caption.index("<answer>")
    instrument_start = caption.index("grasper", answer_start)
    target_start = caption.index("gallbladder", answer_start)
    with pytest.raises(
        ProtocolError,
        match="phase label 'gallbladder_dissection' is not present",
    ):
        build_canonical_frame(
            caption,
            {
                "instrument:grasper": {
                    "token_positives": [
                        instrument_start,
                        instrument_start + len("grasper"),
                    ]
                },
                "target:gallbladder": {
                    "token_positives": [
                        target_start,
                        target_start + len("gallbladder"),
                    ]
                },
            },
            {
                "instrument": ["grasper"],
                "verb": ["dissect"],
                "target": ["gallbladder"],
                "phase": ["gallbladder_dissection"],
            },
        )


def test_dataset_has_one_strict_five_frame_sample(dataset, sample, tokenizer):
    assert len(dataset) == dataset.real_len() == 1
    assert dataset.sliding_windows == [("SYNTHETIC_VIDEO", 0)]
    assert sample["video_id"] == "SYNTHETIC_VIDEO"
    assert sample["frame_ids"] == [f"{index:06d}" for index in range(5)]
    assert sample["frame_id"] == sample["frame_ids"]
    assert sample["frame_metadata"] == [
        {"video_id": "SYNTHETIC_VIDEO", "frame_id": f"{index:06d}"}
        for index in range(5)
    ]

    assert sample["num_segs_per_frame"] == [2] * 5
    expected_keys = ["instrument:grasper", "target:gallbladder"]
    assert sample["role_keys_per_frame"] == [expected_keys] * 5
    assert sample["mask_role_keys"] == expected_keys * 5
    assert all("verb:" not in key for key in sample["mask_role_keys"])
    assert all(labels == {
        "instrument": ["grasper"],
        "verb": ["retract"],
        "target": ["gallbladder"],
        "phase": ["preparation"],
    } for labels in sample["frame_labels"])

    assert sample["masks"].dtype == torch.bool
    assert sample["masks"].shape == (10, 4, 4)
    # Spatial overlaps remain present in two independent binary channels.
    assert sample["masks"][0, 1, 1] and sample["masks"][1, 1, 1]
    assert sample["answer_text"].count("[SEG]") == sample["masks"].shape[0]

    assert len(sample["g_pixel_values"]) == 5
    assert all(value.shape == (3, 1024, 1024) for value in sample["g_pixel_values"])
    assert len(sample["pixel_values"]) == 5
    assert all(value.shape == (1, 3, 448, 448) for value in sample["pixel_values"])
    assert sample["num_patches_per_frame"] == [1] * 5
    assert sample["num_image_tokens_per_frame"] == [256] * 5
    assert sample["prompt_text"].count("<IMG_CONTEXT>") == 5 * 256
    assert sample["prompt_text"].count("<img>") == 5

    for role in ENTITY_ROLES:
        assert len(sample["entity_token_ranges"][role]) == 5
        for span, expected_label in zip(
            sample["entity_token_ranges"][role],
            sample["entity_range_labels"][role],
        ):
            start, end = span
            decoded = tokenizer.decode(sample["input_ids"][start:end].tolist())
            assert normalize_label(decoded) == expected_label
            assert torch.equal(
                sample["input_ids"][start:end], sample["labels"][start:end]
            )


def test_stride_selects_contiguous_annotation_windows(tokenizer):
    annotations = json.loads(SYNTHETIC_JSON.read_text(encoding="utf-8"))
    for frame_index in (5, 6):
        frame = copy.deepcopy(annotations[-1])
        frame["frame_id"] = f"{frame_index:06d}"
        frame["file_name"] = f"synthetic_video/frame_{frame_index:03d}.png"
        annotations.append(frame)

    dataset = SurgMLLMGCGVideoDataset(
        annotations=annotations,
        tokenizer=tokenizer,
        stride=2,
        image_loader=lambda _: torch.zeros(3, 4, 4, dtype=torch.uint8),
        image_processor=lambda _: torch.zeros(1, 3, 8, 8),
        image_tokens_per_tile=1,
        grounding_image_size=8,
        max_length=4096,
    )
    assert dataset.sliding_windows == [
        ("SYNTHETIC_VIDEO", 0),
        ("SYNTHETIC_VIDEO", 2),
    ]
    assert dataset[0]["frame_ids"] == [f"{index:06d}" for index in range(5)]
    assert dataset[1]["frame_ids"] == [f"{index:06d}" for index in range(2, 7)]


def test_dataset_rejects_unknown_configuration_keys(tokenizer):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SurgMLLMGCGVideoDataset(
            annotations=[],
            tokenizer=tokenizer,
            legacy_experiment_switch=True,
        )


def test_dataset_rejects_conflicting_record_identities(tokenizer):
    annotation = copy.deepcopy(
        json.loads(SYNTHETIC_JSON.read_text(encoding="utf-8"))[0]
    )
    annotation["video_id"] = "VID01"
    annotation["image_id"] = "VID02_000000"
    with pytest.raises(ProtocolError, match="Conflicting frame video identities"):
        SurgMLLMGCGVideoDataset(
            annotations=[annotation],
            tokenizer=tokenizer,
            image_loader=lambda _: Image.new("RGB", (4, 4)),
        )


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/external/root/SYNTHETIC_VIDEO/frame_000.png",
        "../SYNTHETIC_VIDEO/frame_000.png",
    ),
)
def test_dataset_rejects_images_outside_image_root(tokenizer, unsafe_path):
    annotations = json.loads(SYNTHETIC_JSON.read_text(encoding="utf-8"))
    annotations[0]["file_name"] = unsafe_path
    dataset = SurgMLLMGCGVideoDataset(
        annotations=annotations,
        tokenizer=tokenizer,
        image_loader=lambda _: Image.new("RGB", (4, 4)),
        max_length=4096,
    )
    with pytest.raises(ProtocolError, match="relative to image_root"):
        dataset[0]


def test_windows_never_bridge_a_missing_source_frame(tokenizer):
    annotations = json.loads(SYNTHETIC_JSON.read_text(encoding="utf-8"))
    annotations[-1]["frame_id"] = "000010"
    annotations[-1]["file_name"] = "synthetic_video/frame_010.png"
    dataset = SurgMLLMGCGVideoDataset(
        annotations=annotations,
        tokenizer=tokenizer,
        image_loader=lambda _: torch.zeros(3, 4, 4, dtype=torch.uint8),
        image_processor=lambda _: torch.zeros(1, 3, 8, 8),
        image_tokens_per_tile=1,
        grounding_image_size=8,
        max_length=4096,
    )
    assert dataset.real_len() == 0


def test_dynamic_processor_tiles_non_square_image():
    processor = InternVLDynamicProcessor()
    tiles = processor(Image.new("RGB", (1200, 300), color=(10, 20, 30)))
    assert processor.image_size == 448
    assert processor.max_dynamic_patch == 12
    assert processor.use_thumbnail is True
    assert processor.num_image_tokens_per_tile == 256
    assert tiles.shape[0] > 1
    assert tiles.shape[1:] == (3, 448, 448)


def test_collate_matches_training_model_contract(sample):
    batch = surgmllm_collate_fn([sample, sample])
    data = batch["data"]
    assert data["input_ids"].shape[0] == data["labels"].shape[0] == 2
    assert data["attention_mask"].dtype == torch.bool
    assert data["frames_per_batch"] == [5, 5]
    assert len(data["pixel_values"]) == 10
    assert len(data["g_pixel_values"]) == 10
    assert len(data["masks"]) == 2
    assert data["role_keys_per_frame"] == [sample["role_keys_per_frame"]] * 2
    assert data["num_segs_per_frame"] == [[2] * 5, [2] * 5]
    assert data["entity_token_ranges"] == [sample["entity_token_ranges"]] * 2

    assert "mask_role_keys" not in data
    assert batch["data_samples"][0]["mask_role_keys"] == sample["mask_role_keys"]
    assert batch["data_samples"][0]["video_id"] == "SYNTHETIC_VIDEO"
    assert batch["data_samples"][0]["frame_id"] == sample["frame_ids"]
