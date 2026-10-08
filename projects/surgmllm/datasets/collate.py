"""MMEngine/XTuner-compatible collation for surgical video samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .gcg_video import IGNORE_INDEX, WINDOW_SIZE


def _long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)


def _frame_tile_batches(value: Any) -> list[torch.Tensor]:
    if isinstance(value, list):
        batches = value
    elif isinstance(value, torch.Tensor) and value.ndim == 5:
        batches = list(value)
    elif isinstance(value, torch.Tensor) and value.ndim == 4:
        # A legacy one-tile-per-frame tensor has [frames, C, H, W].
        batches = [frame.unsqueeze(0) for frame in value]
    else:
        raise ValueError("pixel_values must preserve a tile batch for every frame.")
    normalized = []
    for batch in batches:
        tensor = batch if isinstance(batch, torch.Tensor) else torch.as_tensor(batch)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"A frame tile batch must be 4-D, got {tensor.shape}.")
        normalized.append(tensor)
    return normalized


def surgmllm_collate_fn(
    instances: Sequence[Mapping[str, Any]],
    pad_index: int = 0,
    ignore_index: int = IGNORE_INDEX,
    return_hf_format: bool = False,
) -> dict[str, Any]:
    """Pad language tensors and preserve variable tiles/masks as ordered lists."""

    if not instances:
        raise ValueError("Cannot collate an empty batch.")
    input_ids = [_long_tensor(instance["input_ids"]) for instance in instances]
    labels = [_long_tensor(instance["labels"]) for instance in instances]
    lengths = [int(value.numel()) for value in input_ids]
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=ignore_index)
    attention_mask = torch.zeros_like(padded_ids, dtype=torch.bool)
    for batch_index, length in enumerate(lengths):
        attention_mask[batch_index, :length] = True
    position_ids = torch.arange(padded_ids.shape[1], dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(len(instances), -1).clone()

    pixel_values: list[torch.Tensor] = []
    grounding_pixels: list[torch.Tensor] = []
    frames_per_batch: list[int] = []
    masks_batch: list[torch.Tensor] = []
    role_keys_batch: list[list[list[str]]] = []
    counts_batch: list[list[int]] = []
    ranges_batch: list[dict[str, list[list[int]]]] = []
    for instance in instances:
        frame_tiles = _frame_tile_batches(instance["pixel_values"])
        frame_count = int(instance.get("frames_per_sample", len(frame_tiles)))
        if frame_count != WINDOW_SIZE or len(frame_tiles) != WINDOW_SIZE:
            raise ValueError("Every collated training sample must contain five frames.")
        frame_grounding = list(instance["g_pixel_values"])
        if len(frame_grounding) != frame_count:
            raise ValueError("g_pixel_values must contain one tensor per frame.")
        role_keys = [list(keys) for keys in instance["role_keys_per_frame"]]
        counts = [int(value) for value in instance["num_segs_per_frame"]]
        masks = instance["masks"]
        masks = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
        expected = sum(counts)
        if (
            len(role_keys) != frame_count
            or counts != [len(keys) for keys in role_keys]
            or len(masks) != expected
        ):
            raise ValueError("[SEG], role-key, and mask counts disagree during collation.")

        pixel_values.extend(frame_tiles)
        grounding_pixels.extend(frame_grounding)
        frames_per_batch.append(frame_count)
        masks_batch.append(masks)
        role_keys_batch.append(role_keys)
        counts_batch.append(counts)
        ranges_batch.append(dict(instance.get("entity_token_ranges", {})))

    data = {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": padded_labels,
        "pixel_values": pixel_values,
        "g_pixel_values": grounding_pixels,
        "frames_per_batch": frames_per_batch,
        "masks": masks_batch,
        "role_keys_per_frame": role_keys_batch,
        "num_segs_per_frame": counts_batch,
        "entity_token_ranges": ranges_batch,
    }
    metadata = [
        {
            "video_id": instance.get("video_id"),
            "frame_id": list(instance.get("frame_ids", instance.get("frame_id", []))),
            "frame_metadata": list(instance.get("frame_metadata", [])),
            "mask_role_keys": list(instance.get("mask_role_keys", [])),
            "frame_labels": list(instance.get("frame_labels", [])),
            "groundings_per_frame": list(instance.get("groundings_per_frame", [])),
            "entity_range_labels": dict(instance.get("entity_range_labels", {})),
            "num_patches_per_frame": list(instance.get("num_patches_per_frame", [])),
            "num_image_tokens_per_frame": list(
                instance.get("num_image_tokens_per_frame", [])
            ),
            "num_visual_tiles": int(instance.get("num_visual_tiles", 0)),
            "num_image_tokens": int(instance.get("num_image_tokens", 0)),
        }
        for instance in instances
    ]
    if return_hf_format:
        return {**data, "data_samples": metadata}
    return {"data": data, "data_samples": metadata}


class SurgMLLMCollator:
    """Config-buildable callable wrapper around :func:`surgmllm_collate_fn`."""

    def __init__(
        self,
        pad_index: int = 0,
        ignore_index: int = IGNORE_INDEX,
        return_hf_format: bool = False,
    ) -> None:
        self.pad_index = int(pad_index)
        self.ignore_index = int(ignore_index)
        self.return_hf_format = bool(return_hf_format)

    def __call__(self, instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return surgmllm_collate_fn(
            instances,
            pad_index=self.pad_index,
            ignore_index=self.ignore_index,
            return_hf_format=self.return_hf_format,
        )


# Common config spellings.
collate_fn = surgmllm_collate_fn
SurgMLLMDataCollator = SurgMLLMCollator
