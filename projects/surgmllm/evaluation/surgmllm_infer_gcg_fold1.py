"""Distributed five-frame Fold-1 inference with atomic raw-JSON merging."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as torch_functional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from projects.surgmllm.datasets.gcg_video import (  # type: ignore
        DEFAULT_QUESTION,
        IMG_CONTEXT_TOKEN,
        IMG_END_TOKEN,
        IMG_START_TOKEN,
    )
    from projects.surgmllm.datasets.image_processing import (  # type: ignore
        InternVLDynamicProcessor,
    )
    from projects.surgmllm.evaluation.io import (  # type: ignore
        FOLD1_VIDEO_IDS,
        frame_key,
        load_annotation_records,
        load_split_annotations,
    )
    from projects.surgmllm.evaluation.parser import (  # type: ignore
        FrameParseResult,
        WindowParseResult,
        parse_generated_window,
    )
    from projects.surgmllm.evaluation.taxonomy import normalize_label  # type: ignore
    from projects.surgmllm.record_identity import normalize_video_id  # type: ignore
else:
    from ..datasets.gcg_video import (
        DEFAULT_QUESTION,
        IMG_CONTEXT_TOKEN,
        IMG_END_TOKEN,
        IMG_START_TOKEN,
    )
    from ..datasets.image_processing import InternVLDynamicProcessor
    from .io import (
        FOLD1_VIDEO_IDS,
        frame_key,
        load_annotation_records,
        load_split_annotations,
    )
    from .parser import FrameParseResult, WindowParseResult, parse_generated_window
    from .taxonomy import normalize_label
    from ..record_identity import normalize_video_id


SCHEMA_VERSION = "surgmllm.raw-predictions.v1"
WINDOW_SIZE = 5


@dataclass(frozen=True)
class SourceFrame:
    video_id: str
    frame_id: str
    image_path: str
    base_dir: str
    height: int | None
    width: int | None

    @property
    def key(self) -> tuple[str, str]:
        return self.video_id, self.frame_id


@dataclass(frozen=True)
class InferenceWindow:
    index: int
    video_id: str
    frames: tuple[SourceFrame, ...]

    @property
    def window_id(self) -> str:
        return f"{self.video_id}:{self.frames[0].frame_id}"


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized_here: bool


def _frame_sort_key(frame: SourceFrame) -> tuple[int, int | str]:
    return (0, int(frame.frame_id)) if frame.frame_id.isdigit() else (1, frame.frame_id)


def _source_frames(records: Sequence[Mapping[str, Any]]) -> list[SourceFrame]:
    output: list[SourceFrame] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        video_id, frame_id = frame_key(record)
        key = (video_id, frame_id)
        if not video_id or not frame_id:
            raise ValueError(f"annotation has an empty frame identity: {record!r}")
        if key in seen:
            raise ValueError(f"duplicate annotation frame: {video_id}/{frame_id}")
        seen.add(key)
        image_path = record.get("image_path")
        if not image_path:
            image_path = f"{video_id}/{int(frame_id):06d}.png" if frame_id.isdigit() else frame_id
        output.append(
            SourceFrame(
                video_id=video_id,
                frame_id=frame_id,
                image_path=str(image_path),
                base_dir=str(record.get("_base_dir", "")),
                height=int(record["height"]) if record.get("height") is not None else None,
                width=int(record["width"]) if record.get("width") is not None else None,
            )
        )
    return output


def build_windows(
    frames: Sequence[SourceFrame], *, window_size: int = WINDOW_SIZE, stride: int = 5
) -> list[InferenceWindow]:
    if window_size != WINDOW_SIZE:
        raise ValueError(f"the trained generation protocol requires window_size={WINDOW_SIZE}")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("window stride must be a positive integer")
    grouped: dict[str, list[SourceFrame]] = defaultdict(list)
    video_order: list[str] = []
    for frame in frames:
        if frame.video_id not in grouped:
            video_order.append(frame.video_id)
        grouped[frame.video_id].append(frame)
    windows: list[InferenceWindow] = []
    for video_id in video_order:
        video_frames = sorted(grouped[video_id], key=_frame_sort_key)
        contiguous_runs: list[list[SourceFrame]] = []
        current_run: list[SourceFrame] = []
        previous_number: int | None = None
        for frame in video_frames:
            try:
                frame_number = int(frame.frame_id)
            except ValueError:
                # The temporal protocol is defined over consecutive numeric
                # frame IDs.  An unresolvable identity remains visible as
                # missing coverage and also splits the current temporal run.
                if current_run:
                    contiguous_runs.append(current_run)
                current_run = []
                previous_number = None
                continue
            if previous_number is not None and frame_number != previous_number + 1:
                contiguous_runs.append(current_run)
                current_run = []
            current_run.append(frame)
            previous_number = frame_number
        if current_run:
            contiguous_runs.append(current_run)

        # Apply stride independently inside each source-contiguous run.  Doing
        # this on the pre-split annotation list would let an earlier gap shift
        # the stride phase and silently omit later valid windows.
        for run in contiguous_runs:
            starts = list(range(0, len(run) - window_size + 1, stride))
            if starts:
                tail_start = len(run) - window_size
                if tail_start not in starts:
                    starts.append(tail_start)
            for start in starts:
                selected = tuple(run[start : start + window_size])
                windows.append(InferenceWindow(len(windows), video_id, selected))
    return windows


def _init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    if dist.is_initialized():
        return DistributedContext(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=local_rank,
            device=device,
            initialized_here=initialized_here,
        )
    return DistributedContext(0, 1, local_rank, device, initialized_here)


def _broadcast_run_id(context: DistributedContext) -> str:
    values: list[object] = [uuid.uuid4().hex if context.rank == 0 else None]
    if context.world_size > 1:
        dist.broadcast_object_list(values, src=0)
    return str(values[0])


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_image_path(data_root: Path, frame: SourceFrame) -> Path:
    relative = Path(frame.image_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Inference image paths must stay relative to data_root")
    if len(relative.parts) >= 2 and normalize_video_id(relative.parts[0]) != frame.video_id:
        raise ValueError(
            f"Image path video {relative.parts[0]!r} disagrees with {frame.video_id}"
        )
    root = data_root.expanduser().resolve()
    videos_root = (root / "videos").resolve()
    candidates = [(root / relative, root), (videos_root / relative, videos_root)]
    if len(relative.parts) == 1:
        candidates.append((videos_root / frame.video_id / relative, videos_root))
    for candidate, trusted_root in candidates:
        candidate = candidate.resolve()
        try:
            candidate.relative_to(trusted_root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(
        f"image for {frame.video_id}/{frame.frame_id} not found; tried {tried}"
    )


def _portable_image_reference(frame: SourceFrame) -> str:
    """Return a stable logical image identifier."""

    filename = Path(frame.image_path).name
    if not filename:
        filename = (
            f"{int(frame.frame_id):06d}.png"
            if frame.frame_id.isdigit()
            else frame.frame_id
        )
    return f"{frame.video_id}/{filename}"


def _portable_model_reference(value: str) -> str:
    """Preserve a Hub repository id and shorten local paths to a basename."""

    text = str(value).strip()
    path = Path(text).expanduser()
    return path.name if path.is_absolute() or path.exists() else text


def _grounding_tensor(image: Any, size: int) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    resized = torch_functional.interpolate(
        tensor.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized.round().clamp(0, 255).to(torch.uint8)


def _expanded_question(tile_counts: Sequence[int], tokens_per_tile: int) -> str:
    if len(tile_counts) != WINDOW_SIZE:
        raise ValueError(f"exactly {WINDOW_SIZE} tile counts are required")
    question = DEFAULT_QUESTION
    if question.count("<image>") != WINDOW_SIZE:
        raise ValueError("the canonical question must contain five image placeholders")
    for count in tile_counts:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * (int(count) * tokens_per_tile)
            + IMG_END_TOKEN
        )
        question = question.replace("<image>", image_tokens, 1)
    return question


def _prompt_ids(tokenizer: Any, question: str) -> tuple[str, torch.Tensor]:
    messages = [{"role": "user", "content": question}]
    templated = False
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            templated = True
        except (TypeError, ValueError, NotImplementedError):
            prompt = question + "\n"
    else:
        prompt = question + "\n"
    ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if not templated and bos_id is not None:
        ids.insert(0, int(bos_id))
    if not ids:
        raise ValueError("tokenized prompt is empty")
    if "<answer>" in prompt or "<think>" in prompt:
        raise ValueError("inference prompt unexpectedly contains an answer wrapper")
    return prompt, torch.tensor([ids], dtype=torch.long)


def _prepare_window(
    window: InferenceWindow,
    *,
    data_root: Path,
    processor: InternVLDynamicProcessor,
    tokenizer: Any,
    tokens_per_tile: int,
    grounding_size: int,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for inference image loading") from error

    tile_tensors: list[torch.Tensor] = []
    ground_tensors: list[torch.Tensor] = []
    tile_counts: list[int] = []
    original_sizes: list[tuple[int, int]] = []
    image_references: list[str] = []
    for frame in window.frames:
        path = _resolve_image_path(data_root, frame)
        with Image.open(path) as opened:
            image = opened.convert("RGB").copy()
        width, height = image.size
        tiles = processor(image)
        tile_tensors.append(tiles)
        tile_counts.append(int(tiles.shape[0]))
        ground_tensors.append(_grounding_tensor(image, grounding_size))
        original_sizes.append((height, width))
        image_references.append(_portable_image_reference(frame))
    question = _expanded_question(tile_counts, tokens_per_tile)
    prompt, input_ids = _prompt_ids(tokenizer, question)
    expected_context = sum(tile_counts) * tokens_per_tile
    if prompt.count(IMG_CONTEXT_TOKEN) != expected_context:
        raise ValueError("prompt/image context-token count mismatch")
    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.cat(tile_tensors, dim=0),
        "grounding_pixels": torch.stack(ground_tensors, dim=0),
        "original_sizes": original_sizes,
        "image_references": image_references,
        "tile_counts": tile_counts,
    }


def _transition_probabilities(
    model: Any, generated: Any, generated_ids: torch.Tensor
) -> list[float]:
    """Return probabilities along the selected generation path, including beams."""

    step_count = len(generated.scores)
    if not step_count:
        return []
    scorer = getattr(model, "language_model", model)
    compute_scores = getattr(scorer, "compute_transition_scores", None)
    if callable(compute_scores):
        transition_scores = compute_scores(
            generated.sequences,
            generated.scores,
            beam_indices=getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        return (
            transition_scores[0, -step_count:]
            .float()
            .exp()
            .detach()
            .cpu()
            .tolist()
        )

    if any(score.shape[0] != 1 for score in generated.scores):
        raise RuntimeError(
            "beam-search token confidence requires compute_transition_scores"
        )
    probabilities: list[float] = []
    for score, token_id in zip(generated.scores, generated_ids[0]):
        probability = score[0].float().softmax(dim=-1)[int(token_id)]
        probabilities.append(float(probability.detach().cpu()))
    return probabilities


def _text_only_generation(
    model: Any,
    tokenizer: Any,
    prepared: Mapping[str, Any],
    generation_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    generated = model.generate(
        pixel_values=prepared["pixel_values"],
        input_ids=prepared["input_ids"],
        attention_mask=prepared["attention_mask"],
        return_dict_in_generate=True,
        output_scores=True,
        **generation_kwargs,
    )
    scores = list(generated.scores)
    step_count = len(scores)
    generated_ids = (
        generated.sequences[:, -step_count:]
        if step_count
        else generated.sequences[:, :0]
    )
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return {
        "generated_text": text,
        "generated_ids": generated_ids.detach().cpu(),
        "token_probabilities": _transition_probabilities(
            model, generated, generated_ids
        ),
        "role_keys_per_frame": [[] for _ in range(WINDOW_SIZE)],
        "grounding_assignments_per_frame": [[] for _ in range(WINDOW_SIZE)],
        "binary_masks": [],
    }


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_rng(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    torch.random.set_rng_state(state[0])
    if device.type == "cuda" and state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _generate_window(
    model: Any,
    tokenizer: Any,
    prepared: Mapping[str, Any],
    generation_kwargs: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rng_state = _capture_rng(device)
    try:
        result = model.generate_with_grounding(
            tokenizer=tokenizer,
            pixel_values=prepared["pixel_values"],
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            grounding_pixels=prepared["grounding_pixels"],
            **generation_kwargs,
        )
        result = dict(result)
        result["grounding_error"] = None
        return result
    except ValueError as error:
        # The public model API validates all five role/[SEG] groups after text
        # generation.  Restore RNG before the diagnostic text-only replay so a
        # sampled run remains reproducible and the malformed text is not lost.
        _restore_rng(rng_state, device)
        result = _text_only_generation(
            model, tokenizer, prepared, generation_kwargs
        )
        result["grounding_error"] = f"{type(error).__name__}: {error}"
        return result


def _plain_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("generated_ids must contain one sequence")
        value = value[0]
    return [int(item) for item in value]


def _token_rows(
    tokenizer: Any,
    generated_text: str,
    generated_ids: Sequence[int],
    probabilities: Sequence[float],
) -> tuple[list[dict[str, Any]], bool]:
    count = min(len(generated_ids), len(probabilities))
    ids = [int(value) for value in generated_ids[:count]]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    rows = [
        {
            "token_id": token_id,
            "token": str(token),
            "confidence": float(probability),
            "offset": None,
        }
        for token_id, token, probability in zip(ids, tokens, probabilities)
    ]
    try:
        encoded = tokenizer(
            generated_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if encoded_ids and isinstance(encoded_ids[0], list):
            encoded_ids, offsets = encoded_ids[0], offsets[0]
        if [int(value) for value in encoded_ids] != ids:
            return rows, False
        for row, offset in zip(rows, offsets):
            row["offset"] = [int(offset[0]), int(offset[1])]
        return rows, True
    except (KeyError, TypeError, ValueError, NotImplementedError, AttributeError):
        return rows, False


def _frame_token_rows(
    generated_text: str,
    parsed: WindowParseResult,
    token_rows: Sequence[Mapping[str, Any]],
    offsets_valid: bool,
) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = [[] for _ in range(WINDOW_SIZE)]
    if offsets_valid:
        cursor = 0
        spans: dict[int, tuple[int, int]] = {}
        for number, frame_text in zip(parsed.frame_numbers, parsed.frame_texts):
            start = generated_text.find(frame_text, cursor)
            if start < 0:
                continue
            end = start + len(frame_text)
            cursor = end
            if 1 <= number <= WINDOW_SIZE and number not in spans:
                spans[number] = (start, end)
        for number, (start, end) in spans.items():
            output[number - 1] = [
                dict(row)
                for row in token_rows
                if row.get("offset")
                and int(row["offset"][1]) > start
                and int(row["offset"][0]) < end
            ]
        return output

    # Slow tokenizers do not expose offsets.  Preserve every confidence and
    # make the approximation explicit by assigning contiguous equal slices.
    for index in range(WINDOW_SIZE):
        left = math.floor(len(token_rows) * index / WINDOW_SIZE)
        right = math.floor(len(token_rows) * (index + 1) / WINDOW_SIZE)
        output[index] = [dict(row) for row in token_rows[left:right]]
    return output


def _compress_rle_counts(counts: Sequence[int]) -> str:
    encoded: list[str] = []
    for index, raw_count in enumerate(counts):
        value = int(raw_count)
        if index > 2:
            value -= int(counts[index - 2])
        more = True
        while more:
            chunk = value & 0x1F
            value >>= 5
            more = value != (-1 if chunk & 0x10 else 0)
            if more:
                chunk |= 0x20
            encoded.append(chr(chunk + 48))
    return "".join(encoded)


def binary_mask_to_rle(mask: Any) -> dict[str, Any]:
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"binary mask must be two-dimensional, got {array.shape}")
    flat = np.asarray(array > 0, dtype=np.uint8).reshape(-1, order="F")
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(boundaries).astype(np.int64).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {
        "size": [int(array.shape[0]), int(array.shape[1])],
        "counts": _compress_rle_counts(counts),
    }


def _resize_mask(mask: Any, size: tuple[int, int]) -> torch.Tensor:
    tensor = torch.as_tensor(mask).detach().float().squeeze()
    if tensor.ndim != 2:
        raise ValueError(f"decoded mask must be two-dimensional, got {tuple(tensor.shape)}")
    if tuple(tensor.shape) != size:
        tensor = torch_functional.interpolate(
            tensor[None, None], size=size, mode="nearest"
        )[0, 0]
    return tensor > 0.5


def _ordered_role_masks(
    result: Mapping[str, Any],
    original_sizes: Sequence[tuple[int, int]],
) -> tuple[list[list[tuple[tuple[str, str], dict[str, Any]]]], list[list[str]]]:
    """Keep generated marker/mask pairs in exact source ``[SEG]`` order."""

    ordered: list[list[tuple[tuple[str, str], dict[str, Any]]]] = [
        [] for _ in range(WINDOW_SIZE)
    ]
    warnings: list[list[str]] = [[] for _ in range(WINDOW_SIZE)]
    role_frames = list(result.get("role_keys_per_frame") or ())
    mask_frames = list(result.get("binary_masks") or ())
    for frame_index in range(WINDOW_SIZE):
        roles = list(role_frames[frame_index]) if frame_index < len(role_frames) else []
        masks = mask_frames[frame_index] if frame_index < len(mask_frames) else []
        masks = list(masks)
        if len(roles) != len(masks):
            warnings[frame_index].append(
                f"role/mask count mismatch: {len(roles)} roles vs {len(masks)} masks"
            )
        for role_key, mask in zip(roles, masks):
            if ":" not in str(role_key):
                warnings[frame_index].append(f"invalid role key: {role_key!r}")
                continue
            role, label = str(role_key).split(":", 1)
            role = normalize_label(role)
            label = normalize_label(label)
            if role not in {"instrument", "target"} or not label:
                warnings[frame_index].append(f"invalid role key: {role_key!r}")
                continue
            resized = _resize_mask(mask, original_sizes[frame_index])
            ordered[frame_index].append(
                ((role, label), {"rle": binary_mask_to_rle(resized)})
            )
    return ordered, warnings


def _append_error(current: str | None, message: str | None) -> str | None:
    if not message:
        return current
    return f"{current}; {message}" if current else message


def _mean_confidence(rows: Sequence[Mapping[str, Any]]) -> float:
    values = [float(row["confidence"]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _window_records(
    window: InferenceWindow,
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated_text = str(result.get("generated_text", ""))
    parsed = parse_generated_window(generated_text, expected_frames=WINDOW_SIZE)
    generated_ids = _plain_ids(result.get("generated_ids", []))
    probabilities = [float(value) for value in result.get("token_probabilities", ())]
    token_rows, offsets_valid = _token_rows(
        tokenizer, generated_text, generated_ids, probabilities
    )
    tokens_per_frame = _frame_token_rows(
        generated_text, parsed, token_rows, offsets_valid
    )
    ordered_masks, mask_warnings = _ordered_role_masks(
        result, prepared["original_sizes"]
    )
    raw_assignment_frames = result.get("grounding_assignments_per_frame")
    assignment_frames = (
        list(raw_assignment_frames) if raw_assignment_frames is not None else None
    )

    by_number: dict[int, tuple[str, FrameParseResult]] = {}
    for number, frame_text, frame_result in zip(
        parsed.frame_numbers, parsed.frame_texts, parsed.frames
    ):
        if 1 <= number <= WINDOW_SIZE and number not in by_number:
            by_number[number] = (frame_text, frame_result)
    structural_error = "; ".join(
        f"{issue.code}: {issue.message}" for issue in parsed.issues
    ) or None
    window_error = parsed.parse_error if not parsed.ok else None
    grounding_error = result.get("grounding_error")

    predictions: list[dict[str, Any]] = []
    for position, source in enumerate(window.frames, start=1):
        token_slice = tokens_per_frame[position - 1]
        confidence = _mean_confidence(token_slice)
        matched = by_number.get(position)
        if matched is None:
            parse_error = _append_error(
                structural_error, f"missing_frame_block: Frame {position} was not generated"
            )
            predictions.append(
                {
                    "video_id": source.video_id,
                    "frame_id": source.frame_id,
                    "window_id": window.window_id,
                    "window_index": window.index,
                    "window_position": position,
                    "generated_text": None,
                    "parse_error": parse_error,
                    "parser_result": {"ok": False, "parse_error": parse_error},
                    "phase": {"label": None, "confidence": 0.0},
                    "triplets": [],
                    "token_ids": [row["token_id"] for row in token_slice],
                    "tokens": [row["token"] for row in token_slice],
                    "token_confidences": [row["confidence"] for row in token_slice],
                    "image_path": prepared["image_references"][position - 1],
                }
            )
            continue

        frame_text, frame_result = matched
        parse_error = _append_error(frame_result.parse_error, window_error)
        effective_ok = parsed.ok and frame_result.ok
        frame_masks = ordered_masks[position - 1]
        expected_keys = [
            (grounding.role, grounding.label)
            for grounding in frame_result.groundings
        ]
        actual_keys = [key for key, _ in frame_masks]
        if effective_ok and expected_keys != actual_keys:
            parse_error = _append_error(
                parse_error,
                f"grounding_alignment: expected ordered {expected_keys}, "
                f"got {actual_keys}",
            )
        expected_assignments = [
            {
                "role": grounding.role,
                "label": grounding.label,
                "triplet_index": grounding.triplet_index,
            }
            for grounding in frame_result.groundings
        ]
        if assignment_frames is not None:
            raw_frame_assignments = (
                assignment_frames[position - 1]
                if position - 1 < len(assignment_frames)
                else []
            )
            try:
                actual_assignments = [
                    {
                        "role": normalize_label(item["role"]),
                        "label": normalize_label(item["label"]),
                        "triplet_index": int(item["triplet_index"]),
                    }
                    for item in raw_frame_assignments
                ]
            except (KeyError, TypeError, ValueError):
                actual_assignments = []
                parse_error = _append_error(
                    parse_error,
                    "grounding_alignment: malformed generated grounding assignments",
                )
            if effective_ok and expected_assignments != actual_assignments:
                parse_error = _append_error(
                    parse_error,
                    f"grounding_alignment: expected assignments "
                    f"{expected_assignments}, got {actual_assignments}",
                )
        for warning in mask_warnings[position - 1]:
            parse_error = _append_error(parse_error, f"grounding_alignment: {warning}")
        if grounding_error and frame_result.triplets:
            parse_error = _append_error(
                parse_error, f"grounding_generation: {grounding_error}"
            )

        assigned_masks: dict[tuple[int, str], dict[str, Any]] = {}
        if effective_ok and expected_keys == actual_keys:
            for grounding, (_, mask_payload) in zip(
                frame_result.groundings, frame_masks
            ):
                slot = (grounding.triplet_index, grounding.role)
                if slot in assigned_masks:
                    parse_error = _append_error(
                        parse_error,
                        "grounding_alignment: duplicate mask assignment for "
                        f"triplet {grounding.triplet_index + 1} {grounding.role}",
                    )
                    continue
                assigned_masks[slot] = mask_payload

        triplets: list[dict[str, Any]] = []
        if effective_ok:
            for triplet_index, triplet in enumerate(frame_result.triplets):
                value: dict[str, Any] = {
                    "instrument": triplet.instrument,
                    "verb": triplet.verb,
                    "target": triplet.target,
                    "confidence": confidence,
                }
                if triplet.instrument_grounded:
                    instrument_mask = assigned_masks.get(
                        (triplet_index, "instrument")
                    )
                    if instrument_mask is not None:
                        value["instrument_mask"] = instrument_mask
                if triplet.target_grounded:
                    target_mask = assigned_masks.get((triplet_index, "target"))
                    if target_mask is not None:
                        value["target_mask"] = target_mask
                triplets.append(value)
        parser_payload = frame_result.as_dict()
        if not parsed.ok:
            parser_payload["ok"] = False
            parser_payload["window_parse_error"] = parsed.parse_error
            parser_payload["parse_error"] = parse_error
        elif parse_error:
            parser_payload["ok"] = False
            parser_payload["parse_error"] = parse_error
        predictions.append(
            {
                "video_id": source.video_id,
                "frame_id": source.frame_id,
                "window_id": window.window_id,
                "window_index": window.index,
                "window_position": position,
                "generated_text": frame_text if parsed.ok else None,
                "raw_generated_text": frame_text,
                "parse_error": parse_error,
                "parser_result": parser_payload,
                "phase": {
                    "label": frame_result.phase if effective_ok else None,
                    "confidence": confidence if effective_ok else 0.0,
                },
                "triplets": triplets,
                "token_ids": [row["token_id"] for row in token_slice],
                "tokens": [row["token"] for row in token_slice],
                "token_confidences": [row["confidence"] for row in token_slice],
                "token_alignment": "offsets" if offsets_valid else "equal_slices",
                "image_path": prepared["image_references"][position - 1],
            }
        )

    window_record = {
        "window_id": window.window_id,
        "window_index": window.index,
        "video_id": window.video_id,
        "frame_ids": [frame.frame_id for frame in window.frames],
        "generated_text": generated_text,
        "parser_result": parsed.as_dict(),
        "grounding_error": grounding_error,
        "role_keys_per_frame": [
            list(frame) for frame in (result.get("role_keys_per_frame") or ())
        ],
        "grounding_assignments_per_frame": [
            list(frame)
            for frame in (result.get("grounding_assignments_per_frame") or ())
        ],
        "token_ids": generated_ids,
        "token_confidences": probabilities,
        "token_alignment": "offsets" if offsets_valid else "equal_slices",
        "tile_counts": list(prepared["tile_counts"]),
    }
    return predictions, window_record


def _failed_window(
    window: InferenceWindow, error: BaseException
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    message = f"inference_error: {type(error).__name__}: {error}"
    predictions = [
        {
            "video_id": frame.video_id,
            "frame_id": frame.frame_id,
            "window_id": window.window_id,
            "window_index": window.index,
            "window_position": position,
            "generated_text": None,
            "parse_error": message,
            "parser_result": {"ok": False, "parse_error": message},
            "phase": {"label": None, "confidence": 0.0},
            "triplets": [],
            "token_ids": [],
            "tokens": [],
            "token_confidences": [],
        }
        for position, frame in enumerate(window.frames, start=1)
    ]
    return predictions, {
        "window_id": window.window_id,
        "window_index": window.index,
        "video_id": window.video_id,
        "frame_ids": [frame.frame_id for frame in window.frames],
        "generated_text": None,
        "inference_error": message,
    }


def _set_window_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    value = values[name]
    if device.type == "cpu" and value == torch.float16:
        raise ValueError("float16 inference is not supported on CPU")
    return value


def _load_model(args: argparse.Namespace, context: DistributedContext) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for HF inference") from error
    tensor_dtype = _dtype(args.dtype, context.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=tensor_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device=context.device, dtype=tensor_dtype)
    if not hasattr(model, "generate_with_grounding"):
        raise TypeError("loaded HF model does not expose generate_with_grounding")
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference(tokenizer)
    return model, tokenizer


def _generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        output.update(temperature=args.temperature, top_p=args.top_p)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if eos_id is not None:
        output["eos_token_id"] = int(eos_id)
    if pad_id is not None:
        output["pad_token_id"] = int(pad_id)
    elif eos_id is not None:
        output["pad_token_id"] = int(eos_id)
    return output


def _processor(args: argparse.Namespace, model: Any) -> InternVLDynamicProcessor:
    config = model.config
    image_size = int(
        args.image_size
        or getattr(config, "force_image_size", None)
        or config.vision_config.image_size
    )
    patch_size = int(config.vision_config.patch_size)
    downsample_ratio = float(config.downsample_ratio)
    processor = InternVLDynamicProcessor(
        image_size=image_size,
        min_dynamic_patch=args.min_dynamic_patch,
        max_dynamic_patch=args.max_dynamic_patch,
        use_thumbnail=args.use_thumbnail,
        patch_size=patch_size,
        downsample_ratio=downsample_ratio,
    )
    model_tokens = int(getattr(model, "num_image_token"))
    if processor.num_image_tokens_per_tile != model_tokens:
        raise ValueError(
            "image processor/model token mismatch: "
            f"{processor.num_image_tokens_per_tile} vs {model_tokens}"
        )
    return processor


def _prediction_score(record: Mapping[str, Any]) -> tuple[int, float, int]:
    parser_ok = bool(record.get("parser_result", {}).get("ok"))
    confidence = float(record.get("phase", {}).get("confidence", 0.0))
    return int(parser_ok), confidence, -int(record.get("window_index", 0))


def _coverage(
    expected_frames: Sequence[SourceFrame], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {frame.key for frame in expected_frames}
    predicted = {
        (str(record.get("video_id", "")), str(record.get("frame_id", "")))
        for record in predictions
    }
    parsed = {
        (str(record.get("video_id", "")), str(record.get("frame_id", "")))
        for record in predictions
        if bool(record.get("parser_result", {}).get("ok"))
    }

    def summarize(video_id: str | None) -> dict[str, Any]:
        selected_expected = {
            key for key in expected if video_id is None or key[0] == video_id
        }
        selected_predicted = {
            key for key in predicted if video_id is None or key[0] == video_id
        }
        selected_parsed = {
            key for key in parsed if video_id is None or key[0] == video_id
        }
        recorded = selected_expected & selected_predicted
        usable = selected_expected & selected_parsed
        missing = sorted(selected_expected - selected_predicted)
        unparsed = sorted(selected_expected - selected_parsed)
        denominator = len(selected_expected)
        return {
            "expected_frames": denominator,
            "recorded_frames": len(recorded),
            "parsed_frames": len(usable),
            "record_coverage": len(recorded) / denominator if denominator else 0.0,
            "parsed_coverage": len(usable) / denominator if denominator else 0.0,
            "missing_frames": len(missing),
            "missing_frame_ids": [f"{video}/{frame}" for video, frame in missing],
            "missing_or_unparsed_frames": len(unparsed),
            "missing_or_unparsed_frame_ids": [
                f"{video}/{frame}" for video, frame in unparsed
            ],
        }

    video_ids = sorted({key[0] for key in expected})
    return {
        "overall": summarize(None),
        "per_video": {video_id: summarize(video_id) for video_id in video_ids},
    }


def _merge_shards(
    *,
    shard_dir: Path,
    output_path: Path,
    run_id: str,
    context: DistributedContext,
    expected_frames: Sequence[SourceFrame],
    total_windows: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected_names = {
        f"rank-{rank:05d}.json" for rank in range(context.world_size)
    }
    present_names = {path.name for path in shard_dir.glob("rank-*.json")}
    if present_names != expected_names:
        raise RuntimeError(
            f"incomplete or unexpected current-run shards: "
            f"missing={sorted(expected_names - present_names)}, "
            f"unexpected={sorted(present_names - expected_names)}"
        )
    all_predictions: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    processed_indices: list[int] = []
    for rank in range(context.world_size):
        path = shard_dir / f"rank-{rank:05d}.json"
        with path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        metadata = shard.get("metadata", {})
        if (
            metadata.get("run_id") != run_id
            or int(metadata.get("rank", -1)) != rank
            or int(metadata.get("world_size", -1)) != context.world_size
        ):
            raise RuntimeError(f"shard provenance mismatch: {path}")
        processed_indices.extend(int(value) for value in metadata["window_indices"])
        all_predictions.extend(dict(value) for value in shard.get("predictions", ()))
        all_windows.extend(dict(value) for value in shard.get("windows", ()))
    if sorted(processed_indices) != list(range(total_windows)):
        raise RuntimeError("current-run shards do not cover every planned window exactly once")

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_candidates = 0
    for record in all_predictions:
        key = (str(record.get("video_id", "")), str(record.get("frame_id", "")))
        if key in selected:
            duplicate_candidates += 1
            if _prediction_score(record) > _prediction_score(selected[key]):
                selected[key] = record
        else:
            selected[key] = record
    video_order = {video_id: index for index, video_id in enumerate(dict.fromkeys(
        frame.video_id for frame in expected_frames
    ))}
    predictions = sorted(
        selected.values(),
        key=lambda record: (
            video_order.get(str(record.get("video_id")), len(video_order)),
            0 if str(record.get("frame_id", "")).isdigit() else 1,
            int(record["frame_id"])
            if str(record.get("frame_id", "")).isdigit()
            else str(record.get("frame_id", "")),
        ),
    )
    coverage = _coverage(expected_frames, predictions)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": _portable_model_reference(args.model),
            "split": args.split,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "world_size": context.world_size,
            "total_windows": total_windows,
            "max_windows": args.max_windows,
            "dtype": str(_dtype(args.dtype, context.device)).removeprefix("torch."),
            "device_type": context.device.type,
            "duplicate_window_candidates": duplicate_candidates,
            "confidence": "mean_generated_token_probability",
            "mask_encoding": "coco_rle_compressed",
            "coverage": coverage,
        },
        "predictions": predictions,
        "windows": sorted(all_windows, key=lambda value: int(value["window_index"])),
    }
    _atomic_write_json(output_path, payload)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distributed prompt-only five-frame SurgMLLM inference"
    )
    parser.add_argument("--model", required=True, help="converted HF model directory")
    parser.add_argument("--data-root", required=True, help="scene dataset root")
    parser.add_argument("--annotations", help="optional canonical/legacy annotation path")
    parser.add_argument(
        "--split", choices=("fold1",), default="fold1"
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument(
        "--max-windows",
        type=int,
        help="process only the first N global windows for a bounded smoke run",
    )
    parser.add_argument("--output", required=True, help="atomic merged raw prediction JSON")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--min-dynamic-patch", type=int, default=1)
    parser.add_argument("--max-dynamic-patch", type=int, default=5)
    parser.add_argument(
        "--use-thumbnail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.window_size != WINDOW_SIZE:
        raise ValueError(f"--window-size must be {WINDOW_SIZE}")
    if args.window_stride <= 0 or args.max_new_tokens <= 0:
        raise ValueError("window stride and max new tokens must be positive")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("max windows must be positive when provided")
    if args.num_beams <= 0 or args.min_dynamic_patch <= 0:
        raise ValueError("num beams and minimum dynamic patches must be positive")
    if args.max_dynamic_patch < args.min_dynamic_patch:
        raise ValueError("dynamic patch bounds are invalid")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    _validate_args(args)
    context = _init_distributed()
    run_id = _broadcast_run_id(context)
    output_path = Path(args.output).expanduser().resolve()
    shard_dir = output_path.parent / f".{output_path.name}.shards" / run_id
    if context.rank == 0:
        shard_dir.mkdir(parents=True, exist_ok=False)
    if context.world_size > 1:
        dist.barrier()

    annotations = (
        load_annotation_records(args.annotations)
        if args.annotations
        else load_split_annotations(args.data_root, args.split)
    )
    allowed = set(FOLD1_VIDEO_IDS)
    annotations = [record for record in annotations if frame_key(record)[0] in allowed]
    expected_frames = _source_frames(annotations)
    windows = build_windows(
        expected_frames,
        window_size=args.window_size,
        stride=args.window_stride,
    )
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    assigned = [window for window in windows if window.index % context.world_size == context.rank]
    model, tokenizer = _load_model(args, context)
    processor = _processor(args, model)
    tokens_per_tile = int(model.num_image_token)
    grounding_size = int(getattr(model.config, "sam2_image_size", 1024))
    generation_kwargs = _generation_kwargs(args, tokenizer)
    tensor_dtype = _dtype(args.dtype, context.device)

    predictions: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    for local_index, window in enumerate(assigned, start=1):
        _set_window_seed(args.seed + window.index)
        try:
            prepared = _prepare_window(
                window,
                data_root=Path(args.data_root).expanduser().resolve(),
                processor=processor,
                tokenizer=tokenizer,
                tokens_per_tile=tokens_per_tile,
                grounding_size=grounding_size,
            )
            prepared["input_ids"] = prepared["input_ids"].to(context.device)
            prepared["attention_mask"] = prepared["attention_mask"].to(context.device)
            prepared["pixel_values"] = prepared["pixel_values"].to(
                context.device, dtype=tensor_dtype
            )
            prepared["grounding_pixels"] = prepared["grounding_pixels"].to(
                context.device
            )
            result = _generate_window(
                model,
                tokenizer,
                prepared,
                generation_kwargs,
                context.device,
            )
            frame_predictions, window_record = _window_records(
                window, result, prepared, tokenizer
            )
        except Exception as error:  # one bad window must remain visible in coverage
            frame_predictions, window_record = _failed_window(window, error)
            if context.device.type == "cuda":
                torch.cuda.empty_cache()
        predictions.extend(frame_predictions)
        window_records.append(window_record)
        if args.log_interval > 0 and (
            local_index % args.log_interval == 0 or local_index == len(assigned)
        ):
            print(
                f"rank {context.rank}: {local_index}/{len(assigned)} windows",
                flush=True,
            )

    shard_payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "run_id": run_id,
            "rank": context.rank,
            "world_size": context.world_size,
            "window_indices": [window.index for window in assigned],
        },
        "predictions": predictions,
        "windows": window_records,
    }
    _atomic_write_json(shard_dir / f"rank-{context.rank:05d}.json", shard_payload)
    if context.world_size > 1:
        dist.barrier()

    merge_status: list[object] = [None]
    if context.rank == 0:
        try:
            payload = _merge_shards(
                shard_dir=shard_dir,
                output_path=output_path,
                run_id=run_id,
                context=context,
                expected_frames=expected_frames,
                total_windows=len(windows),
                args=args,
            )
            overall_coverage = payload["metadata"]["coverage"]["overall"]
            merge_status[0] = {
                "output": str(output_path),
                "coverage": {
                    key: value
                    for key, value in overall_coverage.items()
                    if not key.endswith("_ids")
                },
            }
        except Exception as error:
            merge_status[0] = {"error": f"{type(error).__name__}: {error}"}
    if context.world_size > 1:
        dist.broadcast_object_list(merge_status, src=0)
    status = dict(merge_status[0])
    if "error" in status:
        raise RuntimeError(f"atomic shard merge failed: {status['error']}")
    if context.rank == 0:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    if context.initialized_here:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
