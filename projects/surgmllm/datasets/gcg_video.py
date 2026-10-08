"""Five-frame surgical grounded-caption dataset."""

from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset

from ..record_identity import normalize_video_id, record_identity
from .protocol import (
    ENTITY_ROLES,
    MASK_ROLES,
    CanonicalFrame,
    EntityCharSpan,
    ProtocolError,
    build_canonical_frame,
    map_entity_spans_to_tokens,
)
from .image_processing import InternVLDynamicProcessor
from .splits import format_video_id, get_fold1_video_ids
from .tokens import SEG_TOKEN, SPECIAL_TOKENS


IGNORE_INDEX = -100
WINDOW_SIZE = 5
DEFAULT_QUESTION = (
    "Frame 1: <image>\nFrame 2: <image>\nFrame 3: <image>\n"
    "Frame 4: <image>\nFrame 5: <image>\n"
    "Describe every frame, reasoning briefly before its grounded surgical phase "
    "and action labels."
)
IMG_START_TOKEN = "<img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_END_TOKEN = "</img>"


def _build_configurable(value: Any) -> Any:
    if not isinstance(value, Mapping) or "type" not in value:
        return value
    component_type = value["type"]
    kwargs = {key: item for key, item in value.items() if key != "type"}
    if callable(component_type):
        return component_type(**kwargs)
    try:
        from xtuner.registry import BUILDER  # type: ignore

        return BUILDER.build(dict(value))
    except ImportError as error:
        raise RuntimeError(
            "A mapping component with a string type requires XTuner to be installed; "
            "a callable config type works without it."
        ) from error


def _video_number(video_id: str) -> int | None:
    match = re.fullmatch(r"VID0*(\d+)", video_id.upper())
    return int(match.group(1)) if match else None


def _frame_sort_key(item: Mapping[str, Any]) -> tuple[int, int | str]:
    frame_id = str(item["_frame_id"])
    match = re.search(r"(\d+)(?!.*\d)", frame_id)
    if match:
        return 0, int(match.group(1))
    return 1, frame_id


def _numeric_frame_index(item: Mapping[str, Any]) -> int | None:
    """Return the trailing source-frame index used to prevent gap bridging."""

    match = re.search(r"(\d+)(?!.*\d)", str(item["_frame_id"]))
    return int(match.group(1)) if match else None


def _is_contiguous_window(frames: Sequence[Mapping[str, Any]]) -> bool:
    indices = [_numeric_frame_index(frame) for frame in frames]
    if any(index is None for index in indices):
        raise ProtocolError(
            "Five-frame clips require numeric source frame IDs so temporal gaps "
            "cannot be bridged silently."
        )
    return all(right == left + 1 for left, right in zip(indices, indices[1:]))


def _extract_video_and_frame(item: Mapping[str, Any]) -> tuple[str, str]:
    try:
        video_id, _ = record_identity(item)
    except ValueError as error:
        raise ProtocolError(str(error)) from error
    raw_frame = item.get("frame_id", item.get("frame"))
    if raw_frame in (None, ""):
        file_name = item.get("file_name", item.get("image"))
        if file_name not in (None, "") and not Path(str(file_name)).is_absolute():
            raw_frame = Path(str(file_name)).stem
    if raw_frame in (None, ""):
        image_id = str(item.get("image_id", ""))
        raw_frame = image_id.rsplit("_", 1)[1]
    return video_id, str(raw_frame)


def _to_numpy_hwc(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ProtocolError(f"An image must be HWC or CHW, got shape {array.shape}.")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array)


def _to_chw_tensor(value: Any, *, normalize: bool) -> torch.Tensor:
    if isinstance(value, Mapping):
        for key in ("pixel_values", "g_pixel_values", "image"):
            if key in value:
                value = value[key]
                break
    source_is_tensor = isinstance(value, torch.Tensor)
    tensor = (
        value.detach().clone()
        if source_is_tensor
        else torch.as_tensor(np.array(value, copy=True))
    )
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    if tensor.ndim != 3:
        raise ProtocolError(f"Processed image must have three dimensions, got {tensor.shape}.")
    if tensor.shape[-1] in {1, 3, 4} and (
        not source_is_tensor or tensor.shape[0] not in {1, 3, 4}
    ):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    tensor = tensor.contiguous()
    if normalize:
        original_dtype = tensor.dtype
        tensor = tensor.float()
        if original_dtype == torch.uint8 or (tensor.numel() and tensor.max() > 1):
            tensor = tensor / 255.0
    return tensor


def _decode_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
    size = rle.get("size")
    counts = rle.get("counts")
    if not (
        isinstance(size, Sequence)
        and len(size) == 2
        and isinstance(counts, Sequence)
        and not isinstance(counts, (str, bytes))
    ):
        raise ProtocolError("Pure-Python RLE decoding requires integer counts and [H, W].")
    height, width = int(size[0]), int(size[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    foreground = False
    for raw_run in counts:
        run = int(raw_run)
        if run < 0 or cursor + run > flat.size:
            raise ProtocolError("Invalid uncompressed RLE run length.")
        if foreground:
            flat[cursor : cursor + run] = 1
        cursor += run
        foreground = not foreground
    if cursor != flat.size:
        raise ProtocolError(
            f"RLE counts cover {cursor} pixels, expected {flat.size}."
        )
    return flat.reshape((height, width), order="F")


def _decode_one_rle(rle: Mapping[str, Any]) -> np.ndarray:
    if isinstance(rle.get("counts"), Sequence) and not isinstance(
        rle.get("counts"), (str, bytes)
    ):
        return _decode_uncompressed_rle(rle)
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Compressed COCO RLE masks require pycocotools; inline binary masks and "
            "uncompressed integer RLEs do not."
        ) from error
    decoded = np.asarray(mask_utils.decode(dict(rle)))
    return decoded[..., 0] if decoded.ndim == 3 else decoded


def _decode_grounding_mask(
    payload: Mapping[str, Any],
    height: int,
    width: int,
    custom_decoder: Callable[..., Any] | None,
) -> torch.Tensor:
    if custom_decoder is not None:
        decoded = custom_decoder(payload, height=height, width=width)
        array = np.asarray(decoded)
    elif "mask" in payload or "binary_mask" in payload:
        array = np.asarray(payload.get("mask", payload.get("binary_mask")))
    else:
        raw_components = payload.get("rle_masks", payload.get("rle_mask"))
        if isinstance(raw_components, Mapping):
            components = [raw_components]
        elif isinstance(raw_components, Sequence) and not isinstance(
            raw_components, (str, bytes)
        ):
            components = list(raw_components)
        else:
            raise ProtocolError("Each instrument/target grounding requires one mask.")
        if not components:
            raise ProtocolError("A grounding mask component list cannot be empty.")
        decoded_components = [_decode_one_rle(component) for component in components]
        array = np.logical_or.reduce([component > 0 for component in decoded_components])

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != (height, width):
        raise ProtocolError(
            f"Grounding mask shape {array.shape} does not match frame {(height, width)}."
        )
    return torch.from_numpy(np.ascontiguousarray(array > 0))


class SurgMLLMGCGVideoDataset(Dataset):
    """Create strict, temporally contiguous five-frame training samples.

    ``annotation_file`` accepts either a JSON list of frame records or a mapping
    containing ``annotations``/``frames``.  ``data_path`` and ``image_folder`` are
    accepted as config-friendly aliases.
    """

    def __init__(
        self,
        annotation_file: str | os.PathLike[str] | None = None,
        image_root: str | os.PathLike[str] = "",
        tokenizer: Any = None,
        *,
        data_path: str | os.PathLike[str] | None = None,
        image_folder: str | os.PathLike[str] | None = None,
        annotations: Sequence[Mapping[str, Any]] | None = None,
        stride: int = 1,
        window_stride: int | None = None,
        num_frames: int = WINDOW_SIZE,
        sampled_frames: int | None = None,
        split: str | None = None,
        video_ids: Sequence[int | str] | None = None,
        prompt_template: Any = None,
        system_prompt: str = "",
        question: str = DEFAULT_QUESTION,
        max_length: int = 8192,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        image_loader: Callable[[str], Any] | None = None,
        image_processor: Any = None,
        extra_image_processor: Any = None,
        mask_decoder: Callable[..., Any] | None = None,
        grounding_image_size: int = 1024,
        image_size: int = 448,
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 12,
        use_thumbnail: bool = True,
        patch_size: int = 14,
        downsample_ratio: float = 0.5,
        image_tokens_per_tile: int | None = None,
        strict_token_decode: bool = True,
        repeats: float = 1.0,
        name: str = "SurgMLLMGCGVideoDataset",
    ) -> None:
        super().__init__()
        if annotation_file is None:
            annotation_file = data_path
        elif data_path is not None and Path(annotation_file) != Path(data_path):
            raise ValueError("annotation_file and data_path refer to different files.")
        if image_folder is not None:
            if image_root and Path(image_root) != Path(image_folder):
                raise ValueError("image_root and image_folder refer to different directories.")
            image_root = image_folder
        if sampled_frames is not None:
            if num_frames != WINDOW_SIZE and num_frames != sampled_frames:
                raise ValueError("num_frames and sampled_frames disagree.")
            num_frames = sampled_frames
        if num_frames != WINDOW_SIZE:
            raise ValueError(f"This protocol requires exactly {WINDOW_SIZE} frames.")
        if window_stride is not None:
            if stride != 1 and stride != window_stride:
                raise ValueError("stride and window_stride disagree.")
            stride = window_stride
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if tuple(special_tokens) != SPECIAL_TOKENS:
            raise ValueError(
                "The data protocol uses exactly the seven exported SPECIAL_TOKENS."
            )
        if max_length <= 0 or repeats <= 0:
            raise ValueError("max_length and repeats must be positive.")

        self.annotation_file = str(annotation_file) if annotation_file is not None else None
        self.image_root = str(image_root)
        self.stride = stride
        self.num_frames = WINDOW_SIZE
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.question = question
        self.max_length = int(max_length)
        self.special_tokens = SPECIAL_TOKENS
        self.image_loader = image_loader
        self.image_processor = _build_configurable(image_processor)
        if self.image_processor is None:
            self.image_processor = InternVLDynamicProcessor(
                image_size=image_size,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                use_thumbnail=use_thumbnail,
                patch_size=patch_size,
                downsample_ratio=downsample_ratio,
            )
        processor_token_count = getattr(
            self.image_processor, "num_image_tokens_per_tile", None
        )
        self.image_tokens_per_tile = int(
            image_tokens_per_tile
            if image_tokens_per_tile is not None
            else (processor_token_count if processor_token_count is not None else 256)
        )
        if self.image_tokens_per_tile <= 0:
            raise ValueError("image_tokens_per_tile must be positive.")
        self.extra_image_processor = _build_configurable(extra_image_processor)
        self.mask_decoder = mask_decoder
        self.grounding_image_size = int(grounding_image_size)
        self.strict_token_decode = strict_token_decode
        self.repeats = float(repeats)
        self.name = name

        self.tokenizer = _build_configurable(tokenizer)
        if self.tokenizer is None:
            raise ValueError("tokenizer is required; a lightweight fake tokenizer is supported.")
        if hasattr(self.tokenizer, "add_tokens"):
            try:
                self.tokenizer.add_tokens(list(SPECIAL_TOKENS), special_tokens=True)
            except TypeError:
                self.tokenizer.add_tokens(list(SPECIAL_TOKENS))

        loaded = list(annotations) if annotations is not None else self._load_annotations()
        allowed_video_ids = self._resolve_video_filter(split, video_ids)
        self.video_infos = self._organize_frames(loaded, allowed_video_ids)
        self.videos = tuple(self.video_infos)
        self.sliding_windows: list[tuple[str, int]] = []
        self._windows: list[tuple[Mapping[str, Any], ...]] = []
        for video_id, frames in self.video_infos.items():
            for start in range(0, len(frames) - WINDOW_SIZE + 1, self.stride):
                selected = tuple(frames[start : start + WINDOW_SIZE])
                # Invalid annotations split temporal runs; removing a frame before
                # window construction would silently bridge across that gap.
                if all(frame.get("groundings") for frame in selected) and _is_contiguous_window(
                    selected
                ):
                    self.sliding_windows.append((video_id, start))
                    self._windows.append(selected)

    def _load_annotations(self) -> list[Mapping[str, Any]]:
        if self.annotation_file is None:
            raise ValueError("annotation_file/data_path or annotations must be provided.")
        with open(self.annotation_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload = payload.get("annotations", payload.get("frames"))
        if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
            raise ProtocolError("Annotation JSON must contain a list of frame mappings.")
        return payload

    @staticmethod
    def _resolve_video_filter(
        split: str | None,
        video_ids: Sequence[int | str] | None,
    ) -> set[str] | None:
        split_ids = get_fold1_video_ids(split) if split is not None else None
        if split_ids is not None and video_ids is not None:
            supplied = {_video_number(format_video_id(item)) for item in video_ids}
            if supplied != set(split_ids):
                raise ValueError("video_ids conflicts with the canonical requested split.")
        selected = split_ids if split_ids is not None else video_ids
        return {format_video_id(item) for item in selected} if selected is not None else None

    @staticmethod
    def _organize_frames(
        loaded: Sequence[Mapping[str, Any]],
        allowed_video_ids: set[str] | None,
    ) -> OrderedDict[str, list[Mapping[str, Any]]]:
        grouped: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
        for source in loaded:
            video_id, frame_id = _extract_video_and_frame(source)
            if allowed_video_ids is not None and video_id not in allowed_video_ids:
                continue
            item = dict(source)
            item["_video_id"] = video_id
            item["_frame_id"] = frame_id
            grouped.setdefault(video_id, []).append(item)
        for frames in grouped.values():
            frames.sort(key=_frame_sort_key)
        return grouped

    def real_len(self) -> int:
        return len(self._windows)

    def __len__(self) -> int:
        return int(math.ceil(self.real_len() * self.repeats))

    @property
    def modality_length(self) -> list[int]:
        return [10000] * len(self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(videos={len(self.videos)}, "
            f"windows={self.real_len()}, stride={self.stride})"
        )

    def _load_image(self, frame: Mapping[str, Any]) -> Any:
        relative = str(frame.get("file_name", frame.get("image", "")))
        if not relative:
            raise ProtocolError("Each frame requires file_name or image.")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ProtocolError("Frame image paths must stay relative to image_root.")
        if len(relative_path.parts) >= 2:
            path_video = normalize_video_id(relative_path.parts[0])
            if path_video != normalize_video_id(frame["_video_id"]):
                raise ProtocolError(
                    f"Image path video {path_video} disagrees with "
                    f"record video {frame['_video_id']}."
                )
        image_root = Path(self.image_root).expanduser().resolve()
        path = (image_root / relative_path).resolve()
        try:
            path.relative_to(image_root)
        except ValueError as error:
            raise ProtocolError("Frame image path escapes image_root.") from error
        if self.image_loader is not None:
            return self.image_loader(str(path))
        try:
            from PIL import Image

            with Image.open(path) as opened:
                return opened.convert("RGB").copy()
        except (OSError, ImportError) as error:
            raise RuntimeError(f"Unable to load frame image {str(path)!r}.") from error

    @staticmethod
    def _frame_hw(frame: Mapping[str, Any], image: Any) -> tuple[int, int]:
        if frame.get("height") is not None and frame.get("width") is not None:
            return int(frame["height"]), int(frame["width"])
        array = _to_numpy_hwc(image)
        return int(array.shape[0]), int(array.shape[1])

    def _process_pixel_image(self, image: Any) -> torch.Tensor:
        processor = self.image_processor
        if hasattr(processor, "preprocess"):
            try:
                value = processor.preprocess(image, return_tensors="pt")
            except TypeError:
                value = processor.preprocess(image)
        else:
            try:
                value = processor(images=image, return_tensors="pt")
            except TypeError:
                value = processor(image)
        if isinstance(value, Mapping):
            for key in ("pixel_values", "image"):
                if key in value:
                    value = value[key]
                    break
        tensor = (
            value.detach().clone()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(np.array(value, copy=True))
        )
        if tensor.ndim == 3:
            tensor = _to_chw_tensor(tensor, normalize=False).unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[1] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(0, 3, 1, 2)
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            if tensor.shape[1] == 4:
                tensor = tensor[:, :3]
            tensor = tensor.contiguous()
        else:
            raise ProtocolError(
                f"Visual processor must return [C,H,W] or [tiles,C,H,W], got {tensor.shape}."
            )
        return tensor

    def _process_ground_image(self, image: Any) -> torch.Tensor:
        array = _to_numpy_hwc(image)
        processor = self.extra_image_processor
        if processor is not None:
            if hasattr(processor, "apply_image"):
                value = processor.apply_image(array)
            else:
                try:
                    value = processor(array)
                except TypeError:
                    value = processor(image=image)
            return _to_chw_tensor(value, normalize=False)

        tensor = _to_chw_tensor(array, normalize=False)
        if tensor.dtype != torch.uint8:
            tensor = tensor.float()
            if tensor.numel() and tensor.max() <= 1:
                tensor = tensor * 255.0
            tensor = tensor.round().clamp(0, 255).to(torch.uint8)
        resized = torch_functional.interpolate(
            tensor.unsqueeze(0).float(),
            size=(self.grounding_image_size, self.grounding_image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        return resized.round().clamp(0, 255).to(torch.uint8)

    def _expanded_question(self, tile_counts: Sequence[int]) -> str:
        if len(tile_counts) != WINDOW_SIZE:
            raise ProtocolError("Exactly five tile counts are required.")
        question = self.question
        placeholder_count = question.count("<image>")
        if placeholder_count == 0:
            placeholders = "\n".join(
                f"Frame {index}: <image>" for index in range(1, WINDOW_SIZE + 1)
            )
            question = placeholders + "\n" + question
        elif placeholder_count == 1:
            replacement = "\n".join(
                f"Frame {index}: <image>" for index in range(1, WINDOW_SIZE + 1)
            )
            question = question.replace("<image>", replacement, 1)
        elif placeholder_count != WINDOW_SIZE:
            raise ProtocolError(
                f"A five-frame question needs 0, 1, or 5 image placeholders, got "
                f"{placeholder_count}."
            )
        for tile_count in tile_counts:
            visual_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * (int(tile_count) * self.image_tokens_per_tile)
                + IMG_END_TOKEN
            )
            question = question.replace("<image>", visual_tokens, 1)
        if "<image>" in question:
            raise ProtocolError("Not all frame image placeholders were expanded.")
        return question

    def _format_prompt(self, tile_counts: Sequence[int]) -> tuple[str, bool, str]:
        expanded_question = self._expanded_question(tile_counts)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": expanded_question})
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return str(prompt), True, expanded_question
            except (TypeError, ValueError, NotImplementedError):
                pass

        template = self.prompt_template
        instruction = None
        if isinstance(template, Mapping):
            instruction = template.get("INSTRUCTION")
        elif template is not None:
            instruction = getattr(template, "INSTRUCTION", None)
        if instruction:
            return (
                str(instruction).format(input=expanded_question, round=1),
                False,
                expanded_question,
            )
        prefix = f"{self.system_prompt}\n" if self.system_prompt else ""
        return prefix + expanded_question + "\n", False, expanded_question

    def _tokenize_sample(
        self,
        answer_text: str,
        entity_char_spans: Mapping[str, Sequence[EntityCharSpan]],
        tile_counts: Sequence[int],
    ) -> dict[str, Any]:
        prompt, prompt_has_template_tokens, expanded_question = self._format_prompt(
            tile_counts
        )
        expected_visual_tokens = sum(tile_counts) * self.image_tokens_per_tile
        if prompt.count(IMG_CONTEXT_TOKEN) != expected_visual_tokens:
            raise ProtocolError(
                "Visual context-token count does not match the dynamic tile count."
            )
        prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        bos = [] if prompt_has_template_tokens or bos_id is None else [int(bos_id)]
        token_base = len(bos) + len(prompt_ids)
        answer_ids, ranges, range_labels = map_entity_spans_to_tokens(
            answer_text,
            entity_char_spans,
            self.tokenizer,
            token_base=token_base,
            strict_decode=self.strict_token_decode,
        )
        eos = [] if eos_id is None else [int(eos_id)]
        input_ids = bos + [int(item) for item in prompt_ids] + answer_ids + eos
        labels = [IGNORE_INDEX] * token_base + list(answer_ids) + list(eos)
        if len(input_ids) > self.max_length:
            raise ProtocolError(
                f"Five-frame sample has {len(input_ids)} tokens, exceeding max_length "
                f"{self.max_length}; truncation would break grounding alignment."
            )
        for role_ranges in ranges.values():
            for start, end in role_ranges:
                if not (0 <= start < end <= len(input_ids)):
                    raise ProtocolError("An entity token range lies outside input_ids.")
                if any(value == IGNORE_INDEX for value in labels[start:end]):
                    raise ProtocolError("Entity supervision cannot point into prompt labels.")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "entity_token_ranges": {
                role: [list(span) for span in ranges[role]] for role in ENTITY_ROLES
            },
            "entity_range_labels": range_labels,
            "prompt_text": prompt,
            "answer_text": answer_text,
            "conversation": [
                {"role": "user", "content": expanded_question},
                {"role": "assistant", "content": answer_text},
            ],
        }

    def prepare_data(self, index: int) -> dict[str, Any]:
        if not self._windows:
            raise IndexError("The dataset contains no complete valid five-frame windows.")
        frames = self._windows[index % self.real_len()]
        images = [self._load_image(frame) for frame in frames]
        pixel_values = [self._process_pixel_image(image) for image in images]
        tile_counts = [int(value.shape[0]) for value in pixel_values]
        if any(count <= 0 for count in tile_counts):
            raise ProtocolError("Every frame must produce at least one visual tile.")
        ground_pixels = [self._process_ground_image(image) for image in images]

        canonical_frames: list[CanonicalFrame] = []
        flat_masks: list[torch.Tensor] = []
        role_keys_per_frame: list[list[str]] = []
        frame_labels: list[dict[str, list[str]]] = []
        groundings_per_frame: list[list[dict[str, Any]]] = []
        num_segs_per_frame: list[int] = []

        for frame, image in zip(frames, images):
            raw_labels = frame.get(
                "labels", frame.get("frame_labels", frame.get("semantic_labels"))
            )
            canonical = build_canonical_frame(
                str(frame["caption"]),
                frame["groundings"],
                raw_labels=raw_labels,
                explicit_think=frame.get("think"),
            )
            height, width = self._frame_hw(frame, image)
            frame_masks = [
                _decode_grounding_mask(
                    spec.payload,
                    height,
                    width,
                    self.mask_decoder,
                )
                for spec in canonical.groundings
            ]
            if len(frame_masks) != len(canonical.role_keys):
                raise ProtocolError("One mask is required for every frame segment marker.")
            canonical_frames.append(canonical)
            flat_masks.extend(frame_masks)
            role_keys = list(canonical.role_keys)
            role_keys_per_frame.append(role_keys)
            num_segs_per_frame.append(len(role_keys))
            frame_labels.append(
                {role: list(canonical.labels[role]) for role in ENTITY_ROLES}
            )
            groundings_per_frame.append(
                [
                    {
                        "role_key": spec.role_key,
                        "role": spec.role,
                        "label": spec.label,
                        "caption_span": [spec.start, spec.end],
                    }
                    for spec in canonical.groundings
                ]
            )

        answer_parts: list[str] = []
        combined_spans: dict[str, list[EntityCharSpan]] = {
            role: [] for role in ENTITY_ROLES
        }
        cursor = 0
        for frame_index, canonical in enumerate(canonical_frames, start=1):
            prefix = f"Frame {frame_index}:\n"
            part = prefix + canonical.text
            if answer_parts:
                cursor += 2
            base = cursor + len(prefix)
            for role in ENTITY_ROLES:
                combined_spans[role].extend(
                    EntityCharSpan(base + span.start, base + span.end, span.label)
                    for span in canonical.entity_char_spans[role]
                )
            answer_parts.append(part)
            cursor += len(part)
        answer_text = "\n\n".join(answer_parts)

        mask_role_keys = [key for frame_keys in role_keys_per_frame for key in frame_keys]
        if not (
            answer_text.count(SEG_TOKEN)
            == len(mask_role_keys)
            == len(flat_masks)
            == sum(num_segs_per_frame)
        ):
            raise ProtocolError("[SEG], role-key, and binary-mask counts are inconsistent.")
        mask_prefixes = tuple(f"{role}:" for role in MASK_ROLES)
        if any(not key.startswith(mask_prefixes) for key in mask_role_keys):
            raise ProtocolError("Only instrument and target role keys may own masks.")

        tokenized = self._tokenize_sample(answer_text, combined_spans, tile_counts)
        video_id = str(frames[0]["_video_id"])
        frame_ids = [str(frame["_frame_id"]) for frame in frames]
        output: dict[str, Any] = {
            **tokenized,
            "pixel_values": pixel_values,
            "g_pixel_values": ground_pixels,
            "masks": torch.stack(flat_masks, dim=0),
            "role_keys_per_frame": role_keys_per_frame,
            "mask_role_keys": mask_role_keys,
            "num_segs_per_frame": num_segs_per_frame,
            "frame_labels": frame_labels,
            "labels_per_frame": frame_labels,
            "groundings_per_frame": groundings_per_frame,
            "video_id": video_id,
            "frame_id": frame_ids,
            "frame_ids": frame_ids,
            "frame_metadata": [
                {"video_id": video_id, "frame_id": frame_id}
                for frame_id in frame_ids
            ],
            "frames_per_sample": WINDOW_SIZE,
            "num_patches_per_frame": tile_counts,
            "num_image_tokens_per_frame": [
                count * self.image_tokens_per_tile for count in tile_counts
            ],
            "num_visual_tiles": sum(tile_counts),
            "num_image_tokens": sum(tile_counts) * self.image_tokens_per_tile,
            "type": "video",
        }
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.prepare_data(index % self.real_len())
