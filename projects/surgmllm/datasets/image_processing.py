"""Lightweight InternVL-compatible dynamic image tiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Match the reference InternVL tie-breaking rule."""

    best_difference = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio


def _as_rgb_pil(image: Any):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required by the default dynamic image processor.") from error

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC/CHW image, got {array.shape}.")
    if isinstance(image, torch.Tensor) and array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    elif array.shape[-1] not in {1, 3, 4} and array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and array.max() <= 1:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def dynamic_preprocess(
    image: Any,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    """Split one image into aspect-matched square tiles plus an optional thumbnail."""

    if min_num <= 0 or max_num < min_num:
        raise ValueError("Dynamic patch bounds must satisfy 0 < min_num <= max_num.")
    pil_image = _as_rgb_pil(image)
    width, height = pil_image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    target_ratios = sorted(
        {
            (columns, rows)
            for count in range(min_num, max_num + 1)
            for columns in range(1, count + 1)
            for rows in range(1, count + 1)
            if min_num <= columns * rows <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = find_closest_aspect_ratio(
        width / height,
        target_ratios,
        width,
        height,
        image_size,
    )
    target_width, target_height = image_size * columns, image_size * rows
    resized = pil_image.resize((target_width, target_height))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(pil_image.resize((image_size, image_size)))
    return tiles


@dataclass
class InternVLDynamicProcessor:
    """Callable default processor producing normalized ``[tiles, 3, 448, 448]``."""

    image_size: int = 448
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    use_thumbnail: bool = True
    patch_size: int = 14
    downsample_ratio: float = 0.5
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive.")
        tokens = (self.image_size // self.patch_size) ** 2 * self.downsample_ratio**2
        if int(tokens) != tokens:
            raise ValueError("Dynamic processor settings must produce an integer token count.")
        self.num_image_tokens_per_tile = int(tokens)

    def __call__(self, image: Any) -> torch.Tensor:
        tiles = dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=self.max_dynamic_patch,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        tensors = []
        for tile in tiles:
            array = np.asarray(tile, dtype=np.uint8).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors, dim=0)

