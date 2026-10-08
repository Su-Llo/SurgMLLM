"""Canonical, cross-checked frame identity for data and evaluation records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any


def normalize_video_id(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:vid(?:eo)?[_ -]?)?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"VID{int(match.group(1)):02d}"
    return text.upper()


def normalize_frame_id(value: object) -> str:
    text = Path(str(value)).stem.strip()
    numeric_suffix = re.search(r"(\d+)$", text)
    if numeric_suffix:
        return str(int(numeric_suffix.group(1)))
    return text.upper()


def _one_consistent_identity(
    kind: str, candidates: list[tuple[str, object]], normalizer
) -> str:
    normalized = [
        (source, normalizer(value))
        for source, value in candidates
        if value not in (None, "")
    ]
    values = {value for _, value in normalized if value}
    if len(values) > 1:
        details = ", ".join(f"{source}={value}" for source, value in normalized)
        raise ValueError(f"Conflicting frame {kind} identities: {details}")
    if not values:
        raise ValueError(f"A frame needs a {kind} identity")
    return values.pop()


def record_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve identity and reject disagreement among explicit, path, and ID fields."""

    video_candidates: list[tuple[str, object]] = []
    frame_candidates: list[tuple[str, object]] = []
    for field in ("video_id", "video"):
        if record.get(field) not in (None, ""):
            video_candidates.append((field, record[field]))
    for field in ("frame_id", "frame"):
        if record.get(field) not in (None, ""):
            frame_candidates.append((field, record[field]))

    file_name = record.get("file_name", record.get("image_path", record.get("image")))
    if file_name not in (None, ""):
        file_path = Path(str(file_name))
        # Absolute paths are local inputs, not portable identity claims. Formal
        # training/inference paths reject them separately before opening a file.
        if not file_path.is_absolute():
            clean_parts = tuple(part for part in file_path.parts if part not in ("", "."))
            if len(clean_parts) >= 2 and clean_parts[0] != "..":
                video_candidates.append(("file_name", clean_parts[0]))
            frame_candidates.append(("file_name", file_path.stem))

    image_id = str(record.get("image_id", "")).strip()
    if "_" in image_id:
        image_video, image_frame = image_id.rsplit("_", 1)
        video_candidates.append(("image_id", image_video))
        frame_candidates.append(("image_id", image_frame))

    return (
        _one_consistent_identity("video", video_candidates, normalize_video_id),
        _one_consistent_identity("frame", frame_candidates, normalize_frame_id),
    )
