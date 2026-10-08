"""JSON adapters and optional mask decoding for the Fold-1 evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from ..record_identity import normalize_frame_id, normalize_video_id, record_identity
from ..datasets.splits import FOLD1_TEST_VIDEO_IDS, format_video_id
from .parser import parse_legacy_annotation_caption
from .taxonomy import canonical_triplet, normalize_label


FOLD1_VIDEO_IDS = tuple(format_video_id(video_id) for video_id in FOLD1_TEST_VIDEO_IDS)


def frame_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return normalize_video_id(record.get("video_id", "")), normalize_frame_id(
        record.get("frame_id", "")
    )


def _identity_from_record(record: Mapping[str, Any]) -> tuple[str, str]:
    return record_identity(record)


def _phase_value(value: object) -> tuple[str | None, object]:
    confidence: object = 1.0
    label: object = value
    if isinstance(value, Mapping):
        label = value.get("label", value.get("name"))
        confidence = value.get("confidence", value.get("probability"))
    if label is None:
        return None, confidence
    return normalize_label(label), confidence


def _triplet_dict(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) != 3:
            return {"raw": value, "confidence": 1.0}
        value = {"instrument": parts[0], "verb": parts[1], "target": parts[2]}
    if isinstance(value, (list, tuple)):
        if len(value) < 3:
            return {"raw": list(value), "confidence": 1.0}
        confidence = value[3] if len(value) > 3 else None
        value = {
            "instrument": value[0],
            "verb": value[1],
            "target": value[2],
            "confidence": confidence,
        }
    if not isinstance(value, Mapping):
        return {"raw": value, "confidence": 1.0}
    nested = value.get("triplet")
    if nested is not None and nested is not value:
        result = _triplet_dict(nested)
        outer_confidence = value.get(
            "confidence", value.get("probability", value.get("prob"))
        )
        if outer_confidence is not None:
            result["confidence"] = outer_confidence
        for key in ("instrument_mask", "target_mask"):
            if key in value:
                result[key] = value[key]
        return result
    result = dict(value)
    if "action" in result and "verb" not in result:
        result["verb"] = result["action"]
    if any(key in result for key in ("confidence", "probability", "prob")):
        result["confidence"] = result.get(
            "confidence", result.get("probability", result.get("prob"))
        )
    return result


def canonicalize_prediction(record: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    video_id, frame_id = _identity_from_record(record)
    phase, phase_confidence = _phase_value(record.get("phase", record.get("pred_phase")))
    generated_text = record.get(
        "generated_text", record.get("raw_output", record.get("caption", record.get("text")))
    )
    raw_triplets = record.get("triplets", record.get("pred_triplets", ()))
    if raw_triplets is None:
        raw_triplets = ()
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "generated_text": generated_text,
        "parse_error": record.get("parse_error"),
        "phase": {"label": phase, "confidence": phase_confidence},
        "triplets": [_triplet_dict(value) for value in raw_triplets],
        "groundings": list(record.get("groundings", ()))
        if isinstance(record.get("groundings", ()), (list, tuple))
        else record.get("groundings", {}),
        "image_path": record.get("image_path", record.get("file_name")),
        "_base_dir": str(base_dir),
    }


def _mask_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, Mapping) and "rle_masks" in value:
        masks = value.get("rle_masks")
        return list(masks) if isinstance(masks, (list, tuple)) else [masks]
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        first = value[0]
        if isinstance(first, Mapping):
            return list(value)
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
            return list(value)
    return [value]


def _groundings_from_triplets(triplets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groundings: list[dict[str, Any]] = []
    for triplet in triplets:
        instrument = normalize_label(triplet.get("instrument", ""))
        target = normalize_label(triplet.get("target", ""))
        for mask in _mask_values(triplet.get("instrument_mask")):
            groundings.append({"role": "instrument", "label": instrument, "mask": mask})
        if target != "null target":
            for mask in _mask_values(triplet.get("target_mask")):
                groundings.append({"role": "target", "label": target, "mask": mask})
    return groundings


def _legacy_groundings(
    raw_groundings: object, triplets: Sequence[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    if not isinstance(raw_groundings, Mapping):
        return []
    roles_by_label: dict[str, set[str]] = {}
    for instrument, _, target in triplets:
        roles_by_label.setdefault(instrument, set()).add("instrument")
        if target != "null target":
            roles_by_label.setdefault(target, set()).add("target")
    result: list[dict[str, Any]] = []
    for raw_label, raw_masks in raw_groundings.items():
        label = normalize_label(raw_label)
        for role in sorted(roles_by_label.get(label, ())):
            for mask in _mask_values(raw_masks):
                result.append({"role": role, "label": label, "mask": mask})
    return result


def canonicalize_annotation(record: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    video_id, frame_id = _identity_from_record(record)
    raw_triplets = record.get("triplets", record.get("gt_triplets"))
    legacy_phase: str | None = None
    legacy_triplets: tuple[tuple[str, str, str], ...] = ()
    if raw_triplets is None and record.get("caption"):
        legacy_phase, legacy_triplets = parse_legacy_annotation_caption(record["caption"])
        raw_triplets = legacy_triplets
    if raw_triplets is None:
        raw_triplets = ()
    triplet_dicts = [
        _triplet_dict(value) for value in raw_triplets if str(value).upper() != "NULL"
    ]
    phase, _ = _phase_value(record.get("phase", record.get("gt_phase", legacy_phase)))

    explicit_groundings = record.get("groundings", ())
    if isinstance(explicit_groundings, Mapping):
        tuple_values = [
            canonical_triplet(
                triplet.get("instrument", ""),
                triplet.get("verb", ""),
                triplet.get("target", ""),
            )
            for triplet in triplet_dicts
            if all(key in triplet for key in ("instrument", "verb", "target"))
        ]
        groundings = _legacy_groundings(explicit_groundings, tuple_values)
    else:
        groundings = [dict(value) for value in explicit_groundings] if explicit_groundings else []
    groundings.extend(_groundings_from_triplets(triplet_dicts))
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "phase": phase,
        "triplets": triplet_dicts,
        "groundings": groundings,
        "image_path": record.get("image_path", record.get("file_name")),
        "height": record.get("height"),
        "width": record.get("width"),
        "_base_dir": str(base_dir),
    }


def _unwrap_payload(payload: object, kind: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [value for value in payload if isinstance(value, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("JSON root must be an object or list")
    preferred = "predictions" if kind == "prediction" else "annotations"
    if isinstance(payload.get(preferred), list):
        return [value for value in payload[preferred] if isinstance(value, Mapping)]
    if isinstance(payload.get("frames"), Mapping):
        video_id = payload.get("video_id", "")
        records: list[Mapping[str, Any]] = []
        for frame_id, value in payload["frames"].items():
            if not isinstance(value, Mapping):
                continue
            merged = dict(value)
            merged.setdefault("video_id", video_id)
            merged.setdefault("frame_id", frame_id)
            records.append(merged)
        return records
    if kind == "prediction" and isinstance(payload.get("frame_results"), list):
        frame_results: list[Mapping[str, Any]] = []
        for value in payload["frame_results"]:
            if not isinstance(value, Mapping):
                continue
            merged = dict(value)
            frame_file = merged.get("frame_file", "")
            merged.setdefault("frame_id", Path(str(frame_file)).stem)
            frame_results.append(merged)
        return frame_results
    if "video_id" in payload and "frame_id" in payload:
        return [payload]
    return []


def _read_records(path: Path, kind: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = _unwrap_payload(payload, kind)
    path_video = next(
        (
            normalize_video_id(part)
            for part in reversed(path.parts)
            if re.fullmatch(r"VID\d+", part, flags=re.IGNORECASE)
        ),
        "",
    )
    prepared: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        if path_video and value.get("video_id", value.get("video")) in (None, ""):
            value["video_id"] = path_video
        prepared.append(value)
    converter = canonicalize_prediction if kind == "prediction" else canonicalize_annotation
    return [converter(record, base_dir=path.parent) for record in prepared]


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        return _read_records(source, "prediction")
    if not source.is_dir():
        raise FileNotFoundError(f"prediction path does not exist: {source}")
    records: list[dict[str, Any]] = []
    for json_path in sorted(source.rglob("*.json")):
        records.extend(_read_records(json_path, "prediction"))
    return records


def load_annotation_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        return _read_records(source, "annotation")
    if not source.is_dir():
        raise FileNotFoundError(f"annotation path does not exist: {source}")
    records: list[dict[str, Any]] = []
    for json_path in sorted(source.rglob("*.json")):
        records.extend(_read_records(json_path, "annotation"))
    return records


def _annotation_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    candidate_roots = (
        root,
        root / "annotations",
    )
    for candidate in candidate_roots:
        if candidate.is_dir() and any(candidate.glob("VID*_GCG.json")):
            return candidate
    raise FileNotFoundError(
        "could not locate VID*_GCG.json annotations under the data root"
    )


def load_video_annotations(
    data_root: str | Path, video_ids: Sequence[str]
) -> list[dict[str, Any]]:
    annotation_root = _annotation_root(data_root)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw_video_id in video_ids:
        video_id = normalize_video_id(raw_video_id)
        path = annotation_root / f"{video_id}_GCG.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        records.extend(_read_records(path, "annotation"))
    if missing:
        raise FileNotFoundError(
            "missing requested annotation files: " + ", ".join(missing)
        )
    return records


def load_fold1_annotations(data_root: str | Path) -> list[dict[str, Any]]:
    return load_video_annotations(data_root, FOLD1_VIDEO_IDS)


def load_split_annotations(data_root: str | Path, split: str) -> list[dict[str, Any]]:
    normalized = split.strip().lower()
    if normalized in {"fold1", "test", "val", "validation"}:
        return load_video_annotations(data_root, FOLD1_VIDEO_IDS)
    raise ValueError(f"unknown evaluation split: {split!r}")


def _decode_uncompressed_rle(rle: Mapping[str, Any]) -> object:
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("RLE size must contain [height, width]")
    if not isinstance(counts, (list, tuple)):
        raise TypeError("compressed RLE requires the optional pycocotools dependency")
    height, width = int(size[0]), int(size[1])
    total = height * width
    lengths = [int(run) for run in counts]
    if any(length < 0 for length in lengths):
        raise ValueError("RLE counts cannot be negative")
    decoded_size = sum(lengths)
    if decoded_size != total:
        raise ValueError(f"RLE decodes to {decoded_size} pixels, expected {total}")
    try:
        import numpy as np  # type: ignore
    except ImportError:
        flat: list[bool] = []
        value = False
        for length in lengths:
            flat.extend([value] * length)
            value = not value
        # COCO RLE is column-major.
        return [
            [flat[column * height + row] for column in range(width)]
            for row in range(height)
        ]
    values = np.arange(len(lengths), dtype=np.uint8) % 2
    flat_array = np.repeat(values, lengths).astype(bool, copy=False)
    return flat_array.reshape((height, width), order="F")


def _decode_compressed_counts(value: str | bytes) -> list[int]:
    """Decode the compact COCO counts string without importing image packages."""

    text = value.decode("ascii") if isinstance(value, bytes) else value
    counts: list[int] = []
    cursor = 0
    while cursor < len(text):
        decoded = 0
        shift = 0
        more = True
        last = 0
        while more:
            if cursor >= len(text):
                raise ValueError("truncated compressed COCO RLE")
            last = ord(text[cursor]) - 48
            if not 0 <= last <= 63:
                raise ValueError("invalid character in compressed COCO RLE")
            decoded |= (last & 0x1F) << (5 * shift)
            cursor += 1
            shift += 1
            more = bool(last & 0x20)
        if last & 0x10:
            decoded |= -1 << (5 * shift)
        if len(counts) > 2:
            decoded += counts[-2]
        if decoded < 0:
            raise ValueError("compressed COCO RLE produced a negative run")
        counts.append(decoded)
    return counts


def materialize_mask(value: object, *, base_dir: str | Path | None = None) -> object:
    """Load an in-memory mask, COCO RLE, or a path with optional dependencies."""

    if isinstance(value, Mapping):
        if "mask" in value and not ("size" in value and "counts" in value):
            return materialize_mask(value["mask"], base_dir=base_dir)
        if "array" in value:
            return value["array"]
        if "rle" in value:
            return materialize_mask(value["rle"], base_dir=base_dir)
        if "path" in value:
            return materialize_mask(value["path"], base_dir=base_dir)
        if "size" in value and "counts" in value:
            if isinstance(value["counts"], (list, tuple)):
                return _decode_uncompressed_rle(value)
            if isinstance(value["counts"], (str, bytes)):
                return _decode_uncompressed_rle(
                    {"size": value["size"], "counts": _decode_compressed_counts(value["counts"])}
                )
            raise TypeError("COCO RLE counts must be a sequence, string, or bytes")
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "shape") and hasattr(value, "ravel")
    ):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return materialize_mask(json.load(handle), base_dir=path.parent)
        if suffix == ".npy":
            try:
                import numpy as np  # type: ignore
            except ImportError as error:
                raise RuntimeError(".npy masks require numpy") from error
            return np.load(path, allow_pickle=False)
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("image mask paths require Pillow") from error
        with Image.open(path) as image:
            return [[bool(pixel) for pixel in row] for row in _image_rows(image.convert("L"))]
    raise TypeError(f"unsupported mask value: {type(value).__name__}")


def _image_rows(image: object) -> Iterable[tuple[int, ...]]:
    width, height = image.size  # type: ignore[attr-defined]
    pixels = list(image.getdata())  # type: ignore[attr-defined]
    for row in range(height):
        yield tuple(pixels[row * width : (row + 1) * width])


def materialize_groundings(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("groundings", ())
    if isinstance(raw, Mapping):
        values = [
            {"role": value.get("role", ""), "label": label, "mask": value.get("mask")}
            for label, value in raw.items()
            if isinstance(value, Mapping)
        ]
    else:
        values = list(raw) if isinstance(raw, (list, tuple)) else []
    base_dir = record.get("_base_dir")
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping) or value.get("mask") is None:
            continue
        result.append(
            {
                "role": normalize_label(value.get("role", "")),
                "label": normalize_label(value.get("label", value.get("entity", ""))),
                "mask": materialize_mask(value["mask"], base_dir=base_dir),
            }
        )
    return result
