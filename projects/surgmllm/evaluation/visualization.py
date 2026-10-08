"""Pillow-only qualitative visualization for SurgMLLM frame predictions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import colorsys
import hashlib
import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import frame_key, materialize_groundings, materialize_mask
from .parser import parse_generated_frame
from .taxonomy import normalize_label


VISUALIZATION_SCHEMA_VERSION = "surgmllm.visualization.v1"
_PANEL_MAX_SIZE = (640, 360)
_TEXT_HEIGHT = 252


class VisualizationRenderError(RuntimeError):
    """Raised after all requested frames finish when one or more failed to render."""

    def __init__(self, manifest: Mapping[str, Any], manifest_path: str | Path):
        self.manifest = dict(manifest)
        self.manifest_path = str(manifest_path)
        count = int(manifest.get("error_count", 0))
        super().__init__(
            f"{count} visualization frame(s) failed; details: {self.manifest_path}"
        )


def _frame_sort_key(key: tuple[str, str]) -> tuple[str, int, int | str]:
    video_id, frame_id = key
    return (video_id, 0, int(frame_id)) if frame_id.isdigit() else (video_id, 1, frame_id)


def _role_label_color(role: object, label: object) -> tuple[int, int, int]:
    """Return a stable role-aware RGB color without process-randomized ``hash``."""

    normalized_role = normalize_label(role)
    normalized_label = normalize_label(label)
    digest = hashlib.sha256(f"{normalized_role}:{normalized_label}".encode("utf-8")).digest()
    # Instruments use warm hues; targets use cool hues. Unknown roles use the full wheel.
    if normalized_role == "instrument":
        hue = (350.0 + (digest[0] / 255.0) * 70.0) % 360.0
    elif normalized_role == "target":
        hue = 145.0 + (digest[0] / 255.0) * 95.0
    else:
        hue = (digest[0] / 255.0) * 360.0
    saturation = 0.72 + (digest[1] / 255.0) * 0.18
    value = 0.88 + (digest[2] / 255.0) * 0.10
    red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    return int(red * 255), int(green * 255), int(blue * 255)


def _resolve_image_path(
    prediction: Mapping[str, Any], annotation: Mapping[str, Any]
) -> Path:
    for record in (prediction, annotation):
        raw_path = record.get("image_path", record.get("file_name"))
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if path.is_absolute():
            candidates = (path,)
        else:
            base_dir = Path(str(record.get("_base_dir", ".")))
            candidates = (
                base_dir / path,
                base_dir.parent / "videos" / path,
                base_dir.parent.parent / "videos" / path,
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    video_id, frame_id = frame_key(annotation)
    if not video_id or not frame_id:
        video_id, frame_id = frame_key(prediction)
    identity = f"{video_id}/{frame_id}" if video_id or frame_id else "unknown frame"
    raise FileNotFoundError(f"source image not found for {identity}")


def _triplet_groundings(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    base_dir = record.get("_base_dir")
    triplets = record.get("triplets", ())
    if not isinstance(triplets, (list, tuple)):
        return result
    for value in triplets:
        if not isinstance(value, Mapping):
            continue
        instrument = normalize_label(value.get("instrument", ""))
        target = normalize_label(value.get("target", ""))
        for role, label, field in (
            ("instrument", instrument, "instrument_mask"),
            ("target", target, "target_mask"),
        ):
            if value.get(field) is None or (role == "target" and label == "null target"):
                continue
            masks = value[field]
            if isinstance(masks, Mapping) and "rle_masks" in masks:
                masks = masks["rle_masks"]
            if not isinstance(masks, (list, tuple)) or (
                masks and not isinstance(masks[0], Mapping)
            ):
                masks = [masks]
            for mask in masks:
                result.append(
                    {
                        "role": role,
                        "label": label,
                        "mask": materialize_mask(mask, base_dir=base_dir),
                    }
                )
    return result


def _groundings(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = materialize_groundings(record)
    return explicit if explicit else _triplet_groundings(record)


def _mask_image(mask: object, size: tuple[int, int]) -> Image.Image:
    array = np.asarray(mask)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D after squeezing, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("mask cannot be empty")
    binary = Image.fromarray((array.astype(bool) * 255).astype(np.uint8), mode="L")
    if binary.size != size:
        binary = binary.resize(size, resample=Image.Resampling.NEAREST)
    return binary


def _overlay(
    image: Image.Image, groundings: Sequence[Mapping[str, Any]]
) -> tuple[Image.Image, list[dict[str, Any]]]:
    rendered = image.convert("RGBA")
    legend: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for grounding in groundings:
        role = normalize_label(grounding.get("role", ""))
        label = normalize_label(grounding.get("label", ""))
        color = _role_label_color(role, label)
        mask = _mask_image(grounding.get("mask"), rendered.size)
        layer = Image.new("RGBA", rendered.size, (*color, 0))
        layer.putalpha(mask.point(lambda value: 104 if value else 0))
        rendered = Image.alpha_composite(rendered, layer)
        key = (role, label)
        if key not in seen:
            legend.append({"role": role, "label": label, "color": list(color)})
            seen.add(key)
    return rendered.convert("RGB"), legend


def _fit_panel(image: Image.Image) -> Image.Image:
    scale = min(_PANEL_MAX_SIZE[0] / image.width, _PANEL_MAX_SIZE[1] / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _phase(record: Mapping[str, Any]) -> str:
    value = record.get("phase")
    if isinstance(value, Mapping):
        value = value.get("label")
    return normalize_label(value) if value is not None else "<missing>"


def _triplet_text(record: Mapping[str, Any]) -> str:
    values: list[str] = []
    triplets = record.get("triplets", ())
    if not isinstance(triplets, (list, tuple)):
        return "<none>"
    for value in triplets:
        if isinstance(value, Mapping):
            values.append(
                "(" + ", ".join(
                    normalize_label(value.get(field, ""))
                    for field in ("instrument", "verb", "target")
                ) + ")"
            )
        elif isinstance(value, (list, tuple)) and len(value) >= 3:
            values.append("(" + ", ".join(normalize_label(item) for item in value[:3]) + ")")
        else:
            values.append(str(value))
    return "; ".join(values) if values else "<none>"


def _parse_status(prediction: Mapping[str, Any]) -> str:
    if prediction.get("parse_error"):
        return f"FAILED: {prediction['parse_error']}"
    generated = prediction.get("generated_text")
    if not isinstance(generated, str):
        return "FAILED: missing generated_text"
    parsed = parse_generated_frame(generated)
    return "OK" if parsed.ok else f"FAILED: {parsed.parse_error or 'invalid grammar'}"


def _draw_text_block(
    canvas: Image.Image,
    *,
    top: int,
    prediction: Mapping[str, Any],
    annotation: Mapping[str, Any],
    frame_miou: float | None,
    gt_legend: Sequence[Mapping[str, Any]],
    pred_legend: Sequence[Mapping[str, Any]],
) -> None:
    draw = ImageDraw.Draw(canvas)
    regular = _font(15)
    bold = _font(17)
    video_id, frame_id = frame_key(annotation)
    miou = "n/a" if frame_miou is None else f"{frame_miou:.4f}"
    lines = [
        f"{video_id} / frame {frame_id}    frame mIoU: {miou}    parse: {_parse_status(prediction)}",
        f"GT   phase: {_phase(annotation)}    IVT: {_triplet_text(annotation)}",
        f"Pred phase: {_phase(prediction)}    IVT: {_triplet_text(prediction)}",
    ]
    generated = str(prediction.get("generated_text") or prediction.get("raw_generated_text") or "")
    generated = re.sub(r"\s+", " ", generated).strip()
    lines.extend("Generated: " + line if index == 0 else line for index, line in enumerate(
        textwrap.wrap(generated or "<missing>", width=155)[:4]
    ))
    draw.text((12, top + 8), lines[0], fill="white", font=bold)
    y = top + 34
    for line in lines[1:]:
        draw.text((12, y), line, fill=(225, 225, 225), font=regular)
        y += 23
    legend_x = 12
    legend_y = top + _TEXT_HEIGHT - 28
    for prefix, entries in (("GT", gt_legend), ("Pred", pred_legend)):
        for entry in entries:
            color = tuple(entry["color"])
            draw.rectangle((legend_x, legend_y, legend_x + 13, legend_y + 13), fill=color)
            label = f"{prefix} {entry['role']}:{entry['label']}"
            draw.text((legend_x + 18, legend_y - 2), label, fill="white", font=regular)
            legend_x += max(135, 18 + len(label) * 8)


def _render_frame(
    prediction: Mapping[str, Any],
    annotation: Mapping[str, Any],
    output_path: Path,
    frame_miou: float | None,
) -> dict[str, Any]:
    source_path = _resolve_image_path(prediction, annotation)
    with Image.open(source_path) as opened:
        original = opened.convert("RGB")
    gt_overlay, gt_legend = _overlay(original, _groundings(annotation))
    pred_overlay, pred_legend = _overlay(original, _groundings(prediction))
    panels = [_fit_panel(value) for value in (original, gt_overlay, pred_overlay)]
    panel_width, panel_height = panels[0].size
    panels = [value.resize((panel_width, panel_height)) for value in panels]
    header_height = 30
    canvas = Image.new(
        "RGB", (panel_width * 3, header_height + panel_height + _TEXT_HEIGHT), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(zip(("Original", "Ground truth", "Prediction"), panels)):
        left = index * panel_width
        draw.text((left + 10, 6), title, fill="white", font=_font(16))
        canvas.paste(panel, (left, header_height))
    _draw_text_block(
        canvas,
        top=header_height + panel_height,
        prediction=prediction,
        annotation=annotation,
        frame_miou=frame_miou,
        gt_legend=gt_legend,
        pred_legend=pred_legend,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    video_id, frame_id = frame_key(annotation)
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "output_path": f"{video_id}/{output_path.name}",
        "source_path": f"{video_id}/{source_path.name}",
        "frame_mIoU": frame_miou,
        "parse_status": _parse_status(prediction),
        "gt_legend": gt_legend,
        "pred_legend": pred_legend,
    }


def _frame_mious(report: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    grounding = report.get("overall", {}).get("grounding", {})
    frames = grounding.get("frames", ()) if isinstance(grounding, Mapping) else ()
    result: dict[tuple[str, str], float] = {}
    if isinstance(frames, (list, tuple)):
        for value in frames:
            if isinstance(value, Mapping):
                key = frame_key(value)
                try:
                    result[key] = float(value.get("mIoU", 0.0))
                except (TypeError, ValueError):
                    continue
    return result


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "frame"


def render_visualizations(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    output_dir: str | Path,
    workers: int = 8,
) -> dict[str, Any]:
    """Render every matched frame and write a deterministic JSON manifest.

    Missing annotation frames are recorded only in the manifest. If any matched
    frame fails, all other work is allowed to finish, the manifest is written,
    and :class:`VisualizationRenderError` is raised.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"

    annotation_by_key = {frame_key(value): value for value in annotations}
    prediction_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in predictions:
        prediction_by_key.setdefault(frame_key(value), value)
    annotation_keys = {key for key in annotation_by_key if all(key)}
    prediction_keys = {key for key in prediction_by_key if all(key)}
    matched_keys = sorted(annotation_keys & prediction_keys, key=_frame_sort_key)
    missing_keys = sorted(annotation_keys - prediction_keys, key=_frame_sort_key)
    miou_by_key = _frame_mious(report)

    rendered: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    future_to_key: dict[Any, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for key in matched_keys:
            video_id, frame_id = key
            frame_name = f"{int(frame_id):06d}" if frame_id.isdigit() else _safe_component(frame_id)
            output_path = destination / _safe_component(video_id) / f"{frame_name}.png"
            future = executor.submit(
                _render_frame,
                prediction_by_key[key],
                annotation_by_key[key],
                output_path,
                miou_by_key.get(key),
            )
            future_to_key[future] = key
        for future in as_completed(future_to_key):
            video_id, frame_id = future_to_key[future]
            try:
                rendered.append(future.result())
            except Exception as error:  # Aggregate per-frame I/O, mask, and Pillow failures.
                errors.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )

    rendered.sort(key=lambda value: _frame_sort_key(frame_key(value)))
    errors.sort(key=lambda value: _frame_sort_key(frame_key(value)))
    manifest: dict[str, Any] = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "status": "failed" if errors else "complete",
        "output_dir": destination.name,
        "annotated_frames": len(annotation_keys),
        "predicted_frames": len(prediction_keys),
        "matched_frames": len(matched_keys),
        "rendered_frames": len(rendered),
        "missing_frames": len(missing_keys),
        "missing_frame_ids": [f"{video_id}/{frame_id}" for video_id, frame_id in missing_keys],
        "unexpected_frame_ids": [
            f"{video_id}/{frame_id}"
            for video_id, frame_id in sorted(prediction_keys - annotation_keys, key=_frame_sort_key)
        ],
        "error_count": len(errors),
        "errors": errors,
        "frames": rendered,
        "manifest_path": manifest_path.name,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if errors:
        raise VisualizationRenderError(manifest, manifest_path)
    return manifest


__all__ = [
    "VISUALIZATION_SCHEMA_VERSION",
    "VisualizationRenderError",
    "render_visualizations",
]
