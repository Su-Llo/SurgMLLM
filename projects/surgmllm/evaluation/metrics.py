"""Dependency-light metrics for phase, triplet, and grounding evaluation.

This module intentionally does not import image or plotting libraries.  Masks
may be nested Python sequences or array-like objects that provide ``shape`` and
``ravel``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .taxonomy import (
    CHOLECT100_COMPONENT_MAP,
    INSTRUMENT_CLASSES,
    PHASE_CLASSES,
    TARGET_CLASSES,
    TRIPLET_CLASSES,
    TRIPLET_TO_ID,
    VERB_CLASSES,
    canonical_triplet,
    normalize_label,
    triplet_name,
)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return fmean(materialized) if materialized else 0.0


def _flatten_nested(mask: object) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    if hasattr(mask, "shape") and hasattr(mask, "ravel"):
        shape = tuple(int(value) for value in mask.shape)  # type: ignore[attr-defined]
        flat = tuple(bool(value) for value in mask.ravel())  # type: ignore[attr-defined]
        return shape, flat

    if not isinstance(mask, (list, tuple)):
        raise TypeError("mask must be an array-like object or a nested sequence")

    def visit(value: object) -> tuple[tuple[int, ...], list[bool]]:
        if not isinstance(value, (list, tuple)):
            return (), [bool(value)]
        if not value:
            return (0,), []
        children = [visit(child) for child in value]
        child_shape = children[0][0]
        if any(shape != child_shape for shape, _ in children[1:]):
            raise ValueError("mask is ragged")
        flat_values: list[bool] = []
        for _, child_values in children:
            flat_values.extend(child_values)
        return (len(value),) + child_shape, flat_values

    shape, values = visit(mask)
    return shape, tuple(values)


def mask_iou(prediction: object, target: object) -> float:
    """Return binary intersection-over-union for two equally shaped masks."""

    if all(hasattr(value, "shape") and hasattr(value, "ravel") for value in (prediction, target)):
        pred_shape = tuple(int(value) for value in prediction.shape)  # type: ignore[attr-defined]
        target_shape = tuple(int(value) for value in target.shape)  # type: ignore[attr-defined]
        if pred_shape != target_shape:
            raise ValueError(
                f"mask shapes differ: prediction={pred_shape}, target={target_shape}"
            )
        pred_values = prediction.ravel() != 0  # type: ignore[attr-defined,operator]
        target_values = target.ravel() != 0  # type: ignore[attr-defined,operator]
        intersection = int((pred_values & target_values).sum())
        union = int((pred_values | target_values).sum())
        return intersection / union if union else 0.0

    pred_shape, pred_values = _flatten_nested(prediction)
    target_shape, target_values = _flatten_nested(target)
    if pred_shape != target_shape:
        raise ValueError(f"mask shapes differ: prediction={pred_shape}, target={target_shape}")
    intersection = sum(left and right for left, right in zip(pred_values, target_values))
    union = sum(left or right for left, right in zip(pred_values, target_values))
    return intersection / union if union else 0.0


binary_iou = mask_iou


def _mask_shape(mask: object) -> tuple[int, int]:
    if hasattr(mask, "shape"):
        shape = tuple(int(value) for value in mask.shape)  # type: ignore[attr-defined]
    else:
        shape, _ = _flatten_nested(mask)
    if len(shape) != 2:
        raise ValueError(f"mask must be two-dimensional, got shape {shape}")
    return shape


@dataclass(frozen=True)
class GroundingInstance:
    role: str
    label: str
    mask: object

    @classmethod
    def from_value(cls, value: object) -> "GroundingInstance":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("grounding instance must be a mapping")
        return cls(
            role=normalize_label(value.get("role", "")),
            label=normalize_label(value.get("label", value.get("entity", ""))),
            mask=value.get("mask"),
        )


def _optimal_pairs(matrix: Sequence[Sequence[float]]) -> list[tuple[int, int, float]]:
    """Maximum-IoU one-to-one assignment, preferring cardinality on ties."""

    gt_count = len(matrix)
    pred_count = len(matrix[0]) if matrix else 0
    if not gt_count or not pred_count:
        return []

    # Entity multiplicity is normally tiny.  A deterministic greedy fallback
    # bounds the pathological cost of malformed files with many duplicate masks.
    if pred_count > 14 or gt_count > 14:
        candidates = sorted(
            (
                (float(matrix[gt_index][pred_index]), gt_index, pred_index)
                for gt_index in range(gt_count)
                for pred_index in range(pred_count)
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for score, gt_index, pred_index in candidates:
            if score < 0.0:
                continue
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            pairs.append((gt_index, pred_index, score))
            if len(pairs) == min(gt_count, pred_count):
                break
        return sorted(pairs)

    @lru_cache(maxsize=None)
    def solve(gt_index: int, used: int) -> tuple[float, int, tuple[tuple[int, int], ...]]:
        if gt_index == gt_count:
            return 0.0, 0, ()
        best = solve(gt_index + 1, used)
        for pred_index in range(pred_count):
            bit = 1 << pred_index
            if used & bit:
                continue
            suffix_score, suffix_count, suffix_pairs = solve(gt_index + 1, used | bit)
            candidate = (
                float(matrix[gt_index][pred_index]) + suffix_score,
                suffix_count + 1,
                ((gt_index, pred_index),) + suffix_pairs,
            )
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    _, _, index_pairs = solve(0, 0)
    return [
        (gt_index, pred_index, float(matrix[gt_index][pred_index]))
        for gt_index, pred_index in index_pairs
    ]


def match_grounding_instances(
    ground_truth: Sequence[object], predictions: Sequence[object]
) -> dict[str, Any]:
    """Match masks only when both semantic role and canonical label agree.

    Every ground-truth mask contributes exactly once.  A ground-truth mask with
    no compatible prediction contributes IoU zero.
    """

    gt_instances = [GroundingInstance.from_value(value) for value in ground_truth]
    pred_instances = [GroundingInstance.from_value(value) for value in predictions]
    valid_roles = {"instrument", "target"}
    for instance in gt_instances:
        if instance.role not in valid_roles:
            raise ValueError(f"invalid ground-truth role: {instance.role!r}")
        valid_labels = INSTRUMENT_CLASSES if instance.role == "instrument" else TARGET_CLASSES
        if instance.label not in valid_labels:
            raise ValueError(
                f"invalid ground-truth {instance.role} label: {instance.label!r}"
            )
        if instance.mask is None:
            raise ValueError("ground-truth grounding instance has no mask")

    gt_shapes = [_mask_shape(instance.mask) for instance in gt_instances]
    expected_shape = gt_shapes[0] if gt_shapes else None
    if expected_shape is not None and any(
        shape != expected_shape for shape in gt_shapes[1:]
    ):
        raise ValueError(f"ground-truth mask shapes differ: {sorted(set(gt_shapes))}")

    invalid_predictions: set[int] = set()
    valid_prediction_masks: set[int] = set()
    for index, instance in enumerate(pred_instances):
        if instance.mask is None:
            continue
        try:
            shape = _mask_shape(instance.mask)
        except (TypeError, ValueError):
            invalid_predictions.add(index)
            continue
        if expected_shape is not None and shape != expected_shape:
            invalid_predictions.add(index)
            continue
        valid_prediction_masks.add(index)

    grouped_gt: dict[tuple[str, str], list[tuple[int, GroundingInstance]]] = defaultdict(list)
    grouped_pred: dict[tuple[str, str], list[tuple[int, GroundingInstance]]] = defaultdict(list)
    for index, instance in enumerate(gt_instances):
        grouped_gt[(instance.role, instance.label)].append((index, instance))
    for index, instance in enumerate(pred_instances):
        if instance.role in valid_roles and index in valid_prediction_masks:
            grouped_pred[(instance.role, instance.label)].append((index, instance))

    gt_scores = [0.0] * len(gt_instances)
    matches: list[dict[str, Any]] = []
    used_predictions: set[int] = set()
    for key, gt_group in grouped_gt.items():
        pred_group = grouped_pred.get(key, [])
        matrix: list[list[float]] = []
        for _, gt_instance in gt_group:
            row: list[float] = []
            for pred_index, pred_instance in pred_group:
                try:
                    row.append(mask_iou(gt_instance.mask, pred_instance.mask))
                except (TypeError, ValueError):
                    # A malformed prediction is auditable but must not abort
                    # the remaining videos or accidentally match at IoU zero.
                    row.append(-1.0)
                    invalid_predictions.add(pred_index)
            matrix.append(row)
        for gt_local, pred_local, score in _optimal_pairs(matrix):
            gt_index = gt_group[gt_local][0]
            pred_index = pred_group[pred_local][0]
            gt_scores[gt_index] = score
            used_predictions.add(pred_index)
            matches.append(
                {
                    "gt_index": gt_index,
                    "prediction_index": pred_index,
                    "role": key[0],
                    "label": key[1],
                    "iou": score,
                }
            )

    instrument_scores = [
        score
        for score, instance in zip(gt_scores, gt_instances)
        if instance.role == "instrument"
    ]
    target_scores = [
        score
        for score, instance in zip(gt_scores, gt_instances)
        if instance.role == "target"
    ]
    by_label: dict[str, list[float]] = defaultdict(list)
    for score, instance in zip(gt_scores, gt_instances):
        by_label[f"{instance.role}:{instance.label}"].append(score)

    return {
        "IoU_I": _mean(instrument_scores),
        "IoU_T": _mean(target_scores),
        "mIoU": _mean(gt_scores),
        "num_gt_masks": len(gt_scores),
        "num_instrument_gt_masks": len(instrument_scores),
        "num_target_gt_masks": len(target_scores),
        "num_pred_masks": len(pred_instances),
        "num_matched": len(matches),
        "num_unmatched_gt": len(gt_scores) - len(matches),
        "num_unmatched_pred": len(pred_instances) - len(used_predictions),
        "invalid_prediction_mask_count": len(invalid_predictions),
        "gt_ious": gt_scores,
        "gt_keys": [f"{instance.role}:{instance.label}" for instance in gt_instances],
        "instrument_ious": instrument_scores,
        "target_ious": target_scores,
        "per_label": {label: _mean(scores) for label, scores in sorted(by_label.items())},
        "matches": sorted(matches, key=lambda item: item["gt_index"]),
    }


def evaluate_grounding_frames(frames: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool IoUs over all GT masks, rather than averaging frame means."""

    all_scores: list[float] = []
    instrument_scores: list[float] = []
    target_scores: list[float] = []
    per_label: dict[str, list[float]] = defaultdict(list)
    pred_masks = 0
    matched = 0
    unmatched_predictions = 0
    invalid_prediction_masks = 0
    frame_details: list[dict[str, Any]] = []
    for frame in frames:
        result = match_grounding_instances(
            frame.get("gt_groundings", frame.get("ground_truth", ())),
            frame.get("pred_groundings", frame.get("predictions", ())),
        )
        all_scores.extend(result["gt_ious"])
        instrument_scores.extend(result["instrument_ious"])
        target_scores.extend(result["target_ious"])
        pred_masks += result["num_pred_masks"]
        matched += result["num_matched"]
        unmatched_predictions += result["num_unmatched_pred"]
        invalid_prediction_masks += result["invalid_prediction_mask_count"]
        for label, score in zip(
            (
                f"{GroundingInstance.from_value(value).role}:"
                f"{GroundingInstance.from_value(value).label}"
                for value in frame.get("gt_groundings", frame.get("ground_truth", ()))
            ),
            result["gt_ious"],
        ):
            per_label[label].append(score)
        frame_details.append(
            {
                "video_id": frame.get("video_id"),
                "frame_id": frame.get("frame_id"),
                "mIoU": result["mIoU"],
                "num_gt_masks": result["num_gt_masks"],
                "num_pred_masks": result["num_pred_masks"],
                "num_unmatched_gt": result["num_unmatched_gt"],
                "num_unmatched_pred": result["num_unmatched_pred"],
                "invalid_prediction_mask_count": result[
                    "invalid_prediction_mask_count"
                ],
            }
        )

    return {
        "IoU_I": _mean(instrument_scores),
        "IoU_T": _mean(target_scores),
        "mIoU": _mean(all_scores),
        "num_gt_masks": len(all_scores),
        "num_instrument_gt_masks": len(instrument_scores),
        "num_target_gt_masks": len(target_scores),
        "num_pred_masks": pred_masks,
        "num_matched": matched,
        "num_unmatched_gt": len(all_scores) - matched,
        "num_unmatched_pred": unmatched_predictions,
        "invalid_prediction_mask_count": invalid_prediction_masks,
        "per_label": {label: _mean(scores) for label, scores in sorted(per_label.items())},
        "frames": frame_details,
    }


def evaluate_phase_video(
    ground_truth: Sequence[object],
    predictions: Sequence[object],
    *,
    classes: Sequence[str] = PHASE_CLASSES,
) -> dict[str, Any]:
    """Compute unrelaxed phase metrics for one video only."""

    if len(ground_truth) != len(predictions):
        raise ValueError("phase ground truth and predictions must have equal length")
    gt_labels = [normalize_label(value) for value in ground_truth]
    pred_labels = [normalize_label(value) if value is not None else "" for value in predictions]
    canonical_classes = tuple(normalize_label(value) for value in classes)
    invalid_gt = sorted(set(gt_labels).difference(canonical_classes))
    if invalid_gt:
        raise ValueError(f"unknown ground-truth phase labels: {invalid_gt}")

    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in canonical_classes:
        tp = sum(gt == label and pred == label for gt, pred in zip(gt_labels, pred_labels))
        fp = sum(gt != label and pred == label for gt, pred in zip(gt_labels, pred_labels))
        fn = sum(gt == label and pred != label for gt, pred in zip(gt_labels, pred_labels))
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / support if support else None
        jaccard = tp / (tp + fp + fn) if tp + fp + fn else None
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "jaccard": jaccard,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        }

    total = len(gt_labels)
    correct = sum(gt == pred for gt, pred in zip(gt_labels, pred_labels))
    def macro(metric: str) -> float:
        return _mean(
            float(values[metric])
            for values in per_class.values()
            if values[metric] is not None
        )

    macro_precision = macro("precision")
    macro_recall = macro("recall")
    macro_jaccard = macro("jaccard")
    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_jaccard": macro_jaccard,
        "precision": macro_precision,
        "recall": macro_recall,
        "jaccard": macro_jaccard,
        "per_class": per_class,
        "invalid_prediction_count": sum(pred not in canonical_classes for pred in pred_labels),
        "protocol": "unrelaxed",
    }


def evaluate_phase_videos(
    videos: Mapping[str, tuple[Sequence[object], Sequence[object]]],
    *,
    classes: Sequence[str] = PHASE_CLASSES,
) -> dict[str, Any]:
    """Evaluate each video independently and aggregate without joining timelines."""

    per_video = {
        video_id: evaluate_phase_video(gt, pred, classes=classes)
        for video_id, (gt, pred) in videos.items()
    }
    correct = sum(result["correct"] for result in per_video.values())
    total = sum(result["total"] for result in per_video.values())
    per_class: dict[str, dict[str, float | None]] = {}
    for label in classes:
        label = normalize_label(label)
        per_class[label] = {}
        for metric in ("precision", "recall", "jaccard"):
            values = [
                result["per_class"][label][metric]
                for result in per_video.values()
                if result["per_class"][label][metric] is not None
            ]
            per_class[label][metric] = _mean(float(value) for value in values) if values else None

    def class_macro(metric: str) -> float:
        return _mean(
            float(values[metric])
            for values in per_class.values()
            if values[metric] is not None
        )

    video_accuracy = _mean(result["accuracy"] for result in per_video.values())
    macro_precision = class_macro("precision")
    macro_recall = class_macro("recall")
    macro_jaccard = class_macro("jaccard")
    return {
        "video_accuracy": video_accuracy,
        "accuracy": video_accuracy,
        "frame_accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_jaccard": macro_jaccard,
        "precision": macro_precision,
        "recall": macro_recall,
        "jaccard": macro_jaccard,
        "correct": correct,
        "total": total,
        "num_videos": len(per_video),
        "invalid_prediction_count": sum(
            result["invalid_prediction_count"] for result in per_video.values()
        ),
        "per_class": per_class,
        "per_video": per_video,
        "protocol": "unrelaxed_video_independent",
    }


_COMPONENT_COLUMNS = {"I": 1, "V": 2, "T": 3, "IV": 4, "IT": 5, "IVT": 0}


def _component_label(row: tuple[int, int, int, int, int, int], component: str) -> str:
    _, instrument, verb, target, _, _ = row
    if component == "I":
        return INSTRUMENT_CLASSES[instrument]
    if component == "V":
        return VERB_CLASSES[verb]
    if component == "T":
        return TARGET_CLASSES[target]
    if component == "IV":
        return f"{INSTRUMENT_CLASSES[instrument]},{VERB_CLASSES[verb]}"
    if component == "IT":
        return f"{INSTRUMENT_CLASSES[instrument]},{TARGET_CLASSES[target]}"
    return triplet_name(TRIPLET_CLASSES[row[0]])


def _component_banks() -> dict[str, tuple[dict[int, int], tuple[str, ...]]]:
    banks: dict[str, tuple[dict[int, int], tuple[str, ...]]] = {}
    for component, column in _COMPONENT_COLUMNS.items():
        ids = sorted({row[column] for row in CHOLECT100_COMPONENT_MAP})
        compact = {class_id: index for index, class_id in enumerate(ids)}
        first_rows = {
            class_id: next(row for row in CHOLECT100_COMPONENT_MAP if row[column] == class_id)
            for class_id in ids
        }
        labels = tuple(_component_label(first_rows[class_id], component) for class_id in ids)
        banks[component] = compact, labels
    return banks


_COMPONENT_BANKS = _component_banks()


def _coerce_predicted_triplet(value: object) -> tuple[tuple[str, str, str], float] | None:
    confidence: object = None
    raw: object = value
    if isinstance(value, Mapping):
        confidence = value.get(
            "confidence", value.get("probability", value.get("prob"))
        )
        raw = value.get("triplet", value)
        if isinstance(raw, Mapping):
            raw = (
                raw.get("instrument", ""),
                raw.get("verb", raw.get("action", "")),
                raw.get("target", ""),
            )
    if isinstance(raw, str):
        parts = raw.split(",")
        if len(parts) != 3:
            return None
        raw = tuple(parts)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    if not isinstance(value, Mapping) and len(raw) > 3:
        confidence = raw[3]
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return canonical_triplet(raw[0], raw[1], raw[2]), score


def _coerce_gt_triplet(value: object) -> tuple[str, str, str]:
    raw = value
    if isinstance(value, Mapping):
        raw = value.get("triplet", value)
        if isinstance(raw, Mapping):
            raw = (
                raw.get("instrument", ""),
                raw.get("verb", raw.get("action", "")),
                raw.get("target", ""),
            )
    if isinstance(raw, str):
        raw = tuple(raw.split(","))
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise ValueError(f"invalid ground-truth triplet value: {value!r}")
    triplet = canonical_triplet(raw[0], raw[1], raw[2])
    if triplet not in TRIPLET_TO_ID:
        raise ValueError(f"ground-truth triplet is outside CholecT100: {triplet!r}")
    return triplet


def sparse_average_precision(
    positive_frames: set[int],
    detections: Sequence[tuple[float, int]],
    *,
    total_frames: int | None = None,
) -> float | None:
    """Compute non-interpolated AP from sparse emitted scores.

    Unemitted frame/class pairs have confidence zero.  Equal scores are handled
    as one threshold, matching the precision-recall integration used by the
    dense CholecT100 recognition evaluator.
    """

    if not positive_frames:
        return None
    inferred_frames = max(
        [*positive_frames, *(frame_index for _, frame_index in detections)],
        default=-1,
    ) + 1
    frame_count = inferred_frames if total_frames is None else int(total_frames)
    if frame_count < inferred_frames:
        raise ValueError("total_frames does not cover every AP frame index")
    scores = [0.0] * frame_count
    for confidence, frame_index in detections:
        scores[frame_index] = max(scores[frame_index], float(confidence))

    grouped: dict[float, list[int]] = defaultdict(list)
    for frame_index, confidence in enumerate(scores):
        grouped[confidence].append(frame_index)
    true_positives = 0
    predicted = 0
    previous_recall = 0.0
    average_precision = 0.0
    for confidence in sorted(grouped, reverse=True):
        frame_indices = grouped[confidence]
        true_positives += sum(index in positive_frames for index in frame_indices)
        predicted += len(frame_indices)
        recall = true_positives / len(positive_frames)
        precision = true_positives / predicted
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


average_precision = sparse_average_precision


def _triplet_video_records(
    frames: Sequence[Mapping[str, Any]], external_invalid: int
) -> dict[str, Any]:
    gt_by_component: dict[str, list[set[int]]] = {
        component: [] for component in _COMPONENT_COLUMNS
    }
    pred_by_component: dict[str, list[dict[int, float]]] = {
        component: [] for component in _COMPONENT_COLUMNS
    }
    invalid = int(external_invalid)
    valid_predictions = 0
    duplicate_predictions = 0

    for frame in frames:
        gt_ids = {
            TRIPLET_TO_ID[_coerce_gt_triplet(value)]
            for value in frame.get("gt_triplets", frame.get("ground_truth", ()))
        }
        pred_scores: dict[int, float] = {}
        for value in frame.get("pred_triplets", frame.get("predictions", ())):
            coerced = _coerce_predicted_triplet(value)
            if coerced is None:
                invalid += 1
                continue
            triplet, confidence = coerced
            triplet_id = TRIPLET_TO_ID.get(triplet)
            if triplet_id is None:
                invalid += 1
                continue
            if triplet_id in pred_scores:
                duplicate_predictions += 1
                invalid += 1
                pred_scores[triplet_id] = max(pred_scores[triplet_id], confidence)
                continue
            valid_predictions += 1
            pred_scores[triplet_id] = max(pred_scores.get(triplet_id, -math.inf), confidence)

        for component, column in _COMPONENT_COLUMNS.items():
            compact, _ = _COMPONENT_BANKS[component]
            gt_by_component[component].append(
                {compact[CHOLECT100_COMPONENT_MAP[triplet_id][column]] for triplet_id in gt_ids}
            )
            projected: dict[int, float] = {}
            for triplet_id, confidence in pred_scores.items():
                class_index = compact[CHOLECT100_COMPONENT_MAP[triplet_id][column]]
                projected[class_index] = max(
                    projected.get(class_index, -math.inf), confidence
                )
            pred_by_component[component].append(projected)

    component_results: dict[str, dict[str, Any]] = {}
    penalty_factor = (
        valid_predictions / (valid_predictions + invalid)
        if valid_predictions + invalid
        else 1.0
    )
    for component in _COMPONENT_COLUMNS:
        _, labels = _COMPONENT_BANKS[component]
        per_class: dict[str, float | None] = {}
        for class_index, label in enumerate(labels):
            positives = {
                frame_index
                for frame_index, frame_classes in enumerate(gt_by_component[component])
                if class_index in frame_classes
            }
            detections = [
                (frame_scores[class_index], frame_index)
                for frame_index, frame_scores in enumerate(pred_by_component[component])
                if class_index in frame_scores
            ]
            per_class[label] = sparse_average_precision(
                positives, detections, total_frames=len(frames)
            )
        values = [value for value in per_class.values() if value is not None]
        raw_map = _mean(float(value) for value in values)
        component_results[component] = {
            "mAP": raw_map,
            "penalized_mAP": raw_map * penalty_factor,
            "per_class": per_class,
            "fixed_class_count": len(labels),
            "evaluated_class_count": len(values),
        }

    result = {
        "components": component_results,
        "invalid_prediction_count": invalid,
        "valid_prediction_count": valid_predictions,
        "duplicate_prediction_count": duplicate_predictions,
        "invalid_penalty_factor": penalty_factor,
        "num_frames": len(frames),
    }
    for component, metrics in component_results.items():
        result[f"AP_{component}"] = metrics["mAP"]
        result[f"AP_{component}_penalized"] = metrics["penalized_mAP"]
    return result


def evaluate_triplet_ap(
    videos: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    invalid_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Compute CholecT100 component AP independently for every video.

    Unknown labels, invalid combinations, malformed confidence values, and
    externally reported parse failures are counted.  In addition to the fixed
    taxonomy mAP, ``penalized_mAP`` applies the documented factor
    ``valid / (valid + invalid)`` so hallucinations cannot disappear silently.
    """

    invalid_counts = invalid_counts or {}
    per_video = {
        video_id: _triplet_video_records(frames, invalid_counts.get(video_id, 0))
        for video_id, frames in videos.items()
    }
    valid = sum(result["valid_prediction_count"] for result in per_video.values())
    invalid = sum(result["invalid_prediction_count"] for result in per_video.values())
    duplicates = sum(result["duplicate_prediction_count"] for result in per_video.values())
    penalty_factor = valid / (valid + invalid) if valid + invalid else 1.0

    components: dict[str, dict[str, Any]] = {}
    flat_metrics: dict[str, float] = {}
    for component in _COMPONENT_COLUMNS:
        _, labels = _COMPONENT_BANKS[component]
        per_class: dict[str, float | None] = {}
        for label in labels:
            values = [
                result["components"][component]["per_class"][label]
                for result in per_video.values()
                if result["components"][component]["per_class"][label] is not None
            ]
            per_class[label] = _mean(float(value) for value in values) if values else None
        present_values = [value for value in per_class.values() if value is not None]
        raw_map = _mean(float(value) for value in present_values)
        penalized = raw_map * penalty_factor
        components[component] = {
            "mAP": raw_map,
            "penalized_mAP": penalized,
            "per_class": per_class,
            "fixed_class_count": len(labels),
            "evaluated_class_count": len(present_values),
        }
        flat_metrics[f"AP_{component}"] = raw_map
        flat_metrics[f"AP_{component}_penalized"] = penalized

    return {
        **flat_metrics,
        "components": components,
        "invalid_prediction_count": invalid,
        "valid_prediction_count": valid,
        "duplicate_prediction_count": duplicates,
        "invalid_penalty_factor": penalty_factor,
        "per_video": per_video,
        "protocol": "video_wise_sparse_AP_fixed_CholecT100",
    }


compute_triplet_metrics = evaluate_triplet_ap
compute_grounding_metrics = evaluate_grounding_frames
compute_phase_metrics = evaluate_phase_videos
