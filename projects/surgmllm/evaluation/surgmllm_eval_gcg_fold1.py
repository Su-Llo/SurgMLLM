"""Fold-1 raw-prediction evaluator for unified surgical scene understanding."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    # Support direct execution from the repository checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from projects.surgmllm.evaluation.io import (  # type: ignore
        FOLD1_VIDEO_IDS,
        canonicalize_annotation,
        canonicalize_prediction,
        frame_key,
        load_annotation_records,
        load_split_annotations,
        load_prediction_records,
        materialize_groundings,
        materialize_mask,
    )
    from projects.surgmllm.evaluation.metrics import (  # type: ignore
        evaluate_phase_videos,
        evaluate_triplet_ap,
        match_grounding_instances,
    )
    from projects.surgmllm.evaluation.parser import parse_generated_frame  # type: ignore
    from projects.surgmllm.evaluation.taxonomy import (  # type: ignore
        PHASE_CLASSES,
        TRIPLET_TO_ID,
        canonical_triplet,
        normalize_label,
    )
else:
    from .io import (
        FOLD1_VIDEO_IDS,
        canonicalize_annotation,
        canonicalize_prediction,
        frame_key,
        load_annotation_records,
        load_split_annotations,
        load_prediction_records,
        materialize_groundings,
        materialize_mask,
    )
    from .metrics import evaluate_phase_videos, evaluate_triplet_ap, match_grounding_instances
    from .parser import parse_generated_frame
    from .taxonomy import PHASE_CLASSES, TRIPLET_TO_ID, canonical_triplet, normalize_label


SCHEMA_VERSION = "surgmllm.fold1-evaluation.v1"


def _sort_frame(record: Mapping[str, Any]) -> tuple[int, int | str]:
    frame = frame_key(record)[1]
    return (0, int(frame)) if frame.isdigit() else (1, frame)


def _triplet_tuple(value: object) -> tuple[str, str, str] | None:
    raw: object = value
    if isinstance(value, Mapping):
        raw = value.get("triplet", value)
        if isinstance(raw, Mapping):
            raw = (
                raw.get("instrument", ""),
                raw.get("verb", raw.get("action", "")),
                raw.get("target", ""),
            )
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    return canonical_triplet(raw[0], raw[1], raw[2])


def _confidence(value: object, default: object = None) -> object:
    if not isinstance(value, Mapping):
        return default
    return value.get("confidence", value.get("probability", value.get("prob", default)))


def _bounded_confidence(value: object) -> tuple[float, bool]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    return (score, True) if math.isfinite(score) and 0.0 <= score <= 1.0 else (0.0, False)


def _safe_mask(value: object, record: Mapping[str, Any]) -> tuple[object | None, str | None]:
    if value is None:
        return None, None
    try:
        return materialize_mask(value, base_dir=record.get("_base_dir")), None
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return None, str(error)


def _structured_semantics(record: Mapping[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    phase_value = record.get("phase")
    if isinstance(phase_value, Mapping):
        phase = phase_value.get("label")
    else:
        phase = phase_value
    normalized_phase = normalize_label(phase) if phase is not None else None
    triplets = [
        dict(value) if isinstance(value, Mapping) else {"triplet": value}
        for value in record.get("triplets", ())
    ]
    return normalized_phase, triplets


def _derive_prediction(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {
            "phase": None,
            "triplets": [],
            "groundings": [],
            "invalid_events": 0,
            "parse_failed": False,
            "issues": [],
            "invalid_masks": 0,
        }

    generated_text = record.get("generated_text")
    provided_phase, provided_triplets = _structured_semantics(record)
    issues: list[dict[str, str]] = []
    invalid_events = 0
    parse_failed = False

    semantic_phase = provided_phase
    semantic_triplets: list[tuple[str, str, str] | None] = [
        _triplet_tuple(value) for value in provided_triplets
    ]
    semantic_grounding_flags: list[tuple[bool, bool] | None] = [
        None for _ in semantic_triplets
    ]
    structured_for_semantics: list[Mapping[str, Any] | None] = [
        value for value in provided_triplets
    ]

    if isinstance(generated_text, str):
        parsed = parse_generated_frame(generated_text)
        issues.extend(issue.as_dict() for issue in parsed.issues)
        if parsed.ok:
            semantic_phase = parsed.phase
            semantic_triplets = [triplet.as_tuple() for triplet in parsed.triplets]
            semantic_grounding_flags = [
                (triplet.instrument_grounded, triplet.target_grounded)
                for triplet in parsed.triplets
            ]
            structured_for_semantics = []
            for index, triplet in enumerate(semantic_triplets):
                structured = provided_triplets[index] if index < len(provided_triplets) else None
                if structured is not None and _triplet_tuple(structured) != triplet:
                    invalid_events += 1
                    issues.append(
                        {
                            "code": "structured_text_mismatch",
                            "message": (
                                f"structured triplet {index + 1} disagrees with generated text"
                            ),
                        }
                    )
                    structured = None
                structured_for_semantics.append(structured)
            if provided_phase is not None and provided_phase != semantic_phase:
                invalid_events += 1
                issues.append(
                    {
                        "code": "structured_phase_mismatch",
                        "message": "structured phase disagrees with generated text",
                    }
                )
            if len(provided_triplets) > len(semantic_triplets):
                extra = len(provided_triplets) - len(semantic_triplets)
                invalid_events += extra
                issues.append(
                    {
                        "code": "extra_structured_triplets",
                        "message": f"{extra} structured triplet(s) have no generated counterpart",
                    }
                )
        else:
            parse_failed = True
            invalid_events += max(1, len(parsed.triplets))
            semantic_phase = None
            semantic_triplets = []
            semantic_grounding_flags = []
            structured_for_semantics = []
    else:
        parse_failed = True
        invalid_events += 1
        issues.append(
            {
                "code": "missing_generated_text"
                if generated_text is None
                else "generated_text_not_string",
                "message": "strict generated text is required for semantic scoring",
            }
        )
        semantic_phase = None
        semantic_triplets = []
        semantic_grounding_flags = []
        structured_for_semantics = []

    if (
        not isinstance(generated_text, str)
        and semantic_phase is not None
        and semantic_phase not in PHASE_CLASSES
    ):
        invalid_events += 1
        issues.append(
            {"code": "invalid_phase", "message": f"unknown phase label: {semantic_phase!r}"}
        )

    if record.get("parse_error"):
        invalid_events += 1
        issues.append(
            {"code": "reported_parse_error", "message": str(record.get("parse_error"))}
        )

    raw_phase_confidence: object = 1.0
    if isinstance(record.get("phase"), Mapping):
        raw_phase_confidence = record["phase"].get("confidence")
    phase_confidence, phase_confidence_ok = _bounded_confidence(raw_phase_confidence)
    if not phase_confidence_ok:
        invalid_events += 1
        issues.append(
            {
                "code": "invalid_phase_confidence",
                "message": "phase confidence must be a finite number in [0, 1]",
            }
        )

    output_triplets: list[dict[str, Any]] = []
    triplet_groundings: list[dict[str, Any]] = []
    invalid_masks = 0
    for index, triplet in enumerate(semantic_triplets):
        structured = (
            structured_for_semantics[index]
            if index < len(structured_for_semantics)
            else None
        )
        grounding_flags = (
            semantic_grounding_flags[index]
            if index < len(semantic_grounding_flags)
            else None
        )
        if triplet is None:
            output_triplets.append({"raw": provided_triplets[index], "confidence": 1.0})
            continue
        prediction = {
            "instrument": triplet[0],
            "verb": triplet[1],
            "target": triplet[2],
            "confidence": _confidence(structured),
        }
        output_triplets.append(prediction)
        if triplet not in TRIPLET_TO_ID:
            continue
        raw_instrument_mask = (
            structured.get("instrument_mask")
            if isinstance(structured, Mapping)
            else None
        )
        if (
            grounding_flags is not None
            and not grounding_flags[0]
            and raw_instrument_mask is not None
        ):
            invalid_masks += 1
            issues.append(
                {
                    "code": "mask_without_instrument_seg",
                    "message": "instrument mask has no grounded instrument occurrence",
                }
            )
            raw_instrument_mask = None
        instrument_mask, error = _safe_mask(raw_instrument_mask, record)
        if error:
            invalid_masks += 1
            issues.append({"code": "invalid_instrument_mask", "message": error})
        elif instrument_mask is not None:
            triplet_groundings.append(
                {"role": "instrument", "label": triplet[0], "mask": instrument_mask}
            )
        elif grounding_flags is not None and grounding_flags[0]:
            invalid_masks += 1
            issues.append(
                {
                    "code": "missing_instrument_mask",
                    "message": "grounded instrument occurrence has no mask",
                }
            )
        raw_target_mask = (
            structured.get("target_mask")
            if isinstance(structured, Mapping)
            else None
        )
        if triplet[2] == "null target" and raw_target_mask is not None:
            invalid_masks += 1
            issues.append(
                {
                    "code": "null_target_mask",
                    "message": "null target must not own a target mask",
                }
            )
        elif triplet[2] != "null target":
            if (
                grounding_flags is not None
                and not grounding_flags[1]
                and raw_target_mask is not None
            ):
                invalid_masks += 1
                issues.append(
                    {
                        "code": "mask_without_target_seg",
                        "message": "target mask has no grounded target occurrence",
                    }
                )
                raw_target_mask = None
            target_mask, error = _safe_mask(raw_target_mask, record)
            if error:
                invalid_masks += 1
                issues.append({"code": "invalid_target_mask", "message": error})
            elif target_mask is not None:
                triplet_groundings.append(
                    {"role": "target", "label": triplet[2], "mask": target_mask}
                )
            elif grounding_flags is not None and grounding_flags[1]:
                invalid_masks += 1
                issues.append(
                    {
                        "code": "missing_target_mask",
                        "message": "grounded target occurrence has no mask",
                    }
                )

    explicit_groundings: list[dict[str, Any]] = []
    raw_explicit = record.get("groundings", ())
    if (parse_failed or isinstance(generated_text, str)) and raw_explicit:
        explicit_count = (
            len(raw_explicit)
            if isinstance(raw_explicit, (Mapping, list, tuple))
            else 1
        )
        invalid_masks += max(1, explicit_count)
        issues.append(
            {
                "code": "unbound_explicit_grounding",
                "message": (
                    "generated predictions must bind each mask to its tagged "
                    "triplet occurrence"
                ),
            }
        )
    else:
        try:
            explicit_groundings = materialize_groundings(record)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            invalid_masks += 1
            issues.append({"code": "invalid_explicit_grounding", "message": str(error)})

    invalid_events += invalid_masks
    return {
        "phase": semantic_phase,
        "phase_confidence": phase_confidence,
        "triplets": output_triplets,
        "groundings": explicit_groundings + triplet_groundings,
        "invalid_events": invalid_events,
        "parse_failed": parse_failed,
        "issues": issues,
        "invalid_masks": invalid_masks,
    }


def _annotation_triplets(record: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    for value in record.get("triplets", ()):
        triplet = _triplet_tuple(value)
        if triplet is None:
            raise ValueError(
                f"malformed ground-truth triplet at {frame_key(record)}: {value!r}"
            )
        if triplet not in TRIPLET_TO_ID:
            raise ValueError(
                f"ground-truth triplet outside CholecT100 at {frame_key(record)}: {triplet!r}"
            )
        triplets.append(triplet)
    return triplets


def _coverage(
    annotation_keys: set[tuple[str, str]],
    prediction_keys: set[tuple[str, str]],
    duplicate_count: int,
) -> dict[str, Any]:
    matched = annotation_keys.intersection(prediction_keys)
    missing = annotation_keys.difference(prediction_keys)
    unexpected = prediction_keys.difference(annotation_keys)
    return {
        "annotated_frames": len(annotation_keys),
        "predicted_frames": len(prediction_keys),
        "matched_frames": len(matched),
        "missing_frames": len(missing),
        "unexpected_frames": len(unexpected),
        "duplicate_prediction_records": duplicate_count,
        "coverage": len(matched) / len(annotation_keys) if annotation_keys else 0.0,
        "coverage_ratio": len(matched) / len(annotation_keys) if annotation_keys else 0.0,
        "missing_frame_ids": [f"{video}/{frame}" for video, frame in sorted(missing)],
        "unexpected_frame_ids": [
            f"{video}/{frame}" for video, frame in sorted(unexpected)
        ],
    }


def _flat_triplet_video(result: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "invalid_prediction_count": result["invalid_prediction_count"],
        "valid_prediction_count": result["valid_prediction_count"],
        "duplicate_prediction_count": result["duplicate_prediction_count"],
        "invalid_penalty_factor": result["invalid_penalty_factor"],
        "num_frames": result["num_frames"],
        "components": result["components"],
    }
    for component, metrics in result["components"].items():
        output[f"AP_{component}"] = metrics["mAP"]
        output[f"AP_{component}_penalized"] = metrics["penalized_mAP"]
    return output


def _pool_grounding_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    all_scores: list[float] = []
    instrument_scores: list[float] = []
    target_scores: list[float] = []
    per_label_scores: dict[str, list[float]] = defaultdict(list)
    pred_masks = 0
    matched = 0
    unmatched_predictions = 0
    invalid_prediction_masks = 0
    frame_details: list[dict[str, Any]] = []
    for result in results:
        scores = [float(value) for value in result["gt_ious"]]
        all_scores.extend(scores)
        instrument_scores.extend(float(value) for value in result["instrument_ious"])
        target_scores.extend(float(value) for value in result["target_ious"])
        for label, score in zip(result["gt_keys"], scores):
            per_label_scores[str(label)].append(score)
        pred_masks += int(result["num_pred_masks"])
        matched += int(result["num_matched"])
        unmatched_predictions += int(result["num_unmatched_pred"])
        invalid_prediction_masks += int(result["invalid_prediction_mask_count"])
        frame_details.append(
            {
                "video_id": result.get("video_id"),
                "frame_id": result.get("frame_id"),
                "mIoU": sum(scores) / len(scores) if scores else 0.0,
                "num_gt_masks": len(scores),
                "num_pred_masks": result["num_pred_masks"],
                "num_unmatched_gt": result["num_unmatched_gt"],
                "num_unmatched_pred": result["num_unmatched_pred"],
                "invalid_prediction_mask_count": result[
                    "invalid_prediction_mask_count"
                ],
            }
        )

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "IoU_I": mean(instrument_scores),
        "IoU_T": mean(target_scores),
        "mIoU": mean(all_scores),
        "num_gt_masks": len(all_scores),
        "num_instrument_gt_masks": len(instrument_scores),
        "num_target_gt_masks": len(target_scores),
        "num_pred_masks": pred_masks,
        "num_matched": matched,
        "num_unmatched_gt": len(all_scores) - matched,
        "num_unmatched_pred": unmatched_predictions,
        "invalid_prediction_mask_count": invalid_prediction_masks,
        "per_label": {
            label: mean(scores) for label, scores in sorted(per_label_scores.items())
        },
        "frames": frame_details,
    }


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate normalized prediction and annotation records."""

    annotation_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in annotations:
        key = frame_key(record)
        if not all(key):
            raise ValueError(f"annotation has an empty frame identity: {record!r}")
        if key in annotation_by_key:
            raise ValueError(f"duplicate annotation frame: {key}")
        annotation_by_key[key] = record

    prediction_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in predictions:
        key = frame_key(record)
        if not all(key):
            duplicate_records["<invalid-identity>"].append(record)
            continue
        if key in prediction_by_key:
            duplicate_records[key[0]].append(record)
            continue
        prediction_by_key[key] = record

    annotation_keys = set(annotation_by_key)
    prediction_keys = set(prediction_by_key)
    all_coverage = _coverage(
        annotation_keys,
        prediction_keys,
        sum(len(values) for values in duplicate_records.values()),
    )

    frame_records_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phase_pairs: dict[str, tuple[list[object], list[object]]] = {}
    invalid_counts: dict[str, int] = defaultdict(int)
    invalid_audit: dict[str, Counter[str]] = defaultdict(Counter)
    grounding_results_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)

    annotation_videos = sorted({key[0] for key in annotation_keys})
    for video_id in annotation_videos:
        phase_pairs[video_id] = ([], [])
    sorted_annotations = sorted(
        annotation_by_key.values(), key=lambda record: (frame_key(record)[0], _sort_frame(record))
    )
    for annotation in sorted_annotations:
        key = frame_key(annotation)
        video_id, frame_id = key
        prediction = prediction_by_key.get(key)
        derived = _derive_prediction(prediction)
        invalid_counts[video_id] += derived["invalid_events"]
        if derived["parse_failed"]:
            invalid_audit[video_id]["parse_failed_frames"] += 1
        invalid_audit[video_id]["invalid_masks"] += derived["invalid_masks"]
        for issue in derived["issues"]:
            invalid_audit[video_id][f"issue:{issue['code']}"] += 1

        gt_phase = annotation.get("phase")
        if gt_phase is not None:
            normalized_gt_phase = normalize_label(gt_phase)
            if normalized_gt_phase not in PHASE_CLASSES:
                raise ValueError(
                    f"unknown ground-truth phase at {key}: {normalized_gt_phase!r}"
                )
            phase_pairs[video_id][0].append(normalized_gt_phase)
            phase_pairs[video_id][1].append(derived["phase"])

        gt_groundings = materialize_groundings(annotation)
        grounding_result = match_grounding_instances(
            gt_groundings, derived["groundings"]
        )
        shape_invalid = int(grounding_result["invalid_prediction_mask_count"])
        if shape_invalid:
            invalid_counts[video_id] += shape_invalid
            invalid_audit[video_id]["invalid_masks"] += shape_invalid
            invalid_audit[video_id]["issue:incompatible_mask_shape"] += shape_invalid
        grounding_result["video_id"] = video_id
        grounding_result["frame_id"] = frame_id
        grounding_results_by_video[video_id].append(grounding_result)
        frame_records_by_video[video_id].append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "gt_triplets": _annotation_triplets(annotation),
                "pred_triplets": derived["triplets"],
                "parse_failed": derived["parse_failed"],
                "parse_issues": derived["issues"],
            }
        )

    # Unexpected frames are retained as empty-GT frames for AP false positives.
    for key in sorted(prediction_keys.difference(annotation_keys)):
        video_id, frame_id = key
        derived = _derive_prediction(prediction_by_key[key])
        invalid_counts[video_id] += derived["invalid_events"] + 1
        invalid_audit[video_id]["unexpected_frames"] += 1
        if derived["parse_failed"]:
            invalid_audit[video_id]["parse_failed_frames"] += 1
        invalid_audit[video_id]["invalid_masks"] += derived["invalid_masks"]
        for issue in derived["issues"]:
            invalid_audit[video_id][f"issue:{issue['code']}"] += 1
        frame_records_by_video[video_id].append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "gt_triplets": [],
                "pred_triplets": derived["triplets"],
                "parse_failed": derived["parse_failed"],
                "parse_issues": derived["issues"],
                "unexpected": True,
            }
        )

    for video_id, records in duplicate_records.items():
        for duplicate_index, record in enumerate(records, start=1):
            derived = _derive_prediction(record)
            invalid_counts[video_id] += derived["invalid_events"] + 1
            invalid_audit[video_id]["duplicate_prediction_records"] += 1
            if derived["parse_failed"]:
                invalid_audit[video_id]["parse_failed_frames"] += 1
            invalid_audit[video_id]["invalid_masks"] += derived["invalid_masks"]
            for issue in derived["issues"]:
                invalid_audit[video_id][f"issue:{issue['code']}"] += 1
            # Treat every discarded duplicate as an additional empty-GT frame,
            # so valid-taxonomy duplicate content becomes an AP false positive
            # and invalid-taxonomy content is counted by the metric adapter.
            frame_records_by_video[video_id].append(
                {
                    "video_id": video_id,
                    "frame_id": f"duplicate-{duplicate_index}",
                    "gt_triplets": [],
                    "pred_triplets": derived["triplets"],
                    "duplicate": True,
                }
            )

    phase_metrics = evaluate_phase_videos(phase_pairs)
    triplet_metrics = evaluate_triplet_ap(
        frame_records_by_video, invalid_counts=invalid_counts
    )

    grounding_by_video = {
        video_id: _pool_grounding_results(grounding_results_by_video[video_id])
        for video_id in annotation_videos
    }
    grounding_overall = _pool_grounding_results(
        result
        for video_id in annotation_videos
        for result in grounding_results_by_video[video_id]
    )

    per_video: dict[str, dict[str, Any]] = {}
    all_video_ids = sorted(set(annotation_videos) | set(frame_records_by_video))
    for video_id in all_video_ids:
        annotation_video_keys = {key for key in annotation_keys if key[0] == video_id}
        prediction_video_keys = {key for key in prediction_keys if key[0] == video_id}
        coverage = _coverage(
            annotation_video_keys,
            prediction_video_keys,
            len(duplicate_records.get(video_id, ())),
        )
        triplet_video = triplet_metrics["per_video"].get(video_id)
        per_video[video_id] = {
            "coverage": coverage,
            "phase": phase_metrics["per_video"].get(video_id),
            "triplet": _flat_triplet_video(triplet_video) if triplet_video else None,
            "grounding": grounding_by_video.get(video_id),
            "invalid": dict(sorted(invalid_audit[video_id].items())),
        }

    overall_invalid = Counter()
    for counter in invalid_audit.values():
        overall_invalid.update(counter)
    overall_invalid["invalid_prediction_events"] = triplet_metrics[
        "invalid_prediction_count"
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "fold": 1,
            "phase_protocol": "unrelaxed_video_independent",
            "triplet_taxonomy": "CholecT100_fixed",
            "grounding_pooling": "all_ground_truth_masks",
            "num_prediction_records": len(predictions),
            "num_annotation_records": len(annotations),
        },
        "overall": {
            "coverage": all_coverage,
            "phase": {key: value for key, value in phase_metrics.items() if key != "per_video"},
            "triplet": {
                key: value
                for key, value in triplet_metrics.items()
                if key != "per_video"
            },
            "grounding": grounding_overall,
            "invalid": dict(sorted(overall_invalid.items())),
        },
        "per_video": per_video,
    }
    return report


def _normalize_direct_records(
    records: Sequence[Mapping[str, Any]], kind: str
) -> list[dict[str, Any]]:
    converter = canonicalize_prediction if kind == "prediction" else canonicalize_annotation
    return [
        converter(record, base_dir=Path.cwd())
        if "_base_dir" not in record
        else dict(record)
        for record in records
    ]


def evaluate(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Public convenience API accepting canonical or adapter-compatible records."""

    return evaluate_predictions(
        _normalize_direct_records(predictions, "prediction"),
        _normalize_direct_records(annotations, "annotation"),
    )


run_evaluation = evaluate


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    per_video_dir = destination / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    combined_path = destination / "evaluation_results.json"
    overall_path = destination / "overall.json"
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with overall_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": report["schema_version"],
                "metadata": report["metadata"],
                "overall": report["overall"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    for video_id, metrics in report["per_video"].items():
        with (per_video_dir / f"{video_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": report["schema_version"],
                    "video_id": video_id,
                    **metrics,
                },
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
    return {
        "combined": str(combined_path),
        "overall": str(overall_path),
        "per_video": str(per_video_dir),
    }


def _visualization_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "requested": True,
        "status": manifest.get("status"),
        "output_dir": manifest.get("output_dir"),
        "manifest_path": manifest.get("manifest_path"),
        "rendered_frames": manifest.get("rendered_frames"),
        "missing_frames": manifest.get("missing_frames"),
        "error_count": manifest.get("error_count"),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SurgMLLM Fold-1 raw prediction JSON files"
    )
    parser.add_argument(
        "--predictions", required=True, help="prediction JSON file or directory"
    )
    parser.add_argument(
        "--split",
        default="fold1",
        choices=("fold1",),
        help="Fold-1 test videos",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", help="canonical/legacy annotation JSON path")
    source.add_argument("--data-root", help="dataset root containing Fold-1 annotations")
    parser.add_argument("--output-dir", required=True, help="directory for metric JSON files")
    parser.add_argument(
        "--vis-dir",
        help="visualization output directory (defaults to OUTPUT_DIR/visualizations)",
    )
    parser.add_argument(
        "--vis-workers",
        type=int,
        default=8,
        help="number of concurrent Pillow rendering workers",
    )
    parser.add_argument(
        "--skip-vis",
        "--skip_vis",
        dest="skip_vis",
        action="store_true",
        help="skip optional visual rendering; all metrics are still computed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.vis_workers <= 0:
        raise ValueError("--vis-workers must be positive")
    predictions = load_prediction_records(args.predictions)
    annotations = (
        load_annotation_records(args.annotations)
        if args.annotations
        else load_split_annotations(args.data_root, args.split)
    )
    if args.annotations:
        expected_videos = set(FOLD1_VIDEO_IDS)
        annotations = [
            record for record in annotations if frame_key(record)[0] in expected_videos
        ]
    report = evaluate_predictions(predictions, annotations)
    report["metadata"]["split"] = args.split
    visualization_manifest: Mapping[str, Any] = {
        "status": "skipped",
        "rendered_frames": 0,
        "error_count": 0,
    }
    render_error: Exception | None = None
    if args.skip_vis:
        report["metadata"]["visualization"] = {
            "requested": False,
            "status": "skipped",
        }
    else:
        if __package__ in (None, ""):
            from projects.surgmllm.evaluation.visualization import (  # type: ignore
                VisualizationRenderError,
                render_visualizations,
            )
        else:
            from .visualization import VisualizationRenderError, render_visualizations

        vis_dir = Path(args.vis_dir) if args.vis_dir else Path(args.output_dir) / "visualizations"
        try:
            visualization_manifest = render_visualizations(
                predictions,
                annotations,
                report,
                vis_dir,
                workers=args.vis_workers,
            )
        except VisualizationRenderError as error:
            visualization_manifest = error.manifest
            render_error = error
        report["metadata"]["visualization"] = _visualization_metadata(
            visualization_manifest
        )

    paths = write_report(report, args.output_dir)
    overall = report["overall"]
    compact_summary = {
        "coverage": {
            key: value
            for key, value in overall["coverage"].items()
            if not key.endswith("_ids")
        },
        "phase": {
            key: overall["phase"][key]
            for key in (
                "video_accuracy",
                "frame_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_jaccard",
            )
        },
        "triplet": {
            key: value
            for key, value in overall["triplet"].items()
            if key.startswith("AP_")
        },
        "grounding": {
            key: overall["grounding"][key]
            for key in ("IoU_I", "IoU_T", "mIoU", "num_gt_masks")
        },
        "invalid": overall["invalid"],
    }
    print(
        json.dumps(
            {
                "outputs": paths,
                "metrics": compact_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if render_error is not None:
        raise render_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
