"""Evaluation APIs for SurgMLLM."""

from .metrics import (
    GroundingInstance,
    compute_grounding_metrics,
    compute_phase_metrics,
    compute_triplet_metrics,
    evaluate_grounding_frames,
    evaluate_phase_video,
    evaluate_phase_videos,
    evaluate_triplet_ap,
    match_grounding_instances,
    mask_iou,
    sparse_average_precision,
)
from .parser import (
    FrameParseResult,
    GenerationParseError,
    ParsedGrounding,
    ParsedTriplet,
    StrictFrameParser,
    WindowParseResult,
    parse_generated_frame,
    parse_generated_text,
    parse_generated_window,
)
from .surgmllm_eval_gcg_fold1 import evaluate, evaluate_predictions, write_report

__all__ = [
    "FrameParseResult",
    "GenerationParseError",
    "GroundingInstance",
    "ParsedGrounding",
    "ParsedTriplet",
    "StrictFrameParser",
    "WindowParseResult",
    "compute_grounding_metrics",
    "compute_phase_metrics",
    "compute_triplet_metrics",
    "evaluate",
    "evaluate_grounding_frames",
    "evaluate_phase_video",
    "evaluate_phase_videos",
    "evaluate_predictions",
    "evaluate_triplet_ap",
    "match_grounding_instances",
    "mask_iou",
    "parse_generated_frame",
    "parse_generated_text",
    "parse_generated_window",
    "sparse_average_precision",
    "write_report",
]
