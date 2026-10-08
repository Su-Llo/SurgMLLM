#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

GPUS=${GPUS:-8}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_CONVERT=${SKIP_CONVERT:-0}
SKIP_VIS=${SKIP_VIS:-1}
VIS_WORKERS=${VIS_WORKERS:-8}
CHECKPOINT=${CHECKPOINT:-}
if [[ -n "$CHECKPOINT" ]]; then SKIP_TRAIN=1; fi

CONFIG=projects/surgmllm/configs/surgmllm_llm_fold1.py
DEFAULT_EPOCH=5
EVAL_SPLIT=fold1

if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$VIS_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "VIS_WORKERS must be a positive integer" >&2
  exit 2
fi
for FLAG_NAME in SKIP_TRAIN SKIP_CONVERT SKIP_VIS; do
  FLAG_VALUE=${!FLAG_NAME}
  if [[ "$FLAG_VALUE" != 0 && "$FLAG_VALUE" != 1 ]]; then
    echo "$FLAG_NAME must be 0 or 1" >&2
    exit 2
  fi
done
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  AVAILABLE_GPUS=$(
    "$REPO_ROOT/.venv/bin/python" -c 'import torch; print(torch.cuda.device_count())'
  )
elif command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
  AVAILABLE_GPUS=0
fi
if [[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]]; then
  if (( GPUS > AVAILABLE_GPUS )); then
    echo "Requested $GPUS GPUs, but PyTorch can see only $AVAILABLE_GPUS" >&2
    exit 2
  fi
fi

WORK_DIR=${WORK_DIR:-$REPO_ROOT/work_dirs/surgmllm_llm_fold1}
HF_DIR=${HF_DIR:-$REPO_ROOT/models/HF_SurgMLLM_fold1}
PREDICTION_DIR=${PREDICTION_DIR:-$REPO_ROOT/predictions/fold1}
METRICS_DIR=${METRICS_DIR:-$REPO_ROOT/metrics/fold1}
VIS_DIR=${VIS_DIR:-$REPO_ROOT/visualizations/fold1}
TRAIN_LOG=${TRAIN_LOG:-$WORK_DIR/train.log}

if [[ "$SKIP_TRAIN" != 1 ]]; then
  : "${SURGMLLM_DATA_ROOT:?Set SURGMLLM_DATA_ROOT to the CholecT45-Scene root}"
  : "${INTERNVL_MODEL_PATH:?Set INTERNVL_MODEL_PATH to the expected OpenGVLab snapshot}"
  : "${SAM2_CHECKPOINT:?Set SAM2_CHECKPOINT to the expected Meta checkpoint}"
  echo "========== Step 1: train Fold 1 =========="
  source "$REPO_ROOT/.venv/bin/activate"
  mkdir -p "$WORK_DIR"
  bash tools/dist.sh train "$CONFIG" "$GPUS" --work-dir "$WORK_DIR" \
    2>&1 | tee "$TRAIN_LOG"
fi

CHECKPOINT=${CHECKPOINT:-$WORK_DIR/epoch_${DEFAULT_EPOCH}.pth}
if [[ "$SKIP_CONVERT" != 1 ]]; then
  : "${SURGMLLM_DATA_ROOT:?Set SURGMLLM_DATA_ROOT for config reconstruction}"
  : "${INTERNVL_MODEL_PATH:?Set INTERNVL_MODEL_PATH for checkpoint reconstruction}"
  : "${SAM2_CHECKPOINT:?Set SAM2_CHECKPOINT for checkpoint reconstruction}"
  [[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
  echo "========== Step 2: checkpoint to Hugging Face =========="
  source "$REPO_ROOT/.venv/bin/activate"
  PYTHONPATH="$REPO_ROOT" python tools/convert_surgmllm_to_hf.py \
    "$CONFIG" "$CHECKPOINT" --save-path "$HF_DIR"
fi

[[ -d "$HF_DIR" ]] || { echo "HF model not found: $HF_DIR" >&2; exit 2; }
: "${SURGMLLM_DATA_ROOT:?Set SURGMLLM_DATA_ROOT for inference and metrics}"
echo "========== Step 3: distributed $EVAL_SPLIT inference =========="
source "$REPO_ROOT/.venv/bin/activate"
PYTHONPATH="$REPO_ROOT" torchrun --standalone --nproc_per_node="$GPUS" \
  projects/surgmllm/evaluation/surgmllm_infer_gcg_fold1.py \
  --model "$HF_DIR" --data-root "$SURGMLLM_DATA_ROOT" \
  --split "$EVAL_SPLIT" --window-size 5 --window-stride 5 \
  --output "$PREDICTION_DIR/raw_predictions.json"

echo "========== Step 4: metrics and optional visualization =========="
EVAL_ARGS=(
  --predictions "$PREDICTION_DIR/raw_predictions.json"
  --data-root "$SURGMLLM_DATA_ROOT"
  --split "$EVAL_SPLIT"
  --output-dir "$METRICS_DIR"
)
if [[ "$SKIP_VIS" == 1 ]]; then
  EVAL_ARGS+=(--skip-vis)
else
  EVAL_ARGS+=(--vis-dir "$VIS_DIR" --vis-workers "$VIS_WORKERS")
fi
PYTHONPATH="$REPO_ROOT" python \
  projects/surgmllm/evaluation/surgmllm_eval_gcg_fold1.py \
  "${EVAL_ARGS[@]}"

echo "========== SurgMLLM pipeline complete =========="
