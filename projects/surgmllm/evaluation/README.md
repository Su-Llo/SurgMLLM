# SurgMLLM Fold-1 Evaluation

Run the commands from the repository root after activating `.venv` and loading `.env`.

## Inference

Generate predictions with the converted Hugging Face model:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node="$GPUS" \
  projects/surgmllm/evaluation/surgmllm_infer_gcg_fold1.py \
  --model models/HF_SurgMLLM_fold1 \
  --data-root "$SURGMLLM_DATA_ROOT" \
  --split fold1 \
  --window-size 5 \
  --window-stride 5 \
  --output predictions/fold1/raw_predictions.json
```

Add `--max-windows 1` for a bounded smoke run.

## Metrics

Evaluate the generated predictions without rendering visualizations:

```bash
PYTHONPATH=. python -m projects.surgmllm.evaluation \
  --predictions predictions/fold1/raw_predictions.json \
  --data-root "$SURGMLLM_DATA_ROOT" \
  --split fold1 \
  --output-dir metrics/fold1 \
  --skip-vis
```

The evaluator writes `evaluation_results.json`, `overall.json`, and per-video metric files. Remove `--skip-vis` and add `--vis-dir visualizations/fold1` to render mask overlays.
