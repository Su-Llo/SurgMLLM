#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UV_BIN=${UV_BIN:-uv}

cd "$REPO_ROOT"
"$UV_BIN" venv --python 3.10 .venv
"$UV_BIN" pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0
"$UV_BIN" pip install --python .venv/bin/python \
  transformers==4.57.0 peft==0.17.1 xtuner==0.1.23 mmengine==0.10.7 \
  deepspeed==0.18.5 hydra-core==1.3.2 omegaconf==2.3.0 \
  timm==1.0.17 pycocotools pillow numpy scipy scikit-learn pytest

# FlashAttention compiles against the already installed PyTorch/CUDA runtime.
"$UV_BIN" pip install --python .venv/bin/python setuptools wheel ninja packaging
"$UV_BIN" pip install --python .venv/bin/python \
  --no-build-isolation flash-attn==2.7.3

echo "Created $REPO_ROOT/.venv"
echo "This script pins direct core packages; it is not a transitive lockfile install."
