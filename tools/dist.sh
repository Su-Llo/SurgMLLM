#!/usr/bin/env bash
set -euo pipefail

FILE=${1:?Python tool name is required}
CONFIG=${2:?Config path is required}
NPROC=${3:?GPU process count is required}
shift 3

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
DEEPSPEED=${DEEPSPEED:-deepspeed_zero2}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NPROC" \
  "tools/${FILE}.py" "$CONFIG" \
  --launcher pytorch --deepspeed "$DEEPSPEED" "$@"
