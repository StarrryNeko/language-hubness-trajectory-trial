#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LHT_MODEL_ROOT="${BEHAVIOR_MODEL_ROOT:-/root/autodl-tmp/models}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${BEHAVIOR_V3_CONFIG:-configs/behavior_association_v3/xglm_1b7.json}"
STAGE="${1:-}"
LID_MODEL="${BEHAVIOR_LID_MODEL:-/root/autodl-tmp/lid/lid.176.bin}"

if [[ -z "$STAGE" ]]; then
  echo "Usage: bash scripts/run_behavior_association_v3_gpu.sh structure|prepare|calibrate|formal-generate|analyze" >&2
  exit 2
fi

if [[ "$STAGE" == "calibrate" || "$STAGE" == "analyze" ]]; then
  if [[ ! -f "$LID_MODEL" ]]; then
    echo "Missing formal fastText LID model: $LID_MODEL" >&2
    exit 2
  fi
fi

"$PYTHON_BIN" src/run_behavior_association_v3_single.py \
  --config "$CONFIG" --stage "$STAGE" --resume
