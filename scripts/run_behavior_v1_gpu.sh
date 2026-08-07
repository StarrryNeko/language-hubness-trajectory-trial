#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CPU_THREADS="${CPU_THREADS:-24}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export RAYON_NUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LHT_MODEL_ROOT="${LHT_MODEL_ROOT:-/root/autodl-tmp/langhub_models}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE="${BEHAVIOR_SUITE:-configs/model_suite_behavior_v1.json}"
LID_MODEL="/root/autodl-tmp/lid/lid.176.bin"
STAGE="${1:-all}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

if [[ ! -f "$LID_MODEL" ]]; then
  echo "Missing fastText LID model: $LID_MODEL" >&2
  exit 2
fi
if [[ ! -d "$LHT_MODEL_ROOT" ]]; then
  echo "Missing portable model root: $LHT_MODEL_ROOT" >&2
  exit 2
fi

"$PYTHON_BIN" -c "import torch, transformers, pandas, sklearn, statsmodels, sacrebleu, fasttext; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available()"

case "$STAGE" in
  all|prepare|generate|analyze)
    "$PYTHON_BIN" src/run_behavior_suite.py \
      --suite "$SUITE" \
      --stage "$STAGE" \
      --resume
    ;;
  *)
    echo "Usage: bash scripts/run_behavior_v1_gpu.sh [all|prepare|generate|analyze]" >&2
    exit 2
    ;;
esac
