#!/usr/bin/env bash
# Fresh single-GPU run template, not the historical experiment command.
set -Eeuo pipefail

: "${TEXT2SQL_MODEL_PATH:?Set a local initialization model directory}"
: "${TEXT2SQL_TRAIN_PARQUET:?Set the training parquet path}"
: "${TEXT2SQL_VAL_PARQUET:?Set the validation parquet path}"
: "${TEXT2SQL_SFT_OUTPUT:?Set a new output directory}"
: "${TEXT2SQL_SFT_LR:?Set an explicitly chosen learning rate}"
: "${CUDA_VISIBLE_DEVICES:?Expose exactly one GPU}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "${PYTHON_BIN:-python}" -m agentic_text2sql.sft.trainer \
    --model-path "$TEXT2SQL_MODEL_PATH" \
    --train-file "$TEXT2SQL_TRAIN_PARQUET" \
    --validation-file "$TEXT2SQL_VAL_PARQUET" \
    --output-dir "$TEXT2SQL_SFT_OUTPUT" \
    --learning-rate "$TEXT2SQL_SFT_LR" \
    --epochs 1 \
    --max-length 4096 \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --logging-steps 10 \
    --eval-steps 50 \
    --save-steps 100 \
    --attn-implementation "${TEXT2SQL_ATTN_IMPLEMENTATION:-sdpa}" \
    --report-to "${TEXT2SQL_REPORT_TO:-none}"
