#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

require_var REPT_ROOT
require_var REPT_VENV
require_var ENV_BASE_URL

echo "=== 2x H100 Lambda Train Wrapper ==="
echo "  REPT_ROOT                 = $REPT_ROOT"
echo "  REPT_PROFILE              = $REPT_PROFILE"
echo "  REPT_MODEL                = $REPT_MODEL"
echo "  REPT_OUTPUT_DIR           = $REPT_OUTPUT_DIR"
echo "  REPT_REWARD_LOG_PATH      = $REPT_REWARD_LOG_PATH"
echo "  REPT_NUM_GENERATIONS      = $REPT_NUM_GENERATIONS"
echo "  REPT_BATCH_SIZE           = $REPT_BATCH_SIZE"
echo "  REPT_GRAD_ACCUM           = $REPT_GRAD_ACCUM"
echo "  REPT_VLLM_MODE            = $REPT_VLLM_MODE"
echo "  REPT_VLLM_GPU_UTIL        = $REPT_VLLM_GPU_UTIL"
echo "  REPT_VLLM_TP              = $REPT_VLLM_TP"
echo "  REPT_COLOCATE_TRAIN_PROCS = $REPT_COLOCATE_TRAIN_PROCS"
echo "  REPT_VLLM_MAX_MODEL_LEN   = ${REPT_VLLM_MAX_MODEL_LEN:-<unset>}"
echo "  REPT_N_PROMPTS            = $REPT_N_PROMPTS"
echo "  REPT_MAX_COMPLETION_LENGTH= $REPT_MAX_COMPLETION_LENGTH"
echo "  REPT_MAX_TOKENS_PER_STEP  = $REPT_MAX_TOKENS_PER_STEP"
echo "  REPT_DEFAULT_BUDGET_MODE  = $REPT_DEFAULT_BUDGET_MODE"
echo "  REPT_REQUIREMENTS_FILE    = $REPT_REQUIREMENTS_FILE"
echo "  ENV_BASE_URL              = $ENV_BASE_URL"
echo "  REASON_BUDGET_NUM_QUESTIONS = $REASON_BUDGET_NUM_QUESTIONS"
echo "  REASON_BUDGET_HARD_CAP_MODE = $REASON_BUDGET_HARD_CAP_MODE"
echo "  REASON_BUDGET_BUDGET_RATIO  = $REASON_BUDGET_BUDGET_RATIO"
echo ""

if (( REPT_BATCH_SIZE % REPT_NUM_GENERATIONS != 0 )); then
    echo "[ERROR] REPT_BATCH_SIZE ($REPT_BATCH_SIZE) must be divisible by REPT_NUM_GENERATIONS ($REPT_NUM_GENERATIONS)"
    exit 1
fi

cd "$REPT_ROOT"
bash scripts/run_grpo_lambda.sh "$@"
