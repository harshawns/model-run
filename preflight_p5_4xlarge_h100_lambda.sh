#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

require_var REPT_ROOT
require_var REPT_VENV
require_var ENV_BASE_URL

echo "=== p5.4xlarge H100 Lambda Preflight Wrapper ==="
echo "  REPT_ROOT            = $REPT_ROOT"
echo "  REPT_VENV            = $REPT_VENV"
echo "  REPT_PROFILE         = $REPT_PROFILE"
echo "  REPT_MODEL           = $REPT_MODEL"
echo "  REPT_NUM_GENERATIONS = $REPT_NUM_GENERATIONS"
echo "  REPT_BATCH_SIZE      = $REPT_BATCH_SIZE"
echo "  REPT_VLLM_MODE       = $REPT_VLLM_MODE"
echo "  REPT_VLLM_GPU_UTIL   = $REPT_VLLM_GPU_UTIL"
echo "  REPT_VLLM_TP         = $REPT_VLLM_TP"
echo "  REPT_REQUIREMENTS_FILE = $REPT_REQUIREMENTS_FILE"
echo "  ENV_BASE_URL         = $ENV_BASE_URL"
echo ""

if (( REPT_BATCH_SIZE % REPT_NUM_GENERATIONS != 0 )); then
    echo "[ERROR] REPT_BATCH_SIZE ($REPT_BATCH_SIZE) must be divisible by REPT_NUM_GENERATIONS ($REPT_NUM_GENERATIONS)"
    exit 1
fi

cd "$REPT_ROOT"
bash scripts/preflight_lambda.sh "$@"
