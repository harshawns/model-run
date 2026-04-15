#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

ENV_HOST="${ENV_HOST:-127.0.0.1}"
ENV_PORT="${ENV_PORT:-8010}"
ENV_VENV="${ENV_VENV:-$ENV_ROOT/.venv-server}"

echo "=== OpenEnv Server Wrapper ==="
echo "  ENV_ROOT                     = $ENV_ROOT"
echo "  ENV_VENV                     = $ENV_VENV"
echo "  REASON_BUDGET_NUM_QUESTIONS  = $REASON_BUDGET_NUM_QUESTIONS"
echo "  REASON_BUDGET_HARD_CAP_MODE  = $REASON_BUDGET_HARD_CAP_MODE"
echo "  REASON_BUDGET_BUDGET_RATIO   = $REASON_BUDGET_BUDGET_RATIO"
echo "  REASON_BUDGET_TOKENIZER_NAME = $REASON_BUDGET_TOKENIZER_NAME"
echo "  Host / Port                  = $ENV_HOST:$ENV_PORT"
echo ""

cd "$ENV_ROOT"
if [[ -f "$ENV_VENV/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_VENV/bin/activate"
fi

export PYTHONPATH="$ENV_ROOT${PYTHONPATH:+:$PYTHONPATH}"

uvicorn server.app:app --host "$ENV_HOST" --port "$ENV_PORT"
