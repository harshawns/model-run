#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

require_var REPT_ROOT
require_var REPT_VENV

echo "=== p5.4xlarge H100 Lambda Bootstrap Wrapper ==="
echo "  REPT_ROOT      = $REPT_ROOT"
echo "  REPT_VENV      = $REPT_VENV"
echo "  REPT_DATA_ROOT = $REPT_DATA_ROOT"
echo "  REPT_PROFILE   = $REPT_PROFILE"
echo "  REPT_MODEL     = $REPT_MODEL"
echo ""

cd "$REPT_ROOT"
bash scripts/bootstrap_lambda.sh "$@"
