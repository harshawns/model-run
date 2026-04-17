#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

require_var REPT_ROOT
require_var REPT_VENV

echo "=== 2x H100 Lambda Bootstrap Wrapper ==="
echo "  REPT_ROOT      = $REPT_ROOT"
echo "  REPT_VENV      = $REPT_VENV"
echo "  REPT_DATA_ROOT = $REPT_DATA_ROOT"
echo "  REPT_PROFILE   = $REPT_PROFILE"
echo "  REPT_MODEL     = $REPT_MODEL"
echo "  REPT_REQUIREMENTS_FILE = $REPT_REQUIREMENTS_FILE"
echo "  REPT_PYTHON_BIN = $REPT_PYTHON_BIN"
echo "  REPT_VENV_SYSTEM_SITE_PACKAGES = $REPT_VENV_SYSTEM_SITE_PACKAGES"
echo "  REPT_SKIP_TORCH_INSTALL = $REPT_SKIP_TORCH_INSTALL"
echo "  REPT_RECREATE_VENV = $REPT_RECREATE_VENV"
echo "  PYTORCH_WHEEL_INDEX = ${PYTORCH_WHEEL_INDEX:-<default pip indexes>}"
echo ""

cd "$REPT_ROOT"
bash scripts/bootstrap_lambda.sh "$@"
