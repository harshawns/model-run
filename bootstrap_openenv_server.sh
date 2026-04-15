#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

resolve_env_python_bin() {
    local requested="$1"
    shift
    local candidate
    local -a candidates=("$@")

    if [[ "$requested" != "auto" ]]; then
        echo "$requested"
        return 0
    fi

    for candidate in "${candidates[@]}"; do
        [[ -z "$candidate" ]] && continue
        if [[ "$candidate" == */* ]]; then
            [[ -x "$candidate" ]] || continue
        else
            command -v "$candidate" >/dev/null 2>&1 || continue
        fi
        if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major, minor) >= (3, 11) else 1)
PY
        then
            echo "$candidate"
            return 0
        fi
    done

    echo "python3.11"
}

ENV_PYTHON_BIN="$(resolve_env_python_bin "$ENV_PYTHON_BIN" python3.11 python python3)"

echo "=== OpenEnv Server Bootstrap ==="
echo "  ENV_ROOT                     = $ENV_ROOT"
echo "  ENV_VENV                     = $ENV_VENV"
echo "  ENV_PYTHON_BIN               = $ENV_PYTHON_BIN"
echo "  ENV_VENV_SYSTEM_SITE_PACKAGES= $ENV_VENV_SYSTEM_SITE_PACKAGES"
echo ""

if [[ ! -d "$ENV_ROOT" ]]; then
    echo "[ERROR] ENV_ROOT does not exist: $ENV_ROOT"
    exit 1
fi

if [[ -d "$ENV_VENV" ]]; then
    echo ">>> Reusing existing env venv at $ENV_VENV"
else
    echo ">>> Creating env venv at $ENV_VENV"
    if [[ "$ENV_VENV_SYSTEM_SITE_PACKAGES" == "1" ]]; then
        "$ENV_PYTHON_BIN" -m venv --system-site-packages "$ENV_VENV"
    else
        "$ENV_PYTHON_BIN" -m venv "$ENV_VENV"
    fi
fi

cd "$ENV_ROOT"
# shellcheck source=/dev/null
source "$ENV_VENV/bin/activate"

echo ">>> Python: $(python --version) ($(which python))"
echo ">>> Installing server dependencies..."
pip install --quiet --upgrade pip
pip install --no-cache-dir --upgrade \
    "openenv-core[core]>=0.2.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.30.0" \
    "numpy>=1.26" \
    "pandas>=2.2" \
    "pyarrow>=16" \
    "datasets>=2.14.0" \
    "transformers>=4.36.0" \
    "sympy>=1.12" \
    "pydantic>=2.0.0" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0" \
    "tqdm>=4.65.0"
pip install --no-deps -e .

echo ""
echo ">>> Smoke-testing env imports..."
python - <<'PY'
import env.config
import fastapi
import numpy
import openenv
import pandas
import pyarrow
import pydantic
import transformers

print("  env.config OK")
print(f"  numpy        {numpy.__version__}")
print(f"  pandas       {pandas.__version__}")
print(f"  pyarrow      {pyarrow.__version__}")
print(f"  pydantic     {pydantic.__version__}")
print(f"  transformers {transformers.__version__}")
print("  fastapi      OK")
print("  openenv-core OK")
PY

echo ""
echo "=== OpenEnv server bootstrap complete ==="
