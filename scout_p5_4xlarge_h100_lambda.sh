#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

require_var REPT_ROOT
require_var REPT_VENV
require_var ENV_BASE_URL

echo "=== p5.4xlarge H100 Seed Scout Wrapper ==="
echo "  REPT_ROOT            = $REPT_ROOT"
echo "  REPT_PROFILE         = $REPT_PROFILE"
echo "  REPT_MODEL           = $REPT_MODEL"
echo "  REPT_TORCH_DTYPE     = $REPT_TORCH_DTYPE"
echo "  REPT_SCOUT_OUTPUT    = $REPT_SCOUT_OUTPUT"
echo "  REPT_SCOUT_N_SEEDS   = $REPT_SCOUT_N_SEEDS"
echo "  REPT_NUM_GENERATIONS = $REPT_NUM_GENERATIONS"
echo "  ENV_BASE_URL         = $ENV_BASE_URL"
echo ""

cd "$REPT_ROOT"
# shellcheck source=/dev/null
source "$REPT_VENV/bin/activate"

python -m training.scout_openenv_seeds \
  --model "$REPT_MODEL" \
  --torch_dtype "$REPT_TORCH_DTYPE" \
  --env_base_url "$ENV_BASE_URL" \
  --n_seeds "$REPT_SCOUT_N_SEEDS" \
  --num_generations "$REPT_NUM_GENERATIONS" \
  --max_completion_length "$REPT_MAX_COMPLETION_LENGTH" \
  --output_path "$REPT_SCOUT_OUTPUT" \
  "$@"
