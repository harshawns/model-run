#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PT_ROOT_DEFAULT="$PROJECT_ROOT/larger-model/ReasoningEconomicsPT"
ENV_ROOT_DEFAULT="$PROJECT_ROOT/larger-model/ReasoningEconomicsEnv"

export REPT_ROOT="${REPT_ROOT:-$PT_ROOT_DEFAULT}"
export ENV_ROOT="${ENV_ROOT:-$ENV_ROOT_DEFAULT}"

if [[ -n "${REPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$REPT_DATA_ROOT"
elif [[ -n "${REPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${REPT_FS_NAME}/rept"
else
    DATA_ROOT="/lambda/nfs/rept"
fi
export REPT_DATA_ROOT="$DATA_ROOT"

export REPT_PROFILE="${REPT_PROFILE:-p5.4xlarge-single-h100}"
export REPT_MODEL="${REPT_MODEL:-Qwen/Qwen3-14B}"
export REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-$REPT_DATA_ROOT/runs/grpo_qwen3_14b_p5_4xlarge}"
export REPT_NUM_EPOCHS="${REPT_NUM_EPOCHS:-1}"
export REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-4}"
export REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-4}"
export REPT_GRAD_ACCUM="${REPT_GRAD_ACCUM:-1}"
export REPT_TORCH_DTYPE="${REPT_TORCH_DTYPE:-bfloat16}"
export REPT_VLLM_MODE="${REPT_VLLM_MODE:-colocate}"
export REPT_VLLM_GPU_MEMORY_UTILIZATION="${REPT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export REPT_VLLM_MAX_MODEL_LENGTH="${REPT_VLLM_MAX_MODEL_LENGTH:-}"
export REPT_VLLM_TENSOR_PARALLEL_SIZE="${REPT_VLLM_TENSOR_PARALLEL_SIZE:-1}"
export REPT_VLLM_ENABLE_SLEEP_MODE="${REPT_VLLM_ENABLE_SLEEP_MODE:-0}"
export REPT_ALPHA="${REPT_ALPHA:-1.0}"
export REPT_LOG_EVERY="${REPT_LOG_EVERY:-1}"
export REPT_N_PROMPTS="${REPT_N_PROMPTS:-50}"
export REPT_MAX_COMPLETION_LENGTH="${REPT_MAX_COMPLETION_LENGTH:-384}"
export REPT_USE_VLLM="${REPT_USE_VLLM:-1}"
export REPT_INSTALL_DEPS_ON_RUN="${REPT_INSTALL_DEPS_ON_RUN:-0}"

export REPT_SCOUT_OUTPUT="${REPT_SCOUT_OUTPUT:-$REPT_DATA_ROOT/runs/openenv_seed_manifest_qwen3_14b_50.json}"
export REPT_SCOUT_N_SEEDS="${REPT_SCOUT_N_SEEDS:-50}"

export ENV_BASE_URL="${ENV_BASE_URL:-http://127.0.0.1:8010}"
export REASON_BUDGET_NUM_QUESTIONS="${REASON_BUDGET_NUM_QUESTIONS:-1}"

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[ERROR] $name is required"
        exit 1
    fi
}
