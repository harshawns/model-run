#!/bin/bash
# run_grpo_lambda.sh -- Direct GRPO launcher for Lambda Cloud VMs (no Slurm).
#
# Usage:
#   export REPT_ROOT=/home/ubuntu/ReasoningEconomicsPT
#   export REPT_VENV=/home/ubuntu/.venvs/rept-lambda
#   export ENV_BASE_URL=http://127.0.0.1:8000
#   export REPT_FS_NAME=<lambda-filesystem-name>   # optional
#   bash scripts/run_grpo_lambda.sh [--dry-run]

set -euo pipefail

DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Runs GRPO training directly on a Lambda VM (no sbatch)."
    echo ""
    echo "Required exports:"
    echo "  REPT_ROOT       absolute path to ReasoningEconomicsPT"
    echo "  REPT_VENV       absolute path to Python venv"
    echo "  ENV_BASE_URL    OpenEnv endpoint base URL"
    echo ""
    echo "Optional exports:"
    echo "  REPT_FS_NAME          Lambda filesystem name (used when REPT_DATA_ROOT unset)"
    echo "  REPT_DATA_ROOT        Base data path (default: /lambda/nfs/<fs>/rept or /lambda/nfs/rept)"
    echo "  REPT_MODEL            default: Qwen/Qwen3-1.7B"
    echo "  REPT_OPENENV_MODE     default: flat (use episode for multi-question rollouts)"
    echo "  REPT_OUTPUT_DIR       default: <DATA_ROOT>/runs/grpo_train_lambda"
    echo "  REPT_NUM_EPOCHS       default: 1"
    echo "  REPT_NUM_GENERATIONS  default: 4"
    echo "  REPT_BATCH_SIZE       default: 2"
    echo "  REPT_GRAD_ACCUM       default: 8"
    echo "  REPT_TORCH_DTYPE      default: bfloat16"
    echo "  REPT_VLLM_MODE        default: colocate"
    echo "  REPT_VLLM_GPU_MEMORY_UTILIZATION  default: unset (TRL default)"
    echo "  REPT_VLLM_MAX_MODEL_LENGTH        default: unset (model default)"
    echo "  REPT_VLLM_TENSOR_PARALLEL_SIZE    default: unset (TRL default)"
    echo "  REPT_VLLM_ENABLE_SLEEP_MODE       default: 0"
    echo "  REPT_ALPHA            default: 1.0"
    echo "  REPT_LOG_EVERY        default: 1"
    echo "  REPT_INSTALL_DEPS_ON_RUN  default: 0 (set to 1 to pip install before run)"
    echo "  REPT_REQUIREMENTS_FILE    default: <REPT_ROOT>/requirements.txt"
    echo "  PYTORCH_WHEEL_INDEX       optional pip extra index URL"
    echo "  REPT_N_PROMPTS            default: 200 (questions pre-fetched from OpenEnv)"
    echo "  REPT_MAX_COMPLETION_LENGTH  default: 384 (tokens per model response)"
    echo "  REPT_REWARD_LOG_PATH      default: <REPT_OUTPUT_DIR>/reward_log.jsonl"
    echo "  REPT_USE_VLLM             default: 1 (set to 0 to disable vLLM generation)"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

: "${REPT_ROOT:?REPT_ROOT is required}"
: "${REPT_VENV:?REPT_VENV is required}"
: "${ENV_BASE_URL:?ENV_BASE_URL is required}"

if [[ -n "${REPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$REPT_DATA_ROOT"
elif [[ -n "${REPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${REPT_FS_NAME}/rept"
else
    DATA_ROOT="/lambda/nfs/rept"
fi

REPT_MODEL="${REPT_MODEL:-Qwen/Qwen3-1.7B}"
REPT_OPENENV_MODE="${REPT_OPENENV_MODE:-flat}"
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-${DATA_ROOT}/runs/grpo_train_lambda}"
REPT_NUM_EPOCHS="${REPT_NUM_EPOCHS:-1}"
REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-4}"
REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-2}"
REPT_GRAD_ACCUM="${REPT_GRAD_ACCUM:-8}"
REPT_TORCH_DTYPE="${REPT_TORCH_DTYPE:-bfloat16}"
REPT_VLLM_MODE="${REPT_VLLM_MODE:-colocate}"
REPT_VLLM_GPU_MEMORY_UTILIZATION="${REPT_VLLM_GPU_MEMORY_UTILIZATION:-}"
REPT_VLLM_MAX_MODEL_LENGTH="${REPT_VLLM_MAX_MODEL_LENGTH:-}"
REPT_VLLM_TENSOR_PARALLEL_SIZE="${REPT_VLLM_TENSOR_PARALLEL_SIZE:-}"
REPT_VLLM_ENABLE_SLEEP_MODE="${REPT_VLLM_ENABLE_SLEEP_MODE:-0}"
REPT_ALPHA="${REPT_ALPHA:-1.0}"
REPT_LOG_EVERY="${REPT_LOG_EVERY:-1}"
REPT_INSTALL_DEPS_ON_RUN="${REPT_INSTALL_DEPS_ON_RUN:-0}"
REPT_REQUIREMENTS_FILE="${REPT_REQUIREMENTS_FILE:-$REPT_ROOT/requirements.txt}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-}"
REPT_N_PROMPTS="${REPT_N_PROMPTS:-200}"
REPT_MAX_COMPLETION_LENGTH="${REPT_MAX_COMPLETION_LENGTH:-384}"
REPT_USE_VLLM="${REPT_USE_VLLM:-1}"
REPT_REWARD_LOG_PATH="${REPT_REWARD_LOG_PATH:-$REPT_OUTPUT_DIR/reward_log.jsonl}"

if (( REPT_BATCH_SIZE % REPT_NUM_GENERATIONS != 0 )); then
    echo "[ERROR] REPT_BATCH_SIZE ($REPT_BATCH_SIZE) must be divisible by REPT_NUM_GENERATIONS ($REPT_NUM_GENERATIONS)."
    exit 1
fi

if [[ "$REPT_OPENENV_MODE" == "episode" && "$REPT_USE_VLLM" != "1" ]]; then
    echo "[ERROR] REPT_OPENENV_MODE=episode requires REPT_USE_VLLM=1."
    exit 1
fi

CACHE_ROOT="${DATA_ROOT}/cache"
mkdir -p "$CACHE_ROOT/pip" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/tmp" "$REPT_OUTPUT_DIR"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TMPDIR="$CACHE_ROOT/tmp"

if [[ ! -d "$REPT_ROOT" ]]; then
    echo "[ERROR] REPT_ROOT does not exist: $REPT_ROOT"
    exit 1
fi

if [[ ! -f "$REPT_VENV/bin/activate" ]]; then
    echo "[ERROR] REPT_VENV has no bin/activate: $REPT_VENV"
    echo "Run scripts/bootstrap_lambda.sh first."
    exit 1
fi

if [[ "$REPT_OUTPUT_DIR" != /lambda/nfs/* ]]; then
    echo "[WARN] REPT_OUTPUT_DIR is not under /lambda/nfs: $REPT_OUTPUT_DIR"
fi

cd "$REPT_ROOT"
mkdir -p logs
# shellcheck source=/dev/null
source "$REPT_VENV/bin/activate"

if [[ "$REPT_INSTALL_DEPS_ON_RUN" == "1" ]]; then
    if [[ ! -f "$REPT_REQUIREMENTS_FILE" ]]; then
        echo "[ERROR] requirements file not found: $REPT_REQUIREMENTS_FILE"
        exit 1
    fi
    echo ">>> Installing dependencies before run..."
    pip install --quiet --upgrade pip
    if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
        pip install -r "$REPT_REQUIREMENTS_FILE" --extra-index-url "$PYTORCH_WHEEL_INDEX"
    else
        pip install -r "$REPT_REQUIREMENTS_FILE"
    fi
fi

echo "=== REPT GRPO Training (Lambda) ==="
echo "  Host:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  Python:      $(python --version)"
echo "  Torch:       $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA avail:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Model:       $REPT_MODEL"
echo "  OpenEnv mode:$REPT_OPENENV_MODE"
echo "  Output dir:  $REPT_OUTPUT_DIR"
echo "  Env URL:     $ENV_BASE_URL"
echo "  DType:       $REPT_TORCH_DTYPE"
echo "  vLLM mode:   $REPT_VLLM_MODE"
echo "  vLLM mem:    ${REPT_VLLM_GPU_MEMORY_UTILIZATION:-trl-default}"
echo "  vLLM TP:     ${REPT_VLLM_TENSOR_PARALLEL_SIZE:-trl-default}"
echo "==============================="

TRAIN_CMD=(
    python -m training.grpo_train
    --model "$REPT_MODEL"
    --openenv_mode "$REPT_OPENENV_MODE"
    --env_base_url "$ENV_BASE_URL"
    --alpha "$REPT_ALPHA"
    --log_every_n_steps "$REPT_LOG_EVERY"
    --num_train_epochs "$REPT_NUM_EPOCHS"
    --num_generations "$REPT_NUM_GENERATIONS"
    --n_prompts "$REPT_N_PROMPTS"
    --max_completion_length "$REPT_MAX_COMPLETION_LENGTH"
    --per_device_train_batch_size "$REPT_BATCH_SIZE"
    --gradient_accumulation_steps "$REPT_GRAD_ACCUM"
    --torch_dtype "$REPT_TORCH_DTYPE"
    --vllm_mode "$REPT_VLLM_MODE"
    --output_dir "$REPT_OUTPUT_DIR"
    --reward_log_path "$REPT_REWARD_LOG_PATH"
)

if [[ "$REPT_USE_VLLM" == "1" ]]; then
    TRAIN_CMD+=(--use_vllm)
    if [[ -n "$REPT_VLLM_GPU_MEMORY_UTILIZATION" ]]; then
        TRAIN_CMD+=(--vllm_gpu_memory_utilization "$REPT_VLLM_GPU_MEMORY_UTILIZATION")
    fi
    if [[ -n "$REPT_VLLM_MAX_MODEL_LENGTH" ]]; then
        TRAIN_CMD+=(--vllm_max_model_length "$REPT_VLLM_MAX_MODEL_LENGTH")
    fi
    if [[ -n "$REPT_VLLM_TENSOR_PARALLEL_SIZE" ]]; then
        TRAIN_CMD+=(--vllm_tensor_parallel_size "$REPT_VLLM_TENSOR_PARALLEL_SIZE")
    fi
    if [[ "$REPT_VLLM_ENABLE_SLEEP_MODE" == "1" ]]; then
        TRAIN_CMD+=(--vllm_enable_sleep_mode)
    fi
fi

if [[ "$REPT_OPENENV_MODE" == "episode" ]]; then
    export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Command:"
    printf '  %q' "${TRAIN_CMD[@]}"
    echo ""
    exit 0
fi

"${TRAIN_CMD[@]}"

echo "=== Training complete. Artifacts at: $REPT_OUTPUT_DIR ==="
