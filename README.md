# Model-Run GPU Bundle

This folder is a runnable wrapper bundle for launching the competition-compliant
OpenEnv GRPO pipeline (episode mode) on a GPU machine with vLLM, with a
hardware-specific default profile for one AWS `p5.4xlarge`
(`1x H100`, `80 GB HBM3`).

It now includes copied source snapshots of the modified forks inside this
folder:

- [ReasoningEconomicsPT](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT)
- [ReasoningEconomicsEnv](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsEnv)

These are copied from the modified forks under `my_forks/`, excluding local-only
artifacts such as `runs/`, venvs, caches, and `__pycache__`.

If you move to a Lambda / GPU VM, clone or copy this whole `model-run/` folder so the
wrappers and the patched PT/Env source stay together.

For a detailed explanation of what was changed in the PT and Env snapshots, see
[MODIFIED_REASONING_ECONOMICS.md](MODIFIED_REASONING_ECONOMICS.md).

## Current Status

As of the latest 2x H100 Lambda testing:

- `Qwen/Qwen3-4B` episode-mode smokes complete through 6-question, 8-question,
  and real 10-question runs with `100%` `env_done`.
- Best completed 10-question smoke so far:
  - `10/10` questions completed
  - mean reward `-8.5949 ± 1.2561`
  - mean completion tokens about `5050`
- vLLM `max_model_len=8192` is needed for the validated 8-question and
  10-question smokes.
- Server-mode vLLM is stable for smoke validation, but on a 2-GPU box it trains
  on only one GPU. It does not solve optimizer-state OOM for longer runs.
- Experimental 2-GPU sharding is now wired:
  - FSDP full-shard is available but currently blocked by a TRL/Torch FSDP
    assertion during the logprob forward pass.
  - DeepSpeed ZeRO-3 colocate is the active scaling path for longer
    10-question training.
- DeepSpeed ZeRO-3 colocate completed a `Qwen/Qwen3-4B`, 10-question,
  `n_prompts=8`, `num_generations=4`, `batch_size=8`, `128cap` run for `2/2`
  optimizer steps and saved artifacts.
- That `128cap` result validates the memory / distributed path, not model
  improvement yet: `clipped_ratio=1.0`, reward std `0.0`, mean reward `-1.0`,
  `loss=0.0`, and `grad_norm=0.0`.
- Rollout debug logging is available and was used to confirm earlier apparent
  hangs were context, memory, stale-process, or config issues rather than env
  deadlocks.

## What This Bundle Depends On

### Modified PT repo required

This bundle uses the copied PT snapshot in
[grpo_train.py](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/training/grpo_train.py),
not the upstream trainer. Important changes already in that file:

- multi-question episode rollout via TRL `rollout_func`
- tokenizer-aligned env resets with `--env_tokenizer_name`
- model-profile support for Qwen think-tag parsing and visible-only grading
- extended episode reward logging plus summary tooling
- LoRA-capable GRPO path
- seed-manifest filtering support in the rollout path
- rollout debug tracing for turn-by-turn episode diagnosis
- experimental FSDP / DeepSpeed sharding pass-through for 2x H100 scaling
- automatic `max_steps` calculation for TRL rollout `IterableDataset` training

The bundle also uses:

- [scout_openenv_seeds.py](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/training/scout_openenv_seeds.py)
- [run_grpo_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/scripts/run_grpo_lambda.sh)
- [summarize_episode_run.py](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/scripts/summarize_episode_run.py)
- [analyze_reward_logs.py](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/scripts/analyze_reward_logs.py)
- [bootstrap_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/scripts/bootstrap_lambda.sh)
- [preflight_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/scripts/preflight_lambda.sh)

### Modified Env repo required

The bundle also expects the copied environment snapshot in
[ReasoningEconomicsEnv](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsEnv).
Important env-side changes already in place:

- tokenizer-native budget alignment via reset-time `tokenizer_name`
- optional `grading_response` for hybrid-thinking models such as Qwen3
- explicit client budget override via `total_budget`
- server-side config defaults through
  [config.py](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsEnv/env/config.py)
- deployment support in
  [server/docker-entrypoint.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsEnv/server/docker-entrypoint.sh)

Without those env changes, the reward behavior and diagnostics described in the
rest of this project will not match.

## Hardware Profiles

### Recommended for your AWS `p5.4xlarge`

Use:

- [p5_4xlarge_h100.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/p5_4xlarge_h100.lambda.env.example)

This profile defaults to:

- `REPT_MODEL=Qwen/Qwen3-14B`
- `REPT_NUM_GENERATIONS=4`
- `REPT_BATCH_SIZE=4`
- `REPT_VLLM_MODE=colocate`
- `REPT_VLLM_GPU_UTIL=0.25`
- `REPT_MAX_COMPLETION_LENGTH=1024`
- `REASON_BUDGET_NUM_QUESTIONS=4`

This profile is tuned for first H100 episode-mode validation with `Qwen/Qwen3-1.7B`
before scaling to `Qwen/Qwen3-14B`.

### Recommended for `2x H100`

Use:

- [2x_h100.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/2x_h100.lambda.env.example)

This profile defaults to a small one-epoch smoke run:

- `REPT_MODEL=Qwen/Qwen3-1.7B`
- `REPT_NUM_GENERATIONS=2`
- `REPT_BATCH_SIZE=2`
- `REPT_GRAD_ACCUM=1`
- `REPT_N_PROMPTS=1`
- `REPT_MAX_EPISODE_TURNS=12`
- `REPT_VLLM_MODE=server`
- `REPT_VLLM_TP=1`
- `REPT_VLLM_GPU_UTIL=0.35`
- `REPT_VLLM_MAX_MODEL_LEN=4096`
- `REPT_MAX_COMPLETION_LENGTH=1024`
- `REASON_BUDGET_NUM_QUESTIONS=10`

The launcher uses a split layout:

- GPU `0`: GRPO training
- GPU `1`: `trl vllm-serve`

Do not set `REPT_VLLM_TP=2` on a two-GPU box with this launcher. Server mode
needs at least one GPU left for training, so `2x H100` should use
`REPT_VLLM_TP=1`.

Validated 2x H100 server-mode progression:

- `Qwen/Qwen3-1.7B`, 2-question smoke: `100%` completion
- `Qwen/Qwen3-1.7B`, 4-question smoke: `100%` completion
- `Qwen/Qwen3-4B`, 4-question smoke: `100%` completion
- `Qwen/Qwen3-4B`, 6-question smoke: `100%` completion
- `Qwen/Qwen3-4B`, 8-question smoke with `vLLM max_model_len=8192`: `100%`
  completion
- `Qwen/Qwen3-4B`, 10-question smoke: `100%` completion

For longer 10-question training, use the experimental colocate sharding section
in `2x_h100.lambda.env.example` rather than server mode.

### Recommended for `8x A100 40GB SXM4`

Use:

- [8x_a100_40gb.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/8x_a100_40gb.lambda.env.example)

This profile is for an `8x A100 SXM4` node with `40 GB` VRAM per GPU, `124`
vCPUs, `1800 GiB` RAM, and `6 TiB` SSD.

Default first-run layout:

- GPUs `0-5`: GRPO training with DeepSpeed ZeRO-3
- GPUs `6-7`: `trl vllm-serve` with tensor parallel size `2`

Default first-run geometry:

- `REPT_MODEL=Qwen/Qwen3-14B`
- `REPT_VLLM_MODE=server`
- `REPT_VLLM_TP=2`
- `REPT_SHARDING_BACKEND=deepspeed`
- `REPT_DEEPSPEED_CONFIG=configs/deepspeed/zero3_8x_a100_40gb.json`
- `REPT_BATCH_SIZE=2`
- `REPT_NUM_GENERATIONS=2`
- `REPT_MAX_EPISODE_TURNS=3`
- `REPT_MAX_STEPS=1`
- `REPT_VLLM_MAX_MODEL_LEN=5120`
- `REPT_MAX_COMPLETION_LENGTH=96`
- `REPT_MAX_TOKENS_PER_STEP=96`

That default is a fit probe, not a training conclusion. After it passes, change
`REPT_MAX_EPISODE_TURNS=12`, unset `REPT_MAX_STEPS`, and write to a new
`REPT_OUTPUT_DIR` for the real 10-question run.

Use the A100-specific wrappers:

```bash
source ./8x_a100_40gb.lambda.env
bash bootstrap_8x_a100_40gb_lambda.sh
bash start_openenv_server.sh
bash preflight_8x_a100_40gb_lambda.sh
bash run_8x_a100_40gb_lambda.sh 2>&1 | tee /tmp/14b_a100x8_probe.log
```

### Kept for reference only: `Qwen/Qwen3-32B`

Use only if you move to a larger-memory or multi-GPU setup:

- [qwen3_32b_reference.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/qwen3_32b_reference.lambda.env.example)

Why it is not the default on `p5.4xlarge`:

- AWS says `p5.4xlarge` has `1 H100` with `80 GB HBM3` GPU memory.
- `Qwen/Qwen3-32B` in bf16 is about `64 GB` of weights by parameter count inference.
- this training path uses LoRA plus vLLM generation, and the current wrapper defaults to `vllm_mode=colocate`
  with no tensor parallel sharding

So a single `80 GB` H100 is not where I would try to run `Qwen3-32B` in this repo.

## What This Bundle Contains

- [qwen3_32b_reference.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/qwen3_32b_reference.lambda.env.example)
  Reference-only env file for a larger-memory or multi-GPU 32B run.
- [p5_4xlarge_h100.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/p5_4xlarge_h100.lambda.env.example)
  Recommended single-H100 env file.
- [2x_h100.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/2x_h100.lambda.env.example)
  Recommended two-H100 server-mode env file.
- [8x_a100_40gb.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/8x_a100_40gb.lambda.env.example)
  Recommended eight-A100-40GB server-mode + DeepSpeed env file.
- [_common.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/_common.sh)
  Shared defaults used by the wrappers.
- [bootstrap_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/bootstrap_p5_4xlarge_h100_lambda.sh)
  One-time dependency/bootstrap wrapper.
- [bootstrap_2x_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/bootstrap_2x_h100_lambda.sh)
  Two-H100 bootstrap wrapper alias.
- [bootstrap_8x_a100_40gb_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/bootstrap_8x_a100_40gb_lambda.sh)
  Eight-A100 bootstrap wrapper alias.
- [preflight_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/preflight_p5_4xlarge_h100_lambda.sh)
  Sanity checks before launch.
- [preflight_2x_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/preflight_2x_h100_lambda.sh)
  Two-H100 sanity checks before launch.
- [preflight_8x_a100_40gb_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/preflight_8x_a100_40gb_lambda.sh)
  Eight-A100 sanity checks before launch.
- [start_openenv_server.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/start_openenv_server.sh)
  Starts the local env server in V1 mode.
- [scout_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/scout_p5_4xlarge_h100_lambda.sh)
  Runs seed scouting for the larger model.
- [run_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/run_p5_4xlarge_h100_lambda.sh)
  Launches the full training run.
- [run_2x_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/run_2x_h100_lambda.sh)
  Launches the full two-H100 server-mode training run.
- [run_8x_a100_40gb_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/run_8x_a100_40gb_lambda.sh)
  Launches the eight-A100 server-mode + DeepSpeed training run.
- [ReasoningEconomicsPT/configs/accelerate/fsdp_2x_h100.yaml](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/configs/accelerate/fsdp_2x_h100.yaml)
  Experimental FSDP full-shard config for two-H100 colocate runs.
- [ReasoningEconomicsPT/configs/deepspeed/zero3_2x_h100.json](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/configs/deepspeed/zero3_2x_h100.json)
  Experimental DeepSpeed ZeRO-3 fallback config for two-H100 colocate runs.
- [ReasoningEconomicsPT/configs/deepspeed/zero3_8x_a100_40gb.json](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/configs/deepspeed/zero3_8x_a100_40gb.json)
  DeepSpeed ZeRO-3 config for eight-A100-40GB training ranks.

## Default Assumptions

The default wrapper profile in [_common.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/_common.sh) is:

- `REPT_PROFILE=p5.4xlarge-single-h100`
- `REPT_MODEL=Qwen/Qwen3-14B`
- `REPT_NUM_GENERATIONS=4`
- `REPT_BATCH_SIZE=4`
- `REPT_GRAD_ACCUM=1`
- `REPT_N_PROMPTS=50`
- `REPT_VLLM_MODE=colocate`
- `REPT_VLLM_GPU_UTIL=0.25`
- `REPT_VLLM_TP=1`
- `REPT_MAX_COMPLETION_LENGTH=2048`
- `REPT_REQUIREMENTS_FILE=$REPT_ROOT/requirements.lambda.txt`
- `ENV_BASE_URL=http://127.0.0.1:8010`
- `REASON_BUDGET_NUM_QUESTIONS=10`

These defaults are deliberate:

1. `batch_size` must be divisible by `num_generations` for GRPO.
2. the current `code` PT path is rollout-based and episode-oriented by default.
3. `2048` is the bundle starting point for 10-question runs, but GPU runs may need 2560–3072 depending on vLLM pressure.
4. all-50 prompts is the current simple bundle baseline.
5. `num_generations=4` is a practical first single-GPU setting.

## Important Constraint

`Qwen/Qwen3-32B` is not guaranteed to fit on your target GPU just because this
bundle exists.

This bundle makes the launch path reproducible. It does not solve memory limits.
If `Qwen/Qwen3-32B` OOMs, the practical options are:

- use a smaller model such as `Qwen/Qwen3-14B`
- use a larger-memory GPU configuration
- or add tensor-parallel / distributed support beyond the current wrapper path

## Quick Start

This is the shortest end-to-end path after you have copied or cloned the
`model-run/` repo to the GPU machine.

### 1. Copy and edit the env file

```bash
cd /path/to/project/model-run
cp p5_4xlarge_h100.lambda.env.example p5_4xlarge_h100.lambda.env
```

Edit at least these variables in `p5_4xlarge_h100.lambda.env`:

- `REPT_ROOT`
  Absolute path to the copied PT snapshot inside `model-run/` on the GPU machine.
- `REPT_VENV`
  Absolute path to the training venv on the GPU machine.
- `ENV_ROOT`
  Absolute path to the copied Env snapshot inside `model-run/` on the GPU machine.
- `REPT_FS_NAME`
  Your Lambda filesystem name, if you use one.
- `REPT_DATA_ROOT`
  Where runs and caches should live.
- `ENV_BASE_URL`
  The OpenEnv endpoint.

If the env server is local on the same machine, keep:

```bash
export ENV_BASE_URL=http://127.0.0.1:8010
```

### 2. Source the env file

```bash
cd /path/to/project/model-run
source ./p5_4xlarge_h100.lambda.env
```

You should do this before every wrapper command in this folder.

### 3. Bootstrap once

```bash
./bootstrap_p5_4xlarge_h100_lambda.sh
```

What this does:

- creates or reuses the Python venv
- installs PT requirements
- configures cache directories under `REPT_DATA_ROOT`
- checks critical imports including `vllm`, `trl`, and `openenv`

### 4. Run preflight

```bash
./preflight_p5_4xlarge_h100_lambda.sh
```

What this checks:

- `REPT_ROOT`, `REPT_VENV`, and `ENV_BASE_URL`
- GPU visibility via `nvidia-smi`
- `torch.cuda.is_available()`
- required imports
- env server health endpoint
- output path placement on the mounted filesystem

If preflight fails, do not start training yet.

## 2x H100 Quick Start

Use this flow on a two-H100 machine.

```bash
cd /home/ubuntu/csci-544/model-run
cp 2x_h100.lambda.env.example 2x_h100.lambda.env
```

Edit `2x_h100.lambda.env`:

- keep `REPT_FS_NAME=csci-544` if that is your Lambda filesystem name; otherwise change it
- keep `REPT_VLLM_MODE=server`
- keep `REPT_VLLM_TP=1`
- start with `REPT_MODEL=Qwen/Qwen3-1.7B`

Then bootstrap:

```bash
source ./2x_h100.lambda.env
export REPT_RECREATE_VENV=1
./bootstrap_2x_h100_lambda.sh
./bootstrap_openenv_server.sh
```

The Lambda PT requirements pin `vllm==0.17.1` for the 2x H100 server-mode path because TRL 1.0.0 warns on newer vLLM server versions.

Start the env server in one shell:

```bash
cd /home/ubuntu/csci-544/model-run
source ./2x_h100.lambda.env
./start_openenv_server.sh
```

In a second shell:

```bash
cd /home/ubuntu/csci-544/model-run
source ./2x_h100.lambda.env
./preflight_2x_h100_lambda.sh
./run_2x_h100_lambda.sh --dry-run
./run_2x_h100_lambda.sh
```

After the run:

```bash
cd "$REPT_ROOT"
source "$REPT_VENV/bin/activate"
python scripts/summarize_episode_run.py "$REPT_REWARD_LOG_PATH"
cat "$(dirname "$REPT_REWARD_LOG_PATH")/episode_summary.md"
```

### 2x H100 DeepSpeed Colocate Scaling Path

Use this after the server-mode smokes are passing and you need both H100s for
training memory. Server mode reserves one GPU for vLLM, so it cannot shard
optimizer state across both GPUs.

Start with the stable 10-question DeepSpeed colocate infrastructure smoke:

```bash
cd /home/ubuntu/csci-544/model-run
source /home/ubuntu/.venvs/rept-2x-h100/bin/activate
source ./2x_h100.lambda.env

export REPT_MODEL=Qwen/Qwen3-4B
export REPT_OUTPUT_DIR=/lambda/nfs/csci-544/rept/runs/grpo_qwen3_4b_REAL10q_ds_n8_g4_b8_128cap_ctx8192_vllm020_2x_h100
unset REPT_REWARD_LOG_PATH
unset REPT_ROLLOUT_DEBUG_PATH

export REPT_VLLM_MODE=colocate
export REPT_COLOCATE_TRAIN_PROCS=2
export REPT_SHARDING_BACKEND=deepspeed
export REPT_DEEPSPEED_CONFIG=/home/ubuntu/csci-544/model-run/ReasoningEconomicsPT/configs/deepspeed/zero3_2x_h100.json
export REPT_VLLM_ENABLE_SLEEP_MODE=1

export REPT_VLLM_GPU_UTIL=0.20
export REPT_VLLM_MAX_MODEL_LEN=8192
export REPT_MAX_COMPLETION_LENGTH=128
export REPT_MAX_TOKENS_PER_STEP=128

export REPT_N_PROMPTS=8
export REPT_NUM_GENERATIONS=4
export REPT_BATCH_SIZE=8
export REPT_GRAD_ACCUM=1
export REPT_NUM_EPOCHS=1
export REPT_DEBUG_ROLLOUT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

./run_2x_h100_lambda.sh
```

Important details:

- `REPT_DEEPSPEED_CONFIG` must point to the JSON file, not the
  `configs/deepspeed/` directory.
- `REPT_OUTPUT_DIR` should be a full run directory, not just
  `/lambda/nfs/.../runs/`.
- If vLLM reports no available KV-cache blocks at `0.20`, retry with
  `REPT_VLLM_GPU_UTIL=0.25`.
- The `128cap` smoke is expected to be memory-stable, but the latest completed
  run clipped every completion and produced zero reward variance. For the next
  quality probe, keep the same `g4/b8` geometry and try `160` or `192` before
  returning to `256`.
- Clean stale GPU state before reruns:

```bash
pkill -f "VLLM::EngineCore" || true
pkill -f "trl vllm-serve" || true
pkill -f "training.grpo_train" || true
pkill -f "accelerate launch" || true
fuser -k 8001/tcp || true
sleep 5
nvidia-smi
```

## Running The Env Server

You have two choices.

### Option A: local env server on the GPU machine

Use this if you have the modified env fork on the same machine.

```bash
cd /path/to/project/model-run
source ./p5_4xlarge_h100.lambda.env
./start_openenv_server.sh
```

Defaults:

- `ENV_ROOT` defaults to `project/model-run/ReasoningEconomicsEnv`
- `ENV_VENV` defaults to `$ENV_ROOT/.venv-server`
- host defaults to `127.0.0.1`
- port defaults to `8010`
- `REASON_BUDGET_NUM_QUESTIONS=10`

If your env fork or venv lives somewhere else, export:

```bash
export ENV_ROOT=/absolute/path/to/ReasoningEconomicsEnv
export ENV_VENV=/absolute/path/to/env-venv
```

### Option B: existing remote env endpoint

If you already have an env server running elsewhere, just set:

```bash
export ENV_BASE_URL=https://your-endpoint
```

and skip `start_openenv_server.sh`.

## Seed Scout

Before a full large-model run, scout the seeds for that same model.

```bash
cd /path/to/project/model-run
source ./p5_4xlarge_h100.lambda.env
./scout_p5_4xlarge_h100_lambda.sh
```

Default output:

```bash
$REPT_DATA_ROOT/runs/openenv_seed_manifest_qwen3_14b_50.json
```

What the scout does:

- probes `REPT_SCOUT_N_SEEDS` deterministic seeds
- generates `REPT_NUM_GENERATIONS` completions per seed
- replays those completions through OpenEnv reward
- buckets each seed as `easy`, `mixed`, `hard`, or `unclear`

Why this matters:

- seed buckets are model-specific
- if you change models, you should not reuse an old seed manifest

## Full Training Run

### Dry run first

```bash
cd /path/to/project/model-run
source ./p5_4xlarge_h100.lambda.env
./run_p5_4xlarge_h100_lambda.sh --dry-run
```

This prints the exact underlying `python -m training.grpo_train ...` command.

### Actual run

```bash
cd /path/to/project/model-run
source ./p5_4xlarge_h100.lambda.env
./run_p5_4xlarge_h100_lambda.sh
```

By default this launches:

- `training.grpo_train`
- `--model Qwen/Qwen3-14B`
- `--num_generations 4`
- `--n_prompts 50`
- `--max_completion_length 1024`
- `--vllm_mode colocate`

The wrapper delegates to the PT script
[run_grpo_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/ReasoningEconomicsPT/scripts/run_grpo_lambda.sh),
so the source of truth is still the PT repo. This folder only packages the
bundle defaults and launch order.

## Optional: Train With A Seed Manifest

The full-run wrapper currently uses the all-prompts recipe by default. If you
want to explicitly train on a manifest bucket, use the trainer directly:

```bash
cd "$REPT_ROOT"
source "$REPT_VENV/bin/activate"

python -m training.grpo_train \
  --model "$REPT_MODEL" \
  --env_base_url "$ENV_BASE_URL" \
  --seed_manifest_path "$REPT_SCOUT_OUTPUT" \
  --seed_bucket mixed \
  --n_prompts 20 \
  --num_generations "$REPT_NUM_GENERATIONS" \
  --max_completion_length "$REPT_MAX_COMPLETION_LENGTH" \
  --per_device_train_batch_size "$REPT_BATCH_SIZE" \
  --gradient_accumulation_steps "$REPT_GRAD_ACCUM" \
  --vllm_mode "$REPT_VLLM_MODE" \
  --output_dir "$REPT_OUTPUT_DIR"
```

Use this only after you have a scout manifest for the same model.

## Where Outputs Go

By default:

- scout manifest:
  `REPT_SCOUT_OUTPUT`
- training outputs:
  `REPT_OUTPUT_DIR`
- cache root:
  `$REPT_DATA_ROOT/cache`

Expected training outputs include:

- trainer checkpoints
- tokenizer files
- model artifacts
- reward logs when enabled

## Episode Mode Status

- Multi-question episode mode is validated locally (4-question, CPU,
  Qwen3-0.6B).
- Full 4-question local episodes complete end-to-end:
  `termination_reason=env_done`, `episode_clipped=False`,
  `final_questions_remaining=0`.
- 2x H100 GPU validation now reaches 10-question `Qwen/Qwen3-4B` smokes with
  `100%` `env_done`.
- The best completed 10-question GPU smoke so far has mean reward
  `-8.5949 ± 1.2561` and mean completion tokens about `5050`.
- DeepSpeed ZeRO-3 colocate now completes `2/2` optimizer steps at
  `max_completion_length=128`, but that cap clipped every completion and
  produced no learning signal.
- Episode-mode eval parity is available: every run produces `reward_log.jsonl`. Summarize with `scripts/summarize_episode_run.py`.
- The next remaining GPU task is recovering reward variance and nonzero
  gradients on the stable sharded path, not basic episode-loop correctness.

## GPU Sequence Length Tuning

Local CPU validation confirmed `max_completion_length=1024` works for
4-question episodes. Current 2x H100 testing shows two different regimes:

- server-mode smokes can complete 8q/10q when vLLM context is raised to `8192`
- DeepSpeed colocate runs need a tighter per-step cap first because vLLM and
  training share both GPUs

Tuning grid for 10-question `Qwen3-4B` colocate runs:

- `128` — stable infrastructure baseline, but all completions clipped
- `160`
- `192`
- `224`
- `256` — retry only after smaller caps complete without OOM

Set `REPT_MAX_COMPLETION_LENGTH` to control. Use `scripts/summarize_episode_run.py` to check
`clipped_rate`, `termination_reasons`, reward std, and trainer `grad_norm`
after each candidate.

> **Memory note:** Server-mode success does not guarantee colocate memory
> safety. In colocate mode, if vLLM reports no KV-cache blocks, tune
> `REPT_VLLM_GPU_UTIL` before raising sequence length.

## First H100 Episode Smoke (Recommended Starting Point)

Use `Qwen/Qwen3-1.7B` for the first smoke — not `Qwen/Qwen3-14B`. Validate the episode path is
stable before scaling up.

```bash
# 1. Start env (4-question mode — default after this sync)
source ./p5_4xlarge_h100.lambda.env
bash start_openenv_server.sh

# 2. Run episode smoke
export REPT_MODEL=Qwen/Qwen3-1.7B
export REPT_NUM_GENERATIONS=4
export REPT_BATCH_SIZE=4
export REPT_GRAD_ACCUM=4
export REPT_MAX_COMPLETION_LENGTH=1024   # raise to 1280 if episodes clip
bash run_p5_4xlarge_h100_lambda.sh

# 3. Summarize results
python ReasoningEconomicsPT/scripts/summarize_episode_run.py \
    "$REPT_OUTPUT_DIR/reward_log.jsonl"
```

Success criteria: `completion_rate=100%`, `termination_reasons={'env_done': N}`, `clipped_rate=0%`,
non-zero `reward_std`.

`Qwen/Qwen3-14B` is the eventual bigger-model target, not the first smoke.

## Troubleshooting

### `REPT_BATCH_SIZE must be divisible by REPT_NUM_GENERATIONS`

That is a GRPO requirement in this setup.
Use values like:

- `4 / 4`
- `8 / 4`
- `8 / 8`

Do not use `2 / 4`.

### `torch.cuda.is_available() is False`

Your GPU stack or venv is not ready. Re-run:

```bash
./bootstrap_p5_4xlarge_h100_lambda.sh
./preflight_p5_4xlarge_h100_lambda.sh
```

### Env server health fails

Either:

- the env server is not running
- `ENV_BASE_URL` is wrong
- or the server is running on a different port/host

### `Qwen3-32B` OOMs

This bundle does not magically solve memory pressure. If it OOMs:

- try `Qwen/Qwen3-14B`
- or increase available GPU memory
- or reduce ambition before adding distributed complexity

### Scout works but full run is unstable

That usually means:

- vLLM / memory pressure at full training scale
- too many completions per group for the hardware
- or the model is too large for the chosen GPU setup

### DeepSpeed says the config path is a directory

`REPT_DEEPSPEED_CONFIG` must be the JSON file path:

```bash
export REPT_DEEPSPEED_CONFIG=/home/ubuntu/csci-544/model-run/ReasoningEconomicsPT/configs/deepspeed/zero3_2x_h100.json
```

This is wrong:

```bash
export REPT_DEEPSPEED_CONFIG=/home/ubuntu/csci-544/model-run/ReasoningEconomicsPT/configs/deepspeed/
```

### vLLM says no available memory for cache blocks

In colocate mode, vLLM could load model weights but have no remaining KV-cache
blocks. Raise the vLLM fraction before reducing context:

```bash
export REPT_VLLM_GPU_UTIL=0.25
```

If that still fails, lower `REPT_VLLM_MAX_MODEL_LEN` from `8192` to `7168` for
the next smoke.

## Recommended Practical Order

1. source the env file
2. bootstrap once
3. preflight
4. start the env server
5. run the 2x H100 server-mode smoke first
6. summarize `reward_log.jsonl`
7. move to DeepSpeed colocate only after the server-mode smoke completes
8. dry-run the trainer before each new scaling run
9. launch the full run

## GPU Memory Guidance

Official AWS docs list `p5.4xlarge` as:

- `1 H100`
- `80 GB HBM3`
- `256 GiB` host memory

Source:

- https://aws.amazon.com/ec2/instance-types/accelerated-computing/

Practical guidance for this repo:

- `Qwen/Qwen3-14B` on one `p5.4xlarge` is the right target
- `Qwen/Qwen3-32B` is not the right default for one `80 GB` H100 in the current colocated-vLLM path

## GH200 Note

If you run this bundle on a Lambda `GH200 (96 GB)` instance instead of an H100 box:

- use `Lambda Stack 22.04`
- use an isolated PT venv (`REPT_VENV_SYSTEM_SITE_PACKAGES=0`)
- install CUDA-enabled `torch` from the PyTorch wheel index before `vllm` can pull a CPU wheel
- keep the Lambda PT stack pinned through `requirements.lambda.txt`; the GH200/aarch64 path uses `torch==2.10.*`, `vllm==0.17.1`, and NumPy 2.x

The PT bootstrap script supports this through:

- `REPT_PYTHON_BIN=auto` (auto-detects a CUDA-capable Python on GH200)
- `REPT_VENV_SYSTEM_SITE_PACKAGES=0`
- `REPT_SKIP_TORCH_INSTALL=0`
- `PYTORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu128`

The easiest path is to start from:

- [gh200.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/gh200.lambda.env.example)

Recommended GH200 flow:

```bash
cd /home/ubuntu/csci-544/model-run
cp gh200.lambda.env.example gh200.lambda.env
source ./gh200.lambda.env
rm -rf "$REPT_VENV"
./bootstrap_p5_4xlarge_h100_lambda.sh
./bootstrap_openenv_server.sh
./start_openenv_server.sh
./preflight_p5_4xlarge_h100_lambda.sh
./run_p5_4xlarge_h100_lambda.sh --dry-run
./run_p5_4xlarge_h100_lambda.sh
```

## Reward Log Analysis

For quick reward-log diagnostics after a run:

```bash
cd "$REPT_ROOT"
source "$REPT_VENV/bin/activate"
pip install -r requirements.analysis.txt
python scripts/analyze_reward_logs.py "$REPT_REWARD_LOG_PATH" --out-dir reward_log_analysis
```

For episode-mode summaries:

```bash
python scripts/summarize_episode_run.py "$REPT_REWARD_LOG_PATH"
```

The reason is not just raw model size. The run also needs:

- the training model
- LoRA adapters and optimizer state
- activations
- vLLM generation memory

That is why the single-H100 profile defaults to `Qwen3-14B`, not `Qwen3-32B`.

That is the intended use of this folder.
