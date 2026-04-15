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
- [_common.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/_common.sh)
  Shared defaults used by the wrappers.
- [bootstrap_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/bootstrap_p5_4xlarge_h100_lambda.sh)
  One-time dependency/bootstrap wrapper.
- [preflight_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/preflight_p5_4xlarge_h100_lambda.sh)
  Sanity checks before launch.
- [start_openenv_server.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/start_openenv_server.sh)
  Starts the local env server in V1 mode.
- [scout_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/scout_p5_4xlarge_h100_lambda.sh)
  Runs seed scouting for the larger model.
- [run_p5_4xlarge_h100_lambda.sh](/Users/harshawnsingh/Desktop/csci-544/project/model-run/run_p5_4xlarge_h100_lambda.sh)
  Launches the full training run.

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
- `REPT_MAX_COMPLETION_LENGTH=1024`
- `REPT_REQUIREMENTS_FILE=$REPT_ROOT/requirements.lambda.txt`
- `ENV_BASE_URL=http://127.0.0.1:8010`
- `REASON_BUDGET_NUM_QUESTIONS=4`

These defaults are deliberate:

1. `batch_size` must be divisible by `num_generations` for GRPO.
2. the current `code` PT path is rollout-based and episode-oriented by default.
3. `1024` is the bundle starting point for 4-question runs, but GPU runs may need 1280–1536 depending on vLLM pressure.
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
- `REASON_BUDGET_NUM_QUESTIONS=4`

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

- Multi-question episode mode is validated locally (4-question, CPU, Qwen3-0.6B).
- Full 4-question episodes complete end-to-end: `termination_reason=env_done`, `episode_clipped=False`, `final_questions_remaining=0`.
- `max_completion_length=1024` is the bundle starting point for 4-question episodes. This is not a proven minimum — GPU runs may need more.
- Episode-mode eval parity is available: every run produces `reward_log.jsonl`. Summarize with `scripts/summarize_episode_run.py`.
- GPU/vLLM validation on H100 is the next remaining step.

## GPU Sequence Length Tuning

Local CPU validation confirmed `max_completion_length=1024` works for 4-question episodes.
GPU/vLLM runs may need more headroom because:
- Answers are longer without a per-step token cap
- vLLM KV-cache pressure under colocate mode adds constraints

Tuning grid (start low, raise if episodes clip or `termination_reason=suffix_too_large` appears):
- `1024` — validated local starting point
- `1280`
- `1536`

Set `REPT_MAX_COMPLETION_LENGTH` to control. Use `scripts/summarize_episode_run.py` to check
`clipped_rate` and `termination_reasons` after each candidate.

> **Memory note:** Local success at 1024 does not guarantee H100 memory safety. vLLM colocate mode
> dominates memory. If OOM, tune `REPT_VLLM_GPU_UTIL` before raising sequence length.

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

## Recommended Practical Order

1. source the env file
2. bootstrap once
3. preflight
4. start the env server
5. scout the 32B model
6. dry-run the trainer
7. launch the full run

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
- prefer a venv created with `--system-site-packages`
- skip reinstalling the pinned `torch` wheel and reuse the image's system torch

The PT bootstrap script supports this through:

- `REPT_VENV_SYSTEM_SITE_PACKAGES=1`
- `REPT_SKIP_TORCH_INSTALL=1`

The easiest path is to start from:

- [gh200.lambda.env.example](/Users/harshawnsingh/Desktop/csci-544/project/model-run/gh200.lambda.env.example)

Recommended GH200 flow:

```bash
cp gh200.lambda.env.example gh200.lambda.env
source ./gh200.lambda.env
./bootstrap_p5_4xlarge_h100_lambda.sh
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
