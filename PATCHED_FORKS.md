# Patched PT and Env Forks

This repository is not a copy of the upstream project in its original state.
It packages two modified source snapshots:

- `ReasoningEconomicsPT`
- `ReasoningEconomicsEnv`

Those modifications are the reason this bundle is usable for the current
OpenEnv RL pipeline. Without them, the local and GPU launch scripts here would
not reproduce the behavior described in the project notes.

This document explains, in detail, what changed in each fork and why those
changes matter.

## Why There Are Two Patched Forks

The training stack depends on both sides of the system:

- the PT repo controls how prompts are built, how completions are generated,
  how rewards are pulled back into GRPO, and how GPU jobs are launched
- the Env repo controls the actual environment behavior, answer parsing, reward
  shaping, token-budget rules, and the metadata returned by the server

The project originally ran into issues on both sides:

- the trainer-side TRL `environment_factory` path was not reliable for this
  setup
- the environment needed better grading consistency, stronger configurability,
  and more diagnostic reward information

The current working solution is:

- trainer-side OpenEnv V1 via flat `reward_funcs` plus deterministic seed replay
- environment-side shared grading, explicit reward shaping, and richer step
  metadata

## PT Fork: `ReasoningEconomicsPT`

The PT fork is the part that turns OpenEnv into an actual GRPO training loop.

### Main architectural change

The most important PT change is that the OpenEnv path no longer depends on
TRL's `environment_factory` integration.

Instead, the working path in `training/grpo_train.py` does this:

1. Pre-fetches one environment observation per deterministic seed.
2. Builds a dataset containing:
   - the full prompt
   - the seed that produced that prompt
3. Lets the model generate a completion for that prompt.
4. Replays the same seed against OpenEnv inside the reward function.
5. Calls `env.step({"response": ...})`.
6. Uses the returned environment reward as the GRPO reward.

This is what makes the current training path competition-compliant while still
being stable enough to run locally and on GPU.

### `training/grpo_train.py`

This file was effectively rebuilt around the OpenEnv V1 path.

Key changes:

- replaced the unreliable `environment_factory` path with flat OpenEnv
  `reward_funcs`
- added deterministic seed replay through `env_seed`
- added prompt construction from real OpenEnv observations rather than a dummy
  or offline prompt source
- added shared extraction of the model's tool response / final response payload
- added local-cache-friendly tokenizer loading logic
- enabled LoRA by default for the local OpenEnv path
- added current best baseline generation defaults such as
  `max_completion_length=384`
- added seed-manifest filtering support:
  - `--seed_manifest_path`
  - `--seed_bucket`
- added GPU-facing vLLM controls:
  - `--vllm_mode`
  - `--vllm_gpu_memory_utilization`
  - `--vllm_max_model_length`
  - `--vllm_tensor_parallel_size`
  - `--vllm_enable_sleep_mode`
- added explicit dtype plumbing through `--torch_dtype`

Why these matter:

- deterministic seed replay is what lets a normal GRPO reward function talk to
  a stateful environment without losing alignment between prompt and reward
- LoRA keeps the path practical on constrained hardware
- vLLM and dtype controls are required for realistic GPU runs
- manifest filtering enables seed scouting and difficulty bucketing without
  changing the core training loop

### `training/scout_openenv_seeds.py`

This file was added to support seed scouting before training.

What it does:

- probes a fixed set of deterministic OpenEnv seeds
- generates multiple completions per seed
- replays each completion through OpenEnv reward
- computes per-seed reward statistics
- buckets each seed into:
  - `easy`
  - `mixed`
  - `hard`
  - `unclear`

Why it exists:

- GRPO learns best when there is within-group reward variance
- some seeds are always-correct or always-wrong for a given model
- scouting lets us understand difficulty structure before training

Important caveat:

- the seed buckets are model-specific
- a manifest created for `Qwen/Qwen3-1.7B` should not be assumed valid for
  `Qwen/Qwen3-14B` or `Qwen/Qwen3-32B`

### `scripts/run_grpo_lambda.sh`

This script was modified so GPU runs use the actual OpenEnv path and expose the
settings needed for vLLM-based training.

Key changes:

- updated defaults to the current OpenEnv path rather than older local defaults
- added OpenEnv-oriented defaults for:
  - model
  - generations
  - completion length
- added dtype control via `REPT_TORCH_DTYPE`
- added vLLM controls via:
  - `REPT_VLLM_MODE`
  - `REPT_VLLM_GPU_MEMORY_UTILIZATION`
  - `REPT_VLLM_MAX_MODEL_LENGTH`
  - `REPT_VLLM_TENSOR_PARALLEL_SIZE`
  - `REPT_VLLM_ENABLE_SLEEP_MODE`
- forwards those settings into `training.grpo_train`

Why this matters:

- the trainer supports the GPU settings, but a reproducible launch path needs a
  wrapper that actually exposes them
- the larger-model bundle depends on this script to remain the source of truth
  for training invocation

### `eval/evaluate.py`

This file was updated to match the current OpenEnv client API so evaluation
paths did not lag behind the training path.

Why it matters:

- if the env client API changes but evaluation code does not, training and eval
  will silently diverge
- keeping eval runnable is important for smoke tests and regression checks

### `training/grpo_train_local.py`

This file was added as a local fallback path.

Important distinction:

- this is not the competition-compliant OpenEnv path
- it exists as a practical local training path when the env-backed path is not
  what you want to run

It should be thought of as auxiliary infrastructure, not the main OpenEnv RL
solution.

### PT-side outcome

The PT fork is what makes these things possible:

- OpenEnv-backed GRPO training runs end to end
- deterministic seed replay works
- local and GPU launches share one trainer path
- seed scouting and bucketed dataset selection work
- the larger-model bundle can launch against the copied PT snapshot instead of a
  hand-edited local repo

## Env Fork: `ReasoningEconomicsEnv`

The Env fork is the part that makes the reward trustworthy, configurable, and
debuggable.

### Main architectural change

The biggest environment-side change is that grading logic is now shared and the
reward is no longer a single opaque wrong-answer bucket.

The environment now distinguishes important failure modes:

- parseable but wrong answer
- malformed / unparsable answer
- truncated answer

That is essential for interpreting RL failures and for avoiding a completely
flat reward landscape on wrong outputs.

### `env/grading.py`

This file was added and then expanded to centralize final-answer parsing and
grading behavior.

Why it exists:

- answer extraction logic must be consistent between predictions and gold labels
- ad hoc parsing in multiple places leads to silent reward mismatch
- a shared parser reduces drift across env components

### `data/loaders.py`

This file was modified so gold-label parsing uses the shared grading logic
rather than duplicating answer parsing inline.

Why it matters:

- if gold labels and model outputs are parsed by different logic, the reward can
  be wrong even when the model is actually correct

### `env/config.py`

This file was modified to make environment behavior controllable from env vars.

Key changes:

- tokenizer default aligned to the working Qwen3 path
- added env-configurable reward knobs for:
  - `correct_reward`
  - `incorrect_penalty`
  - `parseable_bonus`
  - `malformed_penalty`
  - `truncation_penalty`

Why this matters:

- reward shaping should be adjustable without editing code every time
- the server wrapper can now enforce a consistent V1 profile through exports

### `env/reward.py`

This file was upgraded from a simple reward computation path to a richer,
diagnostic reward breakdown.

Key changes:

- added `compute_reward_components(...)`
- reward now distinguishes:
  - correct
  - parseable-wrong
  - malformed-wrong
  - truncated-wrong

Example shaping from the current path:

- parseable wrong: `-0.08`
- malformed wrong: `-0.15`
- truncated wrong: `-0.20`
- correct: `1.00`

Why this matters:

- training logs can now show whether failures are because the model is wrong,
  malformed, or clipping
- this is much more informative than a single flat penalty for all failure
  cases

### `env/models.py`

This file was modified so the environment state includes `episode_seed`.

Why it matters:

- deterministic seed replay is only useful if the environment state can expose
  and track the seed that defines the episode
- this makes debugging and logging much easier

### `env/reason_budget_env.py`

This file received several important changes.

Key changes:

- switched to shared grading / final-answer extraction
- enforces the per-step token cap
- now exposes richer observation and history metadata
- includes seed/question metadata such as:
  - `episode_seed`
  - `question_id`
  - `question_source`
  - `problem_type`
- records step-level reward diagnostics such as:
  - parseability
  - truncation
  - predicted answer
  - reward components

Why this matters:

- training failures are much easier to diagnose when the environment explains
  what happened on the step
- this is what made it possible to distinguish clipping issues from ordinary
  wrong answers during the local RL runs

### `server/app.py`

This file was modified so the environment loads configuration from env vars via
`EnvConfig.from_env()`.

Why it matters:

- the server wrappers in this repo depend on env-var-driven configuration
- the same server code can now be reused across local smoke tests and GPU runs

### `pyproject.toml`

This was modified so `vllm` is optional on the environment side.

Why it matters:

- the environment should not require GPU-only packages when running in lighter
  local configurations
- it makes the env repo more portable across development setups

### Env-side outcome

The Env fork is what makes these things possible:

- shared, consistent grading between data loading and runtime evaluation
- reward shaping that separates malformed, truncated, and parseable-wrong cases
- token-budget enforcement that actually matters for RL behavior
- seed/question metadata that makes training diagnostics interpretable
- server configuration that can be controlled by wrappers instead of manual code
  edits

## How PT and Env Fit Together

The PT and Env changes are complementary.

The PT side:

- knows how to replay a seed and send a completion back to OpenEnv

The Env side:

- knows how to grade that completion consistently and return an interpretable
  reward

If you remove either side, the current pipeline breaks down:

- remove the PT changes and the trainer loses the reliable OpenEnv reward path
- remove the Env changes and the reward becomes less trustworthy and much harder
  to debug

## Why The Current Bundle Uses Both Snapshots

This repository includes copied snapshots of both forks because the larger-model
launch path should be self-contained.

That gives you:

- one repo to clone onto the GPU machine
- wrapper scripts in the root
- PT and Env source snapshots directly alongside those wrappers
- no dependence on a separate local `my_forks/` directory

The copied source trees are snapshots, not a new upstream. If you change the PT
or Env fork elsewhere, this repo does not automatically update. You would need
to copy those changes over again.

## Recommended Reading Order

If you want to understand the system quickly, read in this order:

1. `README.md`
2. `ReasoningEconomicsPT/training/grpo_train.py`
3. `ReasoningEconomicsPT/training/scout_openenv_seeds.py`
4. `ReasoningEconomicsEnv/env/reward.py`
5. `ReasoningEconomicsEnv/env/reason_budget_env.py`
6. `p5_4xlarge_h100.lambda.env.example`

That sequence shows:

- how prompts are built
- how rewards are computed
- how seeds are replayed
- how the server is configured
- how the single-H100 launch path is intended to work
