"""GRPO training script that keeps ReasoningEconomicsEnv in the loop.

This is the competition-compliant training path in this fork: the policy is
trained against the OpenEnv server rather than a direct dataset reward shortcut.

Local defaults are conservative so the script can be smoke-tested on Apple
Silicon without vLLM:
  - standard HF generation (`use_vllm=False`)
  - Qwen instruct model
  - LoRA adapters by default
  - explicit tokenizer loading
  - local-safe sampling / optimizer / dtype defaults

Reward approach (V1 — flat env-backed reward_funcs):
  - Dataset is built by pre-fetching N questions from OpenEnv at startup using
    deterministic seeds (seed i → always same question list).
  - At training time, reward_funcs replays each seed, steps the env once with
    the model's response, and returns OpenEnv's reward.
  - environment_factory is NOT used. TRL's tool-routing layer is bypassed entirely.
  - If the model emits a <tool_call> block, arguments.response is extracted and
    sent to the env. If not, the raw completion text is sent as a fallback.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from training.config import TrainingRuntimeConfig
from training.openenv_runtime import ReasonBudgetClient, to_openenv_base_url


# ---------------------------------------------------------------------------
# Module-level config (set in main() before trainer init)
# ---------------------------------------------------------------------------

ENV_BASE_URL: str = ""
RUNTIME_CFG: TrainingRuntimeConfig | None = None


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

TOOL_CALL_EXAMPLE = (
    "<tool_call>\n"
    '{"name": "submit_answer", "arguments": {"response": "Short reasoning. Final answer: \\\\boxed{4}"}}\n'
    "</tool_call>"
)

SYSTEM_PROMPT = (
    "/no_think "
    "You are solving math problems under a shared token budget inside an OpenEnv episode. "
    "Use the available tool exactly once for each question. "
    "Reply with a single XML-wrapped tool call and no other text before or after it. "
    "Inside the tool's `response` argument, keep reasoning brief and end with the final answer in \\boxed{}. "
    "Prefer one or two short sentences of reasoning, not a long chain-of-thought. "
    "Stop immediately after the final answer. "
    "Use exactly this format:\n"
    f"{TOOL_CALL_EXAMPLE}"
)


def format_observation_prompt(obs: dict[str, Any]) -> str:
    """Format an environment observation into a prompt for the model."""
    history = obs.get("history", [])
    if history:
        entries = []
        for i, h in enumerate(history, 1):
            status = "correct" if h.get("was_correct") else "wrong"
            tokens = h.get("tokens_used", "?")
            summary = h.get("question_summary", "")
            entries.append(f"  Q{i}: {summary}... [{tokens} tokens, {status}]")
        history_lines = "\n".join(entries)
    else:
        history_lines = "  (none yet)"

    metadata = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    budget_mode = metadata.get("budget_mode", "hard")
    step_cap = metadata.get("max_tokens_per_step", "?")

    return (
        f"\nOpenEnv episode state\n"
        f"- Budget mode: {budget_mode}\n"
        f"- Remaining budget: {int(obs['remaining_budget'])} tokens\n"
        f"- Questions remaining: {obs['questions_remaining']} (including this one)\n"
        f"- Budget per remaining question: {obs['budget_per_remaining']:.0f} tokens\n"
        f"- Accuracy so far: {obs['accuracy_so_far']:.0%}\n"
        f"- Max tokens per step: {step_cap}\n"
        f"\nPrevious questions:\n{history_lines}\n"
        f"\nCurrent question:\n{obs['question']}\n"
        f"\nCall the `submit_answer` tool exactly once for this question. "
        f"Reply ONLY with this tool-call format:\n{TOOL_CALL_EXAMPLE}\n"
        f"Keep reasoning brief, then give the final answer in \\boxed{{}} inside the `response` argument. "
        f"Do not ramble and do not add extra text outside the tool call."
    )


# ---------------------------------------------------------------------------
# Response extraction and reward function
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _extract_tool_response(text: str) -> str | None:
    """Parse the 'response' argument from a <tool_call>...</tool_call> block."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return obj.get("arguments", {}).get("response")
    except (json.JSONDecodeError, AttributeError):
        return None


def _load_seed_manifest(path: str, bucket: str) -> list[int]:
    """Load and filter a seed manifest produced by scout_openenv_seeds.py."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    records = payload.get("records", [])

    selected = []
    for record in records:
        record_bucket = str(record.get("bucket", "unclear"))
        if bucket != "all" and record_bucket != bucket:
            continue
        seed = record.get("seed")
        if seed is None:
            continue
        selected.append(int(seed))

    if not selected:
        raise ValueError(
            f"No seeds matched bucket={bucket!r} in manifest {manifest_path}"
        )
    return selected


def _load_tokenizer_for_model(model_name: str):
    """Load tokenizer with a local-cache-first strategy.

    Some tokenizer code paths perform Hub lookups even when weights are already
    cached locally. Prefer local_files_only + slow tokenizer first, then fall
    back to the standard behavior if needed.
    """
    from transformers import AutoTokenizer

    attempts = [
        {"local_files_only": True, "use_fast": False},
        {"local_files_only": True},
        {"use_fast": False},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return AutoTokenizer.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load tokenizer for {model_name!r}") from last_error


def openenv_reward(prompts, completions, env_seed, **kwargs):
    """Grade each completion against OpenEnv by replaying the stored seed.

    Version 1: single-step reward only — resets the env with the stored seed
    (reproducing the same first question), steps once with the model's response,
    and returns OpenEnv's reward. No multi-question episode replay.

    If the completion contains a <tool_call> block, arguments.response is
    extracted and sent to the env. Otherwise the raw completion text is used
    as a fallback so reward is never gated on format compliance alone.

    Args:
        prompts: list of prompt messages (unused).
        completions: list of model outputs — strings or [{"role":"assistant",...}].
        env_seed: list of int seeds forwarded from the dataset column, one per
                  completion (TRL repeats dataset values num_generations times).
    """
    rewards = []
    for completion, seed in zip(completions, env_seed):
        if isinstance(completion, list) and isinstance(completion[0], dict):
            text = completion[0].get("content", "")
        else:
            text = str(completion)

        # Prefer structured extraction; fall back to raw text.
        response = _extract_tool_response(text) or text

        client = ReasonBudgetClient(base_url=ENV_BASE_URL)
        try:
            client.connect()
            client.reset(seed=int(seed))
            result = client.step({"response": response})
            reward = float(result.reward or 0.0)
            if RUNTIME_CFG:
                reward *= RUNTIME_CFG.alpha
        except Exception as exc:
            print(f"[openenv_reward] env error (seed={seed}): {exc!r} — assigning reward=0.0")
            reward = 0.0
        finally:
            client.disconnect()
        rewards.append(reward)
    return rewards


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    global ENV_BASE_URL, RUNTIME_CFG

    parser = argparse.ArgumentParser(
        description="GRPO training against ReasoningEconomicsEnv via OpenEnv"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument(
        "--seed_manifest_path",
        type=str,
        default="",
        help="Optional JSON manifest from scout_openenv_seeds.py. If set, seeds are drawn from this manifest.",
    )
    parser.add_argument(
        "--seed_bucket",
        type=str,
        default="all",
        choices=["all", "easy", "mixed", "hard", "unclear"],
        help="Seed bucket to use when --seed_manifest_path is provided.",
    )
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--no_log_rewards", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--reward_log_path", type=str, default="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="runs/grpo_openenv")
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--env_base_url", type=str, default=None)
    parser.add_argument("--space_url", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        choices=["colocate", "server"],
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--vllm_max_model_length", type=int, default=None)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=None)
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    ENV_BASE_URL = to_openenv_base_url(
        env_base_url=args.env_base_url,
        space_url=args.space_url,
    )
    RUNTIME_CFG = TrainingRuntimeConfig(
        alpha=args.alpha,
        log_rewards=not args.no_log_rewards,
        log_every_n_steps=args.log_every_n_steps,
        reward_log_path=args.reward_log_path,
    )

    # ------------------------------------------------------------------
    # Build dataset: pre-fetch one question per prompt from OpenEnv.
    # Seed i always produces the same question list (random.Random(i)),
    # so reward_funcs can replay deterministically at training time.
    # ------------------------------------------------------------------
    if args.seed_manifest_path:
        selected_seeds = _load_seed_manifest(args.seed_manifest_path, args.seed_bucket)
        if len(selected_seeds) > args.n_prompts:
            selected_seeds = selected_seeds[: args.n_prompts]
        print(
            f"Pre-fetching {len(selected_seeds)} questions from OpenEnv "
            f"(bucket={args.seed_bucket}, manifest={args.seed_manifest_path})..."
        )
    else:
        selected_seeds = list(range(args.n_prompts))
        print(
            f"Pre-fetching {args.n_prompts} questions from OpenEnv "
            f"(seeds 0..{args.n_prompts - 1})..."
        )

    prompts, seeds = [], []
    for i in selected_seeds:
        client = ReasonBudgetClient(base_url=ENV_BASE_URL)
        try:
            client.connect()
            obs = client.reset(seed=i).observation
        finally:
            client.disconnect()
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation_prompt(obs)},
        ])
        seeds.append(i)

    dataset = Dataset.from_dict({"prompt": prompts, "env_seed": seeds})

    grpo_config_kwargs = dict(
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        save_strategy="epoch",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        mask_truncated_completions=True,
        model_init_kwargs={"torch_dtype": dtype_map[args.torch_dtype]},
        generation_kwargs={
            "remove_invalid_values": True,
            "renormalize_logits": True,
        },
    )
    if args.use_vllm:
        grpo_config_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_gpu_memory_utilization is not None:
            grpo_config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        if args.vllm_max_model_length is not None:
            grpo_config_kwargs["vllm_max_model_length"] = args.vllm_max_model_length
        if args.vllm_tensor_parallel_size is not None:
            grpo_config_kwargs["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
        if args.vllm_enable_sleep_mode:
            grpo_config_kwargs["vllm_enable_sleep_mode"] = True
    if args.max_steps and args.max_steps > 0:
        grpo_config_kwargs["max_steps"] = args.max_steps
    grpo_config = GRPOConfig(**grpo_config_kwargs)

    tokenizer = _load_tokenizer_for_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType
        except ImportError as exc:
            raise ImportError(
                "LoRA is enabled by default for the local OpenEnv path, but `peft` is not installed. "
                "Install it with `.venv-local/bin/python -m pip install peft` or rerun with `--no_lora`."
            ) from exc

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=openenv_reward,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
