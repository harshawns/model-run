"""GRPO training script that keeps ReasoningEconomicsEnv in the loop.

This fork supports two OpenEnv training modes:

1. ``flat`` (default, stable local baseline)
   - Pre-fetch one deterministic first question per seed.
   - Train with a flat reward function that replays the seed and steps the env
     once with the model response.
   - Works locally without vLLM.

2. ``episode`` (multi-question episodes)
   - Uses TRL's experimental ``rollout_func`` support.
   - Replays the full OpenEnv episode for each repeated prompt / seed.
   - Serializes the whole multi-turn transcript into one completion with an
     ``env_mask`` so only model-generated assistant tokens contribute to the
     GRPO loss.
   - Supports both the regular Transformers path (local MacBook testing) and
     vLLM (GPU runs).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import torch

from training.config import TrainingRuntimeConfig
from training.openenv_runtime import (
    ReasonBudgetClient,
    resolve_budget_mode_from_observation,
    to_openenv_base_url,
)


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
            short_summary = " ".join(str(summary).split())
            if len(short_summary) > 48:
                short_summary = short_summary[:45].rstrip() + "..."
            entries.append(f"Q{i}:{status},{tokens}t,{short_summary}")
        history_lines = " | ".join(entries)
    else:
        history_lines = "none"

    metadata = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    budget_mode = metadata.get("budget_mode", "hard")
    step_cap = metadata.get("max_tokens_per_step", "?")

    return (
        f"STATE budget_mode={budget_mode} remaining_budget={int(obs['remaining_budget'])} "
        f"questions_remaining={obs['questions_remaining']} budget_per_remaining={obs['budget_per_remaining']:.0f} "
        f"accuracy={obs['accuracy_so_far']:.0%} step_cap={step_cap}\n"
        f"HISTORY {history_lines}\n"
        f"QUESTION {obs['question']}\n"
        "Use submit_answer exactly once. Return only one <tool_call> JSON block. "
        "Inside arguments.response: brief reasoning, then final answer in \\boxed{}."
    )


# ---------------------------------------------------------------------------
# Response extraction and reward functions
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


def reward_from_env(completions, env_reward=None, **kwargs):
    """Extract precomputed env rewards from rollout_func extra fields."""
    if env_reward:
        return [float(r) for r in env_reward]
    return [0.0] * len(completions or [])


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


def _sanitize_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop local metadata keys before chat templating / generation."""
    cleaned = []
    for message in messages:
        cleaned.append({k: v for k, v in message.items() if not str(k).startswith("_")})
    return cleaned


def _extract_episode_seed(messages: list[dict[str, Any]]) -> int:
    """Recover the deterministic OpenEnv seed stored on the prompt."""
    for message in messages:
        if "_env_seed" in message:
            return int(message["_env_seed"])
    raise ValueError("Episode prompt is missing '_env_seed' metadata.")


def _encode_chat_messages(
    tokenizer,
    trainer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    extra_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Encode a structured chat prompt using the trainer's template settings."""
    kwargs = dict(trainer.chat_template_kwargs) if trainer.chat_template_kwargs else {}
    if extra_template_kwargs:
        kwargs.update(extra_template_kwargs)
    return tokenizer.apply_chat_template(
        messages,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
        tools=trainer.tools or None,
        chat_template=trainer.chat_template,
        **kwargs,
    )


def _encode_message_suffix(
    tokenizer,
    trainer,
    base_messages: list[dict[str, Any]],
    suffix_messages: list[dict[str, Any]],
) -> list[int]:
    """Encode appended external messages plus the next assistant generation prompt.

    This mirrors TRL's tool-loop packing logic: external environment feedback is
    inserted into the completion stream with mask=0 so later assistant tokens are
    conditioned on the full episode transcript.

    Instead of diffing base vs base+suffix (fragile: Qwen3's template adds <think>
    tags to the last assistant message when it is the final message, making the
    prefix token sequence context-dependent), we take the last N tokens from the
    full encoding, where N = len(encode(suffix_messages + gen_prompt)) in isolation.
    User messages encode identically in isolation vs in-context because the Qwen
    template uses non-mergeable special tokens (<|im_start|>, <|im_end|>) as turn
    delimiters, so BPE boundaries are stable.
    """
    full_ids = list(
        _encode_chat_messages(
            tokenizer,
            trainer,
            base_messages + suffix_messages,
            add_generation_prompt=True,
        )
    )
    suffix_only_ids = list(
        _encode_chat_messages(
            tokenizer,
            trainer,
            suffix_messages,
            add_generation_prompt=True,
        )
    )
    suffix_len = len(suffix_only_ids)
    if suffix_len == 0 or suffix_len > len(full_ids):
        raise ValueError(
            f"Suffix encoding length {suffix_len} is invalid (full_ids len={len(full_ids)})."
        )
    return full_ids[-suffix_len:]


def _per_step_generation_cap(obs: dict[str, Any], trainer: Any, completion_tokens_used: int) -> int:
    """Bound each rollout step by env metadata, remaining budget, and trainer cap."""
    remaining_rollout = max(0, int(trainer.max_completion_length) - int(completion_tokens_used))
    if remaining_rollout <= 0:
        return 0

    metadata = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    step_cap = metadata.get("max_tokens_per_step", remaining_rollout)
    try:
        step_cap = int(step_cap)
    except (TypeError, ValueError):
        step_cap = remaining_rollout

    try:
        remaining_budget = int(obs.get("remaining_budget", remaining_rollout))
    except (AttributeError, TypeError, ValueError):
        remaining_budget = remaining_rollout

    default_mode = RUNTIME_CFG.normalized_default_mode() if RUNTIME_CFG else "hard"
    budget_mode = resolve_budget_mode_from_observation(obs, default_mode=default_mode, strict=False)

    if budget_mode == "hard":
        return max(1, min(step_cap, remaining_budget, remaining_rollout))
    return max(1, min(step_cap, remaining_rollout))


def _failure_rollout_sample(prompt_ids: list[int], tokenizer) -> tuple[list[int], list[float], list[int]]:
    """Return a minimal masked completion when episode rollout fails."""
    fallback_token = tokenizer.eos_token_id
    if fallback_token is None:
        fallback_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    completion_ids = [int(fallback_token)]
    return completion_ids, [0.0], [0]


def _generate_episode_turn(
    trainer: Any,
    tokenizer,
    current_messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    """Generate a single episode step with either vLLM or regular HF generation."""
    if trainer.use_vllm:
        from trl.experimental.openenv.utils import generate_rollout_completions

        return generate_rollout_completions(
            trainer,
            [current_messages],
            generation_overrides={"max_tokens": max_tokens},
            as_chat=True,
        )[0]

    from trl.models import unwrap_model_for_generation

    prompt_ids = _encode_chat_messages(
        tokenizer,
        trainer,
        current_messages,
        add_generation_prompt=True,
    )

    with torch.no_grad(), unwrap_model_for_generation(
        trainer.model_wrapped,
        trainer.accelerator,
        gather_deepspeed3_params=trainer.args.ds3_gather_for_generation,
        generation_kwargs=trainer.generation_kwargs,
    ) as unwrapped_model:
        if os.environ.get("ACCELERATE_USE_CPU", "0") == "1":
            model_device = torch.device("cpu")
            unwrapped_model = unwrapped_model.to(model_device)
        else:
            model_device = next(unwrapped_model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
        generation_config = copy.deepcopy(trainer.generation_config)
        generation_config.max_new_tokens = max_tokens
        prompt_completion_ids = unwrapped_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    prompt_length = len(prompt_ids)
    completion_tensor = prompt_completion_ids[:, prompt_length:]
    completion_ids = completion_tensor[0].tolist()
    eos_token_id = trainer.eos_token_id
    if eos_token_id is not None and eos_token_id in completion_ids:
        first_eos = completion_ids.index(eos_token_id)
        completion_ids = completion_ids[: first_eos + 1]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": None,
        "text": text,
    }


def build_episode_rollout_func(*, env_base_url: str, runtime_cfg: TrainingRuntimeConfig, output_dir: str):
    """Build a full-episode rollout function for TRL's experimental rollout API."""
    reward_log_path = runtime_cfg.resolved_reward_log_path(output_dir)
    reward_log_file = Path(reward_log_path)
    reward_log_file.parent.mkdir(parents=True, exist_ok=True)
    debug_episode = os.environ.get("REPT_DEBUG_EPISODE", "0") == "1"

    def rollout_func(prompts: list[list[dict[str, Any]]], trainer: Any):
        tokenizer = trainer.processing_class
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] | None = [] if trainer.use_vllm else None
        all_env_rewards: list[float] = []
        all_env_masks: list[list[int]] = []

        for episode_idx, raw_prompt in enumerate(prompts):
            seed = _extract_episode_seed(raw_prompt)
            base_messages = _sanitize_prompt_messages(raw_prompt)
            system_messages = [m for m in base_messages if m.get("role") == "system"] or [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            client = ReasonBudgetClient(base_url=env_base_url)
            episode_reward = 0.0
            step_idx = 0
            prompt_ids: list[int] | None = None
            completion_ids: list[int] = []
            completion_logprobs: list[float] = []
            env_mask: list[int] = []
            obs = None
            termination_reason = "exception_fallback"
            initial_questions_remaining: int | None = None
            executed_steps = 0
            any_step_hit_generation_cap = False

            try:
                if debug_episode:
                    print(
                        f"[episode_rollout] seed={seed} episode_idx={episode_idx} start",
                        flush=True,
                    )
                client.connect()
                reset_result = client.reset(seed=seed)
                obs = reset_result.observation
                initial_questions_remaining = int(obs.get("questions_remaining", 0))
                done = bool(reset_result.done)
                current_messages = system_messages + [{"role": "user", "content": format_observation_prompt(obs)}]
                prompt_ids = _encode_chat_messages(
                    tokenizer,
                    trainer,
                    current_messages,
                    add_generation_prompt=True,
                )

                while not done:
                    max_tokens = _per_step_generation_cap(obs, trainer, len(completion_ids))
                    if max_tokens <= 0:
                        termination_reason = "step_cap_hit"
                        break
                    if debug_episode:
                        print(
                            f"[episode_rollout] seed={seed} step={step_idx} generate_start max_tokens={max_tokens}",
                            flush=True,
                        )

                    output = _generate_episode_turn(
                        trainer,
                        tokenizer,
                        current_messages,
                        max_tokens,
                    )
                    if debug_episode:
                        print(
                            f"[episode_rollout] seed={seed} step={step_idx} generate_done completion_tokens={len(output['completion_ids'])}",
                            flush=True,
                        )
                    if prompt_ids is None:
                        prompt_ids = list(output["prompt_ids"])

                    turn_text = output["text"]
                    turn_ids = list(output["completion_ids"])
                    if output["logprobs"] is None:
                        turn_logprobs = [0.0] * len(turn_ids)
                    else:
                        turn_logprobs = list(output["logprobs"])

                    completion_ids.extend(turn_ids)
                    completion_logprobs.extend(turn_logprobs)
                    env_mask.extend([1] * len(turn_ids))
                    current_messages = current_messages + [{"role": "assistant", "content": turn_text}]

                    response = _extract_tool_response(turn_text) or turn_text
                    if debug_episode:
                        print(
                            f"[episode_rollout] seed={seed} step={step_idx} env_step_start",
                            flush=True,
                        )
                    step_result = client.step({"response": response})
                    obs = step_result.observation
                    done = bool(step_result.done)
                    step_reward = float(step_result.reward or 0.0)
                    weighted_reward = runtime_cfg.alpha * step_reward
                    episode_reward += weighted_reward
                    if debug_episode:
                        print(
                            f"[episode_rollout] seed={seed} step={step_idx} env_step_done reward={step_reward:.4f} done={done} questions_remaining={obs.get('questions_remaining')}",
                            flush=True,
                        )

                    executed_steps += 1
                    if len(turn_ids) == max_tokens:
                        any_step_hit_generation_cap = True

                    if runtime_cfg.log_rewards and step_idx % max(1, runtime_cfg.log_every_n_steps) == 0:
                        _step_meta = obs.get("metadata", {}) if isinstance(obs, dict) else {}
                        with reward_log_file.open("a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "mode": "episode",
                                        "seed": seed,
                                        "episode_idx": episode_idx,
                                        "step_idx": step_idx,
                                        "step_reward": step_reward,
                                        "weighted_step_reward": weighted_reward,
                                        "episode_reward_so_far": episode_reward,
                                        "done": done,
                                        "questions_remaining": obs.get("questions_remaining"),
                                        "step_completion_tokens": len(turn_ids),
                                        "step_max_tokens": max_tokens,
                                        "step_hit_generation_cap": len(turn_ids) == max_tokens,
                                        "completion_tokens_so_far": sum(env_mask),
                                        "remaining_rollout_after_step": int(trainer.max_completion_length) - len(completion_ids),
                                        "remaining_budget": obs.get("remaining_budget"),
                                        "budget_per_remaining": obs.get("budget_per_remaining"),
                                        "budget_mode": _step_meta.get("budget_mode"),
                                        "question_id": _step_meta.get("question_id"),
                                        "episode_seed": seed,
                                    }
                                )
                                + "\n"
                            )

                    if done:
                        termination_reason = "env_done"
                        break

                    next_user_message = {"role": "user", "content": format_observation_prompt(obs)}
                    suffix_ids = _encode_message_suffix(
                        tokenizer,
                        trainer,
                        current_messages,
                        [next_user_message],
                    )
                    remaining_rollout = int(trainer.max_completion_length) - len(completion_ids)
                    if remaining_rollout <= 0:
                        termination_reason = "rollout_cap_exhausted"
                        break
                    if len(suffix_ids) > remaining_rollout:
                        termination_reason = "suffix_too_large"
                        break

                    completion_ids.extend(suffix_ids)
                    completion_logprobs.extend([0.0] * len(suffix_ids))
                    env_mask.extend([0] * len(suffix_ids))
                    current_messages = current_messages + [next_user_message]
                    step_idx += 1

                if prompt_ids is None:
                    prompt_ids = _encode_chat_messages(
                        tokenizer,
                        trainer,
                        system_messages,
                        add_generation_prompt=True,
                    )
            except Exception as exc:
                print(f"[episode_rollout] seed={seed} failed: {exc!r} — assigning reward=0.0")
                if prompt_ids is None:
                    prompt_ids = _encode_chat_messages(
                        tokenizer,
                        trainer,
                        system_messages,
                        add_generation_prompt=True,
                    )
                completion_ids, completion_logprobs, env_mask = _failure_rollout_sample(prompt_ids, tokenizer)
                episode_reward = 0.0
            finally:
                client.disconnect()

            if not completion_ids:
                completion_ids, completion_logprobs, env_mask = _failure_rollout_sample(prompt_ids, tokenizer)

            if runtime_cfg.log_rewards:
                _final_questions_remaining = int(obs.get("questions_remaining", 0)) if obs else 0
                _questions_completed = (initial_questions_remaining or 0) - _final_questions_remaining
                _prompt_tokens = len(prompt_ids) if prompt_ids else 0
                with reward_log_file.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "mode": "episode",
                                "seed": seed,
                                "episode_idx": episode_idx,
                                "event": "episode_end",
                                "episode_weighted_reward": episode_reward,
                                "questions_completed": _questions_completed,
                                "initial_questions_remaining": initial_questions_remaining,
                                "final_questions_remaining": _final_questions_remaining,
                                "total_completion_tokens": sum(env_mask),
                                "total_tokens_serialized": len(completion_ids),
                                "prompt_tokens": _prompt_tokens,
                                "num_steps_executed": executed_steps,
                                "any_step_hit_generation_cap": any_step_hit_generation_cap,
                                "episode_clipped": termination_reason in {"rollout_cap_exhausted", "suffix_too_large"},
                                "termination_reason": termination_reason,
                                "max_completion_length": int(trainer.max_completion_length),
                            }
                        )
                        + "\n"
                    )
            if debug_episode:
                print(
                    f"[episode_rollout] seed={seed} episode_idx={episode_idx} end reward={episode_reward:.4f} completion_tokens={sum(env_mask)}",
                    flush=True,
                )

            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(completion_ids)
            if all_logprobs is not None:
                all_logprobs.append(completion_logprobs)
            all_env_rewards.append(episode_reward)
            all_env_masks.append(env_mask)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_env_rewards,
            "env_mask": all_env_masks,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    global ENV_BASE_URL, RUNTIME_CFG

    parser = argparse.ArgumentParser(
        description="GRPO training against ReasoningEconomicsEnv via OpenEnv"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--openenv_mode",
        type=str,
        default="flat",
        choices=["flat", "episode"],
        help="flat = single-step reward replay; episode = full multi-question rollouts via rollout_func.",
    )
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
    if args.use_vllm and importlib.util.find_spec("vllm") is None:
        system_name = platform.system()
        extra = ""
        if system_name != "Linux":
            extra = (
                f" Detected platform={system_name}. For MacBook testing, keep `--use_vllm` off and let "
                "`--openenv_mode episode` use the regular Transformers generation path instead."
            )
        raise ImportError(
            "vLLM is required when `--use_vllm` is set, but it is not installed in the active environment. "
            "Install it on the GPU machine with `pip install 'trl[vllm]'` or a compatible `vllm` build."
            + extra
        )
    if args.openenv_mode == "episode":
        os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    RUNTIME_CFG = TrainingRuntimeConfig(
        alpha=args.alpha,
        log_rewards=not args.no_log_rewards,
        log_every_n_steps=args.log_every_n_steps,
        reward_log_path=args.reward_log_path,
    )

    # ------------------------------------------------------------------
    # Build dataset: pre-fetch deterministic first observations by seed.
    # flat mode replays the first step only.
    # episode mode reuses the same seed but rolls the whole episode inside
    # rollout_func.
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
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation_prompt(obs)},
        ]
        if args.openenv_mode == "episode":
            prompt_messages[0]["_env_seed"] = int(i)
        prompts.append(prompt_messages)
        seeds.append(i)
    dataset_columns = {"prompt": prompts, "env_seed": seeds}
    dataset = Dataset.from_dict(dataset_columns)

    _force_cpu = os.environ.get("ACCELERATE_USE_CPU") == "1"
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        mask_truncated_completions=True,
        # When ACCELERATE_USE_CPU=1: force the Trainer to CPU and load the model
        # onto CPU before Accelerate wraps it, preventing the MPS/CPU device split
        # that causes "Calculated loss must be on the original device: mps:0" errors.
        **({"use_cpu": True} if _force_cpu else {}),
        model_init_kwargs={
            "torch_dtype": dtype_map[args.torch_dtype],
            **({"device_map": "cpu"} if _force_cpu else {}),
        },
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
    if args.openenv_mode == "episode" and hasattr(tokenizer, "response_schema"):
        # Multi-question episode completions contain both assistant tokens and
        # environment feedback tokens; force batch_decode instead of parse_response.
        tokenizer.response_schema = None

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

    trainer_kwargs = dict(
        model=args.model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
    )
    if args.openenv_mode == "episode":
        trainer_kwargs["reward_funcs"] = reward_from_env
        trainer_kwargs["rollout_func"] = build_episode_rollout_func(
            env_base_url=ENV_BASE_URL,
            runtime_cfg=RUNTIME_CFG,
            output_dir=args.output_dir,
        )
    else:
        trainer_kwargs["reward_funcs"] = openenv_reward

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
