#!/usr/bin/env python3
"""Summarize an episode-mode reward log into JSON + Markdown reports.

Usage:
    python scripts/summarize_episode_run.py runs/grpo_openenv_episode_cpu_smoke4q/reward_log.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev


def safe_mean(vals: list) -> float | None:
    return mean(vals) if vals else None


def safe_std(vals: list) -> float:
    return stdev(vals) if len(vals) >= 2 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an episode-mode reward_log.jsonl.")
    parser.add_argument("log_path", type=Path, help="Path to reward_log.jsonl")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.log_path.read_text().splitlines()
        if line.strip()
    ]
    episodes = [r for r in records if r.get("event") == "episode_end"]

    if not episodes:
        print("No episode_end records found in log.")
        return

    episodes.sort(key=lambda e: (e.get("seed", -1), e.get("episode_idx", -1)))

    rewards = [e["episode_weighted_reward"] for e in episodes]
    completed = [e.get("questions_completed", 0) for e in episodes]
    tokens = [e.get("total_completion_tokens", 0) for e in episodes]
    completion_rate = (
        sum(1 for e in episodes if e.get("final_questions_remaining", 1) == 0) / len(episodes)
    )
    clipped_rate = sum(1 for e in episodes if e.get("episode_clipped", False)) / len(episodes)
    termination_counts = Counter(e.get("termination_reason", "unknown") for e in episodes)

    summary = {
        "num_episodes": len(episodes),
        "mean_reward": safe_mean(rewards),
        "std_reward": safe_std(rewards),
        "mean_questions_completed": safe_mean(completed),
        "completion_rate": completion_rate,
        "mean_completion_tokens": safe_mean(tokens),
        "clipped_rate": clipped_rate,
        "termination_reasons": dict(termination_counts),
        "episodes": episodes,
    }

    out_dir = args.log_path.parent
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2))

    md_rows = "\n".join(
        f"| {e.get('seed')} | {e.get('episode_weighted_reward', 0):.4f} | "
        f"{e.get('questions_completed', '?')} | {e.get('final_questions_remaining', '?')} | "
        f"{e.get('total_completion_tokens', '?')} | {e.get('total_tokens_serialized', '?')} | "
        f"{e.get('episode_clipped', '?')} | {e.get('termination_reason', '?')} |"
        for e in episodes
    )

    md = f"""# Episode Run Summary

## Aggregates
- Episodes: {len(episodes)}
- Completion rate: {completion_rate:.0%}
- Mean reward: {safe_mean(rewards):.4f} ± {safe_std(rewards):.4f}
- Mean questions completed: {safe_mean(completed):.2f}
- Mean completion tokens (model-only): {safe_mean(tokens):.1f}
- Clipped rate: {clipped_rate:.0%}
- Termination reasons: {dict(termination_counts)}

## Per-Episode
| seed | total_reward | questions_completed | final_questions_remaining | completion_tokens | serialized_tokens | clipped | termination_reason |
|------|-------------:|--------------------:|--------------------------:|------------------:|------------------:|:-------:|-------------------|
{md_rows}
"""
    (out_dir / "episode_summary.md").write_text(md)
    print(f"Written: {out_dir}/episode_summary.json")
    print(f"Written: {out_dir}/episode_summary.md")


if __name__ == "__main__":
    main()
