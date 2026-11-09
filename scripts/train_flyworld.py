#!/usr/bin/env python3
"""Train/evaluate Aiko's MaleCNS mushroom-body action bias in FlyWorld.

By default this runs bounded simulated episodes in shadow mode: it evaluates
the current policy and reports the outcome without changing plasticity. Pass
``--apply --user-id <identity>`` to persist the simulated MB learning for one
identity. Live writes also require MEMORY_FLYMB_MODE=live and
AIKO_FLY_DOPAMINE_MODE=live in the environment.

This trains the fly action-selection/associative layer only. It does not
fine-tune the LLM, train language style, use human feedback, or alter persona
traits. FlyWorld's synthetic credit is scope-isolated and persona plasticity
explicitly ignores simulated events.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20,
                        help="bounded simulated training episodes (default: 20; max: 1000)")
    parser.add_argument("--steps", type=int, default=8,
                        help="simulated turns per episode (default: 8; max: 32)")
    parser.add_argument("--seed", type=int, default=1000,
                        help="first deterministic training seed")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="held-out evaluation episodes before and after training")
    parser.add_argument("--eval-seed", type=int, default=900000,
                        help="first held-out evaluation seed; kept separate from training seeds")
    parser.add_argument("--user-id", default=None,
                        help="identity whose existing fly policy is evaluated; required with --apply")
    parser.add_argument("--apply", action="store_true",
                        help="persist simulated MaleCNS MB plasticity (requires both live gates and --user-id)")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional path for the JSON summary report")
    return parser


def _configure_modes(*, apply: bool, user_id: str | None) -> None:
    """Fail closed by default; require explicit live gates before any writes."""
    if apply:
        if not user_id or not user_id.strip():
            raise ValueError("--apply requires --user-id so learned state has an explicit owner")
        if os.getenv("MEMORY_FLYMB_MODE", "off").strip().lower() != "live":
            raise ValueError("--apply requires MEMORY_FLYMB_MODE=live")
        if os.getenv("AIKO_FLY_DOPAMINE_MODE", "shadow").strip().lower() != "live":
            raise ValueError("--apply requires AIKO_FLY_DOPAMINE_MODE=live")
        os.environ["AIKO_FLYWORLD_MODE"] = "live"
        return

    # These explicit process-local overrides make the default command safe
    # even when config/memory.yaml enables live learning for the running agent.
    os.environ["AIKO_FLYWORLD_MODE"] = "shadow"
    os.environ["AIKO_FLY_DOPAMINE_MODE"] = "shadow"
    os.environ["MEMORY_FLYMB_MODE"] = "shadow"


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 1000:
        parser.error("--episodes must be between 1 and 1000")
    if not 1 <= args.steps <= 32:
        parser.error("--steps must be between 1 and 32")
    if not 1 <= args.eval_episodes <= 1000:
        parser.error("--eval-episodes must be between 1 and 1000")
    try:
        _configure_modes(apply=args.apply, user_id=args.user_id)
    except ValueError as exc:
        parser.error(str(exc))

    # Import only after the mode gates are set so module-level defaults cannot
    # accidentally observe a live setting during a shadow run.
    from cognition.flyworld.loop import evaluate_policy, run_episode

    identity = args.user_id.strip() if args.user_id else "flyworld-training"
    eval_seeds = tuple(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    # The baseline must never learn, even when the caller opted into apply.
    os.environ["AIKO_FLYWORLD_MODE"] = "shadow"
    os.environ["AIKO_FLY_DOPAMINE_MODE"] = "shadow"
    before = evaluate_policy(identity, seeds=eval_seeds, max_steps=args.steps)

    if args.apply:
        os.environ["AIKO_FLYWORLD_MODE"] = "live"
        os.environ["AIKO_FLY_DOPAMINE_MODE"] = "live"
    episode_summaries: list[dict] = []
    for offset in range(args.episodes):
        result = run_episode(
            identity,
            seed=args.seed + offset,
            max_steps=args.steps,
        )
        if not result.get("ran"):
            print(f"episode {offset + 1} skipped: {result.get('reason', 'unknown')}", file=sys.stderr)
            continue
        episode_summaries.append({
            "seed": args.seed + offset,
            "mean_reward": float(result.get("mean_reward", 0.0)),
            "steps": len(result.get("steps", [])),
            "n_pulsed": int(result.get("n_pulsed", 0)),
            "flushed": bool(result.get("flushed", False)),
        })

    # Hold the learned state fixed while measuring transfer on the same
    # deterministic, disjoint seeds used for the baseline.
    os.environ["AIKO_FLYWORLD_MODE"] = "shadow"
    os.environ["AIKO_FLY_DOPAMINE_MODE"] = "shadow"
    after = evaluate_policy(identity, seeds=eval_seeds, max_steps=args.steps)
    rewards = [float(item["mean_reward"]) for item in episode_summaries]
    report = {
        "training_layer": "MaleCNS v1.0 mushroom-body plasticity via FlyWorld + dopamine eligibility credit",
        "simulated": True,
        "applied": bool(args.apply),
        "identity": identity,
        "llm_finetuned": False,
        "persona_traits_changed": False,
        "training": {
            "requested_episodes": args.episodes,
            "completed_episodes": len(episode_summaries),
            "steps_per_episode": args.steps,
            "mean_reward": _mean(rewards),
            "n_pulsed": sum(item["n_pulsed"] for item in episode_summaries),
            "n_flushed_episodes": sum(item["flushed"] for item in episode_summaries),
            "seed_start": args.seed,
        },
        "held_out_evaluation": {
            "seeds": list(eval_seeds),
            "before": before,
            "after": after,
            "mean_reward_delta": round(
                float(after.get("mean_reward", 0.0)) - float(before.get("mean_reward", 0.0)), 4
            ),
        },
        "limits": [
            "FlyWorld uses synthetic scenarios and a hand-authored reward model, not human demonstrations.",
            "This updates only the identity-scoped MaleCNS mushroom-body plastic overlay; the connectome is read-only.",
            "It does not train LLM language generation or persona traits.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if len(episode_summaries) == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
