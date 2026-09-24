"""Phase 9 — closed-loop runner: act -> world responds -> fly learns.

The loop:
  1. sim state -> Phase-5 candidates (one per action; seeded "LLM prior"
     models an imperfect proposer: the best action gets the mass but with
     noise, so the fly must still discriminate).
  2. action_select.score_candidates (the REAL actuator) scores them.
  3. POLICY = argmax over the fly scores (record["candidates"][cid]["fly"]),
     not the blended winner. The blended winner is recorded for comparison.
     This isolates what the fly policy itself learns — which is exactly what
     the transfer test measures.
  4. sim.step(action) -> outcome + reward.
  5. Credit: sim-local eligibility buffer with the same decay math as
     eligibility.py (w = decay**age, cutoff 0.05). Each credited step gets
     dopamine.pulse(reward, kc=kc, weight=w, source="flyworld-sim") — the
     canonical teaching path. The shared real-turn eligibility trail is
     NEVER touched: synthetic steps must not pollute real credit assignment.
     (Unlike eligibility.assign_credit, age-0 IS included here at w=1.0 —
     there is no separate immediate-reinforce path in sim.)
  6. One flush_mb at episode end (Phase 2 convention).

Mode AIKO_FLYWORLD_MODE=off|shadow|live (default shadow):
  off    : return immediately, zero overhead.
  shadow : run episodes, log trails, apply ZERO weight changes
           ("would_pulse" — the pulse is computed, never applied).
  live   : pulses apply through the canonical path, which itself only
           writes when MEMORY_FLYMB_MODE=live.

Every record carries simulated=True: episode records, step records,
neural-state influence events, and replay-bridge candidates. Simulated
experience is never mixed indistinguishably with real experience.

Fail-soft: never raises. Per-episode cost is trivial (a few candidates x
a few steps of sparse KC ops). No ML weights are loaded.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque

log = logging.getLogger("aiko.flyworld.loop")

_lock = threading.RLock()
_EPISODES: dict[str, deque] = {}
_LAST: dict[str, dict] = {}
_EPISODE_TRAIL_MAX = 20


def _uid(user_id: str | None) -> str:
    return (user_id or "").strip() or "default"


def _env_str(name: str, default: str) -> str:
    try:
        from system.config import env_str
        return env_str(name, default)
    except Exception:
        return os.getenv(name, default) or default


def flyworld_mode() -> str:
    """AIKO_FLYWORLD_MODE: off | shadow | live (default shadow)."""
    m = (_env_str("AIKO_FLYWORLD_MODE", "shadow") or "shadow").strip().lower()
    return m if m in ("off", "shadow", "live") else "shadow"


def _elig_decay() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("FLY_ELIGIBILITY_DECAY", "0.7"))))
    except Exception:
        return 0.7


def _max_steps() -> int:
    try:
        return max(1, min(32, int(os.getenv("FLYWORLD_MAX_STEPS", "8"))))
    except Exception:
        return 8


def _prior_for(action: str, best: str, rng) -> float:
    """Seeded imperfect-proposer prior: best action gets the mass + noise."""
    try:
        if action == best:
            return max(0.05, min(0.95, 0.55 + rng.uniform(-0.1, 0.1)))
        return max(0.01, min(0.4, rng.uniform(0.02, 0.18)))
    except Exception:
        return 0.2


def _candidates_for(state: dict, rng) -> list:
    """Phase-5 candidates for the five sim actions."""
    from cognition.fly_behavior.action_select import Candidate
    from .sim import ACTIONS, action_kind

    cands = []
    for a in ACTIONS:
        cands.append(Candidate(
            id=a,
            kind=action_kind(a),
            label=a.replace("_", " "),
            description=f"sim action {a} for request: {state.get('request', '')}",
            llm_prior=_prior_for(a, state.get("best", ""), rng),
            hints={"energy": "med", "speed": "med",
                   "external": a == "reply_with_tool"},
        ))
    return cands


def _outcome_sentence(outcome, best: bool) -> str:
    """Faithful natural-language verbalization of the numeric outcome.

    Uses affect words from the same tiny lexicon the MB's synthetic
    front-end reads (circuit._POS_WORDS/_NEG_WORDS), so the learned
    association lands on the feature dimension that actually separates
    good from bad experience — the way real episodic traces do
    ("the user thanked me" / "that was wrong").
    """
    if outcome.vetoed:
        return "terrible outcome — conscience veto, the user was frustrated"
    if best and outcome.task_success >= 1.0:
        return "good outcome — task complete, the user was pleased"
    if best:
        return "bad outcome — the tool failed, the user was frustrated"
    return "bad outcome — wrong approach, the user was angry"


def _trace_text(state: dict, action: str, outcome) -> str:
    """MB-learnable trace of one sim turn. Synthetic by construction.

    Written as a natural-language experience summary (the form real
    episodic traces take), with the outcome verbalized honestly from the
    numeric result — never invented.
    """
    best = (action == state.get("best"))
    return (
        f"flyworld simulated turn: the user made a {state.get('kind')} request "
        f"('{state.get('request', '')}'). I chose to {action.replace('_', ' ')}. "
        f"{_outcome_sentence(outcome, best)}. "
        f"reward {outcome.reward:+.2f}"
    )


def _record_episode(user_id: str, record: dict) -> None:
    with _lock:
        dq = _EPISODES.get(user_id)
        if dq is None:
            dq = deque(maxlen=_EPISODE_TRAIL_MAX)
            _EPISODES[user_id] = dq
        dq.append(record)
        _LAST[user_id] = {
            "ts": record["ts"], "seed": record["seed"], "mode": record["mode"],
            "mean_reward": record["mean_reward"],
            "n_steps": len(record["steps"]),
        }
    try:
        from cognition.neural_state import get_neural_state
        get_neural_state(user_id).record_influence({
            "kind": "flyworld_episode",
            "simulated": True,
            "mode": record["mode"],
            "seed": record["seed"],
            "mean_reward": record["mean_reward"],
            "total_reward": record["total_reward"],
            "n_steps": len(record["steps"]),
            "n_pulsed": record["n_pulsed"],
            "veto_rate": record["veto_rate"],
        })
    except Exception:
        pass


def run_episode(
    user_id: str | None = None,
    seed: int = 0,
    *,
    max_steps: int | None = None,
    force: bool = False,
) -> dict:
    """Run one closed-loop sim episode. Never raises.

    force=True bypasses the mode gate for tests/harness (still fail-soft;
    pulses apply only when the MB layer itself is live).
    """
    mode = flyworld_mode()
    out: dict = {
        "ran": False, "mode": mode, "simulated": True, "seed": seed,
        "steps": [], "total_reward": 0.0, "mean_reward": 0.0,
        "n_pulsed": 0, "veto_rate": 0.0, "reason": "",
    }
    if mode == "off" and not force:
        out["reason"] = "mode_off"
        return out
    live = (mode == "live") or force
    try:
        from . import sim as _sim
        from cognition.fly_behavior.action_select import score_candidates
        from cognition.fly_registry import get_flymb, get_fly_store
        from cognition.flymemory.circuit import text_features
        from cognition.flymemory.dopamine import pulse

        rng = _sim.new_rng(seed)
        state = _sim.initial_state(rng)
        n_steps = max_steps or _max_steps()
        decay = _elig_decay()

        mb = get_flymb(user_id)
        mb_ok = mb is not None and hasattr(mb, "encode")

        # Sim-local eligibility buffer: [(kc, trace_text)]. Same decay math
        # as eligibility.py; kept local so synthetic steps never pollute the
        # real turn trail.
        elig_buf: list = []
        steps: list[dict] = []
        total = 0.0
        n_pulsed = 0
        n_vetoed = 0

        for i in range(n_steps):
            cands = _candidates_for(state, rng)
            try:
                rec = score_candidates(
                    cands, user_id=user_id,
                    context_text=str(state.get("request", "")),
                    source="flyworld",
                )
                scored = rec.get("candidates", {}) or {}
            except Exception as exc:
                log.debug("flyworld scoring skipped at step %d: %s", i, exc)
                scored = {}
            if scored:
                fly_pick = max(scored, key=lambda cid: scored[cid].get("fly", 0.0))
                blended = rec.get("winner")
            else:
                # Scoring unavailable: fall back to the seeded prior order so
                # the episode still exercises the world dynamics.
                fly_pick = max(cands, key=lambda c: c.llm_prior).id
                blended = fly_pick
            action = fly_pick

            next_state, outcome = _sim.step(state, action, rng)
            total += outcome.reward
            if outcome.vetoed:
                n_vetoed += 1

            trace = _trace_text(state, action, outcome)
            kc = None
            if mb_ok:
                try:
                    kc = mb.encode(text_features(trace))
                    elig_buf.append((kc, trace))
                except Exception as exc:
                    log.debug("flyworld encode skipped: %s", exc)
                    kc = None

            # Delayed credit over the sim-local trail (age-0 included at
            # w=1.0). Shadow mode records what WOULD be taught.
            pulsed_here: list[dict] = []
            for age, (bkc, _btrace) in enumerate(reversed(elig_buf)):
                w = decay ** age
                if w < 0.05:
                    break
                if live and mb_ok:
                    try:
                        res = pulse(
                            outcome.reward, user_id=user_id, kc=bkc,
                            weight=w, source="flyworld-sim",
                        )
                        if res.get("applied"):
                            n_pulsed += 1
                        pulsed_here.append({
                            "age": age, "weight": round(w, 4),
                            "applied": bool(res.get("applied")),
                            "reason": str(res.get("reason") or ""),
                        })
                    except Exception as exc:
                        pulsed_here.append({
                            "age": age, "weight": round(w, 4),
                            "applied": False,
                            "reason": f"error: {exc}",
                        })
                else:
                    pulsed_here.append({
                        "age": age, "weight": round(w, 4),
                        "applied": False,
                        "reason": "would_pulse" if mb_ok else "mb_unavailable",
                    })

            steps.append({
                "step": i,
                "simulated": True,
                "kind": state.get("kind"),
                "request": state.get("request"),
                "action": action,
                "fly_pick": fly_pick,
                "blended_winner": blended,
                "agreed": bool(blended and blended == fly_pick),
                "task_success": outcome.task_success,
                "satisfaction": round(outcome.satisfaction, 3),
                "conscience": outcome.conscience,
                "vetoed": outcome.vetoed,
                "veto_rules": list(outcome.veto_rules),
                "reward": round(outcome.reward, 4),
                "trace": trace,
                "pulsed": pulsed_here,
            })
            state = next_state

        if live and mb_ok and n_pulsed:
            try:
                store = get_fly_store(user_id)
                if store is not None:
                    store.flush_mb(mb)
                    out["flushed"] = True
            except Exception as exc:
                log.debug("flyworld flush skipped: %s", exc)

        out.update({
            "ran": True,
            "reason": "ok",
            "ts": time.time(),
            "steps": steps,
            "total_reward": round(total, 4),
            "mean_reward": round(total / max(1, len(steps)), 4),
            "n_pulsed": n_pulsed,
            "veto_rate": round(n_vetoed / max(1, len(steps)), 4),
        })
        _record_episode(_uid(user_id), out)
    except Exception as exc:
        log.debug("flyworld episode failed: %s", exc)
        out["reason"] = f"error: {exc}"
    return out


def evaluate_policy(
    user_id: str | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    *,
    max_steps: int | None = None,
    force: bool = False,
) -> dict:
    """Run episodes over seeds; return the transfer metric summary.

    mean_reward is the headline number: a policy that learned the loop
    should score higher than an untrained one on held-out seeds.
    """
    rewards: list[float] = []
    veto_rates: list[float] = []
    ran = 0
    for s in seeds:
        try:
            ep = run_episode(user_id, s, max_steps=max_steps, force=force)
        except Exception:
            continue
        if not ep.get("ran"):
            continue
        ran += 1
        rewards.append(float(ep.get("mean_reward", 0.0) or 0.0))
        veto_rates.append(float(ep.get("veto_rate", 0.0) or 0.0))
    return {
        "simulated": True,
        "n_episodes": ran,
        "seeds": list(seeds),
        "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
        "min_reward": round(min(rewards), 4) if rewards else 0.0,
        "mean_veto_rate": round(sum(veto_rates) / max(1, len(veto_rates)), 4),
    }


def past_episodes(user_id: str | None = None, n: int = 20) -> list[dict]:
    """Newest-first episode records (for the replay bridge / Studio)."""
    with _lock:
        dq = _EPISODES.get(_uid(user_id))
        items = list(dq) if dq else []
    return items[-max(1, n):][::-1]


def last_episode(user_id: str | None = None) -> dict:
    """Last episode summary (for harness/Studio)."""
    with _lock:
        return dict(_LAST.get(_uid(user_id), {}))


def clear_episodes(user_id: str | None = None) -> None:
    """Drop episode trails (tests / identity switch)."""
    with _lock:
        _EPISODES.pop(_uid(user_id), None)
        _LAST.pop(_uid(user_id), None)
