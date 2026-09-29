# Phase 12 — Fly metrics + safe dial

**Status:** shipped on branch `feat/phase12-metrics-safe-dial`

After Phases 1–11, the fly stack is largely **live**. Phase 12 does not add
new circuits. It adds **observability and Jetson-safe throttles** so you can
measure whether the fly layer is helping before (or while) dialing weights.

## What landed

| Piece | Role |
|-------|------|
| `cognition/fly_behavior/metrics.py` | Process + per-user counters |
| Action trail hooks | `score_candidates` / `note_feedback` record metrics |
| Fullbrain throttle | Skip whole-brain steps when interval/N not met |
| Studio `GET /studio/fly/api/metrics` | JSON snapshot for the dashboard |
| Config keys | See below |

## Metrics (process-wide)

| Counter | Meaning |
|---------|---------|
| `action_scores` | Times candidates were scored |
| `action_applied` | Live mode actually changed the winner vs LLM prior |
| `action_agree_llm` / `action_disagree_llm` | Fly final rank vs LLM prior rank |
| `action_ambiguous` | Close call / veto → caller fallback |
| `action_veto` | Any candidate vetoed (external + low valence) |
| `feedback_praise` / `feedback_correction` / `feedback_taught` | Outcome → teach path |
| `fullbrain_steps` / `fullbrain_skipped_throttle` | Whole-brain step accounting |
| `fullbrain_ms_total` | Sum of step wall times (avg in `rates`) |

Derived rates:

- **applied_rate** — fraction of scores where the fly decision was applied
- **agree_rate** — how often fly ranking matches the LLM prior ranking
- **fullbrain_avg_ms** — mean whole-brain step cost when it runs

## Safe dial (env / `config/memory.yaml`)

| Key | Default | Effect |
|-----|---------|--------|
| `AIKO_FLY_ACTION_MODE` | `live` (yaml) / code default `shadow` | `off` \| `shadow` \| `live` |
| `AIKO_FLY_ACTION_WEIGHT` | `0.25` | Blend weight toward fly score |
| `AIKO_FULLBRAIN_EVERY_N` | `3` (yaml Phase 12) | Run fullbrain at most every N MB turns |
| `AIKO_FULLBRAIN_MIN_INTERVAL_S` | `2.0` | Minimum seconds between fullbrain steps |

Recommended Jetson 8 GB starting point:

```yaml
AIKO_FULLBRAIN_EVERY_N: "3"
AIKO_FULLBRAIN_MIN_INTERVAL_S: "2.0"
AIKO_FLY_ACTION_MODE: "live"   # or shadow while validating
AIKO_FLY_ACTION_WEIGHT: "0.25"
```

If chat latency spikes, raise `EVERY_N` / `MIN_INTERVAL_S` or set action mode
back to `shadow`.

## Studio

```
GET /studio/fly/api/metrics
GET /studio/fly/api/fly_actions?n=20
GET /studio/fly/api/state   # modes + weights expanded
```

## Acceptance checklist (manual)

1. Open Fly Studio metrics after ~20 turns — counters move.
2. Teach (“don’t bring X up”) → `feedback_taught` increments; recall changes.
3. With throttle on, `fullbrain_skipped_throttle` > 0 and chat stays responsive.
4. If `applied_rate` is high but behavior feels wrong → lower weight or shadow.

## Out of scope

- Offline vote-weight learning (log first; fit later)
- LIF / spiking
- New anatomical circuits
