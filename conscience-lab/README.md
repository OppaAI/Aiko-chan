# conscience-lab/ — fresh start (2026-10-05)

The previous conscience line (v5–v11 label schemes, 2x2/3x2 moral-judgment fine-tunes)
is retired and archived under `old-conscience-lab/` (reference only — do not build on it).

## Doctrine

Laya is a **malicious-content detector**, not a moral philosopher.

- One binary decision per check: **refusal category or none**. No `unknown`/`cannot-tell` label anywhere.
- 16 refusal categories: Ten Commandments construed broadly (v) + harm to any lives
  including animals (h). See `categories.md`. Each category is marked **TERMINAL**
  (hard refuse, template, no override) or **SPEAK UP** (decline + brief concern + legitimate adjacent).
- Judge **stated content only**. No hidden-intent detection, no suspicion-policing.
- Hard refusals are terminal: no escalation, no user-approval override.
  Threat model is future untrusted users, not Oppa. See `threat_model.md`.

## Architecture

```
user input ──▶ [L0 patterns] ──▶ [Laya pre-gate] ──▶ agent reasoning
                                                        │
untrusted content (web/fetch) ──▶ [Laya trust-boundary] ┤
                                                        ▼
                                            [Tool gate: policy table]
                                            tool+args → allow / owner-approval / refuse
                                                        │
                                                        ▼
                                            [Laya egress check] ──▶ output
                                   (response, file writes exc. scratch,
                                    image prompts, external sends)
                                                        │
                              doubt → [L3 ministral] → [L4 Oppa]
```

- Checks sit at **boundaries** (trust entry, egress exit), not wrapped around every tool.
- **One Laya instance** (Jetson RAM ~857MB free). Different prompts per task:
  prompt A = refusal classification, prompt B = outcome verification (selective).
- L0: deterministic vetoes (injection, concealment, known-bad). Tool gate: deterministic.
- Harness enforcement is planned for Phase 4, with the goal of preventing the model from skipping checks; it is not yet implemented.
- Build order and methodology: see `build_plan.md`.

## Status

- Phase 0A: `categories.md` v1 drafted, in Oppa's review (terminal/speak-up split is the open decision).
- Eval set: `eval/laya_refusal_eval.jsonl` — 503 cases (304 harmful: 16 x 19; 199 benign), QC'd.
  Not yet split into dev/locked-test (Phase 0C).
- Phase 0E skeletons: `threat_model.md`, `tool_policy.md`.

## Files

| file | what |
|---|---|
| `categories.md` | 16 categories, one-line definitions, egg tests, terminal/speak-up mapping, animal-cruelty + idolatry deep dives, observable-content rule |
| `build_plan.md` | adopted build plan: Phase 0–5 + Phase 0 checklist |
| `threat_model.md` | who we defend against, what's out of scope |
| `tool_policy.md` | deterministic tool policy table (skeleton) |
| `eval/laya_refusal_eval.jsonl` | 503-case eval set (pre-split) |
