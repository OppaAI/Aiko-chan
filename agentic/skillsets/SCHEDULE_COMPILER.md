---
id: SCHEDULE_COMPILER
name: Schedule Compiler — natural language to mechanical scheduled workflows
summary: Turn the owner's scheduling sentences ("check Kp every hour, TG me when >= 4") into compiled tool-chain schedules that run mechanically with no LLM per firing.
triggers: schedule, every hour, remind me when, notify me when, check.*if, alert me, cron, recurring
tools: schedule_job, list_schedule, cancel_schedule
---

# Schedule Compiler

Use this skill when Oppa asks for something recurring with a condition, e.g.
"check KP every hour, if >= 4 then email and TG me". Your job is to compile
his sentence into a mechanical tool chain — once — so every firing runs
without waking the LLM.

## The pattern

Every such sentence decomposes into three parts:

1. **Trigger** — cadence/window: "every hour", "daily at 9", "weekdays 18:00→01:00".
   Maps to `schedule_job`'s `frequency` / `time_of_day` / `interval_seconds`.
2. **Check** — a tool call whose output decides: "KP", "new email from X".
   Maps to a `{"tool", "args", "as"}` step; the output is bound to a name.
3. **Actions** — what to do when the condition holds: "email me", "TG me".
   Maps to `{"tool", "args"}` steps inside a `{"if", "then"}` branch.

Conditions are structured, never prose:
`{"field": "report.kp_index", "op": ">=", "value": 4}`.
Ops: `== != > >= < < <= in not_in contains`.
String args support `{dotted.path}` substitution from bound outputs,
e.g. `"message": "Aurora Kp {report.kp_index}: {report.summary}"`.

## Compilation steps

0. **Confirm the tools exist.** The check tool and every action tool must be
   registered. If one is missing, say so — do not invent tool names.
1. **Extract the three parts** from his sentence. Ask one short question only
   if a part is genuinely ambiguous (which channel? what threshold?).
2. **Build the chain spec** (see `agentic/workflows/common/chain.py`):
   ```json
   [
     {"tool": "check_aurora", "as": "report"},
     {"if": {"field": "report.kp_index", "op": ">=", "value": 4}, "then": [
        {"tool": "email_send", "args": {"subject": "Aurora Kp {report.kp_index}", "body": "{report.summary}"}},
        {"tool": "telegram_send", "args": {"message": "Aurora Kp {report.kp_index}: {report.summary}"}}
     ]}
   ]
   ```
3. **Call `schedule_job`** with `action="chain"`, the `tool_chain`, and the
   trigger fields. `task` should read like his sentence for the Studio view.
4. **Confirm back in his words**: what will run, how often, what triggers the
   notification. Mention it is editable in Studio and by sentence
   ("make it 5", "switch TG to Discord" → edit the chain fields, or cancel
   and recompile).

## Rules

- Prefer `action="chain"` over `action="agentic"` for check-then-notify
  patterns: the mechanical path costs nothing per firing and cannot drift.
- Use `action="agentic"` only when the firing itself needs judgment
  (e.g. "summarize anything interesting"). Never use agentic just to
  re-evaluate a fixed threshold.
- Parameters (threshold, channels, cadence) live as chain/spec fields, not in
  prose — so later edits are field updates, not recompiles.
- Timed changes ("stop alerting until Friday") are schedule edits or
  `cancel_schedule`, never code.
- Every compiled schedule appears in the end-of-day report's activity log —
  the owner must always be able to see what his sentences became.
