"""Mechanical tool-chain interpreter for scheduled jobs (action="chain").

A chain is a list of steps executed with zero LLM involvement — the compiled
form of "when X every Y, if <condition>, do <actions>". The agent compiles
the user's sentence into this spec once (see skillsets/SCHEDULE_COMPILER.md);
the scheduler then runs it mechanically on every firing.

Spec:

    [
      {"tool": "check_aurora", "args": {...}, "as": "report"},
      {"if": {"field": "report.kp_index", "op": ">=", "value": 4},
       "then": [
         {"tool": "telegram_send",
          "args": {"message": "Aurora Kp {report.kp_index}: {report.summary}"}},
       ]},
    ]

Step kinds:
- {"tool", "args"?, "as"?}: invoke a registered tool; optionally bind its
  output to a name for later steps.
- {"if", "then"?, "else"?}: evaluate a structured condition against bound
  outputs and run the matching branch. The condition shape {"field", "op",
  "value"} mirrors the aurora workflow config's "when" clause. No eval().

String args support {dotted.path} substitution from bound outputs, e.g.
"{report.kp_index}". Unknown paths are left as-is so a misnamed field is
visible in the sent text rather than silently blanked.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

log = logging.getLogger(__name__)

_CONDITION_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains"}

# Maximum nesting depth for if/then/else branches. Chains are validated at
# creation, but a hand-edited schedule.json can bypass that — bound the
# recursion so a pathological record cannot blow the stack.
_MAX_NESTING = 10

_TEMPLATE_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}")


def _coerce_output(raw: Any) -> Any:
    """Make a tool's raw output navigable: parse JSON strings to dicts."""
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                pass
    return raw


def resolve_path(dotted: str, bindings: dict[str, Any]) -> tuple[bool, Any]:
    """Resolve "a.b.c" against bound outputs. Returns (found, value)."""
    parts = dotted.split(".")
    cur: Any = bindings
    for part in parts:
        cur = _coerce_output(cur) if isinstance(cur, str) else cur
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def eval_condition(cond: dict[str, Any], bindings: dict[str, Any]) -> bool:
    """Evaluate {"field", "op", "value"} against bindings. Unknown field -> False."""
    if not isinstance(cond, dict):
        return False
    field = cond.get("field", "")
    if not isinstance(field, str) or not field:
        return False
    op = cond.get("op", "==")
    expected = cond.get("value")
    found, actual = resolve_path(field, bindings)
    if not found:
        return False
    try:
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        if op in (">", ">=", "<", "<="):
            a, e = float(actual), float(expected)
            return {">": a > e, ">=": a >= e, "<": a < e, "<=": a <= e}[op]
        if op == "in":
            return actual in expected
        if op == "not_in":
            return actual not in expected
        if op == "contains":
            return expected in actual
    except (TypeError, ValueError):
        return False
    return False


def render_template(text: str, bindings: dict[str, Any]) -> str:
    """Substitute {dotted.path} placeholders from bindings."""

    def _sub(match: re.Match) -> str:
        found, value = resolve_path(match.group(1), bindings)
        return str(value) if found else match.group(0)

    return _TEMPLATE_RE.sub(_sub, text)


def _render_args(args: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(args, str):
        return render_template(args, bindings)
    if isinstance(args, dict):
        return {k: _render_args(v, bindings) for k, v in args.items()}
    if isinstance(args, list):
        return [_render_args(v, bindings) for v in args]
    return args


def validate_chain(chain: Any) -> list[str]:
    """Return a list of spec errors; empty means valid."""
    errors: list[str] = []
    if not isinstance(chain, list) or not chain:
        return ["tool_chain must be a non-empty list of steps"]
    for i, step in enumerate(chain):
        errors.extend(_validate_step(step, f"step {i}"))
    return errors


def _validate_step(step: Any, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(step, dict):
        return [f"{where}: must be an object"]
    if "tool" in step:
        if not isinstance(step["tool"], str) or not step["tool"].strip():
            errors.append(f"{where}: 'tool' must be a non-empty string")
        if "args" in step and not isinstance(step["args"], (dict, list, str)):
            errors.append(f"{where}: 'args' must be an object, list, or string")
        if "as" in step and not isinstance(step["as"], str):
            errors.append(f"{where}: 'as' must be a string")
    elif "if" in step:
        cond = step["if"]
        if not isinstance(cond, dict):
            errors.append(f"{where}: 'if' must be {{field, op, value}}")
        else:
            if not isinstance(cond.get("field"), str) or not cond["field"].strip():
                errors.append(f"{where}: 'if.field' must be a non-empty string")
            if cond.get("op", "==") not in _CONDITION_OPS:
                errors.append(f"{where}: 'if.op' must be one of {sorted(_CONDITION_OPS)}")
            if "value" not in cond:
                errors.append(f"{where}: 'if' needs a 'value'")
        for branch in ("then", "else"):
            if branch in step:
                if not isinstance(step[branch], list):
                    errors.append(f"{where}: '{branch}' must be a list of steps")
                else:
                    for j, sub in enumerate(step[branch]):
                        errors.extend(_validate_step(sub, f"{where}.{branch}[{j}]"))
    else:
        errors.append(f"{where}: must have 'tool' or 'if'")
    return errors


def run_chain(
    chain: list[dict[str, Any]],
    invoke: Callable[[str, dict[str, Any]], Any],
) -> dict[str, Any]:
    """Execute a chain. Returns a summary of what ran.

    Defensive by design: chains are validated at schedule creation, but a
    hand-edited schedule.json can bypass that. Malformed steps are recorded
    in ``errors`` and skipped — one bad step never aborts the rest of the
    chain.
    """
    bindings: dict[str, Any] = {}
    ran: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    def _run_steps(steps: Any, depth: int) -> None:
        if depth > _MAX_NESTING:
            errors.append(f"chain nesting exceeds {_MAX_NESTING}; deeper branch skipped")
            return
        if not isinstance(steps, list):
            errors.append(f"chain branch is not a list ({type(steps).__name__}); skipped")
            return
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"step {idx}: not an object; skipped")
                continue
            if "tool" in step:
                name = step["tool"]
                if not isinstance(name, str) or not name.strip():
                    errors.append(f"step {idx}: 'tool' must be a non-empty string; skipped")
                    continue
                args = _render_args(step.get("args") or {}, bindings)
                if not isinstance(args, dict):
                    args = {}
                try:
                    result = invoke(name, args)
                except Exception as exc:  # noqa: BLE001 — one bad tool must not kill the chain
                    errors.append(f"{name}: {exc}")
                    log.warning("chain tool %r failed: %s", name, exc)
                    continue
                ran.append(name)
                as_name = step.get("as")
                if isinstance(as_name, str) and as_name:
                    bindings[as_name] = _coerce_output(result)
            elif "if" in step:
                cond = step["if"]
                if not isinstance(cond, dict):
                    errors.append(f"step {idx}: 'if' must be {{field, op, value}}; skipped")
                    continue
                matched = eval_condition(cond, bindings)
                branch = step.get("then", []) if matched else step.get("else", [])
                label = f"if {cond.get('field')} {cond.get('op')} {cond.get('value')}"
                (ran if matched else skipped).append(label)
                _run_steps(branch, depth + 1)
            else:
                errors.append(f"step {idx}: has neither 'tool' nor 'if'; skipped")
                log.warning("chain %s", errors[-1])

    _run_steps(chain if isinstance(chain, list) else [], 0)
    return {"ok": not errors, "ran": ran, "skipped": skipped, "errors": errors}
