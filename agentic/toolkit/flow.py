"""
agentic/toolkit/flow.py

n8n-style core nodes for Aiko's DAG executor (agentic/graph_engine.py).

WHY THIS EXISTS
---------------
Aiko already had *domain* nodes (job_hunt, aurora, research, social) and the
shared Layer-1 five (ingest → store → synth → verify → output). What she did
NOT have is the boring, universal plumbing every real workflow engine ships:
IF, Switch, Merge, Filter, Set, Sort, Limit, Dedupe, Aggregate, Split Out,
Split In Batches, HTTP Request, Wait, Code, Template, No-Op, Stop-and-Error.

Those are the nodes engineers actually wire in n8n/Zapier/Airflow. Without
them every new workflow needs a bespoke Python tool; with them, most of a
workflow is drag-and-drop in the DAG studio.

DATA MODEL — "items"
--------------------
Every node here speaks the same envelope, mirroring n8n's item stream:

    {"ok": true, "count": 12, "items": [ {...}, {...} ], "meta": {...}}

Nodes accept EITHER:
  - ``items_json``  — a ``$result:<node>`` string from an upstream node, or
  - ``from_state``  — a GraphState key (preferred for anything large, because
                      graph_engine._substitute truncates ``$result:`` at
                      4000 chars).

Nodes always write the FULL item list to ``to_state`` (default ``items``) and
return a size-capped preview as node content. So the canvas stays readable
while the data stays intact. On an 8GB Jetson that difference matters.

CONTROL FLOW
------------
``if_condition`` returns the literal string ``true``/``false`` and
``switch_route`` returns a route name, so downstream nodes gate with the
engine's existing conditions — no engine change needed::

    {"id": "vip", "tool": "set_fields", "depends_on": ["router"],
     "run_if": {"node": "router", "equals": "vip"}, "args": {...}}

``split_in_batches`` pairs with the engine's bounded ``loop_to`` /
``max_visits`` to walk a large list a chunk at a time.

EXPRESSIONS
-----------
``{{ ... }}`` templates and ``=`` expressions are evaluated by a restricted
AST interpreter (see ``evaluate``). No ``exec``, no imports, no dunder access,
no arbitrary calls — a whitelist of pure builtins and safe str/list methods
only. This is a Code node you can hand to an LLM without losing sleep.

All nodes are ``graph=True, react=False``: they belong on the canvas, not in
the ReAct tool list (which a 3B model already struggles to choose from).
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from datetime import datetime, timezone as dt_timezone
from typing import Any

from agentic.registry import TOOLS, tool
from system.log import get_logger

log = get_logger(__name__)

# ── budgets (Jetson-safe defaults) ───────────────────────────────────────────
FLOW_MAX_ITEMS = int(os.getenv("FLOW_MAX_ITEMS", "500"))
FLOW_INLINE_ITEMS = int(os.getenv("FLOW_INLINE_ITEMS", "20"))
FLOW_INLINE_CHARS = int(os.getenv("FLOW_INLINE_CHARS", "3500"))
FLOW_HTTP_TIMEOUT = float(os.getenv("FLOW_HTTP_TIMEOUT", "15"))
FLOW_HTTP_MAX_BYTES = int(os.getenv("FLOW_HTTP_MAX_BYTES", "5000000"))
FLOW_MAX_WAIT_SECONDS = float(os.getenv("FLOW_MAX_WAIT_SECONDS", "60"))
FLOW_EXPR_MAX_CHARS = int(os.getenv("FLOW_EXPR_MAX_CHARS", "2000"))


def _spec(name: str, description: str):
    """Use the tools.yaml catalog entry when present, else register by name."""
    return TOOLS[name] if name in TOOLS else name


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _loads(raw: Any, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


# ── item envelope ────────────────────────────────────────────────────────────

_ITEM_LIST_KEYS = ("items", "results", "records", "selection", "data", "rows", "messages")


def _coerce_item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return {"value": list(value)}
    return {"value": value}


def load_items(
    items_json: Any = "",
    state: Any = None,
    from_state: str = "",
    default: list | None = None,
) -> list[dict[str, Any]]:
    """Resolve an item list from a GraphState key or an upstream node result.

    Accepts every shape the rest of the codebase already emits: the flow
    envelope, ``{"results": [...]}`` from the shared Layer-1 nodes, a bare
    JSON list, a single object, or plain text (wrapped as ``{"text": ...}``).
    """
    raw: Any = None
    if from_state and state is not None and isinstance(getattr(state, "data", None), dict):
        raw = state.data.get(from_state)
    if raw is None:
        raw = _loads(items_json, None)
    if raw is None:
        text = items_json if isinstance(items_json, str) else ""
        text = text.strip()
        if text:
            return [{"text": text}]
        return list(default or [])

    if isinstance(raw, dict):
        for key in _ITEM_LIST_KEYS:
            value = raw.get(key)
            if isinstance(value, list):
                return [_coerce_item(v) for v in value][:FLOW_MAX_ITEMS]
        return [raw]
    if isinstance(raw, list):
        return [_coerce_item(v) for v in raw][:FLOW_MAX_ITEMS]
    return [_coerce_item(raw)]


def emit(
    items: list[dict[str, Any]],
    *,
    state: Any = None,
    to_state: str = "items",
    ok: bool = True,
    **meta: Any,
) -> str:
    """Write the full list to GraphState, return a size-capped preview.

    The preview is what lands in NodeResult.content (and therefore in
    ``$result:`` substitution and the studio run panel); the state copy is
    what the next node should actually consume via ``from_state``.
    """
    items = list(items or [])[:FLOW_MAX_ITEMS]
    if to_state and state is not None and isinstance(getattr(state, "data", None), dict):
        state.data[to_state] = items
        state.data[f"{to_state}_count"] = len(items)

    preview = items[: max(1, FLOW_INLINE_ITEMS)]
    payload = {
        "ok": ok,
        "count": len(items),
        "shown": len(preview),
        "state_key": to_state or None,
        "items": preview,
    }
    if meta:
        payload["meta"] = meta
    text = _dumps(payload)
    if len(text) > FLOW_INLINE_CHARS:
        payload["items"] = preview[: max(1, len(preview) // 2)]
        payload["truncated"] = True
        text = _dumps(payload)
        if len(text) > FLOW_INLINE_CHARS:
            payload["items"] = []
            payload["truncated"] = True
            payload["note"] = f"preview omitted — read state key '{to_state}'"
            text = _dumps(payload)
    return text


# ── path resolution ──────────────────────────────────────────────────────────

_INDEX_RE = re.compile(r"\[(-?\d+)\]")


def resolve_path(obj: Any, path: str, default: Any = None) -> Any:
    """Dotted path lookup with list indexing: ``a.b[0].c``."""
    if not path:
        return obj
    cursor = obj
    for raw_part in str(path).split("."):
        part = raw_part.strip()
        if not part:
            continue
        indices = _INDEX_RE.findall(part)
        key = _INDEX_RE.sub("", part)
        if key:
            if isinstance(cursor, dict):
                if key not in cursor:
                    return default
                cursor = cursor[key]
            else:
                return default
        for index in indices:
            if not isinstance(cursor, (list, tuple)):
                return default
            position = int(index)
            if position >= len(cursor) or position < -len(cursor):
                return default
            cursor = cursor[position]
    return cursor


# ── restricted expression evaluator (the "Code" node engine) ─────────────────

_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs, "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "round": round, "min": min, "max": max, "sum": sum, "sorted": sorted,
    "list": list, "dict": dict, "set": set, "any": any, "all": all,
    "enumerate": enumerate, "range": range, "zip": zip, "reversed": reversed,
}

_SAFE_METHODS = frozenset({
    "lower", "upper", "strip", "lstrip", "rstrip", "title", "split", "rsplit",
    "join", "replace", "startswith", "endswith", "find", "count", "format",
    "get", "keys", "values", "items", "index", "isdigit", "isalpha", "capitalize",
})

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare,
    ast.Call, ast.Constant, ast.Attribute, ast.Subscript, ast.Name, ast.Load,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Slice, ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.ListComp, ast.comprehension,
    ast.GeneratorExp, ast.JoinedStr, ast.FormattedValue, ast.Starred,
)


class ExpressionError(ValueError):
    """Raised when an expression is unsafe or cannot be evaluated."""


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ExpressionError("attribute access to private names is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ExpressionError("private names are not allowed")


class _Evaluator(ast.NodeVisitor):
    """Tiny tree-walking interpreter over the whitelisted node set."""

    def __init__(self, env: dict[str, Any]):
        self.env = env

    def generic_visit(self, node):  # pragma: no cover - guarded by _check_ast
        raise ExpressionError(f"unsupported expression node: {type(node).__name__}")

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        return node.value

    def visit_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        item = self.env.get("item")
        if isinstance(item, dict) and node.id in item:
            return item[node.id]
        raise ExpressionError(f"unknown name: {node.id}")

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        if isinstance(value, dict):
            if node.attr in value:
                return value[node.attr]
            if node.attr in _SAFE_METHODS:
                return getattr(value, node.attr)
            return None
        if node.attr in _SAFE_METHODS:
            return getattr(value, node.attr)
        raise ExpressionError(f"attribute not allowed: {node.attr}")

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        key = self.visit(node.slice)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            return None

    def visit_Slice(self, node):
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return slice(lower, upper, step)

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node):
        return tuple(self.visit(e) for e in node.elts)

    def visit_Set(self, node):
        return {self.visit(e) for e in node.elts}

    def visit_Dict(self, node):
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    def visit_JoinedStr(self, node):
        return "".join(str(self.visit(v)) for v in node.values)

    def visit_FormattedValue(self, node):
        return str(self.visit(node.value))

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand

    def visit_BinOp(self, node):
        left, right = self.visit(node.left), self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            if isinstance(right, (int, float)) and right > 64:
                raise ExpressionError("exponent too large")
            return left ** right
        raise ExpressionError("unsupported operator")

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for value in values:
                if not value:
                    return value
                result = value
            return result
        for value in values:
            if value:
                return value
        return values[-1] if values else False

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            elif isinstance(op, ast.Is):
                ok = left is right
            else:
                ok = left is not right
            if not ok:
                return False
            left = right
        return True

    def visit_IfExp(self, node):
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def visit_Call(self, node):
        func = self.visit(node.func)
        if not callable(func):
            raise ExpressionError("target is not callable")
        name = getattr(func, "__name__", "")
        if name not in _SAFE_FUNCS and name not in _SAFE_METHODS:
            raise ExpressionError(f"call not allowed: {name or 'anonymous'}")
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)

    def visit_ListComp(self, node):
        return list(self._comprehend(node))

    def visit_GeneratorExp(self, node):
        return list(self._comprehend(node))

    def _comprehend(self, node):
        if len(node.generators) != 1:
            raise ExpressionError("only single-generator comprehensions are allowed")
        gen = node.generators[0]
        if not isinstance(gen.target, ast.Name):
            raise ExpressionError("comprehension target must be a simple name")
        iterable = self.visit(gen.iter)
        out = []
        saved = self.env.get(gen.target.id)
        try:
            for value in list(iterable)[:FLOW_MAX_ITEMS]:
                self.env[gen.target.id] = value
                if all(self.visit(cond) for cond in gen.ifs):
                    out.append(self.visit(node.elt))
        finally:
            if saved is None:
                self.env.pop(gen.target.id, None)
            else:
                self.env[gen.target.id] = saved
        return out


def build_env(
    item: dict[str, Any] | None = None,
    index: int = 0,
    items: list[dict[str, Any]] | None = None,
    state: Any = None,
) -> dict[str, Any]:
    """Names visible to an expression: item/json/index/items/state/now."""
    data = {}
    if state is not None and isinstance(getattr(state, "data", None), dict):
        data = {k: v for k, v in state.data.items() if not str(k).startswith("_")}
    return {
        "item": item or {},
        "json": item or {},
        "index": index,
        "items": items or [],
        "state": data,
        "now": datetime.now(dt_timezone.utc).isoformat(),
        "true": True,
        "false": False,
        "null": None,
        "none": None,
    }


def evaluate(expression: str, env: dict[str, Any]) -> Any:
    """Evaluate one restricted expression. Raises ExpressionError on refusal."""
    text = (expression or "").strip()
    if not text:
        return ""
    if len(text) > FLOW_EXPR_MAX_CHARS:
        raise ExpressionError("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc
    _check_ast(tree)
    # The tree-walk itself can still raise builtin errors on hostile but
    # syntactically valid input (unhashable set/dict keys, BinOp type
    # mismatches, ...). Normalize everything to ExpressionError so callers
    # (resolve_value, code_transform) degrade to a clean refusal instead of
    # crashing the node with an execution_error.
    try:
        return _Evaluator(dict(env)).visit(tree)
    except ExpressionError:
        raise
    except Exception as exc:
        raise ExpressionError(f"evaluation failed: {exc}") from exc


_TEMPLATE_RE = re.compile(r"\{\{(.+?)\}\}", re.DOTALL)


def render_template(text: Any, env: dict[str, Any]) -> Any:
    """Replace ``{{ expr }}`` spans. A lone span returns its native value."""
    if not isinstance(text, str):
        return text
    matches = list(_TEMPLATE_RE.finditer(text))
    if not matches:
        return text
    if len(matches) == 1 and matches[0].group(0).strip() == text.strip():
        try:
            return evaluate(matches[0].group(1), env)
        except ExpressionError:
            return ""

    def _sub(match: re.Match) -> str:
        try:
            value = evaluate(match.group(1), env)
        except ExpressionError as exc:
            log.debug("flow: template expression failed (%s)", exc)
            return ""
        if isinstance(value, (dict, list)):
            return _dumps(value)
        return "" if value is None else str(value)

    return _TEMPLATE_RE.sub(_sub, text)


def resolve_value(raw: Any, env: dict[str, Any]) -> Any:
    """Resolve a studio-supplied value.

    ``"=price * 2"``      → expression
    ``"{{ first }} x"``   → template
    anything else         → literal (recursively for list/dict)
    """
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("="):
            try:
                return evaluate(stripped[1:], env)
            except ExpressionError as exc:
                log.debug("flow: expression failed (%s)", exc)
                return None
        return render_template(raw, env)
    if isinstance(raw, list):
        return [resolve_value(v, env) for v in raw]
    if isinstance(raw, dict):
        return {k: resolve_value(v, env) for k, v in raw.items()}
    return raw


# ── conditions ───────────────────────────────────────────────────────────────

def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare(actual: Any, op: str, expected: Any = None) -> bool:
    """One condition comparison. Unknown operators fall back to equality."""
    op = (op or "eq").strip().lower()
    text = "" if actual is None else str(actual)
    want = "" if expected is None else str(expected)

    if op in {"exists", "is_not_empty", "not_empty", "truthy"}:
        return bool(actual) if not isinstance(actual, (int, float)) else actual is not None
    if op in {"not_exists", "is_empty", "empty", "falsy"}:
        return not bool(actual)
    if op == "is_true":
        return actual is True or text.strip().lower() in {"true", "1", "yes", "on"}
    if op == "is_false":
        return actual is False or text.strip().lower() in {"false", "0", "no", "off", ""}
    if op in {"eq", "equals", "="}:
        if isinstance(actual, (int, float)) and _as_float(expected) is not None:
            return float(actual) == _as_float(expected)
        return text.casefold() == want.casefold()
    if op in {"ne", "not_equals", "!="}:
        return not compare(actual, "eq", expected)
    if op == "contains":
        return want.casefold() in text.casefold()
    if op in {"not_contains", "does_not_contain"}:
        return want.casefold() not in text.casefold()
    if op in {"starts_with", "startswith"}:
        return text.casefold().startswith(want.casefold())
    if op in {"ends_with", "endswith"}:
        return text.casefold().endswith(want.casefold())
    if op in {"regex", "matches"}:
        try:
            return bool(re.search(want, text, re.IGNORECASE))
        except re.error:
            return False
    if op in {"gt", "gte", "lt", "lte"}:
        left, right = _as_float(actual), _as_float(expected)
        if left is None or right is None:
            return False
        return {
            "gt": left > right, "gte": left >= right,
            "lt": left < right, "lte": left <= right,
        }[op]
    if op == "in":
        pool = expected if isinstance(expected, (list, tuple, set)) else [
            p.strip() for p in want.split(",") if p.strip()
        ]
        return any(text.casefold() == str(p).casefold() for p in pool)
    if op == "not_in":
        return not compare(actual, "in", expected)
    return text == want


def match_conditions(
    item: dict[str, Any],
    conditions: Any,
    combinator: str = "and",
    env: dict[str, Any] | None = None,
) -> bool:
    """Evaluate a condition group against one item.

    Accepted shapes::

        {"combinator": "and", "conditions": [{"field": "x", "op": "eq", "value": 1}]}
        [{"field": "x", "op": "eq", "value": 1}]
        {"field": "x", "op": "eq", "value": 1}
        "=price > 100"          (bare expression)
    """
    env = env or build_env(item)
    if isinstance(conditions, str):
        text = conditions.strip()
        if not text:
            return True
        parsed = _loads(text, None)
        if parsed is None:
            value = resolve_value(text if text.startswith("=") else f"={text}", env)
            return bool(value)
        conditions = parsed
    if isinstance(conditions, dict):
        if "conditions" in conditions:
            combinator = str(conditions.get("combinator") or combinator)
            conditions = conditions.get("conditions") or []
        else:
            conditions = [conditions]
    if not isinstance(conditions, list) or not conditions:
        return True

    results: list[bool] = []
    for rule in conditions:
        if not isinstance(rule, dict):
            continue
        if rule.get("expression"):
            results.append(bool(resolve_value(f"={rule['expression']}", env)))
            continue
        field = str(rule.get("field") or rule.get("path") or "")
        actual = resolve_path(item, field) if field else item
        if isinstance(rule.get("value"), str):
            expected = resolve_value(rule["value"], env)
        else:
            expected = rule.get("value")
        results.append(compare(actual, str(rule.get("op") or rule.get("operator") or "eq"), expected))

    if not results:
        return True
    return all(results) if str(combinator).strip().lower() != "or" else any(results)


# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER / UTILITY NODES
# ═════════════════════════════════════════════════════════════════════════════

@tool(
    _spec("trigger_manual", "Workflow entry point: emit literal/pinned items into the pipeline."),
    description="Workflow entry point. Emits the items you paste in (or a state key) so downstream nodes have data. The n8n 'Manual Trigger' / pinned-data equivalent.",
    graph=True, react=False, domain="flow",
)
def trigger_manual(
    items_json: str = "[]",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    """Entry node. With no input it emits one empty item, like n8n does."""
    items = load_items(items_json, state, from_state, default=[{}])
    if not items:
        items = [{}]
    return emit(items, state=state, to_state=to_state, source="manual")


@tool(
    _spec("no_op", "Pass items through unchanged (placeholder / join point)."),
    description="Do nothing. Useful as a join point, a placeholder while building, or a stable anchor for several branches to converge on.",
    graph=True, react=False, domain="flow",
)
def no_op(items_json: str = "", from_state: str = "", to_state: str = "items", *, state=None) -> str:
    return emit(load_items(items_json, state, from_state), state=state, to_state=to_state)


@tool(
    _spec("sticky_note", "Canvas annotation. Never executes as real work."),
    description="Sticky note for the canvas. Documentation only — the runner skips it.",
    graph=True, react=False, domain="note",
)
def sticky_note(content: str = "", **_kwargs) -> str:
    return ""


@tool(
    _spec("wait_delay", "Pause the branch for N seconds (bounded)."),
    description="Wait before continuing. Bounded by FLOW_MAX_WAIT_SECONDS so a canvas edit cannot stall the Jetson.",
    graph=True, react=False, domain="flow",
)
def wait_delay(
    seconds: float = 1.0,
    items_json: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    try:
        wait = max(0.0, min(float(seconds or 0), FLOW_MAX_WAIT_SECONDS))
    except (TypeError, ValueError):
        wait = 1.0
    time.sleep(wait)
    return emit(load_items(items_json, state, from_state), state=state, to_state=to_state, waited_seconds=wait)


@tool(
    _spec("stop_and_error", "Fail the branch deliberately when a condition holds."),
    description="Stop the workflow with an explicit error. Downstream nodes are marked dependency_failed, which is how you surface a bad precondition instead of silently continuing.",
    graph=True, react=False, domain="flow",
)
def stop_and_error(
    message: str = "workflow stopped",
    items_json: str = "",
    conditions_json: str = "",
    combinator: str = "and",
    from_state: str = "",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    if conditions_json:
        env = build_env(items[0] if items else {}, 0, items, state)
        triggered = any(
            match_conditions(item, conditions_json, combinator, build_env(item, i, items, state))
            for i, item in enumerate(items or [{}])
        )
        if not triggered:
            return emit(items, state=state, to_state="items", stopped=False)
        message = str(render_template(message, env))
    raise RuntimeError(f"stop_and_error: {message}")


# ═════════════════════════════════════════════════════════════════════════════
# BRANCHING
# ═════════════════════════════════════════════════════════════════════════════

@tool(
    _spec("if_condition", "IF node: returns 'true'/'false' for run_if gating."),
    description="IF node. Evaluates conditions and returns the literal string true or false so downstream nodes can gate with run_if {\"node\":\"<id>\",\"equals\":\"true\"}. Matching and non-matching items are written to separate state keys.",
    graph=True, react=False, domain="flow",
)
def if_condition(
    items_json: str = "",
    conditions_json: str = "",
    combinator: str = "and",
    mode: str = "any",
    from_state: str = "",
    true_state: str = "items_true",
    false_state: str = "items_false",
    to_state: str = "",
    *,
    state=None,
) -> str:
    """mode: any (default) | all | first — how item-level matches roll up.

    ``to_state`` is empty by default *on purpose*: an IF must not quietly
    narrow the shared ``items`` stream for every node downstream. Branches
    read ``true_state`` / ``false_state``; set ``to_state`` only when you
    deliberately want the winning branch written back to a shared key.
    """
    items = load_items(items_json, state, from_state)
    matched: list[dict] = []
    unmatched: list[dict] = []
    for index, item in enumerate(items):
        env = build_env(item, index, items, state)
        (matched if match_conditions(item, conditions_json, combinator, env) else unmatched).append(item)

    mode = (mode or "any").strip().lower()
    if not items:
        verdict = False
    elif mode == "all":
        verdict = len(matched) == len(items)
    elif mode == "first":
        verdict = bool(matched) and matched[0] is items[0]
    else:
        verdict = bool(matched)

    if state is not None and isinstance(getattr(state, "data", None), dict):
        if true_state:
            state.data[true_state] = matched
        if false_state:
            state.data[false_state] = unmatched
        if to_state:
            state.data[to_state] = matched if verdict else unmatched
        state.data["if_matched_count"] = len(matched)
    log.info("[flow.if] %d/%d matched -> %s", len(matched), len(items), verdict)
    return "true" if verdict else "false"


@tool(
    _spec("switch_route", "Switch node: returns the first matching route name."),
    description="Switch node. Walks ordered rules and returns the first matching route name (or the fallback). Gate each branch with run_if {\"node\":\"<id>\",\"equals\":\"<route>\"}.",
    graph=True, react=False, domain="flow",
)
def switch_route(
    items_json: str = "",
    rules_json: str = "[]",
    fallback: str = "default",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    """rules: [{"route":"vip","field":"tier","op":"eq","value":"gold"}, ...]"""
    items = load_items(items_json, state, from_state)
    rules = _loads(rules_json, []) or []
    if isinstance(rules, dict):
        rules = rules.get("rules") or []
    probe = items[0] if items else {}
    env = build_env(probe, 0, items, state)

    route = str(fallback or "default")
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("route") or rule.get("output") or "").strip()
        if not name:
            continue
        group = rule.get("conditions") or rule
        if any(match_conditions(item, group, str(rule.get("combinator") or "and"),
                                build_env(item, i, items, state)) for i, item in enumerate(items or [probe])):
            route = name
            break

    if state is not None and isinstance(getattr(state, "data", None), dict):
        state.data[to_state] = items
        state.data["switch_route"] = route
    log.info("[flow.switch] route=%s over %d item(s)", route, len(items))
    return route


@tool(
    _spec("filter_items", "Keep only items matching the conditions."),
    description="Filter node. Drops items that fail the conditions and passes the rest on. Use this instead of IF when you want to narrow the stream rather than branch it.",
    graph=True, react=False, domain="transform",
)
def filter_items(
    items_json: str = "",
    conditions_json: str = "",
    combinator: str = "and",
    invert: bool = False,
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    kept = []
    for index, item in enumerate(items):
        env = build_env(item, index, items, state)
        ok = match_conditions(item, conditions_json, combinator, env)
        if bool(invert):
            ok = not ok
        if ok:
            kept.append(item)
    return emit(kept, state=state, to_state=to_state, input_count=len(items), dropped=len(items) - len(kept))


@tool(
    _spec("merge_items", "Merge up to four upstream branches into one stream."),
    description="Merge node. Combines several branches: append, combine_by_key, combine_by_position, or pick_first_non_empty. Point depends_on at every branch you want joined.",
    graph=True, react=False, domain="flow",
)
def merge_items(
    input_1: str = "",
    input_2: str = "",
    input_3: str = "",
    input_4: str = "",
    mode: str = "append",
    key: str = "id",
    from_states: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    """from_states: comma-separated state keys, an alternative to $result wiring."""
    branches: list[list[dict]] = []
    if from_states.strip():
        for name in [n.strip() for n in from_states.split(",") if n.strip()]:
            branches.append(load_items("", state, name))
    for raw in (input_1, input_2, input_3, input_4):
        if raw:
            branches.append(load_items(raw, None, ""))
    branches = [b for b in branches if b]

    mode = (mode or "append").strip().lower()
    merged: list[dict] = []
    if not branches:
        merged = []
    elif mode in {"combine_by_key", "by_key"}:
        index: dict[str, dict] = {}
        order: list[str] = []
        for branch in branches:
            for item in branch:
                ident = str(resolve_path(item, key) or "")
                if not ident:
                    order.append(f"__anon_{len(order)}")
                    index[order[-1]] = dict(item)
                    continue
                if ident in index:
                    index[ident].update(item)
                else:
                    index[ident] = dict(item)
                    order.append(ident)
        merged = [index[i] for i in order]
    elif mode in {"combine_by_position", "by_position", "zip"}:
        longest = max(len(b) for b in branches)
        for position in range(longest):
            row: dict[str, Any] = {}
            for branch in branches:
                if position < len(branch):
                    row.update(branch[position])
            merged.append(row)
    elif mode in {"pick_first_non_empty", "choose_branch"}:
        merged = branches[0]
    else:  # append
        for branch in branches:
            merged.extend(branch)

    return emit(merged, state=state, to_state=to_state, mode=mode, branches=len(branches))


@tool(
    _spec("split_in_batches", "Loop over a large list one batch at a time."),
    description="Split In Batches. Returns {\"done\": false, \"batch\": [...]} until the list is exhausted. Pair it with loop_to on itself plus loop_condition {\"not\":{\"contains\":\"\\\"done\\\": true\"}} to walk big inputs without blowing the context budget.",
    graph=True, react=False, domain="flow",
)
def split_in_batches(
    items_json: str = "",
    batch_size: int = 10,
    from_state: str = "items",
    to_state: str = "batch",
    reset: bool = False,
    assignments_json: str = "{}",
    accumulate_to: str = "",
    *,
    state=None,
) -> str:
    """Walk one slice per pass. The engine only re-runs the loop-carrying
    node each pass — downstream nodes run ONCE after the loop exits — so a
    per-batch transform cannot live in a downstream node. Instead:
      - ``assignments_json`` (set_fields syntax) transforms each batch
        in place before it is stored;
      - ``accumulate_to`` appends every pass's transformed batch to a named
        state key, so a terminal node (e.g. Aggregate) sees ALL batches,
        not just the last one.
    """
    if state is None or not isinstance(getattr(state, "data", None), dict):
        return _dumps({"done": True, "reason": "no_state"})
    pool_key = f"_batch_pool_{to_state}"
    cursor_key = f"_batch_cursor_{to_state}"

    pool = state.data.get(pool_key)
    if reset or not isinstance(pool, list):
        pool = load_items(items_json, state, from_state)
        state.data[pool_key] = pool
        state.data[cursor_key] = 0

    try:
        size = max(1, int(batch_size or 10))
    except (TypeError, ValueError):
        size = 10
    cursor = int(state.data.get(cursor_key) or 0)
    batch = pool[cursor:cursor + size]
    state.data[cursor_key] = cursor + len(batch)
    transformed = batch
    raw_assignments = (assignments_json or "").strip()
    if raw_assignments not in ("", "{}"):
        assignments = _loads(assignments_json, {}) or {}
        if isinstance(assignments, dict) and assignments:
            transformed = apply_assignments(batch, assignments, state=state)
    state.data[to_state] = transformed
    state.data[f"{to_state}_index"] = cursor // size
    done = state.data[cursor_key] >= len(pool)
    accumulated = len(transformed)
    if (accumulate_to or "").strip():
        acc = state.data.get(accumulate_to)
        if not isinstance(acc, list):
            acc = []
            state.data[accumulate_to] = acc
        acc.extend(transformed)
        if len(acc) > FLOW_MAX_ITEMS:
            del acc[FLOW_MAX_ITEMS:]
        accumulated = len(acc)

    log.info("[flow.batch] %d-%d of %d (done=%s)", cursor, cursor + len(batch), len(pool), done)
    return _dumps({
        "done": bool(done),
        "batch_index": cursor // size,
        "batch_size": len(batch),
        "processed": state.data[cursor_key],
        "total": len(pool),
        "state_key": to_state,
        "accumulated": accumulated,
        "accumulate_key": (accumulate_to or "").strip() or None,
        "batch": transformed[:FLOW_INLINE_ITEMS],
    })


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ═════════════════════════════════════════════════════════════════════════════

def apply_assignments(
    items: list[dict[str, Any]],
    assignments: dict[str, Any] | None,
    *,
    keep_only_set: bool = False,
    state: Any = None,
) -> list[dict[str, Any]]:
    """set_fields core, shared with split_in_batches' per-pass transform.

    Values support ``{{ templates }}`` and ``=`` expressions (see
    resolve_value). A lone ``{{ expr }}`` span returns its native value.
    """
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items or [{}]):
        env = build_env(item, index, items, state)
        row = {} if keep_only_set else dict(item)
        for field, raw in (assignments or {}).items():
            row[str(field)] = resolve_value(raw, env)
        out.append(row)
    return out


@tool(
    _spec("set_fields", "Set/Edit Fields: add or overwrite fields on every item."),
    description="Set node. Adds or overwrites fields on each item. Values support {{ templates }} and =expressions, e.g. {\"label\":\"{{ title }} ({{ source }})\",\"double\":\"=price * 2\"}.",
    graph=True, react=False, domain="transform",
)
def set_fields(
    items_json: str = "",
    assignments_json: str = "{}",
    keep_only_set: bool = False,
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    assignments = _loads(assignments_json, {}) or {}
    if not isinstance(assignments, dict):
        assignments = {}
    out = apply_assignments(items, assignments, keep_only_set=bool(keep_only_set), state=state)
    return emit(out, state=state, to_state=to_state, fields=list(assignments.keys()))


@tool(
    _spec("rename_fields", "Rename or drop fields across every item."),
    description="Rename fields with a {\"old\":\"new\"} map, and optionally drop fields listed in drop (comma-separated).",
    graph=True, react=False, domain="transform",
)
def rename_fields(
    items_json: str = "",
    mapping_json: str = "{}",
    drop: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    mapping = _loads(mapping_json, {}) or {}
    drops = {d.strip() for d in (drop or "").split(",") if d.strip()}
    out = []
    for item in items:
        row = {}
        for field, value in item.items():
            if field in drops:
                continue
            row[str(mapping.get(field, field))] = value
        out.append(row)
    return emit(out, state=state, to_state=to_state)


@tool(
    _spec("code_transform", "Code node: run a safe expression over items."),
    description="Code node. Evaluates a restricted expression (no imports, no exec) either per item (each) or once over the whole list (all). Names available: item/json, index, items, state, now.",
    graph=True, react=False, domain="transform",
)
def code_transform(
    expression: str = "item",
    items_json: str = "",
    mode: str = "each",
    to_field: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    mode = (mode or "each").strip().lower()
    try:
        if mode == "all":
            value = evaluate(expression, build_env(items[0] if items else {}, 0, items, state))
            if isinstance(value, list):
                out = [_coerce_item(v) for v in value]
            elif isinstance(value, dict):
                out = [value]
            else:
                out = [{to_field or "result": value}]
        else:
            out = []
            for index, item in enumerate(items):
                value = evaluate(expression, build_env(item, index, items, state))
                if to_field:
                    row = dict(item)
                    row[to_field] = value
                    out.append(row)
                elif isinstance(value, dict):
                    out.append(value)
                else:
                    out.append({"result": value})
    except ExpressionError as exc:
        log.warning("[flow.code] refused expression: %s", exc)
        return _dumps({"ok": False, "error": f"expression rejected: {exc}"})
    return emit(out, state=state, to_state=to_state, mode=mode)


@tool(
    _spec("template_render", "Render a text template per item."),
    description="Template node. Renders a {{ template }} string into a field on each item — the usual way to build post text, email bodies, or LLM evidence.",
    graph=True, react=False, domain="transform",
)
def template_render(
    template: str = "{{ title }}",
    items_json: str = "",
    to_field: str = "text",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    out = []
    for index, item in enumerate(items):
        row = dict(item)
        row[to_field or "text"] = render_template(template, build_env(item, index, items, state))
        out.append(row)
    return emit(out, state=state, to_state=to_state)


@tool(
    _spec("sort_items", "Sort items by a field."),
    description="Sort node. Orders items by a dotted field path, ascending or descending, numeric when possible.",
    graph=True, react=False, domain="transform",
)
def sort_items(
    items_json: str = "",
    field: str = "",
    order: str = "asc",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)

    def _key(item: dict) -> tuple:
        value = resolve_path(item, field) if field else item
        number = _as_float(value)
        if number is not None:
            return (0, number, "")
        return (1, 0.0, str(value or "").casefold())

    reverse = str(order or "asc").strip().lower() in {"desc", "descending", "reverse"}
    return emit(sorted(items, key=_key, reverse=reverse), state=state, to_state=to_state)


@tool(
    _spec("limit_items", "Keep only the first or last N items."),
    description="Limit node. Truncates the stream to max_items, keeping either the first or last ones.",
    graph=True, react=False, domain="transform",
)
def limit_items(
    items_json: str = "",
    max_items: int = 10,
    keep: str = "first",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    try:
        cap = max(1, int(max_items or 10))
    except (TypeError, ValueError):
        cap = 10
    kept = items[-cap:] if str(keep).strip().lower() == "last" else items[:cap]
    return emit(kept, state=state, to_state=to_state, input_count=len(items))


@tool(
    _spec("remove_duplicates", "Drop duplicate items by field or whole-item hash."),
    description="Remove Duplicates node. Keeps the first occurrence per key field (or per whole item when no field is given).",
    graph=True, react=False, domain="transform",
)
def remove_duplicates(
    items_json: str = "",
    field: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    seen: set[str] = set()
    kept = []
    for item in items:
        if field:
            ident = str(resolve_path(item, field) or "").strip().casefold()
        else:
            ident = _dumps(item)
        if ident in seen:
            continue
        seen.add(ident)
        kept.append(item)
    return emit(kept, state=state, to_state=to_state, removed=len(items) - len(kept))


@tool(
    _spec("aggregate_items", "Aggregate/summarize items (count, sum, avg, collect…)."),
    description="Aggregate node. Rolls the stream up: count, sum, avg, min, max, concat, or collect — optionally grouped by a field. The usual last step before a report or a notification.",
    graph=True, react=False, domain="transform",
)
def aggregate_items(
    items_json: str = "",
    mode: str = "count",
    field: str = "",
    group_by: str = "",
    separator: str = "\n",
    to_field: str = "value",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    mode = (mode or "count").strip().lower()

    def _reduce(group: list[dict]) -> Any:
        values = [resolve_path(i, field) for i in group] if field else group
        numbers = [n for n in (_as_float(v) for v in values) if n is not None]
        if mode == "count":
            return len(group)
        if mode == "sum":
            return sum(numbers)
        if mode in {"avg", "mean"}:
            return round(sum(numbers) / len(numbers), 6) if numbers else None
        if mode == "min":
            return min(numbers) if numbers else None
        if mode == "max":
            return max(numbers) if numbers else None
        if mode == "concat":
            return (separator or "\n").join(str(v) for v in values if v not in (None, ""))
        return values  # collect

    if group_by:
        buckets: dict[str, list[dict]] = {}
        for item in items:
            buckets.setdefault(str(resolve_path(item, group_by) or ""), []).append(item)
        out = [{group_by: name, to_field or "value": _reduce(group), "count": len(group)}
               for name, group in buckets.items()]
    else:
        out = [{to_field or "value": _reduce(items), "count": len(items)}]
    return emit(out, state=state, to_state=to_state, mode=mode, input_count=len(items))


@tool(
    _spec("split_out", "Explode a list field into one item per element."),
    description="Split Out node. Turns a list-valued field into separate items (the inverse of Aggregate/collect).",
    graph=True, react=False, domain="transform",
)
def split_out(
    items_json: str = "",
    field: str = "items",
    keep_parent_fields: bool = False,
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    out: list[dict] = []
    for item in items:
        values = resolve_path(item, field)
        if not isinstance(values, (list, tuple)):
            values = [values] if values not in (None, "") else []
        for value in values:
            row = _coerce_item(value)
            if keep_parent_fields:
                parent = {k: v for k, v in item.items() if k != field}
                row = {**parent, **row}
            out.append(row)
    return emit(out, state=state, to_state=to_state, input_count=len(items))


@tool(
    _spec("items_to_text", "Flatten items into one plain-text block for LLM nodes."),
    description="Bridge node. Renders every item through a template and joins them into a single string — feed this into synthesize_report, write_report, or any tool that wants prose rather than items.",
    graph=True, react=False, domain="transform",
)
def items_to_text(
    items_json: str = "",
    template: str = "{{ title }}\n{{ text }}",
    separator: str = "\n\n---\n\n",
    max_items: int = 25,
    max_chars: int = 8000,
    from_state: str = "",
    to_state: str = "text",
    *,
    state=None,
) -> str:
    items = load_items(items_json, state, from_state)
    try:
        cap = max(1, int(max_items or 25))
    except (TypeError, ValueError):
        cap = 25
    blocks = []
    for index, item in enumerate(items[:cap]):
        rendered = render_template(template, build_env(item, index, items, state))
        text = rendered if isinstance(rendered, str) else _dumps(rendered)
        text = text.strip()
        if text:
            blocks.append(text)
    joined = (separator or "\n\n").join(blocks)[: max(200, int(max_chars or 8000))]
    if to_state and state is not None and isinstance(getattr(state, "data", None), dict):
        state.data[to_state] = joined
    return joined


# ═════════════════════════════════════════════════════════════════════════════
# I/O
# ═════════════════════════════════════════════════════════════════════════════

@tool(
    _spec("http_request", "HTTP Request node: call any public JSON/text API."),
    description="HTTP Request node. GET/POST/PUT/PATCH/DELETE against a public URL, with SSRF protection, a byte cap and a timeout. JSON responses become items; response_path drills into a nested list.",
    graph=True, react=False, domain="io",
)
def http_request(
    url: str = "",
    method: str = "GET",
    query_json: str = "{}",
    headers_json: str = "{}",
    body_json: str = "",
    response_path: str = "",
    timeout: float = FLOW_HTTP_TIMEOUT,
    max_items: int = 50,
    items_json: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
) -> str:
    """URL/query/body support {{ templates }} against the first incoming item."""
    incoming = load_items(items_json, state, from_state)
    env = build_env(incoming[0] if incoming else {}, 0, incoming, state)

    resolved_url = str(render_template(url, env) or "").strip()
    if not resolved_url.startswith(("http://", "https://")):
        return _dumps({"ok": False, "error": "url must start with http:// or https://"})

    try:
        import requests
        from urllib.parse import urlparse
        from agentic.toolkit.ingest import _check_host_ssrf, FETCH_URL_USER_AGENT
    except Exception as exc:  # pragma: no cover - requests is a core dep
        return _dumps({"ok": False, "error": f"http dependencies unavailable: {exc}"})

    host = urlparse(resolved_url).hostname
    if not host or _check_host_ssrf(host):
        return _dumps({"ok": False, "error": "URL host is not allowed"})

    params = resolve_value(_loads(query_json, {}) or {}, env)
    headers = resolve_value(_loads(headers_json, {}) or {}, env)
    if not isinstance(headers, dict):
        headers = {}
    headers.setdefault("User-Agent", FETCH_URL_USER_AGENT)
    body = resolve_value(_loads(body_json, None), env) if body_json else None

    verb = (method or "GET").strip().upper()
    if verb not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
        return _dumps({"ok": False, "error": f"unsupported method: {verb}"})

    try:
        seconds = max(1.0, min(float(timeout or FLOW_HTTP_TIMEOUT), 60.0))
    except (TypeError, ValueError):
        seconds = FLOW_HTTP_TIMEOUT

    try:
        response = requests.request(
            verb, resolved_url,
            params=params if isinstance(params, dict) else None,
            headers=headers,
            json=body if isinstance(body, (dict, list)) else None,
            data=body if isinstance(body, str) else None,
            timeout=seconds,
            stream=True,
        )
        raw = response.raw.read(FLOW_HTTP_MAX_BYTES + 1, decode_content=True)
        if len(raw) > FLOW_HTTP_MAX_BYTES:
            return _dumps({"ok": False, "error": "response exceeded FLOW_HTTP_MAX_BYTES"})
        text = raw.decode(response.encoding or "utf-8", errors="replace")
        status = response.status_code
    except Exception as exc:
        log.warning("[flow.http] %s %s failed: %s", verb, resolved_url[:120], exc)
        return _dumps({"ok": False, "error": str(exc)[:300], "url": resolved_url[:200]})
    finally:
        try:
            response.close()  # type: ignore[has-type]
        except Exception:
            pass

    payload = _loads(text, None)
    if payload is None:
        items = [{"status": status, "url": resolved_url, "text": text[:FLOW_INLINE_CHARS * 2]}]
    else:
        target = resolve_path(payload, response_path) if response_path else payload
        if isinstance(target, list):
            items = [_coerce_item(v) for v in target]
        elif isinstance(target, dict):
            for key in _ITEM_LIST_KEYS:
                if isinstance(target.get(key), list):
                    items = [_coerce_item(v) for v in target[key]]
                    break
            else:
                items = [target]
        else:
            items = [{"value": target}]

    try:
        cap = max(1, int(max_items or 50))
    except (TypeError, ValueError):
        cap = 50
    ok = 200 <= status < 300
    return emit(items[:cap], state=state, to_state=to_state, ok=ok, status=status, url=resolved_url[:200])


# ═════════════════════════════════════════════════════════════════════════════
# STUDIO PHASE 2 — triggers, agentic nodes, social/email/messaging, utilities
# ═════════════════════════════════════════════════════════════════════════════
#
# n8n-parity palette: chat/webhook/schedule triggers, a composite AI Agent
# with chat-model + memory sub-nodes, email/social/messaging actions, and
# everyday utilities. Every node speaks the standard items envelope
# (load_items/emit). Nodes without a real backend return a graceful
# {"ok": False, "error": "<service> is not connected"} result — a run never
# crashes because a service is missing.

def _svc_result(service: str, hint: str = "") -> str:
    """Graceful 'not connected' result — never raises, never crashes a run."""
    return _dumps({
        "ok": False,
        "service": service,
        "error": f"{service} is not connected",
        "hint": hint or f"Connect {service} in Aiko's settings, then re-run this workflow.",
    })


# ── triggers ───────────────────────────────────────────────────────────────

@tool(
    _spec("trigger_chat", "Chat message trigger: start the workflow from a chat message."),
    description="Entry node for conversational workflows. Emits one item carrying the incoming chat message — n8n's 'Chat Trigger' equivalent.",
    graph=True, react=False, domain="trigger",
)
def trigger_chat(
    message: str = "$prompt",
    from_user: str = "user",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Emit one chat-message item."""
    item = {"kind": "chat", "message": message or "", "from": from_user or "user"}
    return emit([item], state=state, to_state=to_state, source="chat")


@tool(
    _spec("trigger_webhook", "Webhook trigger: start the workflow from an HTTP call."),
    description="Entry node for webhook-driven workflows. Emits the configured test payload (or one empty item) so downstream nodes have data to shape.",
    graph=True, react=False, domain="trigger",
)
def trigger_webhook(
    path: str = "/webhook/studio",
    method: str = "POST",
    payload_json: str = "[{}]",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Emit the webhook test payload as items."""
    items = load_items(payload_json, state, "", default=[{}])
    if not items:
        items = [{}]
    return emit(items, state=state, to_state=to_state, source="webhook",
                path=path, method=(method or "POST").upper())


@tool(
    _spec("trigger_schedule", "Schedule trigger: start the workflow on a cron timetable."),
    description="Entry node for scheduled workflows. Emits one item stamped with the current time; the cron expression documents the intended timetable.",
    graph=True, react=False, domain="trigger",
)
def trigger_schedule(
    cron: str = "0 9 * * *",
    timezone: str = "America/Los_Angeles",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Emit one schedule-tick item."""
    item = {
        "kind": "schedule",
        "cron": cron or "",
        "timezone": timezone or "",
        "at": datetime.now(dt_timezone.utc).isoformat(),
    }
    return emit([item], state=state, to_state=to_state, source="schedule")


# ── agentic ────────────────────────────────────────────────────────────────

def _extract_chat_text(resp: Any) -> str:
    """Best-effort extraction of assistant text from an OpenAI-style response."""
    try:
        choices = getattr(resp, "choices", None) or (resp.get("choices") if isinstance(resp, dict) else None) or []
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
        if message is not None:
            content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
            if content:
                return str(content)
        text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
        return str(text or "")
    except Exception:
        return ""


@tool(
    _spec("ai_agent", "AI Agent: run the chat model with tools and memory (n8n Tools Agent)."),
    description="Composite agent node. Takes a prompt plus an optional chat-model config ($result:<chat_model node>) and a memory key, calls the LLM, and returns the response. Chat model, memory and tools are attached as sub-nodes on the canvas.",
    graph=True, react=False, domain="ai",
)
def ai_agent(
    prompt: str = "$prompt",
    system_prompt: str = "",
    chat_model_json: str = "",
    tools_json: str = "[]",
    memory_key: str = "",
    memory_window: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    items_json: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    client=None,
    model=None,
    state=None,
    **_kwargs,
) -> str:
    """Run one agent turn: prompt + upstream items (+ memory) through the chat model."""
    items = load_items(items_json, state, from_state)
    model_cfg = _loads(chat_model_json, {}) or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    # $result:<chat_model> arrives as the emit envelope {"items": [...]}
    cfg_items = model_cfg.get("items") if isinstance(model_cfg.get("items"), list) else None
    if cfg_items and isinstance(cfg_items[0], dict):
        model_cfg = cfg_items[0]
    model_name = str(model_cfg.get("model") or model or "").strip()
    try:
        temp = float(model_cfg.get("temperature", temperature))
    except (TypeError, ValueError):
        temp = 0.7
    try:
        mtok = max(1, int(model_cfg.get("max_tokens", max_tokens)))
    except (TypeError, ValueError):
        mtok = 1024
    tools = _loads(tools_json, []) or []
    if not isinstance(tools, list):
        tools = []

    history: list[dict[str, Any]] = []
    if memory_key and state is not None and isinstance(getattr(state, "data", None), dict):
        stored = state.data.get(memory_key) or []
        if isinstance(stored, list):
            try:
                w = max(1, int(memory_window or 10))
            except (TypeError, ValueError):
                w = 10
            history = [m for m in stored[-w:] if isinstance(m, dict) and m.get("role")]

    if client is None or not model_name:
        return emit([], state=state, to_state=to_state, ok=False,
                    error="no LLM client/model available in this run",
                    hint="Attach a chat_model sub-node (or run where the graph runner injects a client). The studio dry-run has no LLM client.")

    item_text = _dumps(items[:5]) if items else ""
    if len(item_text) > 3000:
        item_text = item_text[:3000] + "…"
    user_text = (prompt or "").strip()
    if item_text:
        user_text += f"\n\nInput items:\n{item_text}"
    if tools:
        user_text += f"\n\nAvailable tools: {', '.join(str(t) for t in tools)}"

    messages: list[dict[str, Any]] = []
    messages.append({"role": "system",
                     "content": system_prompt or "You are Aiko, a helpful on-device assistant. Answer concisely."})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        from agentic.toolkit.common import chat_completions_create
        resp = chat_completions_create(
            client, model=model_name, messages=messages,
            temperature=temp, max_tokens=mtok, stream=False,
        )
    except Exception as exc:
        return emit([], state=state, to_state=to_state, ok=False,
                    error=f"LLM call failed: {exc}")
    text = _extract_chat_text(resp)
    if not text:
        return emit([], state=state, to_state=to_state, ok=False,
                    error="LLM returned no text")

    if memory_key and state is not None and isinstance(getattr(state, "data", None), dict):
        buf = list(state.data.get(memory_key) or [])
        buf.append({"role": "user", "content": user_text[:2000]})
        buf.append({"role": "assistant", "content": text[:2000]})
        try:
            w = max(1, int(memory_window or 10)) * 2
        except (TypeError, ValueError):
            w = 20
        state.data[memory_key] = buf[-w:]

    return emit([{"response": text, "model": model_name,
                  "tools": tools, "history_used": len(history)}],
                state=state, to_state=to_state, source="ai_agent")


@tool(
    _spec("chat_model", "Chat Model: configuration sub-node for the AI Agent."),
    description="Sub-node for the AI Agent: which chat model to use, temperature and max tokens. Emits its config so the agent can read it via $result:<node>.",
    graph=True, react=False, domain="ai",
)
def chat_model(
    model: str = "ministral-3b",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Emit the chat-model config as one item."""
    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        temp = 0.7
    try:
        mtok = max(1, int(max_tokens))
    except (TypeError, ValueError):
        mtok = 1024
    cfg = {"kind": "chat_model", "model": model or "ministral-3b",
           "temperature": temp, "max_tokens": mtok}
    return emit([cfg], state=state, to_state=to_state, source="chat_model")


@tool(
    _spec("memory_buffer", "Memory: sliding-window conversation buffer sub-node for the AI Agent."),
    description="Sub-node for the AI Agent: keeps the last N items under a GraphState key so the agent sees recent conversation without re-reading everything.",
    graph=True, react=False, domain="ai",
)
def memory_buffer(
    memory_key: str = "agent_memory",
    window_size: int = 10,
    mode: str = "read_write",
    items_json: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Maintain a bounded item buffer under a state key."""
    items = load_items(items_json, state, from_state)
    try:
        w = max(1, int(window_size or 10))
    except (TypeError, ValueError):
        w = 10
    buf: list[dict[str, Any]] = []
    if state is not None and isinstance(getattr(state, "data", None), dict):
        stored = state.data.get(memory_key or "agent_memory") or []
        buf = list(stored) if isinstance(stored, list) else []
        if mode in ("write", "read_write") and items:
            buf.extend(items)
            buf = buf[-w:]
            state.data[memory_key or "agent_memory"] = buf
    window = buf[-w:] if buf else items
    return emit(window, state=state, to_state=to_state,
                memory_key=memory_key or "agent_memory", mode=mode, source="memory_buffer")


# ── email ──────────────────────────────────────────────────────────────────

@tool(
    _spec("email_check", "Email: check inbox for new messages."),
    description="Poll the owner's inbox via Aiko's email bridge (ProtonMail). Returns one item per new message with id, from, subject and prompt text.",
    graph=True, react=False, domain="email",
)
def email_check(
    max_results: int = 5,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Check the inbox through the owner-email bridge."""
    try:
        from agentic.workflows.owner_email.toolset import check_owner_email
    except Exception as exc:
        return _svc_result("email", f"email bridge unavailable: {exc}")
    try:
        raw = check_owner_email(max_results=max(1, int(max_results or 5)), state=state)
    except Exception as exc:
        return _dumps({"ok": False, "error": f"email check failed: {exc}"})
    parsed = _loads(raw, {}) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    messages = parsed.get("messages") if isinstance(parsed.get("messages"), list) else []
    items = [_coerce_item(m) for m in messages]
    ok = bool(parsed.get("ok")) and not parsed.get("error")
    if parsed.get("error") and not items:
        return _dumps({"ok": False, "error": parsed.get("error"),
                       "hint": "Set AIKO_EMAIL and configure the email MCP adapter."})
    return emit(items, state=state, to_state=to_state, ok=ok, source="email")


@tool(
    _spec("email_reply", "Email: reply to inbox messages via the LLM."),
    description="Generate replies with Aiko's LLM and send them back through the email bridge. Takes the email_check batch ($result:<node>) or reads it from state.",
    graph=True, react=False, domain="email",
)
def email_reply(
    report_json: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Reply through the owner-email bridge."""
    try:
        from agentic.workflows.owner_email.toolset import reply_owner_email
    except Exception as exc:
        return _svc_result("email", f"email bridge unavailable: {exc}")
    try:
        raw = reply_owner_email(report_json=report_json or "", state=state)
    except Exception as exc:
        return _dumps({"ok": False, "error": f"email reply failed: {exc}"})
    parsed = _loads(raw, {}) or {}
    if not isinstance(parsed, dict):
        parsed = {"raw": raw}
    items = [_coerce_item(parsed)]
    return emit(items, state=state, to_state=to_state,
                ok=bool(parsed.get("ok", True)), source="email")


@tool(
    _spec("email_draft", "Email: compose a draft (saved, not sent)."),
    description="Compose an email draft and park it under a GraphState key for review. Nothing is sent — a later email_send step (or a human) delivers it.",
    graph=True, react=False, domain="email",
)
def email_draft(
    subject: str = "",
    body: str = "$prompt",
    to: str = "",
    drafts_key: str = "email_drafts",
    items_json: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Park a draft under a state key instead of sending it."""
    items = load_items(items_json, state, from_state)
    draft = {
        "kind": "email_draft",
        "to": to or "",
        "subject": subject or "",
        "body": body or "",
        "sent": False,
        "created_at": datetime.now(dt_timezone.utc).isoformat(),
    }
    if items and not (subject or body):
        first = items[0] if isinstance(items[0], dict) else {}
        draft["subject"] = str(first.get("subject") or "Re: workflow item")
        draft["body"] = _dumps(first)[:4000]
    if state is not None and isinstance(getattr(state, "data", None), dict):
        key = drafts_key or "email_drafts"
        drafts = list(state.data.get(key) or [])
        drafts.append(draft)
        state.data[key] = drafts
        draft["draft_index"] = len(drafts) - 1
    draft["note"] = "Draft saved — not sent. No mail-provider send is configured."
    return emit([draft], state=state, to_state=to_state, source="email")


# ── social ─────────────────────────────────────────────────────────────────

@tool(
    _spec("threads_post", "Threads: post text (and optional image)."),
    description="Post to Threads through Aiko's social MCP bridge. Returns the provider result — check ok before assuming it went out.",
    graph=True, react=False, domain="social",
)
def threads_post(
    text: str = "$prompt",
    image_path: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Post to Threads via the social MCP bridge."""
    try:
        from agentic.toolkit.social import post_to_social
    except Exception as exc:
        return _svc_result("threads", f"social toolkit unavailable: {exc}")
    try:
        result = post_to_social(text=text or "", services="threads",
                                image_path=image_path or None)
    except Exception as exc:
        return _dumps({"ok": False, "service": "threads", "error": str(exc)})
    if not isinstance(result, dict):
        result = {"result": result}
    items = [_coerce_item(result)]
    return emit(items, state=state, to_state=to_state,
                ok=bool(result.get("ok", True)), source="threads")


@tool(
    _spec("instagram_post", "Instagram: post text/image (not connected)."),
    description="Post to Instagram. No Instagram backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def instagram_post(
    caption: str = "$prompt",
    image_path: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no Instagram backend exists yet."""
    return _svc_result("instagram", "No Instagram adapter is configured. Wire one into agentic/toolkit/social.py, then re-run.")


@tool(
    _spec("messenger_send", "Messenger: send a message (not connected)."),
    description="Send a Facebook Messenger message. No Messenger backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def messenger_send(
    recipient: str = "",
    message: str = "$prompt",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no Messenger backend exists yet."""
    return _svc_result("messenger", "No Messenger adapter is configured.")


@tool(
    _spec("social_read", "Social: read recent mentions/posts (not connected)."),
    description="Read recent social mentions or posts. No social read backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def social_read(
    services: str = "threads",
    max_results: int = 10,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no social read backend exists yet."""
    return _svc_result("social_read", "No social read adapter is configured. Use email_check for inbox triage in the meantime.")


# ── messaging ──────────────────────────────────────────────────────────────

@tool(
    _spec("whatsapp_send", "WhatsApp: send a message (not connected)."),
    description="Send a WhatsApp message. No WhatsApp backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def whatsapp_send(
    recipient: str = "",
    message: str = "$prompt",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no WhatsApp backend exists yet."""
    return _svc_result("whatsapp", "No WhatsApp adapter is configured.")


@tool(
    _spec("telegram_send", "Telegram: send a message (not connected)."),
    description="Send a Telegram message. No Telegram backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def telegram_send(
    chat_id: str = "",
    message: str = "$prompt",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no Telegram backend exists yet."""
    return _svc_result("telegram", "No Telegram bot adapter is configured.")


# ── utilities ──────────────────────────────────────────────────────────────

@tool(
    _spec("calendar_create", "Calendar: create an event (not connected)."),
    description="Create a Google Calendar event. No calendar backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="scheduling",
)
def calendar_create(
    title: str = "$prompt",
    start: str = "",
    end: str = "",
    description: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no calendar backend exists yet."""
    return _svc_result("google_calendar", "No Google Calendar adapter is configured.")


@tool(
    _spec("calendar_list", "Calendar: list upcoming events (not connected)."),
    description="List upcoming Google Calendar events. No calendar backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="scheduling",
)
def calendar_list(
    max_results: int = 10,
    time_min: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no calendar backend exists yet."""
    return _svc_result("google_calendar", "No Google Calendar adapter is configured.")


@tool(
    _spec("rss_read", "RSS: read headlines from a feed."),
    description="Fetch an RSS or Atom feed with the standard library (no extra deps) and emit one item per entry: title, link, summary, published.",
    graph=True, react=False, domain="research",
)
def rss_read(
    url: str = "https://feeds.bbci.co.uk/news/rss.xml",
    max_items: int = 20,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Read an RSS/Atom feed using urllib + ElementTree."""
    import urllib.request
    import xml.etree.ElementTree as ET

    target = (url or "").strip()
    if not target or not target.startswith(("http://", "https://")):
        return emit([], state=state, to_state=to_state, ok=False,
                    error="rss_read needs an http(s) feed URL")
    try:
        cap = max(1, min(int(max_items or 20), 100))
    except (TypeError, ValueError):
        cap = 20
    try:
        req = urllib.request.Request(target, headers={"User-Agent": "Aiko-chan/1.0 (RSS reader)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(2_000_000)
        root = ET.fromstring(raw)
    except Exception as exc:
        return emit([], state=state, to_state=to_state, ok=False,
                    error=f"feed fetch failed: {exc}")

    def _text(el: Any, names: list[str]) -> str:
        for name in names:
            child = el.find(name)
            if child is not None and child.text:
                return str(child.text).strip()
        return ""

    items: list[dict[str, Any]] = []
    # RSS 2.0
    for entry in root.findall("./channel/item")[:cap]:
        link = _text(entry, ["link"])
        if not link:
            guid = entry.find("guid")
            if guid is not None and guid.text and str(guid.text).startswith("http"):
                link = str(guid.text).strip()
        items.append({
            "title": _text(entry, ["title"]),
            "link": link,
            "summary": _text(entry, ["description"]),
            "published": _text(entry, ["pubDate"]),
        })
    # Atom
    if not items:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("./a:entry", ns)[:cap]:
            link_el = entry.find("a:link", ns)
            link = (link_el.get("href") or "") if link_el is not None else ""
            summary = entry.find("a:summary", ns)
            items.append({
                "title": (entry.findtext("a:title", default="", namespaces=ns) or "").strip(),
                "link": link.strip(),
                "summary": ((summary.text if summary is not None else "") or "").strip(),
                "published": (entry.findtext("a:updated", default="", namespaces=ns) or "").strip(),
            })
    items = [i for i in items if i.get("title") or i.get("link")][:cap]
    return emit(items, state=state, to_state=to_state, ok=True,
                source="rss", feed=target[:200], count=len(items))


@tool(
    _spec("file_write", "File: write or append a workspace file."),
    description="Write text to a file under Aiko's workspace root. Paths are jailed to the workspace — traversal outside is rejected.",
    graph=True, react=False, domain="io",
)
def file_write(
    path: str = "notes/draft.txt",
    content: str = "$prompt",
    mode: str = "write",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Write/append a workspace-jailed file."""
    try:
        from agentic.toolkit.common import safe_path
        target = safe_path(path or "notes/draft.txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content if isinstance(content, str) else _dumps(content)
        if (mode or "write") == "append":
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(data if data.endswith("\n") else data + "\n")
        else:
            target.write_text(data, encoding="utf-8")
    except Exception as exc:
        return emit([], state=state, to_state=to_state, ok=False,
                    error=f"file_write failed: {exc}")
    return emit([{"path": str(target), "mode": mode or "write",
                  "chars": len(data)}],
                state=state, to_state=to_state, source="file_write")


@tool(
    _spec("notify_user", "Notify: surface a message to the owner."),
    description="Emit a notification item for the owner (title + message). Best-effort delivery via a registered mail tool; the item itself is always emitted so downstream nodes can act on it.",
    graph=True, react=False, domain="everyday",
)
def notify_user(
    title: str = "Aiko notification",
    message: str = "$prompt",
    channel: str = "app",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Emit a notification; best-effort email delivery, never raises."""
    note = {"kind": "notification", "title": title or "Aiko notification",
            "message": message or "", "channel": channel or "app",
            "delivered": False,
            "at": datetime.now(dt_timezone.utc).isoformat()}
    if (channel or "app") == "email":
        try:
            from agentic.workflows.common.notify import notify_email
            result = notify_email(note["title"], note["message"]) or {}
            note["delivered"] = bool(result.get("ok"))
            if result.get("error"):
                note["delivery_error"] = str(result["error"])[:200]
        except Exception as exc:
            note["delivery_error"] = str(exc)[:200]
    return emit([note], state=state, to_state=to_state, source="notify")


__all__ = [
    # helpers other modules may reuse
    "load_items", "emit", "resolve_path", "evaluate", "render_template",
    "resolve_value", "compare", "match_conditions", "build_env", "ExpressionError",
    # nodes
    "trigger_manual", "no_op", "sticky_note", "wait_delay", "stop_and_error",
    "if_condition", "switch_route", "filter_items", "merge_items", "split_in_batches",
    "set_fields", "rename_fields", "code_transform", "template_render", "sort_items",
    "limit_items", "remove_duplicates", "aggregate_items", "split_out", "items_to_text",
    "http_request",
    # studio phase 2
    "trigger_chat", "trigger_webhook", "trigger_schedule",
    "ai_agent", "chat_model", "memory_buffer",
    "email_check", "email_reply", "email_draft",
    "threads_post", "instagram_post", "messenger_send", "social_read",
    "instagram_read", "threads_read", "messenger_read",
    "gmail_search", "gmail_send", "gmail_draft",
    "tool_router", "reminder", "file_read",
    "whatsapp_send", "telegram_send",
    "calendar_create", "calendar_list",
    "rss_read", "file_write", "notify_user",
]


# ── gmail ──────────────────────────────────────────────────────────────────

def _gmail_adapter():
    """Return a Gmail adapter module if one is installed, else None."""
    try:
        from agentic.integrations import gmail as adapter  # type: ignore
        return adapter
    except Exception:
        return None


@tool(
    _spec("gmail_search", "Gmail: search messages (not connected)."),
    description="Search Gmail messages. No Gmail adapter is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="email",
)
def gmail_search(
    query: str = "is:unread",
    max_results: int = 5,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Search Gmail; graceful when no adapter is configured."""
    adapter = _gmail_adapter()
    if adapter is None or not hasattr(adapter, "search"):
        return _svc_result("Gmail", "No Gmail adapter is configured. Connect Gmail in Aiko's settings, then re-run.")
    try:
        messages = adapter.search(query=query or "is:unread",
                                  max_results=max(1, int(max_results or 5)))
    except Exception as exc:
        return _dumps({"ok": False, "error": f"gmail search failed: {exc}"})
    items = [_coerce_item(m) for m in (messages or [])]
    return emit(items, state=state, to_state=to_state, ok=True, source="gmail")


@tool(
    _spec("gmail_send", "Gmail: send a message (not connected)."),
    description="Send a Gmail message. No Gmail adapter is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="email",
)
def gmail_send(
    to: str = "",
    subject: str = "",
    body: str = "$prompt",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Send via Gmail; graceful when no adapter is configured."""
    adapter = _gmail_adapter()
    if adapter is None or not hasattr(adapter, "send"):
        return _svc_result("Gmail", "No Gmail adapter is configured. Connect Gmail in Aiko's settings, then re-run.")
    try:
        result = adapter.send(to=to or "", subject=subject or "", body=body or "")
    except Exception as exc:
        return _dumps({"ok": False, "error": f"gmail send failed: {exc}"})
    return emit([_coerce_item(result if isinstance(result, dict) else {"result": result})],
                state=state, to_state=to_state, ok=True, source="gmail")


@tool(
    _spec("gmail_draft", "Gmail: compose a draft (saved locally, not sent)."),
    description="Compose a Gmail draft and park it under a GraphState key for review. Nothing is sent — a later gmail_send step (or a human) delivers it.",
    graph=True, react=False, domain="email",
)
def gmail_draft(
    to: str = "",
    subject: str = "",
    body: str = "$prompt",
    drafts_key: str = "gmail_drafts",
    items_json: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Park a Gmail draft under a state key instead of sending it."""
    items = load_items(items_json, state, from_state)
    draft = {
        "kind": "gmail_draft",
        "to": to or "",
        "subject": subject or "",
        "body": body or "",
        "sent": False,
        "created_at": datetime.now(dt_timezone.utc).isoformat(),
    }
    if items and not (subject or body):
        first = items[0] if isinstance(items[0], dict) else {}
        draft["subject"] = str(first.get("subject") or "Re: workflow item")
        draft["body"] = _dumps(first)[:4000]
    if state is not None and isinstance(getattr(state, "data", None), dict):
        key = drafts_key or "gmail_drafts"
        drafts = list(state.data.get(key) or [])
        drafts.append(draft)
        state.data[key] = drafts
        draft["draft_index"] = len(drafts) - 1
    draft["note"] = "Draft saved — not sent. Connect Gmail to enable gmail_send."
    return emit([draft], state=state, to_state=to_state, source="gmail")


# ── social reads ───────────────────────────────────────────────────────────

@tool(
    _spec("instagram_read", "Instagram: read recent posts/mentions (not connected)."),
    description="Read recent Instagram posts or mentions. No Instagram read backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def instagram_read(
    max_results: int = 10,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no Instagram read backend exists yet."""
    return _svc_result("instagram", "No Instagram read adapter is configured.")


@tool(
    _spec("threads_read", "Threads: read recent posts/mentions (not connected)."),
    description="Read recent Threads posts or mentions. No Threads read backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def threads_read(
    max_results: int = 10,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no Threads read backend exists yet."""
    return _svc_result("threads", "No Threads read adapter is configured.")


@tool(
    _spec("messenger_read", "Messenger: read recent messages (not connected)."),
    description="Read recent Messenger messages. No Messenger backend is configured — the node returns a graceful not-connected result instead of failing the run.",
    graph=True, react=False, domain="social",
)
def messenger_read(
    max_results: int = 10,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Graceful stub: no Messenger backend exists yet."""
    return _svc_result("messenger", "No Messenger adapter is configured.")


# ── router ─────────────────────────────────────────────────────────────────

@tool(
    _spec("tool_router", "Router: send items down the branch whose keyword matches."),
    description="Route items to a named branch. Each route pairs a keyword (or /regex/) with a branch name; the first matching route wins, otherwise the default branch is taken. Emits one item naming the chosen route so downstream if/switch nodes can branch on it.",
    graph=True, react=False, domain="flow",
)
def tool_router(
    input_text: str = "$prompt",
    routes_json: str = '[{"name": "default", "match": ""}]',
    default_route: str = "default",
    case_sensitive: bool = False,
    items_json: str = "",
    from_state: str = "",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Pick a route by keyword/regex match; never raises."""
    import re as _re
    items = load_items(items_json, state, from_state)
    text = input_text if isinstance(input_text, str) else _dumps(input_text)
    if not text and items:
        first = items[0] if isinstance(items[0], dict) else {}
        text = str(first.get("text") or first.get("message") or first.get("body") or "")
    routes = _loads(routes_json, []) or []
    if not isinstance(routes, list):
        routes = []
    haystack = text if case_sensitive else text.lower()
    chosen = default_route or "default"
    matched_rule = None
    for route in routes:
        if not isinstance(route, dict):
            continue
        name = str(route.get("name") or "").strip() or "default"
        match = str(route.get("match") or "")
        if not match:
            continue
        needle = match if case_sensitive else match.lower()
        hit = False
        # /pattern/ (or /pattern/flags) is a regex; anything else is a plain
        # keyword substring. The old rule required two inner slashes, which
        # silently treated "/urgent/" as the literal text "/urgent/".
        if len(match) >= 2 and match.startswith("/") and "/" in match[1:]:
            pattern = match[1:match.rfind("/")]
            if pattern:
                try:
                    hit = _re.search(pattern, text,
                                     0 if case_sensitive else _re.IGNORECASE) is not None
                except _re.error:
                    hit = needle.strip("/") in haystack
        else:
            hit = needle in haystack
        if hit:
            chosen = name
            matched_rule = match
            break
    return emit([{"route": chosen, "matched": matched_rule,
                  "input": text[:500], "item_count": len(items)}],
                state=state, to_state=to_state, source="tool_router")


# ── reminders ──────────────────────────────────────────────────────────────

@tool(
    _spec("reminder", "Reminders: schedule a local reminder."),
    description="Schedule a local reminder (title + message at a time of day, optionally repeating). Uses Aiko's local scheduler while she is running.",
    graph=True, react=False, domain="scheduling",
)
def reminder(
    title: str = "$prompt",
    message: str = "",
    time_of_day: str = "09:00",
    repeat: str = "once",
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Schedule a local reminder via the system scheduler; never raises."""
    try:
        from system.schedule import schedule_reminder_record
        record = schedule_reminder_record(
            title or "Reminder", message or "",
            time_of_day or "09:00", repeat or "once")
        try:
            from system.schedule import notify_scheduler_new_job
            notify_scheduler_new_job()
        except Exception:
            pass
        item = {"kind": "reminder", "ok": True,
                "record": record if isinstance(record, dict) else {"result": record}}
    except Exception as exc:
        item = {"kind": "reminder", "ok": False, "error": f"reminder failed: {exc}"}
    return emit([item], state=state, to_state=to_state, source="reminder")


# ── file read ──────────────────────────────────────────────────────────────

@tool(
    _spec("file_read", "File: read a workspace file."),
    description="Read text from a file under Aiko's workspace root. Paths are jailed to the workspace — traversal outside is rejected. Pairs with file_write.",
    graph=True, react=False, domain="io",
)
def file_read(
    path: str = "notes/draft.txt",
    max_chars: int = 20000,
    to_state: str = "items",
    *,
    state=None,
    **_kwargs,
) -> str:
    """Read a workspace-jailed file; never raises."""
    try:
        from agentic.toolkit.common import safe_path
        target = safe_path(path or "notes/draft.txt")
        if not target.is_file():
            return emit([], state=state, to_state=to_state, ok=False,
                        error=f"file not found: {path}")
        try:
            cap = max(1, int(max_chars or 20000))
        except (TypeError, ValueError):
            cap = 20000
        content = target.read_text(encoding="utf-8")[:cap]
    except Exception as exc:
        return emit([], state=state, to_state=to_state, ok=False,
                    error=f"file_read failed: {exc}")
    return emit([{"path": str(target), "content": content,
                  "chars": len(content)}],
                state=state, to_state=to_state, source="file_read")
