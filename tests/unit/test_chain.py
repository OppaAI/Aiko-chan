"""Unit tests for agentic/workflows/common/chain.py — the mechanical
tool-chain interpreter behind scheduled jobs with action="chain"."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic.workflows.common import chain as c


def test_validate_accepts_good_chain():
    assert c.validate_chain([
        {"tool": "check_aurora", "as": "report"},
        {"if": {"field": "report.kp_index", "op": ">=", "value": 4}, "then": [
            {"tool": "telegram_send", "args": {"message": "Kp {report.kp_index}"}},
        ]},
    ]) == []


def test_validate_rejects_bad_chains():
    assert c.validate_chain([])
    assert c.validate_chain([{"bogus": 1}])
    assert c.validate_chain([{"tool": ""}])
    assert c.validate_chain([{"if": {"field": "a", "op": "??", "value": 1}, "then": []}])
    assert c.validate_chain([{"if": {"field": "", "op": "==", "value": 1}, "then": []}])


def test_eval_condition_operators():
    b = {"report": {"kp_index": 5.0, "tags": ["a", "b"], "summary": "big storm"}}
    assert c.eval_condition({"field": "report.kp_index", "op": ">=", "value": 4}, b)
    assert not c.eval_condition({"field": "report.kp_index", "op": ">", "value": 5}, b)
    assert c.eval_condition({"field": "report.kp_index", "op": "==", "value": 5.0}, b)
    assert c.eval_condition({"field": "report.summary", "op": "contains", "value": "storm"}, b)
    assert c.eval_condition({"field": "report.kp_index", "op": "in", "value": [5.0, 6.0]}, b)
    assert not c.eval_condition({"field": "report.missing", "op": "==", "value": 1}, b)


def test_render_template():
    b = {"report": {"kp_index": 5.0}}
    assert c.render_template("Kp {report.kp_index}!", b) == "Kp 5.0!"
    # unknown paths stay as-is (visible, not silently blanked)
    assert c.render_template("{report.nope}", b) == "{report.nope}"


def _spec():
    return [
        {"tool": "check_aurora", "as": "report"},
        {"if": {"field": "report.kp_index", "op": ">=", "value": 4}, "then": [
            {"tool": "telegram_send",
             "args": {"message": "Kp {report.kp_index}: {report.summary}"}},
        ]},
    ]


def test_run_chain_true_branch():
    calls = []

    def fake_invoke(name, args):
        calls.append((name, args))
        if name == "check_aurora":
            return '{"kp_index": 5.0, "summary": "big storm"}'
        return {"ok": True}

    s = c.run_chain(_spec(), fake_invoke)
    assert s["ok"] and "telegram_send" in s["ran"]
    assert calls[1][1]["message"] == "Kp 5.0: big storm"


def test_run_chain_false_branch_skips():
    calls = []

    def fake_invoke(name, args):
        calls.append((name, args))
        return '{"kp_index": 2.0, "summary": "quiet"}' if name == "check_aurora" else {"ok": True}

    s = c.run_chain(_spec(), fake_invoke)
    assert s["ok"] and "telegram_send" not in s["ran"]
    assert calls == [("check_aurora", {})]


def test_run_chain_failure_isolated():
    def fake_invoke(name, args):
        if name == "check_aurora":
            raise RuntimeError("noaa down")
        return {"ok": True}

    s = c.run_chain(_spec(), fake_invoke)
    assert not s["ok"] and len(s["errors"]) == 1


def test_run_chain_malformed_steps_skipped_not_aborted():
    calls = []

    def fake_invoke(name, args):
        calls.append(name)
        return {"ok": True}

    s = c.run_chain(
        [
            {"bogus": 1},                      # neither tool nor if
            "not-a-dict",                      # not an object at all
            {"tool": ""},                      # empty tool name
            {"if": "not-a-dict", "then": []},  # non-dict condition
            {"tool": "telegram_send"},         # good step still runs
        ],
        fake_invoke,
    )
    assert not s["ok"] and len(s["errors"]) == 4
    assert calls == ["telegram_send"]


def test_run_chain_nesting_bounded():
    def fake_invoke(name, args):
        return 1 if name == "set_flag" else {"ok": True}

    # build a chain nested deeper than _MAX_NESTING, with the flag bound
    # so every level takes the "then" branch
    deep_inner: list = [{"tool": "x"}]
    for _ in range(c._MAX_NESTING + 3):
        deep_inner = [{"if": {"field": "flag", "op": "==", "value": 1}, "then": deep_inner}]
    s = c.run_chain([{"tool": "set_flag", "as": "flag"}] + deep_inner, fake_invoke)
    assert not s["ok"] and any("nesting" in e for e in s["errors"])


def test_eval_condition_rejects_non_dict():
    assert c.eval_condition("nope", {}) is False
    assert c.eval_condition({"field": 123, "op": "==", "value": 1}, {}) is False
