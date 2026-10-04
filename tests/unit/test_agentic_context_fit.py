"""Tests for the assembled-request context guard and the Needle breaker.

Background: the Aiko-Playground self-coding loop fired 72 times and produced
zero code. Every session died the same way — the ReAct loop grew the request
past the llama-server context window, the server answered 400
``exceed_context_size_error``, and the loop broke with no final answer. The
system-prompt budget guard could not prevent it because it runs once, before
the loop appends tool observations.

Covers:
  - _estimate_messages_tokens: counts the payload actually sent.
  - _fit_messages_to_context: degrades instead of shipping a doomed request.
  - _is_context_size_error: recognizes the server's overflow 400.
  - Needle circuit breaker: a dead server costs one probe, not 4x timeout.
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace_ctxfit")
sys.path.insert(0, "/home/oppa-ai/jetson")

from agentic import agentic as AG  # noqa: E402


# ── token estimation ───────────────────────────────────────────────────────

def test_estimate_counts_tool_schemas_and_observations():
    messages = [
        {"role": "system", "content": "S" * 400},
        {"role": "user", "content": "U" * 400},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "repo_read_file", "arguments": '{"path":"x.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "repo_read_file", "content": "T" * 800},
    ]
    without_tools = AG._estimate_messages_tokens(messages)
    with_tools = AG._estimate_messages_tokens(messages, [{"function": {"name": "repo_read_file"}}])
    assert with_tools > without_tools
    # 2000 chars of payload is ~500 tokens; assert it is in the right ballpark.
    assert without_tools > 400


# ── pre-flight fit ─────────────────────────────────────────────────────────

def test_fit_is_noop_when_within_budget():
    messages = [{"role": "system", "content": "short"}, {"role": "user", "content": "hi"}]
    fitted, trimmed = AG._fit_messages_to_context(messages)
    assert trimmed is False
    assert fitted == messages


def test_fit_trims_oversized_tool_output(monkeypatch):
    monkeypatch.setattr(AG, "LLM_CTX_SIZE", 2000)
    monkeypatch.setattr(AG, "AGENT_REQUEST_BUDGET_RATIO", 0.8)
    messages = [
        {"role": "system", "content": "SYSTEM " * 50},
        {"role": "user", "content": "TASK"},
        {"role": "tool", "tool_call_id": "c1", "name": "repo_read_file", "content": "X" * 40000},
    ]
    fitted, trimmed = AG._fit_messages_to_context(messages)
    assert trimmed is True
    assert AG._estimate_messages_tokens(fitted) < AG._estimate_messages_tokens(messages)
    # The task itself must survive — a session with no goal is useless.
    assert any(m.get("content") == "TASK" for m in fitted)


def test_fit_never_drops_the_final_user_task(monkeypatch):
    monkeypatch.setattr(AG, "LLM_CTX_SIZE", 1200)
    monkeypatch.setattr(AG, "AGENT_REQUEST_BUDGET_RATIO", 0.5)
    messages = [
        {"role": "system", "content": "SYSTEM " * 100},
        {"role": "user", "content": "OLD TURN"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "Y" * 20000},
        {"role": "user", "content": "THE REAL TASK"},
    ]
    fitted, _ = AG._fit_messages_to_context(messages)
    assert any(m.get("content") == "THE REAL TASK" for m in fitted)
    assert any(m.get("role") == "system" for m in fitted)


def test_fit_stubs_oldest_tool_output_first(monkeypatch):
    """The newest observation is what the next step reasons about — keep it whole."""
    monkeypatch.setattr(AG, "LLM_CTX_SIZE", 4000)
    monkeypatch.setattr(AG, "AGENT_REQUEST_BUDGET_RATIO", 0.8)
    messages = [{"role": "system", "content": "S" * 200}]
    for i in range(5):
        messages.append({
            "role": "tool", "tool_call_id": f"c{i}", "name": "t",
            "content": f"RESULT-{i} " + "X" * 6000,
        })
    messages.append({"role": "user", "content": "TASK"})
    fitted, trimmed = AG._fit_messages_to_context(messages)
    assert trimmed is True
    assert AG._estimate_messages_tokens(fitted) <= int(4000 * 0.8)
    contents = [str(m.get("content") or "") for m in fitted if m.get("role") == "tool"]
    assert contents, "tool messages must not all be dropped"
    assert "RESULT-4" in contents[-1], "newest observation must survive intact"
    assert any("omitted to fit context" in c for c in contents)


def test_fit_does_not_mutate_input():
    messages = [{"role": "tool", "tool_call_id": "c1", "name": "t", "content": "Z" * 5000}]
    snapshot = [dict(m) for m in messages]
    AG._fit_messages_to_context(messages)
    assert messages == snapshot


def test_fit_honors_active_tool_result_cap(monkeypatch):
    """A lean tick's tighter per-run cap should be the trim ceiling."""
    monkeypatch.setattr(AG, "LLM_CTX_SIZE", 4000)
    monkeypatch.setattr(AG, "AGENT_REQUEST_BUDGET_RATIO", 0.8)
    token = AG._tool_result_max_chars.set(1000)
    try:
        messages = [
            {"role": "system", "content": "S" * 100},
            {"role": "user", "content": "TASK"},
            {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "Q" * 20000},
        ]
        fitted, trimmed = AG._fit_messages_to_context(messages)
        assert trimmed is True
        tool_msgs = [m for m in fitted if m.get("role") == "tool"]
        assert all(len(str(m.get("content") or "")) <= 1000 for m in tool_msgs)
    finally:
        AG._tool_result_max_chars.reset(token)


# ── overflow detection ─────────────────────────────────────────────────────

def test_is_context_size_error_recognized():
    assert AG._is_context_size_error(Exception(
        "{'error': {'code': 400, 'message': 'request (11465 tokens) exceeds the "
        "available context size (10240 tokens)', 'type': 'exceed_context_size_error'}}"
    ))


def test_is_context_size_error_ignores_unrelated():
    assert not AG._is_context_size_error(Exception("connection refused"))
    assert not AG._is_context_size_error(Exception("Read timed out"))


# ── Needle circuit breaker ─────────────────────────────────────────────────

def test_needle_breaker_trips_after_consecutive_failures(monkeypatch):
    monkeypatch.setattr(AG, "NEEDLE_CIRCUIT_FAILURES", 3)
    monkeypatch.setattr(AG, "NEEDLE_CIRCUIT_COOLDOWN", 300)
    AG._needle_circuit_reset()
    assert not AG._needle_circuit_open()
    AG._needle_circuit_record(False)
    AG._needle_circuit_record(False)
    assert not AG._needle_circuit_open(), "must not trip before the threshold"
    AG._needle_circuit_record(False)
    assert AG._needle_circuit_open(), "breaker must open on the configured failure count"


def test_needle_breaker_resets_on_success(monkeypatch):
    monkeypatch.setattr(AG, "NEEDLE_CIRCUIT_FAILURES", 3)
    AG._needle_circuit_reset()
    AG._needle_circuit_record(False)
    AG._needle_circuit_record(False)
    AG._needle_circuit_record(True)
    assert not AG._needle_circuit_open()
    assert AG._needle_state["failures"] == 0


def test_needle_breaker_cooldown_expires(monkeypatch):
    monkeypatch.setattr(AG, "NEEDLE_CIRCUIT_FAILURES", 1)
    monkeypatch.setattr(AG, "NEEDLE_CIRCUIT_COOLDOWN", 300)
    AG._needle_circuit_reset()
    AG._needle_circuit_record(False)
    assert AG._needle_circuit_open()
    # Simulate the cooldown elapsing.
    AG._needle_state["open_until"] = AG.time.monotonic() - 1
    assert not AG._needle_circuit_open()


def test_needle_max_workers_is_usable():
    """int(NEEDLE_MAX_WORKERS) used to raise on every ReAct turn.

    A blank env value reached int() as "" and raised ValueError, which the
    orchestrator reported as "NEEDLE_MAX_WORKERS must be an integer" and
    escalated straight past Needle. The module must always normalize to >= 1.
    """
    assert int(AG.NEEDLE_MAX_WORKERS) >= 1


def test_blank_needle_workers_means_no_workers():
    """Empty NEEDLE_WORKERS is 'no workers', not a crash (documented contract)."""
    from agentic.needle_orchestrator import load_needle_workers

    assert load_needle_workers(
        "", default_timeout=15.0, default_confidence_threshold=0.85, max_workers=4,
    ) == ()


def test_malformed_needle_workers_reports_actionable_error():
    """A typo in NEEDLE_WORKERS must name the variable, not raise raw JSONDecodeError."""
    from agentic.needle_orchestrator import load_needle_workers
    from agentic.needle import NeedleError

    with pytest.raises(NeedleError) as excinfo:
        load_needle_workers("{not json", default_timeout=15.0,
                            default_confidence_threshold=0.85, max_workers=4)
    assert "NEEDLE_WORKERS" in str(excinfo.value)


# ── lean profile plumbing ──────────────────────────────────────────────────

def test_tool_result_cap_defaults_to_global(monkeypatch):
    monkeypatch.setattr(AG, "AGENT_TOOL_RESULT_MAX_CHARS", 50)
    token = AG._tool_result_max_chars.set(None)
    try:
        result = AG.ToolResult(ok=True, tool="t", args={}, content="X" * 500)
        assert "truncated" in result.observation()
        assert "450 more chars" in result.observation()
    finally:
        AG._tool_result_max_chars.reset(token)


def test_tool_result_cap_uses_active_limit(monkeypatch):
    monkeypatch.setattr(AG, "AGENT_TOOL_RESULT_MAX_CHARS", 5000)
    token = AG._tool_result_max_chars.set(100)
    try:
        result = AG.ToolResult(ok=True, tool="t", args={}, content="X" * 500)
        observation = result.observation()
        assert "400 more chars" in observation
        assert AG._tool_result_max_chars.get() == 100
    finally:
        AG._tool_result_max_chars.reset(token)
