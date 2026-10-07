"""Regression coverage for Needle subagent tool boundaries and turn state."""
from __future__ import annotations

import importlib
import json
from unittest.mock import Mock

import pytest

from agentic.needle import NeedleCall, NeedleLowConfidence, NeedleResponse, needle_tools
from agentic.registry import TOOLS, ToolRegistry
from agentic.tool_models import RepoReadFileArgs
from agentic.toolkit import self_improve
from agentic.toolkit.needle_subagent import MAX_TURNS, TOOL_HISTORY_BUDGET, needle_subagent


def _payload(text):
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def _calls(*calls):
    return NeedleResponse("call", tuple(NeedleCall(name, args) for name, args in calls), 0.99, "", "")


def _answer():
    return NeedleResponse("answer", (), 0.99, "finished", "")


@pytest.fixture
def harness(monkeypatch):
    engine = importlib.import_module("agentic.agentic")
    registry_module = importlib.import_module("agentic.registry")
    needle = importlib.import_module("agentic.needle")
    reg = ToolRegistry()
    read = Mock(return_value="page content")
    search = Mock(return_value="search content")
    reg.register("repo_read_file", "Read", handler=read, args_model=RepoReadFileArgs)
    reg.register("repo_search_text", "Search", handler=search)
    monkeypatch.setattr(registry_module, "registry", reg)
    monkeypatch.setattr(engine, "registry", reg)
    monkeypatch.setattr(engine, "_gate_tool_call", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "_preference_requires_approval", lambda name: False)
    monkeypatch.setenv("NEEDLE_MAX_WORKERS", "4")
    monkeypatch.setenv("NEEDLE_WORKERS", json.dumps([
        {"id": "reader", "base_url": "http://reader", "allowed_tools": ["repo_read_file"]},
        {"id": "searcher", "base_url": "http://searcher", "allowed_tools": ["repo_search_text"]},
    ]))
    requests = []
    responses = []

    def complete(client, prompt, schemas):
        # Exercise the actual schema converter, as complete() does on the wire.
        requests.append((client.url, prompt, needle_tools(schemas)))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(needle.NeedleClient, "complete", complete)
    return reg, read, search, requests, responses, engine


def test_multiple_workers_select_first_and_restrict_schemas_and_calls(harness):
    _, read, search, requests, responses, _ = harness
    responses.extend([
        _calls(("repo_search_text", {"query": "forbidden"}), ("repo_read_file", {"relative_path": "a.py"})),
        _answer(),
    ])
    assert _payload(needle_subagent("inspect"))["ok"] is True
    assert all(url == "http://reader/complete" for url, _, _ in requests)
    assert all([tool["name"] for tool in schemas] == ["repo_read_file"] for _, _, schemas in requests)
    read.assert_called_once_with(relative_path="a.py", max_chars=20000, offset=0)
    search.assert_not_called()
    assert "DENIED" in requests[1][1]
    assert "page content" in requests[1][1]


def test_empty_hint_intersection_never_calls_worker(harness):
    _, read, _, requests, _, _ = harness
    result = _payload(needle_subagent("inspect", tools_hint="repo_search_text"))
    assert result == {"ok": False, "error": "no allowed tools resolved"}
    assert requests == []
    read.assert_not_called()


@pytest.mark.parametrize("prior_turns", [0, 1])
def test_low_confidence_is_failure_even_after_tool_results(harness, prior_turns):
    _, _, _, requests, responses, _ = harness
    responses.extend([_calls(("repo_read_file", {"relative_path": "a.py"}))] * prior_turns)
    responses.append(NeedleLowConfidence("below threshold"))
    result = _payload(needle_subagent("inspect"))
    assert result["ok"] is False
    assert "low confidence" in result["error"]
    assert result["turns"] == len(requests) == prior_turns + 1
    assert "summary" not in result


def test_invalid_arguments_are_validated_before_handler(harness):
    _, read, _, requests, responses, _ = harness
    responses.extend([_calls(("repo_read_file", {"offset": "invalid"})), _answer()])
    needle_subagent("inspect")
    read.assert_not_called()
    assert "schema_validation_failed" in requests[1][1]


def test_approval_gate_prevents_handler_execution(harness, monkeypatch):
    reg, read, _, requests, responses, engine = harness
    reg.get("repo_read_file").needs_approval = True
    persist = Mock()
    monkeypatch.setattr(engine, "_persist_pending_approval", persist)
    responses.extend([_calls(("repo_read_file", {"relative_path": "a.py"})), _answer()])
    needle_subagent("inspect")
    read.assert_not_called()
    persist.assert_called_once()
    assert "needs_approval" in requests[1][1]


def test_history_preserves_prior_calls_and_results_and_original_instructions(harness):
    _, read, _, requests, responses, _ = harness
    read.side_effect = ["first page", "second page"]
    responses.extend([
        _calls(("repo_read_file", {"relative_path": "first.py"})),
        _calls(("repo_read_file", {"relative_path": "second.py", "offset": 20})),
        _answer(),
    ])
    needle_subagent("compare pages")
    final_prompt = requests[2][1]
    for text in ["first.py", "first page", "second.py", "second page", '"offset": 20']:
        assert text in final_prompt
    assert final_prompt.startswith(requests[0][1])
    assert "compare pages" in final_prompt
    assert "under 800 characters" in final_prompt


def test_history_and_fallback_are_bounded(harness):
    _, read, _, requests, responses, _ = harness
    read.return_value = "x" * 10000
    responses.extend(_calls(("repo_read_file", {"relative_path": f"page-{i}.py"})) for i in range(MAX_TURNS))
    result = _payload(needle_subagent("inspect"))
    assert len(requests) == MAX_TURNS
    for _, prompt, _ in requests[1:]:
        history = prompt.split("Recent tool calls and results:\n", 1)[1].split("\n\nContinue investigating", 1)[0]
        assert len(history) <= TOOL_HISTORY_BUDGET
    assert "page-0.py" not in requests[-1][1]
    assert f"page-{MAX_TURNS - 2}.py" in requests[-1][1]
    assert result["truncated"] is True
    assert len(result["summary"]) <= 800


def test_registered_reader_schema_and_validation_support_pagination(tmp_path, monkeypatch):
    monkeypatch.setattr(self_improve, "REPO_ROOT", tmp_path)
    (tmp_path / "sample.txt").write_text("abcdefghijkl", encoding="utf-8")
    spec = TOOLS["repo_read_file"]
    assert spec.props["offset"] == {"type": "integer", "default": 0, "minimum": 0}
    schema = spec.to_openai_schema()["function"]["parameters"]
    assert schema["properties"]["offset"]["type"] == "integer"
    assert schema["properties"]["offset"]["default"] == 0
    args = spec.validate_args({"relative_path": "sample.txt", "max_chars": 4})
    assert args["offset"] == 0
    assert self_improve.repo_read_file(**args).startswith("abcd\n[truncated at 4/12")
    args = spec.validate_args({"relative_path": "sample.txt", "max_chars": 4, "offset": 4})
    assert self_improve.repo_read_file(**args).startswith("efgh\n[truncated at 8/12")
