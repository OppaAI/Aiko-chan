"""Unit tests for the optional Needle 3 local tool-worker adapter."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agentic.needle import NeedleClient, NeedleError, NeedleLowConfidence, needle_tools


class _Response:
    def __init__(self, body: dict):
        self.body = body

    def read(self):
        return json.dumps(self.body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_needle_tools_converts_openai_function_schema():
    tools = needle_tools([{
        "type": "function",
        "function": {"name": "save_note", "description": "Save a note", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}},
    }])
    assert tools == [{"name": "save_note", "description": "Save a note", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}]


def test_complete_parses_constrained_call():
    client = NeedleClient("http://needle.test", confidence_threshold=0.8)
    response = {"type": "call", "success": True, "confidence": 0.91, "function_calls": [{"name": "save_note", "arguments": {"text": "hello"}}]}
    with patch("agentic.needle.urlopen", return_value=_Response(response)):
        result = client.complete("save this", [{"function": {"name": "save_note", "parameters": {}}}])
    assert result.confidence == 0.91
    assert result.calls[0].name == "save_note"
    assert result.calls[0].arguments == {"text": "hello"}


def test_complete_escalates_below_confidence_threshold():
    client = NeedleClient("http://needle.test", confidence_threshold=0.9)
    response = {"type": "call", "success": True, "confidence": 0.5, "function_calls": []}
    with patch("agentic.needle.urlopen", return_value=_Response(response)), pytest.raises(NeedleLowConfidence):
        client.complete("save this", [])


@pytest.mark.parametrize("confidence", [int("9" * 400), float("nan")], ids=["overflow", "nan"])
def test_complete_rejects_invalid_confidence_before_parsing_calls(confidence):
    client = NeedleClient("http://needle.test")
    response = {
        "type": "call",
        "success": True,
        "confidence": confidence,
        "function_calls": [{"name": "send_money", "arguments": {}}],
    }
    with patch("agentic.needle.urlopen", return_value=_Response(response)), pytest.raises(NeedleError, match="confidence must be finite"):
        client.complete("pay", [{"function": {"name": "save_note", "parameters": {}}}])


def test_complete_rejects_tool_outside_allowed_subset():
    client = NeedleClient("http://needle.test")
    response = {"type": "call", "success": True, "confidence": 0.99, "function_calls": [{"name": "send_money", "arguments": {}}]}
    with patch("agentic.needle.urlopen", return_value=_Response(response)), pytest.raises(NeedleError, match="outside Aiko's allowed subset"):
        client.complete("pay", [{"function": {"name": "save_note", "parameters": {}}}])


def test_complete_sends_needle3_query_and_tools_contract():
    """Needle 3 playground reads per-request ``query`` + ``tools`` (``input`` kept for v2)."""
    client = NeedleClient("http://needle.test", confidence_threshold=0.8)
    response = {"type": "call", "success": True, "confidence": 0.91, "function_calls": []}
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(response)

    with patch("agentic.needle.urlopen", side_effect=fake_urlopen):
        client.complete("save this", [{"function": {"name": "save_note", "parameters": {}}}])
    assert captured["body"]["query"] == "save this"
    assert captured["body"]["tools"] == [{"name": "save_note", "description": "", "parameters": {}}]


def test_complete_accepts_none_confidence_for_tuned_weights():
    """Needle 3 tuned (.cact) weights report confidence=None; validation alone decides."""
    client = NeedleClient("http://needle.test", confidence_threshold=0.9)
    response = {"type": "call", "success": True, "confidence": None, "function_calls": [{"name": "save_note", "arguments": {"text": "hi"}}]}
    with patch("agentic.needle.urlopen", return_value=_Response(response)):
        result = client.complete("save this", [{"function": {"name": "save_note", "parameters": {}}}])
    assert result.confidence is None
    assert result.calls[0].name == "save_note"
