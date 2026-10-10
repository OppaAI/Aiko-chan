"""Unit tests for the lean worker profile (worker_mode) used by autonomous
coding ticks on small context windows.

Note: agentic/agentic.py cannot be imported in minimal environments
(agentic/tools.py does URL parsing at import time), so the prompt/tool
constants are verified via AST instead of import. The schedule plumbing
is tested via real import.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_REPO = Path(__file__).resolve().parents[2]
_AGentic_SRC = (_REPO / "agentic" / "agentic.py").read_text()
_AGentic_TREE = ast.parse(_AGentic_SRC)


def _const(name):
    for node in ast.walk(_AGentic_TREE):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id == name):
                    continue
                value = node.value
                # frozenset({...}) -> the set elements
                if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "frozenset":
                    return set(ast.literal_eval(value.args[0]))
                # int(os.getenv("X", "default")) -> the default
                if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "int":
                    inner = value.args[0]
                    if isinstance(inner, ast.Call):
                        # os.getenv("X", "default") — last arg is the default
                        return ast.literal_eval(inner.args[-1])
                return ast.literal_eval(value)
    raise AssertionError(f"constant {name} not found in agentic/agentic.py")


def test_worker_prompt_is_lean():
    prompt = _const("WORKER_SYSTEM_PROMPT")
    # The whole point: fixed overhead ~1k tokens, not ~4-6k.
    assert len(prompt) < 1000, f"worker prompt too long: {len(prompt)} chars"
    assert "coding worker" in prompt
    assert "CHECKPOINT.md" in prompt
    assert "final_answer" in prompt


def test_worker_tools_coding_only():
    tools = set(_const("WORKER_TOOLS"))
    assert "final_answer" in tools
    assert "code_apply_patch" in tools
    assert "repo_read_file" in tools
    assert "code_run_tests" in tools
    # No chat/social/research tools in the worker's fixed set.
    assert "adaptive_search" not in tools
    assert "deep_research" not in tools
    assert "telegram_send" not in tools


def test_estimator_uses_conservative_divisor():
    # int(os.getenv(..., "3")) — the AST gives the string default
    assert int(_const("AGENT_TOKEN_ESTIMATE_DIVISOR")) == 3


def test_schedule_record_accepts_worker_mode():
    from system.schedule import schedule_job_record

    rec = schedule_job_record(
        "t", "t", "09:00", action="agentic", worker_mode=True, dedupe=False,
    )
    assert rec.get("worker_mode") is True


def test_duejob_carries_worker_mode():
    from system.schedule import DueJob

    j = DueJob(id="x", title="t", task="t", worker_mode=True)
    assert j.worker_mode is True
    j2 = DueJob(id="x", title="t", task="t")
    assert j2.worker_mode is None


def test_update_path_accepts_worker_mode():
    from system.schedule import schedule_job_record, update_schedule_record

    rec = schedule_job_record(
        "t", "t", "09:00", action="agentic", dedupe=False,
    )
    updated = update_schedule_record(rec["id"], {"worker_mode": True})
    assert updated and updated.get("worker_mode") is True
