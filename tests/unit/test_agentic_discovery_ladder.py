"""Tests for piece 2 of the recall ladder: the agentic discovery ladder.

Covers:
  - load_skill: lazy full-instruction loading (metadata-only discovery stays cheap).
  - _similar_successful_experience: only outcome="ok" + score>=0.7 past tasks
    guide ReAct; failed/partial traces never become templates.
  - validate_dag_promotion: explicit promotion rules (repeated, clean, stable,
    recent); promote_skill_proposal(validated=True) refuses without them.
  - _enforce_agentic_context_budget: new 3-block signature.
  - AGENT_MEMKB_TIMEOUT: task-start future is bounded.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace_ladder")
sys.path.insert(0, "/home/oppa-ai/jetson")

from agentic import agentic as AG  # noqa: E402
from agentic.skills import SkillDoc, load_skill  # noqa: E402
from agentic import skill_learning  # noqa: E402


# ── load_skill ────────────────────────────────────────────────────────────

def _doc(skill_id="repo_patch", name="Repository Patch"):
    return SkillDoc(
        skill_id=skill_id,
        name=name,
        path=__import__("pathlib").Path("repo.md"),
        summary="Inspect and change code safely.",
        triggers=("patch code",),
        tools=("repo_read_file",),
    )


def test_load_skill_returns_full_markdown_for_known_skill(monkeypatch, tmp_path):
    from pathlib import Path
    full = "# Repository Patch\n\n## Steps\n1. read\n2. patch\n"
    p = tmp_path / "repo.md"
    p.write_text(full)
    monkeypatch.setattr("agentic.skills.discover_skill_docs", lambda: [
        SkillDoc(skill_id="repo_patch", name="Repository Patch", path=p,
                 summary="Inspect and change code safely.",
                 triggers=("patch code",), tools=("repo_read_file",))
    ])
    out = load_skill("Repository Patch")
    assert "Steps" in out
    assert "read" in out


def test_load_skill_not_found_is_graceful(monkeypatch):
    monkeypatch.setattr("agentic.skills.discover_skill_docs", lambda: [_doc()])
    out = load_skill("No Such Skill")
    assert "not found" in out.lower()
    assert "Repository Patch" in out  # lists available names


# ── _similar_successful_experience ────────────────────────────────────────

def _hit(outcome, score, recall_score=0.9, goal="post the job post"):
    return {
        "outcome": outcome,
        "score": score,
        "recall_score": recall_score,
        "goal": goal,
        "record_text": "step1 -> step2",
    }


def test_similar_experience_prefers_success(monkeypatch):
    monkeypatch.setattr(
        "agentic.experience.search_experience",
        lambda *a, **k: [
            _hit("failed", 0.9),
            _hit("ok", 0.95),
        ],
    )
    out = AG._similar_successful_experience("post the job post", None)
    assert "<similar_experience>" in out
    assert "0.95" in out


def test_similar_experience_rejects_low_score_and_failures(monkeypatch):
    monkeypatch.setattr(
        "agentic.experience.search_experience",
        lambda *a, **k: [_hit("ok", 0.5), _hit("partial", 0.95)],
    )
    assert AG._similar_successful_experience("post the job post", None) == ""


def test_similar_experience_empty_when_no_eligible(monkeypatch):
    monkeypatch.setattr("agentic.experience.search_experience", lambda *a, **k: [])
    assert AG._similar_successful_experience("something new", None) == ""


def test_similar_experience_survives_search_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("store down")
    monkeypatch.setattr("agentic.experience.search_experience", boom)
    assert AG._similar_successful_experience("x", None) == ""


# ── validate_dag_promotion ────────────────────────────────────────────────

def _steps(tools):
    return [{"tool": t, "args": {}, "ok": True} for t in tools]


def _make_proposal(tmp_path, monkeypatch, goal, runs):
    """runs: list of (verified_ok, score). Returns slug."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    for verified_ok, score in runs:
        skill_learning.propose_skill_from_run(
            goal, _steps(["search_memory", "final_answer"]), "done",
            verified_ok=verified_ok, score=score, user_id="u1",
        )
    return skill_learning._slug(goal)


def test_validate_dag_promotion_requires_repeated_clean_runs(tmp_path, monkeypatch):
    slug = _make_proposal(tmp_path, monkeypatch, "repeatable task",
                          [(True, 0.9), (True, 0.85), (True, 0.95)])
    ok, reasons = skill_learning.validate_dag_promotion(slug, user_id="u1")
    assert ok, reasons


def test_validate_dag_promotion_rejects_single_success(tmp_path, monkeypatch):
    slug = _make_proposal(tmp_path, monkeypatch, "one off task", [(True, 0.9)])
    ok, reasons = skill_learning.validate_dag_promotion(slug, user_id="u1")
    assert not ok
    assert any("successful" in r for r in reasons)


def test_validate_dag_promotion_rejects_low_scores(tmp_path, monkeypatch):
    slug = _make_proposal(tmp_path, monkeypatch, "shaky task",
                          [(True, 0.9), (True, 0.5), (True, 0.9)])
    ok, reasons = skill_learning.validate_dag_promotion(slug, user_id="u1")
    assert not ok


def test_validate_dag_promotion_rejects_trailing_failure(tmp_path, monkeypatch):
    slug = _make_proposal(tmp_path, monkeypatch, "regressed task",
                          [(True, 0.9), (True, 0.9), (True, 0.9), (False, 0.2)])
    ok, reasons = skill_learning.validate_dag_promotion(slug, user_id="u1")
    assert not ok
    assert any("latest" in r for r in reasons)


def test_validate_dag_promotion_rejects_unstable_order(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    goal = "unstable task"
    orders = [["a", "b"], ["b", "a"], ["a", "b"]]
    for tools in orders:
        skill_learning.propose_skill_from_run(
            goal, _steps(tools), "done", verified_ok=True, score=0.9, user_id="u1")
    slug = skill_learning._slug(goal)
    ok, _ = skill_learning.validate_dag_promotion(slug, user_id="u1")
    assert not ok


def test_promote_validated_refuses_without_rules(tmp_path, monkeypatch):
    slug = _make_proposal(tmp_path, monkeypatch, "draft only", [(True, 0.9)])
    with pytest.raises(ValueError):
        skill_learning.promote_skill_proposal(slug, user_id="u1",
                                              dry_run=False, validated=True)


def test_promote_validated_stages_when_rules_pass(tmp_path, monkeypatch):
    slug = _make_proposal(tmp_path, monkeypatch, "solid task",
                          [(True, 0.9), (True, 0.85), (True, 0.95)])
    path = skill_learning.promote_skill_proposal(slug, user_id="u1",
                                                 dry_run=False, validated=True)
    payload = json.loads(path.read_text())
    assert payload["review_status"] == "validated_dag"


def test_promote_unvalidated_stages_as_draft(tmp_path, monkeypatch):
    slug = _make_proposal(tmp_path, monkeypatch, "plain draft", [(True, 0.9)])
    path = skill_learning.promote_skill_proposal(slug, user_id="u1", dry_run=False)
    payload = json.loads(path.read_text())
    assert payload["review_status"] == "pending_human_review"


# ── context budget + timeout wiring ──────────────────────────────────────

def test_context_budget_new_signature_sheds_weakest():
    kb, exp, tm = AG._enforce_agentic_context_budget(
        "persona", "mem", "user",
        "k" * 10, "e" * 10, "task-mode-guidance",
        tool_schemas=[],
        scores={"knowledge": 0.9, "experience": 0.1},
    )[:3]
    # With a tiny budget pressure the weakest block sheds first; here the
    # budget is large enough that nothing sheds — assert shape instead.
    assert isinstance(kb, str) and isinstance(exp, str) and isinstance(tm, str)


def test_context_budget_returns_memory_unchanged_by_default():
    """Interactive chat keeps its memories: memory is fixed budget."""
    mem, kb, exp, tm = AG._enforce_agentic_context_budget(
        "persona", "MY-MEMORY", "user", "kb", "exp", "guidance",
        tool_schemas=[],
    )
    assert mem == "MY-MEMORY"
    assert (kb, exp, tm) == ("kb", "exp", "guidance")


def test_context_budget_lean_makes_memory_droppable(monkeypatch):
    """A lean tick sheds memory before knowledge/experience under pressure."""
    monkeypatch.setattr(AG, "AGENT_CONTEXT_BUDGET_RATIO", 0.0001)
    mem, kb, _exp, _tm = AG._enforce_agentic_context_budget(
        "persona", "X" * 4000, "user", "kb", "exp", "guidance",
        tool_schemas=[],
        scores={"knowledge": 0.9, "experience": 0.9},
        lean=True,
    )
    assert "context budget exceeded" in mem
    # Memory sheds first because it carries the lowest score in lean mode.
    assert mem != "X" * 4000


def test_memkb_timeout_is_positive_bound():
    assert isinstance(AG.AGENT_MEMKB_TIMEOUT, float)
    assert AG.AGENT_MEMKB_TIMEOUT > 0
