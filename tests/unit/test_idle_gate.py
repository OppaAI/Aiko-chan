"""Tests for the requires_idle scheduler gate in think.handle_scheduled_job.

The gate must fail closed: an idle-only job never fires when the user is
active, and never fires merely because the idle tracker is unavailable.
"""
from __future__ import annotations

import os
import sys
import types

# This sandbox sets malformed proxy vars that break openai's client at import
# time (module-level OpenAI() in agentic/toolkit/social.py). Scrub before import.
for _key in [k for k in os.environ if "proxy" in k.lower()]:
    del os.environ[_key]

from cognition.think import AikoThink  # noqa: E402
from system.schedule import DueJob  # noqa: E402


def _think_with_idle(monkeypatch, idle_value):
    """AikoThink with a stubbed system.orchestrate.get_idle_seconds()."""
    fake = types.ModuleType("system.orchestrate")
    fake.get_idle_seconds = lambda: idle_value
    monkeypatch.setitem(sys.modules, "system.orchestrate", fake)
    return object.__new__(AikoThink)


def test_idle_job_skipped_when_user_active(monkeypatch):
    think = _think_with_idle(monkeypatch, 30.0)  # only 30s idle
    job = DueJob(id="j1", title="t", task="t", requires_idle=True, idle_seconds=600)
    assert think._idle_requirement_met(job) is False


def test_idle_job_fires_when_idle_long_enough(monkeypatch):
    think = _think_with_idle(monkeypatch, 1200.0)
    job = DueJob(id="j1", title="t", task="t", requires_idle=True, idle_seconds=600)
    assert think._idle_requirement_met(job) is True


def test_idle_job_uses_default_threshold(monkeypatch):
    think = _think_with_idle(monkeypatch, 599.0)
    job = DueJob(id="j1", title="t", task="t", requires_idle=True, idle_seconds=None)
    assert think._idle_requirement_met(job) is False
    think2 = _think_with_idle(monkeypatch, 600.0)
    assert think2._idle_requirement_met(job) is True


def test_idle_job_fails_closed_without_tracker(monkeypatch):
    think = _think_with_idle(monkeypatch, None)  # tracker running, no data
    job = DueJob(id="j1", title="t", task="t", requires_idle=True, idle_seconds=600)
    assert think._idle_requirement_met(job) is False


def test_non_idle_job_unaffected_by_gate():
    # requires_idle=False jobs never consult the gate in handle_scheduled_job;
    # the DueJob default keeps them fireable.
    job = DueJob(id="j1", title="t", task="t")
    assert job.requires_idle is False
