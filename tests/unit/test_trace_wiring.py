"""Wiring tests: --trace/--debug flags -> env vars -> downstream consumers.

Locks in the TRACE_BRAIN name on both ends (main.py sets it,
system/brain_trace.py reads it) so the two can never silently drift apart
again.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_trace_wiring.py -q --override-ini="addopts="
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import main as main_module
from main import _apply_debug_trace_env, parse_args

REPO_ROOT = Path(__file__).resolve().parents[2]


def _argv(*args):
    return ["main.py", *args]


@pytest.fixture(autouse=True)
def clean_flag_env(monkeypatch):
    for var in ("TRACE_BRAIN", "AIKO_TRACE_BRAIN", "LOG_CONSOLE", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)


class TestFlagToEnv:
    def test_trace_sets_trace_brain(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--trace"))
        _apply_debug_trace_env(parse_args())
        assert os.environ["TRACE_BRAIN"] == "1"

    def test_no_flags_set_nothing(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv())
        _apply_debug_trace_env(parse_args())
        assert "TRACE_BRAIN" not in os.environ
        assert "LOG_CONSOLE" not in os.environ

    def test_debug_sets_log_env(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--debug"))
        _apply_debug_trace_env(parse_args())
        assert os.environ["LOG_CONSOLE"] == "1"
        assert os.environ["LOG_LEVEL"] == "DEBUG"

    def test_flag_beats_shell_export(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--debug"))
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        _apply_debug_trace_env(parse_args())
        assert os.environ["LOG_LEVEL"] == "DEBUG"

    def test_console_on_forces_on(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--console"))
        _apply_debug_trace_env(parse_args())
        assert os.environ["LOG_CONSOLE"] == "1"

    def test_console_off_beats_debug(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--debug", "--no-console"))
        _apply_debug_trace_env(parse_args())
        assert os.environ["LOG_CONSOLE"] == "0"

    def test_console_off_beats_shell_export(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--no-console"))
        monkeypatch.setenv("LOG_CONSOLE", "1")
        _apply_debug_trace_env(parse_args())
        assert os.environ["LOG_CONSOLE"] == "0"

    def test_console_unset_keeps_debug_rule(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--debug"))
        _apply_debug_trace_env(parse_args())
        assert os.environ["LOG_CONSOLE"] == "1"


def _fresh_trace_enabled(extra_env: dict) -> str:
    """Import brain_trace in a fresh interpreter and report TRACE_ENABLED."""
    env = {k: v for k, v in os.environ.items() if k not in ("TRACE_BRAIN", "AIKO_TRACE_BRAIN")}
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", "from system.brain_trace import TRACE_ENABLED; print(TRACE_ENABLED)"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"probe failed: {proc.stderr[-500:]}"
    return proc.stdout.strip()


class TestEnvToConsumer:
    def test_plain_name_enables(self):
        assert _fresh_trace_enabled({"TRACE_BRAIN": "1"}) == "True"

    def test_unset_disables(self):
        assert _fresh_trace_enabled({}) == "False"

    def test_old_prefixed_name_no_longer_enables(self):
        # Guards the rename: the AIKO_ prefixed spelling must not come back.
        assert _fresh_trace_enabled({"AIKO_TRACE_BRAIN": "1"}) == "False"
