"""Tests for main.py helpers: lazy --version, console-gated clear-mem prints.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_main_flags.py -q --override-ini="addopts="
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import main as main_module
from main import _console_enabled, _handle_clear_mem, parse_args

PHRASE = "Clear All Aiko's Memories."


def _argv(*args):
    return ["main.py", *args]


@pytest.fixture(autouse=True)
def clean_console_env(monkeypatch):
    monkeypatch.delenv("LOG_CONSOLE", raising=False)


def _confirm_inputs(monkeypatch):
    answers = iter(["y", PHRASE])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))


@pytest.fixture()
def fake_stores(monkeypatch, tmp_path):
    """Fake all three learned-state stores + redirect the dream dir to tmp."""
    mem_cls = MagicMock()
    know = MagicMock(return_value={"learned_chunks": 2})
    exp = MagicMock(return_value={"experiences": 3})
    mem_mod, know_mod, exp_mod = MagicMock(), MagicMock(), MagicMock()
    mem_mod.AikoMemorize = mem_cls
    know_mod.delete_all = know
    exp_mod.delete_all = exp
    monkeypatch.setitem(sys.modules, "cognition.memory.memorize", mem_mod)
    monkeypatch.setitem(sys.modules, "cognition.knowledge.schema", know_mod)
    monkeypatch.setitem(sys.modules, "agentic.experience.schema", exp_mod)
    dream = tmp_path / "dream"
    dream.mkdir()
    (dream / "scratch.db").write_bytes(b"x")
    monkeypatch.setattr("system.userspace.user_state_path", lambda *a, **k: dream)
    return SimpleNamespace(memorize=mem_cls, knowledge=know, experience=exp, dream=dream)


class TestVersion:
    def test_version_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", _argv("--version"))
        with pytest.raises(SystemExit) as e:
            parse_args()
        assert e.value.code == 0
        captured = capsys.readouterr()
        assert "main.py" in captured.out + captured.err  # parser.exit() writes to stderr

    def test_no_metadata_scan_without_flag(self, monkeypatch):
        # The dist scan must not run on ordinary parses (edge boot cost).
        monkeypatch.setattr(sys, "argv", _argv("--cli"))
        with patch("importlib.metadata.version", side_effect=AssertionError("scanned!")):
            ns = parse_args()
        assert ns.cli is True


class TestConsoleGating:
    def test_console_enabled_truth_table(self, monkeypatch):
        assert _console_enabled() is False
        monkeypatch.setenv("LOG_CONSOLE", "1")
        assert _console_enabled() is True
        monkeypatch.setenv("LOG_CONSOLE", "0")
        assert _console_enabled() is False

    def test_clear_mem_prints_when_console_off(self, monkeypatch, capsys, fake_stores):
        _confirm_inputs(monkeypatch)
        assert _handle_clear_mem(MagicMock()) == 0
        assert "Memory cleared." in capsys.readouterr().out
        fake_stores.memorize.assert_called_once()
        fake_stores.knowledge.assert_called_once_with()
        fake_stores.experience.assert_called_once_with()
        assert list(fake_stores.dream.iterdir()) == []  # scratch wiped

    def test_clear_mem_silent_when_console_on(self, monkeypatch, capsys, fake_stores):
        monkeypatch.setenv("LOG_CONSOLE", "1")
        _confirm_inputs(monkeypatch)
        assert _handle_clear_mem(MagicMock()) == 0
        assert "Memory cleared." not in capsys.readouterr().out
        fake_stores.memorize.assert_called_once()
        fake_stores.knowledge.assert_called_once_with()
        fake_stores.experience.assert_called_once_with()
