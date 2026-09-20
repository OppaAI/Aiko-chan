"""
tests/unit/test_main_sigint.py — unit tests for main._install_two_stage_sigint.

Split from test_main.py (whose top-level imports are stale against current
main.py) so the SIGINT contract is verified independently.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_main_sigint.py -q --override-ini="addopts="
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import main as main_module


class TestTwoStageSigint:
    def test_first_ctrl_c_raises_second_force_exits(self):
        import signal as real_signal

        installed = {}
        calls = []

        class FakeSignal:
            SIGINT = real_signal.SIGINT

            @staticmethod
            def signal(signum, handler):
                installed["handler"] = handler

        # NOTE: handler invocations must stay inside the patch context —
        # the second press calls the real os._exit(130), which would kill
        # the test runner itself.
        # "import signal as _signal" inside the installer hits sys.modules first
        with patch.dict("sys.modules", {"signal": FakeSignal}), \
             patch("os._exit", side_effect=lambda c: calls.append(c)):
            main_module._install_two_stage_sigint(MagicMock())
            handler = installed["handler"]
            with pytest.raises(KeyboardInterrupt):
                handler(real_signal.SIGINT, None)
            handler(real_signal.SIGINT, None)
        assert calls == [130]

    def test_install_failure_never_breaks_boot(self):
        class BadSignal:
            SIGINT = 2

            @staticmethod
            def signal(signum, handler):
                raise RuntimeError("no signal support")

        with patch.dict("sys.modules", {"signal": BadSignal}):
            # Must not raise — shutdown shortcut is best-effort.
            main_module._install_two_stage_sigint(MagicMock())
