"""Run deterministic browser audio state machines when Node is available."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_echo_and_cancellation_state_machines():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for browser voice state-machine regressions')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, str(root / 'tests/browser/voice_audio.test.js')],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
