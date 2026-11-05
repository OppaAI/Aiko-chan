"""Browser attention detection and VLM request lifecycle regressions."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_camera_attention_and_controller():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required for browser camera regressions')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, str(root / 'tests/browser/camera_attention.test.js')],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
