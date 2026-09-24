"""Regression tests for busy-gate timeout validation."""
from __future__ import annotations

import importlib
import math

import pytest

from system import turngate


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_invalid_explicit_timeout_is_rejected_before_lock(monkeypatch, timeout):
    class Lock:
        def acquire(self, **_kwargs):
            raise AssertionError("invalid timeout reached the lock")

    monkeypatch.setattr(turngate, "AIKO_BUSY_LOCK", Lock())
    with pytest.raises(ValueError, match="finite and non-negative"):
        turngate.acquire_busy(timeout=timeout)


@pytest.mark.parametrize("configured", ["nan", "inf", "-inf", "-1"])
def test_invalid_configured_timeout_uses_finite_default(monkeypatch, configured):
    with monkeypatch.context() as patch:
        patch.setenv("AIKO_BUSY_TIMEOUT_S", configured)
        importlib.reload(turngate)
        assert math.isfinite(turngate.BUSY_TIMEOUT_S)
        assert turngate.BUSY_TIMEOUT_S == 600.0
    importlib.reload(turngate)
