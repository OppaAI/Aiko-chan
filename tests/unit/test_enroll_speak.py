"""P2 regression: util/enroll_speak._record_seconds() must reap parec (no
zombie) and must not block forever when the mic produces no audio.

sherpa_onnx isn't installed in CI/dev containers; it's stubbed so the module
imports — the tests only exercise the subprocess plumbing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))

import util.enroll_speak as es


class FakeProc:
    def __init__(self, stdout, hang_on_wait=False):
        self.stdout = stdout
        self.hang_on_wait = hang_on_wait
        self.terminated = False
        self.wait_calls = 0
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.hang_on_wait and self.wait_calls == 1:
            # First wait (after terminate) hangs; the post-kill wait reaps.
            raise subprocess.TimeoutExpired("parec", timeout)
        return 0

    def kill(self):
        self.killed = True

    def poll(self):
        return 0


def _pipe_with(data: bytes):
    r, w = os.pipe()
    os.write(w, data)
    os.close(w)
    return os.fdopen(r, "rb")


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(es.time, "sleep", lambda s: None)  # skip the 3-2-1 countdown


def test_record_reaps_child(monkeypatch, fast):
    n_bytes = int(0.1 * es.SAMPLE_RATE) * 4
    proc = FakeProc(_pipe_with(b"\x00" * n_bytes))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)

    out = es._record_seconds(0.1)

    assert proc.terminated
    assert proc.wait_calls >= 1
    assert not proc.killed
    assert len(out) == n_bytes // 4


def test_record_kills_child_that_ignores_terminate(monkeypatch, fast):
    n_bytes = int(0.1 * es.SAMPLE_RATE) * 4
    proc = FakeProc(_pipe_with(b"\x00" * n_bytes), hang_on_wait=True)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)

    es._record_seconds(0.1)

    assert proc.terminated
    assert proc.killed
    assert proc.wait_calls == 2


def test_record_gives_up_when_mic_silent(monkeypatch, fast, capsys):
    r, w = os.pipe()  # writer held open, never written: select() would wait
    proc = FakeProc(os.fdopen(r, "rb"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    real_monotonic = es.time.monotonic
    ticks = {"n": 0}

    def fake_monotonic():
        ticks["n"] += 1
        return real_monotonic() + ticks["n"] * 100.0  # blow past the deadline at once

    monkeypatch.setattr(es.time, "monotonic", fake_monotonic)
    try:
        out = es._record_seconds(0.1)
    finally:
        os.close(w)

    assert len(out) == 0
    assert proc.terminated
    assert proc.wait_calls >= 1
    assert "no audio in time" in capsys.readouterr().out


def test_record_keeps_partial_data_then_hits_deadline(monkeypatch, fast, capsys):
    """Partial capture then stall: 4 bytes (one float32 sample) arrive, then
    the mic goes silent with the pipe open. Capture must complete at the
    deadline with the partial sample preserved — not hang in read()."""
    import threading

    r, w = os.pipe()
    os.write(w, b"\x01\x02\x03\x04")  # writer stays open: the stall
    proc = FakeProc(os.fdopen(r, "rb"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    real_monotonic = es.time.monotonic
    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        # Allow the deadline computation and the first select+read, then jump
        # past the deadline so the second iteration gives up.
        if calls["n"] <= 2:
            return real_monotonic()
        return real_monotonic() + 10**6

    monkeypatch.setattr(es.time, "monotonic", fake_monotonic)
    result = {}
    t = threading.Thread(
        target=lambda: result.setdefault("out", es._record_seconds(60.0)),
        daemon=True,
    )
    try:
        t.start()
        t.join(timeout=15)
        assert not t.is_alive(), "_record_seconds hung on a stalled mic"
    finally:
        os.close(w)

    out = result["out"]
    assert len(out) == 1
    assert out.tobytes() == b"\x01\x02\x03\x04"
    assert proc.terminated
    assert proc.wait_calls >= 1
    assert "no audio in time" in capsys.readouterr().out
