"""speak.stop() must signal-and-return: a barge-in never blocks the new
turn on audio teardown (the old serial path cost up to ~4s).

Locked here:
  * stop() returns promptly even with a wedged stream thread
  * the audible halt (sd.stop) still happens synchronously
  * the dead thread is joined in the background, not abandoned
  * stop() stays idempotent (no duplicate joiners)
"""
from __future__ import annotations

import threading
import time

import pytest

from sensory.speak import AikoSpeak


@pytest.fixture()
def speak():
    s = object.__new__(AikoSpeak)
    # Minimal init surface that stop() touches.
    s._stop_flag = threading.Event()
    s._lock = threading.Lock()
    s._sd_lock = threading.Lock()
    s._streaming_active = False
    s._stream_chunks = []
    s._stream_on_word = None
    s._stream_queue = None
    s._stream_thread = None
    s._playing = threading.Event()
    s._sd = None  # no ALSA in unit context; sd.stop() skipped
    return s


def _wedged_thread():
    # A thread that ignores the stop signal for a while (worst case).
    t = threading.Thread(target=lambda: time.sleep(30), daemon=True)
    t.start()
    return t


def test_stop_returns_promptly_with_wedged_thread(speak):
    speak._stream_thread = _wedged_thread()
    speak._streaming_active = True
    t0 = time.monotonic()
    speak.stop()
    dt = time.monotonic() - t0
    assert dt < 1.0, f"stop() blocked {dt:.2f}s on teardown"
    assert speak._stream_thread is None  # detached synchronously
    assert speak._stop_flag.is_set()


def test_stop_joins_dead_thread_in_background(speak):
    done = threading.Event()
    t = threading.Thread(target=lambda: done.wait(5), daemon=True)
    t.start()
    speak._stream_thread = t
    done.set()  # let it finish quickly
    speak.stop()
    deadline = time.monotonic() + 5.0
    while t.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not t.is_alive()


def test_stop_idempotent_no_duplicate_joiners(speak, monkeypatch):
    spawned = []
    real_thread = threading.Thread

    def _spy(*a, **kw):
        th = real_thread(*a, **kw)
        spawned.append(th)
        return th

    monkeypatch.setattr(threading, "Thread", _spy)
    speak.stop()
    speak.stop()
    # Nothing playing, no live thread -> no background joiners at all.
    assert spawned == []


def test_audible_halt_stays_synchronous(speak, monkeypatch):
    calls = []

    class FakeSD:
        def stop(self):
            calls.append("sd.stop")

    speak._sd = object()  # non-None so the ALSA path runs
    monkeypatch.setattr(speak, "_load_sd", lambda: FakeSD())
    speak.stop()
    assert calls == ["sd.stop"]  # happened before stop() returned
