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


def test_stop_speech_stream_halts_word_emission(speak):
    """Regression: stop_speech_stream() must synchronously halt the worker's
    on_word emission. Previously it only queued the poison pill, so the worker
    kept emitting for the full in-flight chunk while the calling thread wrote
    the refusal note — word-level interleave corruption on one callback."""
    import queue

    emitted = []
    q = queue.Queue()

    def worker():
        # Mimic _speech_stream_worker: block on the queue, then paced word
        # emission honoring _stop_flag exactly as _emit_words_timed does.
        while not speak._stop_flag.is_set():
            try:
                chunk = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if chunk is None:
                break
            for w in chunk.split():
                if speak._stop_flag.is_set():
                    return
                emitted.append(w)
                time.sleep(0.02)

    t = threading.Thread(target=worker, daemon=True)
    speak._stream_thread = t
    speak._streaming_active = True
    speak._stream_queue = q
    t.start()
    q.put("alpha beta gamma delta epsilon zeta eta theta")
    deadline = time.monotonic() + 2.0
    while not emitted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert emitted, "worker never started emitting"
    t0 = time.monotonic()
    speak.stop_speech_stream()
    dt = time.monotonic() - t0
    assert dt < 5.0, f"stop_speech_stream() blocked {dt:.2f}s"
    assert not t.is_alive(), "worker still alive after stop_speech_stream()"
    assert speak._stop_flag.is_set()
    # Snapshot AFTER stop returns: only emissions past this point count
    # against the shutdown guarantee (the worker may legally emit between
    # an earlier snapshot and the stop request).
    n_after_stop = len(emitted)
    time.sleep(0.3)  # grace window: any post-stop emission would land here
    assert len(emitted) == n_after_stop, (
        f"worker emitted after stop: {emitted[n_after_stop:]}"
    )


def test_stop_speech_stream_idempotent(speak):
    speak.stop_speech_stream()  # not streaming -> no-op, no raise
    assert not speak._stop_flag.is_set()


def test_stop_speech_stream_retains_wedged_worker(speak):
    """If the bounded join times out, the worker handle must be retained
    (not discarded) so a later stop() can reap it — and the flag stays set
    so the worker can't emit once it unsticks."""
    import queue

    q = queue.Queue()
    release = threading.Event()

    def worker():
        # Wedged: ignores the pill, blocks past the 5s join timeout.
        release.wait(30)

    t = threading.Thread(target=worker, daemon=True)
    speak._stream_thread = t
    speak._streaming_active = True
    speak._stream_queue = q
    t.start()
    t0 = time.monotonic()
    speak.stop_speech_stream()
    dt = time.monotonic() - t0
    assert dt < 6.0, f"stop_speech_stream() blocked {dt:.2f}s on a wedged worker"
    assert t.is_alive(), "wedged worker unexpectedly exited"
    assert speak._stream_thread is t, "wedged worker handle was discarded"
    assert speak._stop_flag.is_set(), "flag must stay set for the wedged worker"
    release.set()
    # A later stop() detaches the retained handle for background reaping.
    speak.stop()
    assert speak._stream_thread is None
