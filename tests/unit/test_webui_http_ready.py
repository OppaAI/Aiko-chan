"""P2 regression: AikoWeb._run_http() must signal _loop_ready only after
uvicorn's server.started becomes true (socket bound) — not before serve() starts —
and must close the event loop on shutdown."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
import pytest

from interface.webui import webui as W


def test_start_servers_resets_flag_after_startup_error(monkeypatch):
    inst = W.AikoWeb.__new__(W.AikoWeb)
    inst._lock = threading.Lock()
    inst._server_start_lock = threading.Lock()
    inst._servers_started = False
    def fail():
        raise RuntimeError("bind failed")
    monkeypatch.setattr(inst, "_start_servers", fail)
    with pytest.raises(RuntimeError, match="bind failed"):
        inst.start_servers()
    assert not inst._servers_started


@pytest.mark.parametrize("started", [False, True])
def test_startup_caller_requires_actual_http_readiness(monkeypatch, started):
    inst = W.AikoWeb.__new__(W.AikoWeb)
    inst._loop_ready = threading.Event()
    inst._http_server = None
    monkeypatch.setattr(W, "_make_ssl_context", lambda *args: None)
    monkeypatch.setattr("socket.gethostbyname", lambda host: "127.0.0.1")
    def fake_http():
        inst._http_started = started
        inst._http_error = RuntimeError("bind failed") if not started else None
        inst._loop_ready.set()
    monkeypatch.setattr(inst, "_run_http", fake_http)
    if started:
        inst._start_servers()
    else:
        with pytest.raises(RuntimeError, match="listener failed"):
            inst._start_servers()
    inst._http_thread.join(2)


def test_http_config_failure_still_releases_waiter_and_closes_loop(monkeypatch):
    def bad_config(*args, **kwargs):
        raise ValueError("bad HTTP configuration")
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(Config=bad_config))
    inst = W.AikoWeb.__new__(W.AikoWeb)
    inst._ssl_context = None
    inst._loop_ready = threading.Event()
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: loop)
    with pytest.raises(ValueError, match="bad HTTP configuration"):
        inst._run_http()
    assert inst._loop_ready.is_set()
    assert loop.is_closed()


def test_stop_servers_signals_server_and_joins_thread():
    inst = W.AikoWeb.__new__(W.AikoWeb)
    inst._lock = threading.Lock()
    inst._servers_started = True
    inst._http_server = types.SimpleNamespace(should_exit=False)
    def serve():
        deadline = time.monotonic() + 2
        while not inst._http_server.should_exit and time.monotonic() < deadline:
            time.sleep(0.01)
    inst._http_thread = threading.Thread(target=serve)
    inst._http_thread.start()
    inst.stop_servers()
    assert inst._http_server.should_exit
    assert not inst._http_thread.is_alive()
    assert not inst._servers_started


def _install_fake_uvicorn(monkeypatch):
    created = []

    class FakeServer:
        def __init__(self, config):
            self.config = config
            # Uvicorn exposes a boolean, not an asyncio.Event.
            self.started = False
            self._done = asyncio.Event()
            created.append(self)

        async def serve(self):
            await self._done.wait()

    fake = types.SimpleNamespace(Config=lambda *a, **k: None, Server=FakeServer)
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    return created


def test_run_http_signals_ready_only_after_start(monkeypatch):
    created = _install_fake_uvicorn(monkeypatch)
    inst = W.AikoWeb.__new__(W.AikoWeb)
    inst._ssl_context = None
    loop = asyncio.new_event_loop()
    # _run_http() creates its own loop; hand it ours so the test can drive it.
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: loop)
    inst._loop_ready = threading.Event()

    t = threading.Thread(target=inst._run_http, daemon=True)
    t.start()
    try:
        # Old code set _loop_ready immediately; the socket wasn't bound yet.
        assert not inst._loop_ready.wait(timeout=2), "_loop_ready fired before server.started"

        deadline = time.monotonic() + 10
        while not created and time.monotonic() < deadline:
            time.sleep(0.05)
        server = created[0]

        inst._loop.call_soon_threadsafe(setattr, server, "started", True)
        assert inst._loop_ready.wait(timeout=5), "_loop_ready never fired after started"

        inst._loop.call_soon_threadsafe(server._done.set)
        t.join(timeout=10)
        assert not t.is_alive()
        assert loop.is_closed(), "event loop was not closed on shutdown"
        assert inst._loop is None
    finally:
        if t.is_alive() and created:
            # Best-effort teardown so a failure can't hang the suite.
            inst._loop.call_soon_threadsafe(created[0]._done.set)
            t.join(timeout=10)


def test_run_http_bind_failure_sets_ready_and_clears_loop(monkeypatch):
    """Uvicorn raises SystemExit (not Exception) on bind failure, which
    escapes run_until_complete before _serve_and_signal() resumes: waiters
    must still be released, and self._loop must be cleared so later
    broadcasts hit the None guard instead of a closed loop."""
    created = []

    class DyingServer:
        def __init__(self, config):
            self.started = False
            created.append(self)

        async def serve(self):
            raise SystemExit(3)

    monkeypatch.setitem(
        sys.modules, "uvicorn",
        types.SimpleNamespace(Config=lambda *a, **k: None, Server=DyingServer),
    )
    hook_calls = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: hook_calls.append(args)
    )

    inst = W.AikoWeb.__new__(W.AikoWeb)
    inst._ssl_context = None
    loop = asyncio.new_event_loop()
    # _run_http() creates its own loop; hand it ours so the test can drive it.
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: loop)
    inst._loop_ready = threading.Event()

    t = threading.Thread(target=inst._run_http, daemon=True)
    t.start()
    t.join(timeout=10)

    assert not t.is_alive()
    assert inst._loop_ready.is_set(), "waiters would hang 5s on a dead server"
    assert inst._loop is None, "closed loop left behind for _broadcast()"
    assert loop.is_closed()
    assert any(
        isinstance(c.exc_value, SystemExit) for c in hook_calls
    ), "SystemExit must still propagate (not be swallowed)"
