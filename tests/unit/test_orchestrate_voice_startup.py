from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import MagicMock

import pytest


def _run_commands(monkeypatch, commands, *, speak=None, listen=None, memory_flush=None, scheduler_stop=None, memory=None, reset_context=None, spin_loop=None, wire_backends=None, finish_status=None, background_adapters=None):
    import system.orchestrate as orchestrate
    import system.wakeup as wakeup
    import system.schedule as schedule
    monkeypatch.setattr(schedule, "stop_scheduler", scheduler_stop or (lambda: None))

    messages = []
    backends = []
    inputs = iter(commands)
    def get_input():
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        return value
    think = SimpleNamespace(
        reset_context=reset_context or (lambda: None),
        set_speak=lambda backend: None,
        set_proactive_resting=lambda resting: None,
        wait_for_memory=memory_flush or (lambda: None),
    )
    ui = SimpleNamespace(
        _boot_result=SimpleNamespace(think=think, memorize=memory, speak=None, listen=None),
        _spin_loop=spin_loop or (lambda stop: stop.wait()),
        _stats={},
        add_message=lambda role, text: messages.append(text),
        set_voice_backends=wire_backends or (lambda output, input_backend: backends.append((output, input_backend))),
        status_finish=finish_status or (lambda: None),
        _draw=lambda **kwargs: None,
        get_input=get_input,
        get_voice_input=lambda input_backend, **kwargs: "/quit",
    )
    proactive = SimpleNamespace(start=lambda: None, stop=lambda: None, touch=lambda: None, set_speak=lambda backend: None)
    monkeypatch.setattr(orchestrate, "ProactiveIdleRunner", lambda *args, **kwargs: proactive)
    monkeypatch.setattr(orchestrate, "_brain_trace", MagicMock())
    monkeypatch.setattr(orchestrate.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(wakeup, "start_speak", lambda: speak)
    monkeypatch.setattr(wakeup, "start_listen", lambda: listen)
    adapters = ModuleType("interface.adapter")
    adapters.start_background_adapters = lambda *args: background_adapters or []
    monkeypatch.setitem(sys.modules, "interface.adapter", adapters)
    monitors = ModuleType("interface.mcp_server.social.monitor_daemon")
    monitors.start_threads_monitor_daemon = lambda: None
    monitors.start_bluesky_monitor_daemon = lambda: None
    monitors.start_mastodon_monitor_daemon = lambda: None
    monitors.set_shared_memorize = lambda memory: None
    monkeypatch.setitem(sys.modules, "interface.mcp_server.social.monitor_daemon", monitors)

    orchestrate.run_session(ui, SimpleNamespace(text=True, no_asr=True, debug=False, cli=True))
    return ui, messages, backends


def test_setup_failure_stops_startup_spinner(monkeypatch):
    import threading
    finished = threading.Event()
    def spin(stop):
        if stop.wait(3):
            finished.set()
    def fail_wiring(*args):
        raise RuntimeError("frontend wiring failed")
    with pytest.raises(RuntimeError, match="frontend wiring failed"):
        _run_commands(monkeypatch, [], spin_loop=spin, wire_backends=fail_wiring)
    assert finished.is_set()


def test_frontend_setup_failure_stops_scheduler_and_flushes_memory(monkeypatch):
    events = []
    def fail(*args):
        raise RuntimeError("frontend setup failed")
    with pytest.raises(RuntimeError, match="frontend setup failed"):
        _run_commands(monkeypatch, [], wire_backends=fail,
                      scheduler_stop=lambda: events.append("stop"),
                      memory_flush=lambda: events.append("flush"))
    assert events == ["stop", "flush"]


def test_ready_display_failure_stops_already_started_adapters(monkeypatch):
    events = []
    def fail():
        raise RuntimeError("ready display failed")
    adapter = SimpleNamespace(stop=lambda: events.append("adapter"))
    with pytest.raises(RuntimeError, match="ready display failed"):
        _run_commands(monkeypatch, [], finish_status=fail, background_adapters=[adapter],
                      scheduler_stop=lambda: events.append("scheduler"),
                      memory_flush=lambda: events.append("memory"))
    assert events == ["scheduler", "adapter", "memory"]


def test_stalled_startup_spinner_does_not_prevent_chat(monkeypatch):
    import threading
    release = threading.Event()
    finished = threading.Event()
    def spin(stop):
        release.wait(5)
        finished.set()
    try:
        _, messages, _ = _run_commands(monkeypatch, ["/quit"], spin_loop=spin)
        assert any("Already leaving" in text for text in messages)
        assert not finished.is_set()
    finally:
        release.set()
        assert finished.wait(2)


def test_failed_memory_switch_releases_turn_gate(monkeypatch):
    import system.orchestrate as orchestrate
    events = []
    monkeypatch.setattr(orchestrate, "acquire_busy", lambda **kw: events.append("acquire") or True)
    monkeypatch.setattr(orchestrate, "release_busy", lambda: events.append("release"))
    memory = SimpleNamespace(switch_user=lambda uid: False)
    _run_commands(monkeypatch, ["hello", "/quit"], memory=memory)
    assert events == ["acquire", "release"]


def test_failed_context_reset_releases_turn_gate(monkeypatch):
    import system.orchestrate as orchestrate
    events = []
    monkeypatch.setattr(orchestrate, "acquire_busy", lambda **kw: True)
    monkeypatch.setattr(orchestrate, "release_busy", lambda: events.append("release"))
    def fail_reset():
        raise RuntimeError("context reset failed")
    with pytest.raises(RuntimeError, match="context reset failed"):
        _run_commands(monkeypatch, ["hello"], reset_context=fail_reset)
    assert events == ["release"]


def test_shutdown_requests_scheduler_stop_before_memory_flush(monkeypatch):
    events = []
    _run_commands(monkeypatch, ["/quit"],
                  scheduler_stop=lambda: events.append("scheduler"),
                  memory_flush=lambda: events.append("memory"))
    assert events == ["scheduler", "memory"]


@pytest.mark.parametrize("commands", [[RuntimeError("input failed")], ["/reset"]])
def test_exception_runs_session_cleanup(monkeypatch, commands):
    events = []
    def fail_reset():
        raise RuntimeError("reset failed")
    with pytest.raises(RuntimeError):
        _run_commands(monkeypatch, commands, reset_context=fail_reset,
                      scheduler_stop=lambda: events.append("scheduler"),
                      memory_flush=lambda: events.append("memory"))
    assert events == ["scheduler", "memory"]


def test_shutdown_flushes_memory_even_if_listen_stop_fails(monkeypatch):
    flushed = []
    def fail_stop():
        raise RuntimeError("listener failed to stop")
    listen = SimpleNamespace(ensure_ready=lambda: None, _voice_error=None,
                             stop_barge_in_monitor=fail_stop)
    _run_commands(monkeypatch, ["/listen", "/quit"], listen=listen,
                  memory_flush=lambda: flushed.append(True))
    assert flushed == [True]


def test_lazy_voice_wires_typewriter_and_first_audio(monkeypatch):
    import system.orchestrate as orchestrate

    typewriters = []

    class Typewriter:
        def __init__(self, ui, backend):
            self.audio_started = False
            typewriters.append(self)

        def on_first_audio(self):
            self.audio_started = True

        def stop(self, flush=True):
            pass

    speak = SimpleNamespace(set_first_audio_callback=lambda callback: setattr(speak, "on_audio", callback))
    monkeypatch.setattr(orchestrate, "TypewriterSync", Typewriter)

    ui, messages, backends = _run_commands(monkeypatch, ["/voice", "/quit"], speak=speak)

    assert len(typewriters) == 1
    assert ui._stats["tts_on"] is True
    assert backends[-1] == (speak, None)
    speak.on_audio()
    assert typewriters[0].audio_started


@pytest.mark.parametrize("error", ["ASR model unavailable", None])
def test_lazy_listen_only_enables_after_ready(monkeypatch, error):
    events = []
    listen = SimpleNamespace(_voice_error=None)

    def ensure_ready():
        events.append("ready")
        listen._voice_error = error

    listen.ensure_ready = ensure_ready
    listen.stop_barge_in_monitor = lambda: None
    ui, messages, backends = _run_commands(monkeypatch, ["/listen", "/quit"], listen=listen)

    assert events == ["ready"]
    assert ui._stats["asr_on"] is (error is None)
    if error:
        assert backends == [(None, None)]
        assert any(error in message for message in messages)
    else:
        assert backends[-1] == (None, listen)
