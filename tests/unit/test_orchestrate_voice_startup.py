from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import MagicMock

import pytest


def _run_commands(monkeypatch, commands, *, speak=None, listen=None):
    import system.orchestrate as orchestrate
    import system.wakeup as wakeup

    messages = []
    backends = []
    inputs = iter(commands)
    think = SimpleNamespace(
        set_speak=lambda backend: None,
        set_proactive_resting=lambda resting: None,
        wait_for_memory=lambda: None,
    )
    ui = SimpleNamespace(
        _boot_result=SimpleNamespace(think=think, memorize=None, speak=None, listen=None),
        _spin_loop=lambda stop: stop.wait(),
        _stats={},
        add_message=lambda role, text: messages.append(text),
        set_voice_backends=lambda output, input_backend: backends.append((output, input_backend)),
        status_finish=lambda: None,
        _draw=lambda **kwargs: None,
        get_input=lambda: next(inputs),
        get_voice_input=lambda input_backend, **kwargs: "/quit",
    )
    proactive = SimpleNamespace(start=lambda: None, stop=lambda: None, touch=lambda: None, set_speak=lambda backend: None)
    monkeypatch.setattr(orchestrate, "ProactiveIdleRunner", lambda *args, **kwargs: proactive)
    monkeypatch.setattr(orchestrate, "_brain_trace", MagicMock())
    monkeypatch.setattr(orchestrate.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(wakeup, "start_speak", lambda: speak)
    monkeypatch.setattr(wakeup, "start_listen", lambda: listen)
    adapters = ModuleType("interface.adapter")
    adapters.start_background_adapters = lambda *args: []
    monkeypatch.setitem(sys.modules, "interface.adapter", adapters)
    monitors = ModuleType("interface.mcp_server.social.monitor_daemon")
    monitors.start_threads_monitor_daemon = lambda: None
    monitors.start_bluesky_monitor_daemon = lambda: None
    monitors.start_mastodon_monitor_daemon = lambda: None
    monitors.set_shared_memorize = lambda memory: None
    monkeypatch.setitem(sys.modules, "interface.mcp_server.social.monitor_daemon", monitors)

    orchestrate.run_session(ui, SimpleNamespace(text=True, no_asr=True, debug=False, cli=True))
    return ui, messages, backends


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
