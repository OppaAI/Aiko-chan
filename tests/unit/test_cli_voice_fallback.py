"""CLI voice-input fallback: a broken voice runtime must not spin forever.

When the voice runtime is unavailable, AikoListen.listen() fails fast with
("", {"voice_error": ...}) on every call. Returning "" to the session loop
would print "listening..." forever with no way to type — instead the CLI
must surface the cause and fall back to typed input.
"""
from __future__ import annotations

from types import SimpleNamespace

from interface.cli.cli import AikoSimpleCLI


def _cli_with_typed(monkeypatch, typed="hello typed"):
    cli = AikoSimpleCLI()
    monkeypatch.setattr("builtins.input", lambda _prompt="": typed)
    return cli


def test_broken_voice_falls_back_to_typed_input(monkeypatch, capsys):
    cli = _cli_with_typed(monkeypatch)
    broken = SimpleNamespace(
        listen=lambda: ("", {"voice_error": "voice runtime unavailable (test)"})
    )
    result = cli.get_voice_input(broken)
    text = result[0] if isinstance(result, tuple) else result
    assert text == "hello typed"
    out = capsys.readouterr().out
    assert "voice unavailable" in out


def test_broken_voice_notice_printed_once(monkeypatch, capsys):
    cli = _cli_with_typed(monkeypatch)
    broken = SimpleNamespace(
        listen=lambda: ("", {"voice_error": "voice runtime unavailable (test)"})
    )
    cli.get_voice_input(broken)
    first = capsys.readouterr().out
    assert "voice runtime unavailable (test)" in first
    cli.get_voice_input(broken)
    second = capsys.readouterr().out
    assert "voice runtime unavailable (test)" not in second
    assert "type instead" in second


def test_healthy_silence_still_returns_empty(monkeypatch):
    cli = _cli_with_typed(monkeypatch)
    silent = SimpleNamespace(listen=lambda: ("", {"voice_error": None}))
    result = cli.get_voice_input(silent)
    text = result[0] if isinstance(result, tuple) else result
    # No voice_error: genuine silence — return "" so the loop listens again.
    assert text == ""


def test_voice_transcript_returned(monkeypatch):
    cli = _cli_with_typed(monkeypatch)
    heard = SimpleNamespace(listen=lambda: ("hey aiko", {"voice_error": None}))
    result = cli.get_voice_input(heard)
    text = result[0] if isinstance(result, tuple) else result
    assert text == "hey aiko"
