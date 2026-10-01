"""Piece 3: per-turn latency instrumentation + sentence-stream gating.

Covers:
- TurnLatency tracker semantics (first-write-wins, reset, elapsed, report).
- _SentenceSink bookkeeping wrapper.
- _make_sentence_sink gating: safe sentences are fed to the speech stream in
  order; a sentence failing local review or gate_speak aborts early audio.
- _stream_response integration: incremental sentence parsing, first-token /
  first-sentence / final-token marks, and consumed_chars bookkeeping.
"""
from types import SimpleNamespace

import pytest

import cognition.think as think_module
from cognition.think import AikoThink, _SentenceSink
from sensory import latency as latency_mod


# ── latency tracker ──────────────────────────────────────────────────────

def test_latency_mark_is_first_write_wins():
    t = latency_mod.TurnLatency()
    t.mark("input_received")
    first = t.marks["input_received"]
    t.mark("input_received")
    assert t.marks["input_received"] == first


def test_latency_reset_clears_marks():
    t = latency_mod.TurnLatency()
    t.mark("input_received")
    t.reset()
    assert t.marks == {}


def test_latency_elapsed_and_report():
    t = latency_mod.TurnLatency()
    assert t.elapsed("input_received", "first_audio") is None
    t.mark("input_received")
    t.mark("first_llm_token")
    t.mark("first_audio")
    assert t.elapsed("input_received", "first_audio") >= 0.0
    assert t.elapsed("recall_completed", "first_audio") is None
    report = t.report()
    assert "input→first_audio=" in report
    assert "input_received→first_llm_token=" in report


def test_latency_module_singleton():
    latency_mod.reset()
    latency_mod.mark("input_received")
    assert "input_received" in latency_mod._current.marks
    latency_mod.reset()
    assert latency_mod._current.marks == {}


# ── sentence sink ────────────────────────────────────────────────────────

class FakeSpeak:
    def __init__(self):
        self.started = 0
        self.fed = []
        self.stopped = 0

    def start_speech_stream(self, token_callback=None):
        self.started += 1

    def feed_speech_stream(self, text):
        self.fed.append(text)

    def stop_speech_stream(self):
        self.stopped += 1


def _bare_think(monkeypatch, review_flags=(), gate_result="keep"):
    think = object.__new__(AikoThink)
    think._review_response = lambda user_input, text: {"flags": list(review_flags)}
    think._get_memorize = lambda: None
    think._client = None

    from cognition.conscience import hooks

    def fake_gate_speak(*, draft, **kwargs):
        if gate_result == "keep":
            return None
        return gate_result  # refusal / substitution text

    monkeypatch.setattr(hooks, "gate_speak", fake_gate_speak)
    return think


def _sink_parts(monkeypatch, review_flags=(), gate_result="keep"):
    think = _bare_think(monkeypatch, review_flags=review_flags, gate_result=gate_result)
    speak = FakeSpeak()
    state = {"aborted": False, "streaming": False, "spoken": []}
    sink = _SentenceSink(think._make_sentence_sink("hello", speak, None, state))
    return think, speak, state, sink


def test_sentence_sink_feeds_safe_sentences_in_order(monkeypatch):
    _, speak, state, sink = _sink_parts(monkeypatch)
    assert sink("First sentence.") is True
    assert sink("Second one.") is True
    assert speak.started == 1
    assert speak.fed == ["First sentence.", "Second one."]
    assert state["spoken"] == ["First sentence.", "Second one."]
    assert not state["aborted"]
    assert speak.stopped == 0


def test_sentence_sink_aborts_on_review_flags(monkeypatch):
    _, speak, state, sink = _sink_parts(monkeypatch, review_flags=("a", "b"))
    assert sink("Shady sentence.") is False
    assert speak.fed == []
    assert speak.stopped == 1
    assert state["aborted"] is True
    # further sentences are refused once aborted
    assert sink("Another.") is False
    assert speak.fed == []


def test_sentence_sink_aborts_on_gate_refusal(monkeypatch):
    _, speak, state, sink = _sink_parts(monkeypatch, gate_result="I should hold that thought.")
    assert sink("Borderline sentence.") is False
    assert speak.fed == []
    assert state["aborted"] is True


def test_sentence_sink_keeps_draft_when_gate_keeps_it(monkeypatch):
    _, speak, state, sink = _sink_parts(monkeypatch, gate_result="keep")
    assert sink("Fine.") is True
    assert speak.fed == ["Fine."]


# ── _stream_response integration ─────────────────────────────────────────

class FakeStreamCompletions:
    def __init__(self, tokens):
        self.tokens = tokens

    def create(self, **kwargs):
        for tok in self.tokens:
            delta = SimpleNamespace(content=tok)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeStreamClient:
    def __init__(self, tokens):
        self.chat = SimpleNamespace(completions=FakeStreamCompletions(tokens))


def _stream_think(tokens):
    think = object.__new__(AikoThink)
    think._reasoning = False
    think._deep_think = False
    think._llm_model = "test-model"
    think._client = FakeStreamClient(tokens)
    think._review_response = lambda user_input, text: {"flags": []}
    think._get_memorize = lambda: None
    return think


def test_stream_response_sentence_sink_end_to_end(monkeypatch):
    monkeypatch.setattr(think_module, "_SENTENCE_STREAM", True)
    latency_mod.reset()
    from cognition.conscience import hooks
    monkeypatch.setattr(hooks, "gate_speak", lambda *, draft, **kw: None)

    think = _stream_think(["Hello world. ", "How are you?"])
    speak = FakeSpeak()
    state = {"aborted": False, "streaming": False, "spoken": []}
    sink = _SentenceSink(think._make_sentence_sink("hi", speak, None, state))

    text = think._stream_response([], system="", sentence_sink=sink)

    assert text == "Hello world. How are you?"
    assert speak.fed == ["Hello world.", "How are you?"]
    assert not state["aborted"]
    # whole stripped draft was consumed by the sink
    assert sink.consumed_chars == len(text)
    # latency stages fired in order
    assert latency_mod.elapsed("first_llm_token", "first_sentence") is not None
    assert latency_mod.elapsed("first_sentence", "final_token") is not None


def test_stream_response_sink_ignored_when_flag_off(monkeypatch):
    monkeypatch.setattr(think_module, "_SENTENCE_STREAM", False)
    think = _stream_think(["Hello world. How are you?"])
    calls = []
    sink = _SentenceSink(lambda s: calls.append(s) or True)

    text = think._stream_response([], system="", sentence_sink=sink)

    assert text == "Hello world. How are you?"
    assert calls == []
    assert sink.consumed_chars == 0


# ── _finalize_response spoken-prefix skip ────────────────────────────────

def _finalize_think(monkeypatch, correct_fn=None):
    think = object.__new__(AikoThink)
    think._review_response = lambda u, t: {"flags": []}
    think._correct_response = correct_fn or (lambda u, d, r: d)
    think._get_memorize = lambda: None
    think._get_speak = lambda: None
    think._client = None
    from cognition.conscience import hooks
    monkeypatch.setattr(hooks, "gate_speak", lambda *, draft, **kw: None)
    emitted = []
    monkeypatch.setattr(
        think, "_emit_finalized_response",
        lambda text, token_callback=None: emitted.append(text),
    )
    return think, emitted


def test_finalize_emits_only_unspoken_remainder(monkeypatch):
    think, emitted = _finalize_think(monkeypatch)
    draft = "First sentence. Second sentence."
    out = think._finalize_response("hi", draft, _spoken_prefix="First sentence.")
    assert out == draft
    assert emitted == [" Second sentence."]


def test_finalize_emits_nothing_when_all_spoken(monkeypatch):
    think, emitted = _finalize_think(monkeypatch)
    draft = "Only sentence."
    out = think._finalize_response("hi", draft, _spoken_prefix="Only sentence.")
    assert out == draft
    assert emitted == []


def test_finalize_warns_and_emits_full_on_correction_divergence(monkeypatch):
    think, emitted = _finalize_think(
        monkeypatch, correct_fn=lambda u, d, r: "Completely different."
    )
    draft = "First sentence. Second sentence."
    out = think._finalize_response("hi", draft, _spoken_prefix="First sentence.")
    assert out == "Completely different."
    assert emitted == ["Completely different."]
