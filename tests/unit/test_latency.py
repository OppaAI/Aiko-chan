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


# ---------------------------------------------------------------------------
# Sentence-stream sink: fail-closed on evaluation errors (CodeRabbit fix 5)
# ---------------------------------------------------------------------------


def _fail_closed_sink_parts(monkeypatch, gate_raises=False, review_raises=False):
    think = _bare_think(monkeypatch)
    from cognition.conscience import hooks
    if gate_raises:
        def _boom(*, draft, **kw):
            raise RuntimeError("gate exploded")
        monkeypatch.setattr(hooks, "gate_speak", _boom)
    if review_raises:
        def _boom_review(user_input, text):
            raise RuntimeError("review exploded")
        think._review_response = _boom_review
    speak = FakeSpeak()
    state = {"aborted": False, "streaming": False, "spoken": []}
    sink = think._make_sentence_sink("hi", speak, None, state)
    return think, speak, state, sink


def test_sentence_sink_aborts_on_gate_exception(monkeypatch):
    """A gate_speak error must NOT let an unevaluated sentence reach TTS."""
    _, speak, state, sink = _fail_closed_sink_parts(monkeypatch, gate_raises=True)
    assert sink("Hello world.") is False
    assert state["aborted"] is True
    assert speak.fed == [], f"no unevaluated sentence may reach TTS, got {speak.fed!r}"


def test_sentence_sink_aborts_on_review_exception(monkeypatch):
    """A _review_response error must NOT let an unevaluated sentence reach TTS."""
    _, speak, state, sink = _fail_closed_sink_parts(monkeypatch, review_raises=True)
    assert sink("Hello world.") is False
    assert state["aborted"] is True
    assert speak.fed == [], f"no unevaluated sentence may reach TTS, got {speak.fed!r}"


# ---------------------------------------------------------------------------
# Sentence-stream drain: only accepted spans count as consumed (CodeRabbit 6)
# ---------------------------------------------------------------------------


def test_drain_counts_only_accepted_sentences(monkeypatch):
    """A refused sentence is never spoken, so it must not count as consumed.

    Regression: the old drain counted every parsed sentence (and the final
    tail) as consumed before the sink accepted it, so _finalize_response
    could skip emitting a sentence that was never spoken.
    """
    monkeypatch.setattr(think_module, "_SENTENCE_STREAM", True)
    latency_mod.reset()
    from cognition.conscience import hooks
    monkeypatch.setattr(hooks, "gate_speak", lambda *, draft, **kw: None)

    think = _stream_think(["Hello world. ", "Bad news here. ", "All good."])
    think._review_response = (
        lambda user_input, text: {"flags": ["bad", "worse"] if "Bad" in text else []}
    )
    speak = FakeSpeak()
    state = {"aborted": False, "streaming": False, "spoken": []}
    sink = _SentenceSink(think._make_sentence_sink("hi", speak, None, state))

    text = think._stream_response([], system="", sentence_sink=sink)

    assert text == "Hello world. Bad news here. All good."
    # Only the accepted sentence was spoken...
    assert speak.fed == ["Hello world."]
    assert state["aborted"] is True
    # ...and only its raw span counts as consumed — the refused sentence and
    # everything after it stay un-consumed so finalize can still emit them.
    assert sink.consumed_chars == len("Hello world. "), (
        f"refused sentence counted as consumed: {sink.consumed_chars}"
    )


def test_drain_rejected_first_sentence_counts_nothing(monkeypatch):
    """If the very first sentence is refused, consumed stays 0."""
    monkeypatch.setattr(think_module, "_SENTENCE_STREAM", True)
    latency_mod.reset()
    from cognition.conscience import hooks
    monkeypatch.setattr(hooks, "gate_speak", lambda *, draft, **kw: None)

    think = _stream_think(["Nope. ", "Later."])
    think._review_response = lambda user_input, text: {"flags": ["bad", "worse"]}
    speak = FakeSpeak()
    state = {"aborted": False, "streaming": False, "spoken": []}
    sink = _SentenceSink(think._make_sentence_sink("hi", speak, None, state))

    think._stream_response([], system="", sentence_sink=sink)

    assert speak.fed == []
    assert sink.consumed_chars == 0


# ---------------------------------------------------------------------------
# Latency report: intervals in timestamp order (CodeRabbit fix 8)
# ---------------------------------------------------------------------------


def test_latency_report_sorts_by_timestamp():
    """Out-of-order marks (e.g. first_audio before final_token during
    sentence streaming) must not produce negative intervals."""
    t = latency_mod.TurnLatency()
    # Simulate sentence streaming: audio starts before the stream ends.
    t.marks["input_received"] = 1000.0
    t.marks["first_audio"] = 1001.0
    t.marks["final_token"] = 1002.0
    report = t.report()
    assert "first_audio→final_token=" in report
    assert "=-" not in report, f"negative interval in {report!r}"


# ---------------------------------------------------------------------------
# gate_speak fail-closed mode (CodeRabbit: residual finding on fix 5)
#
# gate_speak swallows conscience_for().evaluate() errors internally and
# returns the draft unchanged — which the sink reads as approval. The
# sink-level try/except alone cannot see that failure, so gate_speak gets
# an opt-in fail-closed mode for early sentence streaming.
# ---------------------------------------------------------------------------


def _failing_conscience(monkeypatch):
    import cognition.conscience as conscience_mod

    def _boom():
        raise RuntimeError("conscience down")

    monkeypatch.setattr(conscience_mod, "conscience_for", _boom)


def test_gate_speak_fail_closed_reraises_eval_errors(monkeypatch):
    from cognition.conscience import hooks

    _failing_conscience(monkeypatch)
    # Default preserves existing fail-open behavior for finalization.
    assert hooks.gate_speak(draft="Hello.") == "Hello."
    # Opt-in fail-closed surfaces the error to the caller.
    with pytest.raises(RuntimeError, match="conscience down"):
        hooks.gate_speak(draft="Hello.", fail_closed=True)


def test_sentence_sink_enables_fail_closed_gate(monkeypatch):
    from cognition.conscience import hooks

    think = _bare_think(monkeypatch)
    seen = {}

    def _rec_gate_speak(*, draft, fail_closed=False, **kw):
        seen["fail_closed"] = fail_closed
        return None

    monkeypatch.setattr(hooks, "gate_speak", _rec_gate_speak)
    speak = FakeSpeak()
    state = {"aborted": False, "streaming": False, "spoken": []}
    sink = think._make_sentence_sink("hi", speak, None, state)
    assert sink("Hello world.") is True
    assert seen.get("fail_closed") is True


def test_sentence_sink_aborts_when_gate_eval_fails_internally(monkeypatch):
    """End-to-end: real gate_speak + failing conscience_for() must abort
    early audio with nothing fed — the unevaluated sentence must not
    reach TTS via gate_speak's internal fail-open draft return."""
    _failing_conscience(monkeypatch)
    # Minimal think WITHOUT _bare_think's gate_speak monkeypatch: use the
    # real gate_speak so its internal except path is exercised.
    think = object.__new__(AikoThink)
    think._review_response = lambda user_input, text: {"flags": []}
    think._get_memorize = lambda: None
    think._client = None
    speak = FakeSpeak()
    state = {"aborted": False, "streaming": False, "spoken": []}
    sink = think._make_sentence_sink("hi", speak, None, state)
    assert sink("Hello world.") is False
    assert state["aborted"] is True
    assert speak.fed == [], f"unevaluated sentence reached TTS: {speak.fed!r}"
