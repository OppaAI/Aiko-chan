"""Grading tests: weak retrieval must never hard-refuse on its own.

Regression cover for the over-refusal episode where benign turns
("Did you see someone in the image?", "Do you know who I am.") were
refused with confidence 0.88 on purely semantic, low-similarity canon
hits while the SLM was down. The rule these tests lock in:

  REFUSE needs trigger-strength evidence (>= 0.45 relevance on a
  prohibition) or a model that actually read the norms (SLM/deliberation).
  Anything weaker degrades to ESCALATE — still default-closed via HITL —
  never a confident wall.
"""
from __future__ import annotations

from types import SimpleNamespace

from types import SimpleNamespace

import cognition.conscience.canon as canon_mod
import cognition.conscience.core as core_mod
import cognition.conscience.judge as judge_mod
from cognition.conscience.core import ConscienceCircuitCore
from cognition.conscience.schema import AXIS_HORIZONTAL, AXIS_VERTICAL, Norm


def _prohibition(norm_id, axis=AXIS_HORIZONTAL, statement="Do not do the bad thing."):
    return Norm(
        id=norm_id, axis=axis, polarity=-1, statement=statement,
        ref="", weight=1.0, tags=("test",), triggers=(),
    )


def _good(norm_id, axis=AXIS_HORIZONTAL):
    return Norm(
        id=norm_id, axis=axis, polarity=1, statement="Do good things kindly.",
        ref="", weight=1.0, tags=("test",), triggers=(),
    )


class _StubStore:
    def __init__(self, retrieved):
        self._retrieved = list(retrieved)

    def retrieve(self, *args, **kwargs):
        return list(self._retrieved)

    def render_block(self, *args, **kwargs):
        return "<canon>\n(stub)\n</canon>"


class _NoLedger:
    def record(self, *args, **kwargs):
        return "stub-row"

    def expire_stale(self):
        return None


def _core(monkeypatch, retrieved):
    monkeypatch.setattr(canon_mod, "get_canon", lambda *a, **k: _StubStore(retrieved))
    monkeypatch.setattr(
        judge_mod, "get_slm",
        lambda *a, **k: SimpleNamespace(available=False),
    )
    monkeypatch.setattr(core_mod, "ledger_for", lambda *a, **k: _NoLedger())
    monkeypatch.setattr(
        ConscienceCircuitCore, "_note_self_decision", lambda *a, **k: None
    )
    return ConscienceCircuitCore("grades-test-user")


def test_lexical_path_allows_benign_turns():
    """With no embedder (pure lexical), these never engage the canon."""
    core = ConscienceCircuitCore("grades-test-user")
    for text in (
        "Did you see someone in the image?",
        "Did you see someone in the image.",
        "Do you know who I am.",
    ):
        verdict = core.evaluate(act="respond", content=text, context={"surface": "chat"})
        assert verdict.decision == "allow", (text, verdict.decision, verdict.reasons)


def test_weak_semantic_evidence_escalates_instead_of_refusing(monkeypatch):
    """Three low-relevance prohibitions summed past REFUSE_AT — the exact
    production shape (h=-0.72, c=0.88) — must ask, not refuse."""
    retrieved = [
        (0.25, _prohibition("T-1")),
        (0.22, _prohibition("T-2")),
        (0.20, _prohibition("T-3")),
    ]
    core = _core(monkeypatch, retrieved)
    verdict = core.evaluate(
        act="respond", content="Do you know who I am.", context={"surface": "chat"}
    )
    assert verdict.decision == "escalate", (verdict.decision, verdict.reasons)
    assert verdict.escalation_id
    assert "weak retrieval" in verdict.reasons[0]


def test_trigger_strength_hit_still_refuses(monkeypatch):
    """A literal trigger-level prohibition refuses exactly as before."""
    retrieved = [(0.70, _prohibition("T-9", axis=AXIS_VERTICAL))]
    core = _core(monkeypatch, retrieved)
    verdict = core.evaluate(
        act="respond", content="do the bad thing now", context={"surface": "chat"}
    )
    assert verdict.decision == "refuse", (verdict.decision, verdict.reasons)


def test_slm_confirmed_refuse_stands(monkeypatch):
    """When the SLM read the norms and agrees, the refusal is not neutered —
    even though no single hit reaches trigger strength on its own."""
    retrieved = [(0.35, _prohibition("T-4")), (0.35, _prohibition("T-5"))]
    monkeypatch.setattr(canon_mod, "get_canon", lambda *a, **k: _StubStore(retrieved))
    monkeypatch.setattr(
        judge_mod, "get_slm",
        lambda *a, **k: SimpleNamespace(
            available=True,
            score=lambda *a, **k: (-0.9, -0.9, 0.9, ["slm: applies"], ["T-4"]),
        ),
    )
    monkeypatch.setattr(core_mod, "ledger_for", lambda *a, **k: _NoLedger())
    monkeypatch.setattr(
        ConscienceCircuitCore, "_note_self_decision", lambda *a, **k: None
    )
    core = ConscienceCircuitCore("grades-test-user")
    verdict = core.evaluate(
        act="respond", content="do the bad thing now", context={"surface": "chat"}
    )
    assert verdict.decision == "refuse", (verdict.decision, verdict.reasons)


def test_noisy_semantic_retrieval_never_hard_refuses(monkeypatch):
    """End to end through real retrieval: mid-strength similarities on many
    norms (the production failure shape) must ask, never wall."""
    import numpy as np

    # Pin semantic retrieval on: the ambient config may disable it (it is
    # off interim while no SLM is deployed) and these tests pin the
    # behaviour they exercise rather than inheriting the box config.
    monkeypatch.setattr(canon_mod, "CANON_SEMANTIC", True)
    store = canon_mod.get_canon()
    n = len(store.all())
    monkeypatch.setattr(
        "cognition.reason.batch_cosine_scores",
        lambda q, vecs: np.full(n, 0.55),
    )
    monkeypatch.setattr(
        judge_mod, "get_slm",
        lambda *a, **k: SimpleNamespace(available=False),
    )
    monkeypatch.setattr(core_mod, "ledger_for", lambda *a, **k: _NoLedger())
    monkeypatch.setattr(
        ConscienceCircuitCore, "_note_self_decision", lambda *a, **k: None
    )

    class _Embedder:
        def embed_batch(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        def embed_query(self, text, instruct=""):
            return np.zeros(4, dtype=np.float32)

    # Semantic vectors are cached per process — force a rebuild so the
    # stubbed similarities take effect for this store.
    store._vectors = None
    core = ConscienceCircuitCore("grades-test-user")
    for text in (
        "Did you see someone in the image?",
        "Do you know who I am.",
    ):
        retrieved = store.retrieve(text, embedder=_Embedder())
        assert any(s > 0.0 for s, _norm in retrieved)  # noise really retrieved
        verdict = core.evaluate(act="respond", content=text, context={"surface": "chat"})
        assert verdict.decision != "refuse", (text, verdict.decision, verdict.reasons)


def test_semantic_floor_drops_embedding_noise(monkeypatch):
    """Similarities that discount below the lexical floor count as nothing."""
    import numpy as np

    # See above: pin the flag these tests exercise.
    monkeypatch.setattr(canon_mod, "CANON_SEMANTIC", True)
    store = canon_mod.get_canon()
    n = len(store.all())
    monkeypatch.setattr(
        "cognition.reason.batch_cosine_scores",
        lambda q, vecs: __import__("numpy").full(n, 0.28),
    )

    class _Embedder:
        def embed_batch(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        def embed_query(self, text, instruct=""):
            return np.zeros(4, dtype=np.float32)

    retrieved = store.retrieve("Do you know who I am.", embedder=_Embedder())
    assert [s for s, _norm in retrieved if s > 0.0] == []

    # ...while a genuine neighbour still counts.
    monkeypatch.setattr(
        "cognition.reason.batch_cosine_scores",
        lambda q, vecs: __import__("numpy").full(n, 0.90),
    )
    retrieved = store.retrieve("Do you know who I am.", embedder=_Embedder())
    assert any(s >= 0.18 for s, _norm in retrieved)


def _deliberating_core(monkeypatch, retrieved, deliberate_result):
    core = _core(monkeypatch, retrieved)
    monkeypatch.setattr(
        judge_mod, "deliberate", lambda *a, **k: deliberate_result
    )
    return core


def test_deliberation_cannot_upgrade_escalate_to_refuse(monkeypatch):
    """The production misfire shape: thin L2 evidence escalates, deliberation
    claims REFUSE — the upgrade must not stand."""
    retrieved = [(0.30, _prohibition("T-1"))]
    core = _deliberating_core(
        monkeypatch, retrieved, (-0.7, 0.0, 0.8, ["deliberation: bad"], ["T-1"])
    )
    verdict = core.evaluate(
        act="respond", content="introduce yourself please", context={"surface": "chat"},
        llm_client=object(),
    )
    assert verdict.decision == "escalate", (verdict.decision, verdict.reasons)
    assert "deliberate-set-aside" in verdict.layers_run
    assert "deliberate" not in verdict.layers_run


def test_deliberation_suspicion_on_allow_goes_to_human(monkeypatch):
    retrieved = [(0.30, _good("G-1"))]
    core = _deliberating_core(
        monkeypatch, retrieved, (-0.8, -0.8, 0.9, ["deliberation: bad"], ["T-1"])
    )
    verdict = core.evaluate(
        act="respond", content="hello there friend", context={"surface": "chat"},
        llm_client=object(),
    )
    assert verdict.decision == "escalate", (verdict.decision, verdict.reasons)


def test_deliberation_cannot_erase_trigger_refuse(monkeypatch):
    """A trigger-level hard signal survives deliberation leniency: the axis
    floor keeps the negative reading, and trigger strength keeps it strong."""
    retrieved = [(0.70, _prohibition("T-9", axis=AXIS_VERTICAL))]
    core = _deliberating_core(
        monkeypatch, retrieved, (0.5, 0.5, 0.9, ["deliberation: fine"], [])
    )
    verdict = core.evaluate(
        act="respond", content="do the thing now", context={"surface": "chat"},
        llm_client=object(),
    )
    assert verdict.decision == "refuse", (verdict.decision, verdict.reasons)
    assert "deliberate" in verdict.layers_run


def test_refusal_notes_rotate_and_carry_why():
    from cognition.conscience.core import ConscienceCircuitCore as Core

    verdict = SimpleNamespace(decision="refuse")
    retrieved = [(0.60, _prohibition("T-7", axis=AXIS_VERTICAL))]
    notes = {Core._note_for(verdict, retrieved) for _ in range(6)}
    assert len(notes) >= 2, notes
    for note in notes:
        assert "Flagged by T-7" in note, note


def test_speak_escalation_keeps_draft(monkeypatch):
    from cognition.conscience import hooks

    class _Core:
        def evaluate(self, **kwargs):
            return SimpleNamespace(
                decision="escalate", escalation_id="abc123",
                pastoral_note="check?", reasons=["unsure"], gate="hitl",
            )

    monkeypatch.setattr(
        "cognition.conscience.conscience_for", lambda *a, **k: _Core()
    )
    assert hooks.gate_speak(draft="hello there") is None
