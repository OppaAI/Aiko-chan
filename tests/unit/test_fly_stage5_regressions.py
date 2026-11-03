"""Regression tests for Stage 5 CX blending review fixes."""
from __future__ import annotations

import json

import pytest


def test_cx_step_falls_back_before_calling_original_once(monkeypatch):
    from cognition import fly_registry
    from cognition.fly_behavior import cx_blend_install, cx_features

    calls = []

    class Compass:
        def step(self, features, pen_drive=0.0, fatigue=0.0):
            calls.append((features, pen_drive, fatigue))
            return "stepped"

    compass = Compass()
    features = [0.25] * 8
    uid = "stage5-blend-fallback"
    cx_blend_install._wrapped.discard(uid)
    monkeypatch.setattr(fly_registry, "get_flycx", lambda _user_id: compass)
    monkeypatch.setattr(
        cx_features,
        "blend_cx_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("blend failed")),
    )

    assert cx_blend_install.install(uid) is True
    assert compass.step(features, 0.2, 0.1) == "stepped"
    assert calls == [(features, 0.2, 0.1)]


def test_cx_step_does_not_retry_original_failure(monkeypatch):
    from cognition import fly_registry
    from cognition.fly_behavior import cx_blend_install, cx_features

    calls = []

    class Compass:
        def step(self, features, pen_drive=0.0, fatigue=0.0):
            calls.append((features, pen_drive, fatigue))
            raise RuntimeError("step failed")

    compass = Compass()
    uid = "stage5-step-failure"
    cx_blend_install._wrapped.discard(uid)
    monkeypatch.setattr(fly_registry, "get_flycx", lambda _user_id: compass)
    monkeypatch.setattr(
        cx_features,
        "blend_cx_features",
        lambda *_args, **_kwargs: ([0.5] * 8, 0.4, 0.3),
    )

    assert cx_blend_install.install(uid) is True
    with pytest.raises(RuntimeError, match="step failed"):
        compass.step([0.25] * 8, 0.2, 0.1)
    assert calls == [([0.5] * 8, 0.4, 0.3)]


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_cx_blend_preserves_inputs_unmodified_when_not_live(monkeypatch, mode):
    from cognition.fly_behavior.cx_features import blend_cx_features

    features = [0.1, 0.2]
    monkeypatch.setenv("MEMORY_FLYCX_MODE", mode)

    blended, pen, fatigue = blend_cx_features(features, 0.2, 0.1, "new text", "user")

    assert blended is features
    assert pen == 0.2
    assert fatigue == 0.1


def test_cx_blend_prefers_supplied_text_over_cached_features(monkeypatch):
    from cognition.fly_behavior import cx_features, semantic_features

    calls = []
    monkeypatch.setenv("MEMORY_FLYCX_MODE", "live")
    monkeypatch.setattr(
        semantic_features,
        "semantic_features",
        lambda text, *, user_id=None: calls.append((text, user_id)) or [1.0] * 8,
    )
    monkeypatch.setattr(
        semantic_features,
        "last_semantic_features",
        lambda _user_id=None: (_ for _ in ()).throw(AssertionError("cache used")),
    )
    monkeypatch.setattr("cognition.neural_state.peek_neural_state", lambda _user_id: None)

    blended, _, _ = cx_features.blend_cx_features(
        [0.0] * 8, 0.2, 0.1, "new text", "user"
    )

    assert calls == [("new text", "user")]
    assert blended == pytest.approx([0.55] * 8)


def test_cx_blend_uses_cached_features_only_without_text(monkeypatch):
    from cognition.fly_behavior import cx_features, semantic_features

    calls = []
    monkeypatch.setenv("MEMORY_FLYCX_MODE", "live")
    monkeypatch.setattr(
        semantic_features,
        "semantic_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("text used")),
    )
    monkeypatch.setattr(
        semantic_features,
        "last_semantic_features",
        lambda user_id=None: calls.append(user_id) or [1.0] * 8,
    )
    monkeypatch.setattr("cognition.neural_state.peek_neural_state", lambda _user_id: None)

    blended, _, _ = cx_features.blend_cx_features(
        [0.0] * 8, 0.2, 0.1, "  ", "user"
    )

    assert calls == ["user"]
    assert blended == pytest.approx([0.55] * 8)


def test_projection_cache_checks_stored_vector_dimension(monkeypatch):
    from cognition.fly_behavior import semantic_features

    cached = [[0.0, 0.0, 0.0] for _ in range(8)]
    monkeypatch.setattr(semantic_features, "_proj", cached)

    assert semantic_features._projection(3) is cached
    regenerated = semantic_features._projection(4)
    assert regenerated is not cached
    assert len(regenerated) == 8
    assert all(len(column) == 4 for column in regenerated)


@pytest.mark.parametrize(
    "topic",
    [
        {"applied": False, "reason": "semantic", "heading": 1.0, "sharpness": 0.5},
        {"applied": True, "reason": "error", "heading": 1.0, "sharpness": 0.5},
        {"applied": True, "reason": "semantic", "heading": None, "sharpness": 0.5},
        {"applied": True, "reason": "semantic", "heading": 1.0, "sharpness": None},
    ],
)
def test_stage5_eval_rejects_incomplete_topic_results(monkeypatch, capsys, topic):
    from cognition.fly_behavior import cx_features, cx_topic, lateral_horn, semantic_features
    from tests.eval import fly_stage5_ab

    monkeypatch.setattr(semantic_features, "semantic_features", lambda *_args, **_kwargs: [0.1] * 8)
    monkeypatch.setattr(cx_topic, "apply_topic_drive", lambda *_args, **_kwargs: topic)
    monkeypatch.setattr(lateral_horn, "context_prior", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        cx_features,
        "blend_cx_features",
        lambda feats, pen, fatigue, *_args: (feats, pen, fatigue),
    )

    assert fly_stage5_ab.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == "cx_topic_failed"
