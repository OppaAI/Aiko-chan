from types import SimpleNamespace

from cognition import neural_state
from cognition.fly_behavior import turn, lateral_horn, giant_fiber, cx_topic, stage4_hooks, circadian
from cognition.flymemory import online_teach, soul_teach
from cognition.flysense import dn, pathways
from cognition.centralcomplex import temporal
from cognition import fly_persona


def test_turn_preserves_zero_dn_inputs_and_shadow_familiarity(monkeypatch):
    monkeypatch.setenv('MEMORY_FLYMB_MODE', 'off')
    monkeypatch.setenv('MEMORY_FLYDN_MODE', 'live')
    monkeypatch.setattr(soul_teach, 'ensure_soul_bootstrap', lambda *a: None)
    monkeypatch.setattr(cx_topic, 'apply_topic_drive', lambda *a, **k: {})
    monkeypatch.setattr(online_teach, 'teach_from_user_text', lambda *a, **k: {})
    monkeypatch.setattr(stage4_hooks, 'after_online_teach', lambda *a: None)
    monkeypatch.setattr(circadian, 'circadian_now', lambda *a, **k: {})
    monkeypatch.setattr(pathways, 'encode_turn_senses', lambda *a, **k: {})
    monkeypatch.setattr(giant_fiber, 'assess_interrupt', lambda *a, **k: {'mode': 'shadow', 'urgency': 1., 'interrupt': False, 'would_interrupt': True})
    monkeypatch.setattr(lateral_horn, 'context_prior', lambda *a, **k: {'mode': 'shadow', 'familiarity': .95})
    monkeypatch.setattr(temporal, 'cx_mode', lambda: 'off')
    monkeypatch.setattr(fly_persona, 'persona_mode', lambda: 'off')
    inputs = []
    def drive(**kwargs):
        inputs.append(kwargs)
        return {'arousal': 0., 'rate_mult': 1.}
    monkeypatch.setattr(dn, 'get_flydn', lambda: SimpleNamespace(drive=drive))
    neural_state.clear_neural_state('zero-dn')
    state = neural_state.get_neural_state('zero-dn')
    state.decisiveness = 0.
    state.context_familiarity = .2
    out = turn.apply_turn_priors('A calm afternoon', user_id=' zero-dn ')
    assert inputs[0]['decisiveness'] == 0.
    assert state.action_drive == 0.
    assert state.context_familiarity == .2

    assert out['urgency'] == 0.
    assert out['gf_prediction']['urgency'] == 1.
    assert out['familiarity'] == .2
