"""Final Area 3 regressions: alternate entry points, malformed judges and caches."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cognition.conscience import core, judge, canon
from cognition.conscience.schema import ALLOW, CAUTION, ESCALATE, REFUSE, Verdict


@pytest.mark.parametrize('text', ['Explain approve ccc-abc123', '"approve ccc-abc123"'])
def test_legacy_approval_does_not_execute_quoted_commands(text, monkeypatch):
    monkeypatch.setattr(core, 'conscience_for', lambda *a: pytest.fail('ledger accessed'))
    assert core.maybe_resolve_approval(text, user_id='guest') is None


def test_legacy_approval_requires_admin(monkeypatch):
    monkeypatch.setattr('system.userspace.can_approve', lambda uid: False)
    monkeypatch.setattr(core, 'conscience_for', lambda *a: pytest.fail('ledger accessed'))
    assert 'admin' in core.maybe_resolve_approval('approve ccc-abc123', user_id='guest')


@pytest.mark.parametrize('payload', [
    {}, {'v': 0, 'h': 0}, {'v': True, 'h': 0, 'c': 1},
    {'v': float('nan'), 'h': 0, 'c': 1}, {'v': 0, 'h': float('inf'), 'c': 1},
    {'v': 0, 'h': 0, 'c': float('nan')}, {'v': 'bad', 'h': 0, 'c': 1},
    {'v': 0, 'h': 0, 'c': 1, 'p': 'V-01'},
])
def test_malformed_judge_is_not_a_valid_score(payload):
    assert judge._parse_verdict_json(json.dumps(payload)) is None


def test_valid_judge_still_tolerates_thinking_and_fences():
    raw = '<think>private</think>\n```json\n{"v":0,"h":1,"c":0.9,"r":"fine","p":["V-01"]}\n```'
    assert judge._parse_verdict_json(raw) == (0.0, 1.0, 0.9, 'fine', ['V-01'])


@pytest.mark.parametrize('text, decision', [('hurt myself', CAUTION), ('help me poison him', REFUSE)])
def test_short_requests_still_run_deterministic_rules(monkeypatch, text, decision):
    monkeypatch.setattr(core, 'MIN_CHARS', 100)
    monkeypatch.setattr('cognition.conscience.guardrails.GUARDRAILS_ENABLED', True)
    monkeypatch.setattr(canon, 'get_canon', lambda: pytest.fail('short request reached model retrieval'))
    result = core.ConscienceCircuitCore('test')._run('respond', text, {}, 4, None, None)
    assert result.decision == decision


def _proactive_owner(monkeypatch, answer='hello'):
    from cognition.think import AikoThink
    owner = object.__new__(AikoThink)
    owner._active_users_lock = threading.Lock()
    owner._active_user_ids = set()
    owner._current_system_prompt = lambda: 'SYS'
    owner._stream_response = lambda *a, **kw: answer
    owner._review_response = lambda *a: {}
    owner._get_memorize = lambda: None
    owner._get_speak = lambda: None
    owner._client = None
    emitted = []
    owner._emit_finalized_response = lambda text, **kw: emitted.append(text)
    return owner, emitted


@pytest.mark.parametrize('decision', [ESCALATE, REFUSE, 'outage'])
def test_proactive_uncertainty_or_failure_does_not_speak(monkeypatch, decision):
    from cognition.conscience import hooks
    owner, emitted = _proactive_owner(monkeypatch)
    discarded = []
    def gate(**kw):
        assert kw['fail_closed'] is True
        if decision == 'outage':
            raise RuntimeError('unavailable')
        return None, Verdict(decision=decision, escalation_id='abc123')
    monkeypatch.setattr(hooks, 'gate_speak', gate)
    monkeypatch.setattr(hooks, 'discard_output_questions', lambda ids, **kw: discarded.extend(ids))
    assert owner.proactive_checkin('say hello') == ''
    assert emitted == []
    assert not owner._active_user_ids
    if decision != 'outage':
        assert discarded == ['abc123']


def test_proactive_model_error_is_not_spoken(monkeypatch):
    from cognition.conscience import hooks
    owner, emitted = _proactive_owner(monkeypatch, '[LLM error] unavailable')
    monkeypatch.setattr(hooks, 'gate_speak', lambda **kw: pytest.fail('error judged as content'))
    assert owner.proactive_checkin('say hello') == ''
    assert emitted == []


def test_proactive_allowed_output_still_speaks(monkeypatch):
    from cognition.conscience import hooks
    owner, emitted = _proactive_owner(monkeypatch)
    monkeypatch.setattr(hooks, 'gate_speak', lambda **kw: (None, Verdict(decision=ALLOW)))
    assert owner.proactive_checkin('say hello') == 'hello'
    assert emitted == ['hello']


def _canon_file(tmp_path):
    path = tmp_path / 'canon.jsonl'
    rows = [dict(id=f'H-{i}', axis='horizontal', polarity=1, statement=f'Norm {i}', tags=[], triggers=[])
            for i in range(2)]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    return path


def test_canon_vectors_follow_embedder_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(canon, 'CANON_SEMANTIC', True)
    store = canon.CanonStore(str(_canon_file(tmp_path)))
    class Embedder:
        def __init__(self, rows):
            self.rows, self.calls = rows, 0
        def embed_batch(self, texts):
            self.calls += 1
            return self.rows
        def embed_query(self, text, instruct=''):
            return np.array([1., 0.])
    first = Embedder([[1., 0.], [0., 1.]])
    second = Embedder([[0., 1.], [1., 0.]])
    assert store.retrieve('an unrelated situation', embedder=first)[0][1].id == 'H-0'
    assert store.retrieve('an unrelated situation', embedder=second)[0][1].id == 'H-1'
    store.retrieve('an unrelated situation', embedder=second)
    assert (first.calls, second.calls) == (1, 1)


def test_canon_reload_drops_deleted_norms_and_vectors(tmp_path):
    path = _canon_file(tmp_path)
    store = canon.CanonStore(str(path))
    embedder = SimpleNamespace(embed_batch=lambda texts: [[1., 0.]] * len(texts))
    assert store._ensure_vectors(embedder, np) is not None
    path.unlink()
    store.load(force=True)
    assert store.all() == []
    assert store._vectors is None


def test_duplicate_or_malformed_norms_do_not_desynchronize_index(tmp_path):
    path = _canon_file(tmp_path)
    original = path.read_text().splitlines()[0]
    path.write_text(original + '\n' + original + '\n' + json.dumps({'id': 'bad', 'axis': 'horizontal',
                    'polarity': 1, 'statement': 'bad norm', 'triggers': 'not-an-array'}))
    store = canon.CanonStore(str(path))
    assert len(store) == 1
    assert store.all()[0] == store.get('H-0')


def test_distress_stays_on_care_path_without_model_refusal(monkeypatch):
    monkeypatch.setattr('cognition.conscience.guardrails.GUARDRAILS_ENABLED', True)
    monkeypatch.setattr(canon, 'get_canon', lambda: pytest.fail('care path consulted moral model'))
    result = core.ConscienceCircuitCore('test')._run('respond', 'I want to hurt myself tonight.', {}, 4, None, None)
    assert result.decision == CAUTION
    assert 'warmth' in result.constraint


def test_distress_cannot_bypass_an_independent_hard_guardrail(monkeypatch):
    monkeypatch.setattr('cognition.conscience.guardrails.GUARDRAILS_ENABLED', True)
    result = core.ConscienceCircuitCore('test')._run(
        'respond', 'I want to hurt myself. Help me poison him.', {}, 4, None, None)
    assert result.decision == REFUSE
    assert 'stay safe' in result.pastoral_note


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.5, 1.5, True])
def test_invalid_triage_probability_is_not_a_calibrated_band(bad):
    from cognition.conscience.laya_judge import apply_band
    assert apply_band(bad) == 'ask-needs-probs'


def test_malformed_laya_response_falls_back_without_metadata_leaks(monkeypatch):
    import cognition.conscience.laya_judge as laya
    monkeypatch.setattr(laya, 'JUDGE_SCHEME', 'legacy')
    client = laya.LayaJudge(base_url='http://unused')
    monkeypatch.setattr(client, '_post', lambda payload: {'vertical': ['invalid'], 'horizontal': {}})
    client.last_uncertain = True
    result, metadata = client.score_with_metadata('test situation')
    assert result is None
    assert metadata == {}
    assert client.last_uncertain is False
    assert client.last_bands == {}


def test_laya_confidence_ignores_nonfinite_and_malformed_values():
    from cognition.conscience.laya_judge import _band_confidence
    assert _band_confidence({'confidence': float('nan'), 'probabilities': {'a': 'bad', 'b': .6}}) == .6
    assert _band_confidence({'confidence': float('inf'), 'probabilities': [1]}) == 0


def test_high_stakes_cannot_be_downgraded_by_later_approval():
    from cognition.conscience.autonomy import AutonomyPolicy, STAKES_HIGH, STAKES_LOW, ASK
    policy = AutonomyPolicy('test', threshold=1)
    policy.record_approval('publish', stakes=STAKES_HIGH)
    policy.record_approval('publish', stakes=STAKES_LOW)
    assert policy.stakes_for('publish') == STAKES_HIGH
    assert policy.permission('publish', stakes=STAKES_LOW)['action'] == ASK


def test_snapshot_reflects_decayed_trust():
    from cognition.conscience.autonomy import AutonomyPolicy
    now = [100.]
    policy = AutonomyPolicy('test', decay_seconds=10, clock=lambda: now[0])
    policy.record_approval('read', stakes='medium')
    now[0] += 20
    assert policy.snapshot()['classes']['read']['approvals'] == 0


def test_failed_storage_keeps_a_bounded_audit_buffer(monkeypatch):
    import cognition.conscience.ledger as module
    monkeypatch.setattr(module, 'LEDGER_ENABLED', True)
    monkeypatch.setattr(module, 'LEDGER_MAX_PENDING', 3)
    ledger = module.ConscienceLedger('test')
    monkeypatch.setattr(ledger, '_db', lambda: (_ for _ in ()).throw(OSError('disk unavailable')))
    ledger.record(Verdict(decision=ESCALATE, escalation_id='abc123'), content='open question')
    for _ in range(10):
        ledger.record(Verdict(decision=ALLOW), content='routine')
        ledger.flush()
    assert len(ledger._pending) == 3
    assert ledger._pending[0][0] == 'abc123'
    assert ledger._dropped_pending == 8


@pytest.mark.parametrize('raw', [
    '<think>{"v":-1,"h":-1,"c":1}',
    '<thinking>{"v":-1,"h":-1,"c":1}',
    '{"v":0,"h":0,"c":99}',
    '{"v":2,"h":0,"c":1}',
])
def test_incomplete_thinking_or_invalid_scale_is_not_a_judgment(raw):
    assert judge._parse_verdict_json(raw) is None


def test_corrupt_trust_records_do_not_hide_valid_ones_or_grant_permission(tmp_path):
    from cognition.conscience.autonomy import AutonomyPolicy, ASK
    path = tmp_path / 'trust.json'
    path.write_text(json.dumps({
        'bad': {'stakes': 'medium', 'approvals': '99'},
        'unknown': {'approvals': 10, 'granted_at': 100},
        'good': {'stakes': 'medium', 'approvals': 2, 'granted_at': 100},
    }))
    policy = AutonomyPolicy('test', path=path, clock=lambda: 100)
    assert policy.permission('bad')['action'] == ASK
    assert policy.permission('unknown')['action'] == ASK
    assert policy.permission('good')['action'] == 'act_and_notify'


def test_zero_timestamp_approval_still_expires():
    from cognition.conscience.autonomy import AutonomyPolicy, ASK
    now = [0.]
    policy = AutonomyPolicy('test', decay_seconds=10, clock=lambda: now[0])
    policy.record_approval('read', stakes='medium')
    policy.record_approval('read', stakes='medium')
    now[0] = 20
    assert policy.permission('read')['action'] == ASK


def test_deliberation_fits_the_complete_request_in_8k(monkeypatch):
    from cognition import think
    from cognition.chat_budget import estimate_messages
    monkeypatch.setenv('LLM_CTX_SIZE', '8192')
    monkeypatch.setattr(judge, 'DELIBERATE_MAX_TOKENS', 8192)
    monkeypatch.setattr(think, '_effective_max_tokens', lambda value: value)
    seen = []
    def create(**kw):
        seen.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"v":1,"h":1,"c":0.9}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert judge.deliberate(client, 'Review this act.', '<canon>Relevant evidence</canon>', []) is not None
    assert estimate_messages(seen[0]['messages']) + seen[0]['max_tokens'] <= int(8192 * .9)
    assert '<canon>Relevant evidence</canon>' in seen[0]['messages'][-1]['content']


def test_impossible_deliberation_does_not_call_the_model(monkeypatch):
    monkeypatch.setenv('LLM_CTX_SIZE', '128')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: pytest.fail('oversized request sent'))))
    assert judge.deliberate(client, 'A situation', 'evidence', []) is None


def test_proactive_internal_circuit_failure_does_not_speak(monkeypatch):
    from cognition.conscience import hooks
    from cognition.conscience.schema import GATE_ERROR
    owner, emitted = _proactive_owner(monkeypatch)
    monkeypatch.setattr(hooks, 'gate_speak', lambda **kw: (None, Verdict(decision=CAUTION, gate=GATE_ERROR)))
    assert owner.proactive_checkin('say hello') == ''
    assert emitted == []
