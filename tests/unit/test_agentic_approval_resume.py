"""Regression tests for durable, one-shot normal tool approvals."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from agentic import agentic


@pytest.fixture
def pending(monkeypatch, tmp_path):
    monkeypatch.setattr(agentic, 'user_state_dir', lambda uid=None: tmp_path)
    monkeypatch.setattr(agentic, '_agent_context', lambda owner=None, run_id=None:
        agentic.AgentContext(user_id='alice', workspace=tmp_path, run_id=run_id))
    ctx = agentic.AgentContext(user_id='alice', run_id='r2')
    agentic._persist_pending_approval(ctx, 'publish', {'value': 1}, agentic.TaskState(goal='approved action'))
    return agentic._pending_approval_path(ctx)


def test_failed_result_cannot_replay_an_approved_side_effect(pending, monkeypatch):
    calls = []
    def execute(name, args, state, **kw):
        calls.append(args)
        return agentic.ToolResult(ok=False, tool=name, args=args, content='delivery outcome unknown', error_type='tool_error')
    monkeypatch.setattr(agentic, 'execute_tool_with_policy', execute)
    assert agentic._maybe_resume_approval(None, 'approve r2') is not None
    assert agentic._maybe_resume_approval(None, 'approve r2') is None
    assert calls == [{'value': 1}]
    assert not pending.exists()


def test_concurrent_approval_claim_is_consumed_before_dispatch(pending, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def execute(name, args, state, **kw):
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return agentic.ToolResult(ok=True, tool=name, args=args, content='done')
    monkeypatch.setattr(agentic, 'execute_tool_with_policy', execute)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(agentic._maybe_resume_approval, None, 'approve r2')
        try:
            assert entered.wait(3)
            second = pool.submit(agentic._maybe_resume_approval, None, 'approve r2')
            assert second.result(timeout=3) is None
        finally:
            release.set()
        assert first.result(timeout=3) is not None
    assert len(calls) == 1


def test_success_cannot_delete_a_new_pause_created_by_the_handler(pending, monkeypatch):
    def execute(name, args, state, **kw):
        agentic._persist_pending_approval(kw['ctx'], 'next_step', {}, state)
        return agentic.ToolResult(ok=True, tool=name, args=args, content='step done')
    monkeypatch.setattr(agentic, 'execute_tool_with_policy', execute)
    agentic._maybe_resume_approval(None, 'approve r2')
    assert json.loads(pending.read_text())['tool'] == 'next_step'


@pytest.mark.parametrize('content', ['[]', '{broken', '{"run_id":"other","args":{}}'])
def test_invalid_saved_approval_never_executes(pending, monkeypatch, content):
    pending.write_text(content)
    monkeypatch.setattr(agentic, 'execute_tool_with_policy', lambda *a, **kw: pytest.fail('invalid action executed'))
    reply = agentic._maybe_resume_approval(None, 'approve r2')
    assert 'invalid' in reply
    assert not pending.exists()


def test_approval_without_run_id_gets_a_resumable_unique_id(monkeypatch, tmp_path):
    monkeypatch.setattr(agentic, 'user_state_dir', lambda uid=None: tmp_path)
    monkeypatch.setattr(agentic, '_preference_requires_approval', lambda name: False)
    monkeypatch.setattr(agentic, '_validate_args', lambda *a: None)
    monkeypatch.setattr(agentic.registry, 'get', lambda name: SimpleNamespace(needs_approval=True))
    waits = [agentic.execute_tool_with_policy('test_action', {}, agentic.TaskState(goal='test'),
             ctx=agentic.AgentContext(user_id='alice'), guards=[]) for _ in range(2)]
    ids = [json.loads(result.content)['run_id'] for result in waits]
    assert len(set(ids)) == 2
    assert all(agentic._looks_like_approval_reply(f'approve {run_id}') for run_id in ids)
    assert all((tmp_path / 'agentic' / 'pending_approvals' / f'{run_id}.json').exists() for run_id in ids)


def test_failed_pause_persistence_returns_honest_nonexecution(monkeypatch):
    monkeypatch.setattr(agentic, '_preference_requires_approval', lambda name: False)
    monkeypatch.setattr(agentic, '_validate_args', lambda *a: None)
    monkeypatch.setattr(agentic.registry, 'get', lambda name: SimpleNamespace(needs_approval=True))
    monkeypatch.setattr(agentic, '_persist_pending_approval', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    result = agentic.execute_tool_with_policy('test_action', {}, agentic.TaskState(goal='test'),
                                             ctx=agentic.AgentContext(user_id='alice'), guards=[])
    assert result.error_type == 'approval_storage'
    assert 'Nothing was executed' in result.content


def test_saved_approval_identity_must_match_current_user(pending, monkeypatch):
    data = json.loads(pending.read_text())
    data['user_id'] = 'someone-else'
    pending.write_text(json.dumps(data))
    monkeypatch.setattr(agentic, 'execute_tool_with_policy', lambda *a, **kw: pytest.fail('foreign approval executed'))
    assert 'invalid' in agentic._maybe_resume_approval(None, 'approve r2')
