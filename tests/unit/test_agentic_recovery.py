"""Execution failures, recovery and resumption must preserve real effects."""
from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agentic import agentic as ag, checkpoint, graph_engine as ge
from agentic.outcomes import tool_outcome
from agentic.registry import ToolRegistry, ToolSpec
from agentic.toolkit import coding


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('USER_SPACE_ROOT', str(tmp_path / 'users'))
    monkeypatch.setattr(checkpoint, '_db_path', lambda: tmp_path / 'checkpoints.db')
    monkeypatch.setattr(ag, '_append_step_trace', lambda *a, **k: None)


@pytest.mark.parametrize('value', [
    {'ok': False, 'error': 'bad'}, {'success': False}, {'returncode': 7},
    '[sandbox_run]\n{"ok": false, "returncode": 1}',
    '{"status": "waiting_for_approval"}', '[fetch failed: timeout]',
])
def test_structured_and_legacy_failures(value):
    assert not tool_outcome(value).ok


@pytest.mark.parametrize('value', ['[]', '[1, 2]', ['item'], {'ok': True}, 'plain result'])
def test_array_data_and_plain_text_are_successful(value):
    assert tool_outcome(value).ok


def test_handler_type_error_does_not_execute_twice(monkeypatch):
    calls = []
    def handler(value):
        calls.append(value)
        raise TypeError('failure after effect')
    monkeypatch.setattr(ag.registry, 'get', lambda _: ToolSpec('custom', '', handler=handler))
    with pytest.raises(TypeError, match='after effect'):
        ag._dispatch_tool_impl('custom', {'value': 3})
    assert calls == [3]


def test_legacy_dict_handler_is_selected_before_execution(monkeypatch):
    handler = lambda args: args['value']
    monkeypatch.setattr(ag.registry, 'get', lambda _: ToolSpec('custom', '', handler=handler))
    assert ag._dispatch_tool_impl('custom', {'value': 3}) == 3


def test_mutation_barriers_allow_retest_without_repeating_external_calls(monkeypatch):
    events = []
    def execute(name, args, state, **kwargs):
        events.append(name)
        return ag.ToolResult(True, name, args, 'done')
    monkeypatch.setattr(ag, 'execute_tool_with_policy', execute)
    seen = set()
    calls = [('1', 'sandbox_run', {'relative_path': 'x.py'}),
             ('2', 'repo_write_file', {'relative_path': 'x.py', 'content': 'fixed'}),
             ('3', 'sandbox_run', {'relative_path': 'x.py'}),
             ('4', 'post_to_social', {'text': 'x'}),
             ('5', 'post_to_social', {'text': 'x'})]
    results = list(ag._run_tool_batch(calls, ag.TaskState('test'), ag.AgentContext(), None, [], seen))
    assert events == ['sandbox_run', 'repo_write_file', 'sandbox_run', 'post_to_social']
    assert results[-1][-1].error_type == 'repeated_tool_call'


def test_corrected_local_failure_is_resolved():
    state = ag.TaskState('fix script')
    state.record(ag.ToolResult(False, 'sandbox_run', {'relative_path': 'x.py'}, 'failed'))
    state.record(ag.ToolResult(False, 'post_to_social', {}, 'uncertain'))
    state.record(ag.ToolResult(True, 'sandbox_run', {'relative_path': 'x.py'}, 'passed'))
    assert [f.tool for f in state.failures] == ['post_to_social']


def graph(*nodes):
    return ge.PlanGraph('test', 'test', 'test', tuple(nodes))


def test_structured_failure_uses_checked_fallback(monkeypatch):
    monkeypatch.setattr(ge, '_tool_map', lambda: {
        'fail': lambda: {'ok': False}, 'backup': lambda: '[backup]\n{"ok":true}',
        'next': lambda: 'done'})
    result = ge.execute_graph(graph(ge.PlanNode('a', 'fail', {}, fallback_to='b'),
        ge.PlanNode('b', 'backup', {}), ge.PlanNode('c', 'next', {}, depends_on=('a',))))
    assert all(r.ok for r in result.results)
    assert result.results[0].error_type == 'recovered_by_fallback'
    assert result.results[-1].content == 'done'


def test_failed_fallback_is_not_a_rescue(monkeypatch):
    monkeypatch.setattr(ge, '_tool_map', lambda: {'fail': lambda: {'ok': False}})
    result = ge.execute_graph(graph(ge.PlanNode('a', 'fail', {}, fallback_to='b'),
        ge.PlanNode('b', 'fail', {})))
    assert not any(r.ok for r in result.results)


def test_recovered_fallback_respects_interrupt(monkeypatch):
    effect = Mock(return_value='effect')
    monkeypatch.setattr(ge, '_tool_map', lambda: {
        'fail': lambda: {'ok': False}, 'backup': lambda: 'review first', 'effect': effect})
    result = ge.execute_graph(graph(ge.PlanNode('a', 'fail', {}, fallback_to='b'),
        ge.PlanNode('b', 'backup', {}, interrupt=True),
        ge.PlanNode('c', 'effect', {}, depends_on=('a',))))
    assert result.interrupted and result.interrupted_at == 'b'
    effect.assert_not_called()


def test_interrupt_does_not_queue_unstarted_independent_nodes(monkeypatch):
    effect = Mock(return_value='effect')
    monkeypatch.setattr(ge, 'GRAPH_MAX_WORKERS', 1)
    monkeypatch.setattr(ge, '_tool_map', lambda: {'pause': lambda: 'review first', 'effect': effect})
    result = ge.execute_graph(graph(ge.PlanNode('a', 'pause', {}, interrupt=True),
                                    ge.PlanNode('b', 'effect', {})))
    assert result.interrupted
    effect.assert_not_called()


def test_timeout_never_retries_or_runs_fallback(monkeypatch):
    handler = Mock(side_effect=TimeoutError('uncertain outcome'))
    backup = Mock()
    monkeypatch.setattr(ge, '_tool_map', lambda: {'slow': handler, 'backup': backup})
    result = ge.execute_graph(graph(ge.PlanNode('a', 'slow', {}, max_retries=3, fallback_to='b'),
        ge.PlanNode('b', 'backup', {})))
    assert result.results[0].error_type == 'timeout'
    assert handler.call_count == 1
    backup.assert_not_called()


def test_cycle_rejected_before_root_effect(monkeypatch):
    handler = Mock()
    monkeypatch.setattr(ge, '_tool_map', lambda: {'effect': handler})
    result = ge.execute_graph(graph(ge.PlanNode('root', 'effect', {}),
        ge.PlanNode('a', 'effect', {}, depends_on=('b',)),
        ge.PlanNode('b', 'effect', {}, depends_on=('a',))))
    assert result.results[0].error_type == 'dependency_error'
    handler.assert_not_called()


@pytest.mark.parametrize('condition', [
    {'node': 'a', 'typo': True}, {'node': 'a', 'not': {'gt': float('nan')}},
    {'node': 'a', 'not': {'matches': '['}},
])
def test_invalid_conditions_cannot_enable_effects(monkeypatch, condition):
    effect = Mock(return_value='changed')
    monkeypatch.setattr(ge, '_tool_map', lambda: {'seed': lambda: '10', 'effect': effect})
    result = ge.execute_graph(graph(ge.PlanNode('a', 'seed', {}),
        ge.PlanNode('b', 'effect', {}, depends_on=('a',), run_if=condition),
        ge.PlanNode('c', 'effect', {}, depends_on=('b',))))
    assert result.results[-1].error_type == 'condition_skipped'
    effect.assert_not_called()


def test_interrupt_records_parallel_siblings_before_resume(monkeypatch):
    barrier = threading.Barrier(2)
    calls = []
    def sibling(name):
        barrier.wait(timeout=5)
        calls.append(name)
        return name
    monkeypatch.setattr(ge, 'GRAPH_MAX_WORKERS', 2)
    monkeypatch.setattr(ge, '_tool_map', lambda: {
        'first': lambda: sibling('first'), 'second': lambda: sibling('second'),
        'after': lambda state: state.get('_resume')})
    plan = graph(ge.PlanNode('a', 'first', {}, interrupt=True), ge.PlanNode('b', 'second', {}),
                 ge.PlanNode('c', 'after', {}, depends_on=('a', 'b')))
    paused = ge.execute_graph(plan, run_id='paused')
    assert paused.interrupted
    assert {r.node_id for r in paused.results} == {'a', 'b'}
    resumed = ge.resume_graph(plan, 'paused', resume_input='continue')
    assert sorted(calls) == ['first', 'second']
    assert resumed.results[-1].content == 'continue'
    with pytest.raises(ValueError, match='no saved'):
        ge.resume_graph(plan, 'unknown')


def test_changed_graph_cannot_resume_saved_effects(monkeypatch):
    effect = Mock(return_value='done')
    monkeypatch.setattr(ge, '_tool_map', lambda: {'effect': effect})
    ge.execute_graph(graph(ge.PlanNode('a', 'effect', {}, interrupt=True)), run_id='pause')
    with pytest.raises(ValueError, match='different graph'):
        ge.resume_graph(graph(ge.PlanNode('a', 'effect', {'changed': True}, interrupt=True)), 'pause')
    assert effect.call_count == 1


def test_stream_yields_before_graph_completion(monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow():
        started.set()
        assert release.wait(5)
        return 'second'
    monkeypatch.setattr(ge, '_tool_map', lambda: {'fast': lambda: 'first', 'slow': slow})
    stream = ge.execute_graph_stream(graph(ge.PlanNode('a', 'fast', {}),
                                          ge.PlanNode('b', 'slow', {}, depends_on=('a',))))
    try:
        assert next(stream).node_id == 'a'
        assert started.wait(5)
        release.set()
        remaining = list(stream)
        assert remaining[-1].results[-1].content == 'second'
    finally:
        release.set()
        stream.close()


def test_parallel_usage_and_reducers_stay_per_node(monkeypatch):
    barrier = threading.Barrier(2)
    def work(tokens, state):
        state.set('_usage', {'total_tokens': tokens})
        barrier.wait(5)
        for _ in range(100):
            state.set('count', 1)
        return 'done'
    monkeypatch.setattr(ge, 'GRAPH_MAX_WORKERS', 2)
    monkeypatch.setattr(ge, '_tool_map', lambda: {'work': work})
    plan = ge.PlanGraph('usage', 'usage', '', (ge.PlanNode('a', 'work', {'tokens': 3}),
        ge.PlanNode('b', 'work', {'tokens': 7})), reducers={'count': 'add'})
    result = ge.execute_graph(plan)
    assert {r.node_id: r.usage['total_tokens'] for r in result.results} == {'a': 3, 'b': 7}
    assert result.final_state['count'] == 200


def test_registry_updates_are_visible_to_cached_graph(monkeypatch):
    registry = ToolRegistry()
    monkeypatch.setattr('agentic.registry.registry', registry)
    monkeypatch.setattr(ge, '_TOOL_MAP_CACHE', None)
    monkeypatch.setattr(ge, '_build_tool_map', registry.get_graph_tool_map)
    assert ge._tool_map() == {}
    handler = lambda: 'late'
    registry.register('late', '', handler=handler, graph=True)
    assert ge._tool_map()['late'] is handler
    registry.register('late', '', handler=handler, graph=False)
    assert 'late' not in ge._tool_map()


def test_interactive_gated_graph_defers_before_any_effect(monkeypatch):
    plan = graph(ge.PlanNode('a', 'effect', {}), ge.PlanNode('b', 'gate', {}, needs_approval=True))
    monkeypatch.setattr(ge, 'plan_from_master', lambda *a, **k: plan)
    execute = Mock()
    monkeypatch.setattr(ge, 'execute_graph', execute)
    assert ge.run_schema_agent('test', interactive_approvals=True) is None
    execute.assert_not_called()


def test_llm_goal_verification_cannot_assume_success(monkeypatch):
    assert ge.goal_verification('evidence').startswith('PARTIAL:')
    monkeypatch.setattr('agentic.toolkit.synthesize.synthesize_report', lambda **k: 'NOT ACHIEVED')
    assert ge.goal_verification('evidence', client=object(), model='test').startswith('PARTIAL:')


def test_verifier_outage_preserves_detected_problems(monkeypatch):
    monkeypatch.setattr(ag, 'AGENT_VERIFY_LLM_MODE', 'on')
    verdict = NS(content='missing evidence')
    monkeypatch.setattr(ag, '_POST_ANSWER_GUARDRAILS', [lambda *a: verdict])
    owner = NS(_client=None, _llm_model='test')
    result = ag._verify_final_answer(owner, 'request', 'claimed done', ag.TaskState('goal'))
    assert not result.ok
    assert 'missing evidence' in result.feedback


@pytest.mark.parametrize('interrupted', [False, True])
def test_hybrid_cannot_replay_graph_actions(monkeypatch, interrupted):
    plan = graph(ge.PlanNode('sent', 'post_to_social', {}), ge.PlanNode('failed', 'repo_read_file', {}))
    result = ge.GraphRunResult(plan, (
        ge.NodeResult('sent', 'post_to_social', True, 'sent'),
        ge.NodeResult('failed', 'repo_read_file', False, 'read failed'),
    ), 'Partial result: sent; read failed.', interrupted=interrupted)
    monkeypatch.setattr(ag, 'AGENT_EXECUTOR_MODE', 'hybrid')
    monkeypatch.setattr(ag, 'AGENT_VERIFY_FINAL', False)
    monkeypatch.setattr(ag, '_maybe_resume_approval', lambda *a, **k: None)
    monkeypatch.setattr(ag, '_owner_embedder', lambda _: None)
    monkeypatch.setattr(ag, 'match_capabilities', lambda *a, **k: [])
    monkeypatch.setattr(ge, 'run_schema_agent', lambda *a, **k: result)
    monkeypatch.setattr(ag, 'CONTEXT_POOL', NS(submit=Mock()))
    react = Mock(side_effect=AssertionError('would replay actions'))
    monkeypatch.setattr(ag, '_stream_agent_message', react)
    owner = NS(user_id='owner', _client=None, _llm_model='mock', _history=[],
               _history_lock=threading.Lock(), _emit=Mock(), _store_async=Mock())
    assert ag.run_agentic_chat(owner, 'do task') == result.final_answer
    react.assert_not_called()
    if interrupted:
        ag.CONTEXT_POOL.submit.assert_not_called()


def test_failed_node_is_never_memoized(monkeypatch):
    monkeypatch.setattr(ge, '_tool_map', lambda: {'fail': lambda: '{"ok":false}'})
    monkeypatch.setattr(ge.graph_memo, '_memo_key', lambda *a: 'key')
    monkeypatch.setattr(ge.graph_memo, '_memo_get', lambda *a: None)
    save = Mock()
    monkeypatch.setattr(ge.graph_memo, '_memo_put', save)
    assert not ge._run_node(ge.PlanNode('a', 'fail', {}), '', {}).ok
    save.assert_not_called()


def test_trace_retention_removes_oldest_files(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(ag, 'AGENT_TRACE_MAX_FILES', 2)
    for index in range(4):
        path = tmp_path / f'{index}.jsonl'
        path.touch()
        os.utime(path, (index, index))
    ag._trim_trace_dir(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['2.jsonl', '3.jsonl']


def test_oversized_verifier_never_sends_request(monkeypatch):
    monkeypatch.setattr(ag, 'AGENT_VERIFY_LLM_MODE', 'on')
    monkeypatch.setattr(ag, 'LLM_CTX_SIZE', 8192)
    client = Mock()
    ag._verify_final_answer(NS(_client=client, _llm_model='test'), 'request', 'x' * 100000, ag.TaskState('goal'))
    client.chat.completions.create.assert_not_called()


def test_bounded_subprocess_replaces_invalid_bytes_and_drains_both_pipes():
    proc = coding._run_bounded([sys.executable, '-c',
        "import os; os.write(1, b'x'*100000+b'\\xffend'); os.write(2, b'y'*100000+b'err')"],
        output_chars=17)
    assert len(proc.stdout) == len(proc.stderr) == 17
    assert proc.stdout.endswith('\ufffdend') and proc.stderr.endswith('err')


def test_timeout_kills_child_even_after_parent_exits(tmp_path):
    pidfile = tmp_path / 'child.pid'
    source = ("import os,time; pid=os.fork(); "
              f"open({str(pidfile)!r}, 'w').write(str(pid)) if pid else None; "
              "os._exit(0) if pid else time.sleep(60)")
    with pytest.raises(subprocess.TimeoutExpired):
        coding._run_bounded([sys.executable, '-c', source], timeout=.5)
    pid = int(pidfile.read_text())
    # A reaped child disappears; a not-yet-reaped killed child may be a zombie.
    from pathlib import Path
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists():
        for _ in range(100):
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            threading.Event().wait(.01)
        assert not stat.exists() or stat.read_text().split()[2] == 'Z'


def test_checkpoint_migrates_old_schema_and_hides_state_sentinel(tmp_path):
    import sqlite3
    conn = sqlite3.connect(tmp_path / 'checkpoints.db')
    conn.execute('CREATE TABLE node_checkpoints (run_id TEXT, node_id TEXT, tool TEXT, '
                 'ok INTEGER, content TEXT, args TEXT, error_type TEXT, seq INTEGER, '
                 'PRIMARY KEY(run_id,node_id))')
    conn.close()
    checkpoint.save_graph_state('test', {'value': 1})
    checkpoint.save_node_result('test', 1, ge.NodeResult('a', 'tool', True, 'done'))
    assert [r.node_id for r in checkpoint.load_checkpoint('test', ge.NodeResult)] == ['a']
    assert checkpoint.load_graph_state('test') == {'value': 1}


def test_skill_vectors_isolate_paths_providers_and_metadata(tmp_path, monkeypatch):
    import numpy as np
    from agentic import skills
    monkeypatch.setattr(skills, '_skill_embed_cache', {})
    class Embedder:
        def __init__(self, vector):
            self.vector, self.calls = vector, 0
        def embed_query(self, text, instruct=''):
            assert instruct == skills._SKILL_INSTRUCT
            self.calls += 1
            return self.vector
    a, b = Embedder([1., 0.]), Embedder([0., 1.])
    doc = skills.SkillDoc('same', 'Same', tmp_path / 'a', summary='a')
    other = skills.SkillDoc('same', 'Same', tmp_path / 'b', summary='b')
    first = skills._get_skill_embedding(doc, a)
    first[:] = 0  # callers cannot mutate the cached vector
    np.testing.assert_array_equal(skills._get_skill_embedding(doc, a), [1, 0])
    np.testing.assert_array_equal(skills._get_skill_embedding(doc, b), [0, 1])
    skills._get_skill_embedding(other, a)
    assert a.calls == 2 and b.calls == 1


def test_concurrent_skill_observations_preserve_runs_and_arguments(tmp_path, monkeypatch):
    import json
    from agentic import skill_learning as sl
    monkeypatch.setattr(sl, 'skill_proposal_dir', lambda user_id=None: tmp_path)
    monkeypatch.setattr(sl, 'skillset_staging_dir', lambda user_id=None: tmp_path)
    steps = [{'tool': 'repo_read_file', 'args': {'relative_path': 'work/g/main.py'}},
             {'tool': 'save_note', 'args': {'title': 'observed', 'content': 'actual'}}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: sl.propose_skill_from_run(
            'same goal', steps, 'done', verified_ok=True, score=.9), range(12)))
    assert len(sl._observed_runs(paths[0].read_text())) == 12
    staged = sl.promote_skill_proposal('same-goal', dry_run=False, validated=True)
    nodes = json.loads(staged.read_text())['nodes']
    assert nodes[0]['args'] == steps[0]['args']
    assert nodes[1]['args'] == steps[1]['args']
    assert nodes[1]['depends_on'] == [nodes[0]['id']]
    with pytest.raises(ValueError, match='slug'):
        sl.promote_skill_proposal('../escape', dry_run=False)
