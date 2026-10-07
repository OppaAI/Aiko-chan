"""Coding tool boundaries and the playbook's single correction budget."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from agentic import graph_engine as graph
from agentic.registry import registry
from agentic.toolkit import coding
from agentic.toolkit.common import json_block
from agentic.toolkit.needle_subagent import SUBAGENT_TOOLS


def payload(text):
    return json.loads(text.split(']', 1)[1])


@pytest.mark.parametrize('name', ['sandbox_run', 'code_apply_correction'])
def test_execution_requires_approval(name, monkeypatch):
    handler = Mock()
    monkeypatch.setattr(graph, '_tool_map', lambda: {name: handler})
    result = graph._run_node(graph.PlanNode('execute', name, {}), '', {})
    assert registry.get(name).needs_approval
    assert result.error_type == 'needs_approval'
    handler.assert_not_called()


def test_subagent_tool_boundary():
    assert SUBAGENT_TOOLS == {
        'repo_read_file', 'repo_search_text', 'codebase_search',
        'code_run_tests', 'adaptive_search',
    }


@pytest.mark.parametrize('kind', ['relative', 'absolute', 'tmp', 'tmp_child', 'missing', 'file'])
def test_shell_allowed_cwd_and_fallback(kind, tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    child = repo / 'child'
    child.mkdir()
    file = repo / 'file'
    file.touch()
    paths = {
        'relative': ('child', child), 'absolute': (str(child), child),
        'tmp': ('/tmp', Path('/tmp')), 'tmp_child': (str(tmp_path), tmp_path),
        'missing': ('missing', repo), 'file': ('file', repo),
    }
    cwd, expected = paths[kind]
    monkeypatch.setattr(coding, 'REPO_ROOT', repo)
    popen = Mock(wraps=coding.subprocess.Popen)
    monkeypatch.setattr(coding.subprocess, 'Popen', popen)
    assert payload(coding.shell_run('pwd', cwd=cwd))['ok']
    assert popen.call_args.kwargs['cwd'] == str(expected)


@pytest.mark.parametrize('cwd', ['/etc', '/tmp-other', '/etc/nonexistent', '../../../../../../etc', 'escape'])
def test_shell_rejects_escape_before_execution(cwd, tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'escape').symlink_to('/etc', target_is_directory=True)
    monkeypatch.setattr(coding, 'REPO_ROOT', repo)
    popen = Mock()
    monkeypatch.setattr(coding.subprocess, 'Popen', popen)
    result = payload(coding.shell_run('pwd', cwd=cwd))
    assert result['ok'] is False
    assert 'outside' in result['error']
    popen.assert_not_called()


def test_correction_uses_patch_checks_and_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(coding, 'REPO_ROOT', tmp_path)
    file = tmp_path / 'a.py'
    file.write_text('x = 1\n')
    correction = json_block('needle_subagent', {'ok': True, 'summary': json.dumps({
        'relative_path': 'a.py', 'old_text': 'x = 1', 'new_text': 'x = 2',
    })})
    assert payload(coding.code_apply_correction(correction))['ok']
    assert file.read_text() == 'x = 2\n'
    assert (tmp_path / 'a.py.bak').read_text() == 'x = 1\n'


@pytest.mark.parametrize('result', [
    {'ok': False, 'summary': '{}'}, {'ok': True, 'summary': '{}', 'truncated': True},
    {'ok': True, 'summary': '{"relative_path":'}, {'ok': True, 'summary': '{}'},
    {'ok': True, 'summary': '{"relative_path":"a.py","old_text":null,"new_text":"x"}'},
    {'ok': True, 'summary': '[]'}, [],
])
def test_invalid_triage_never_writes(result, monkeypatch):
    apply = Mock()
    monkeypatch.setattr(coding, 'code_apply_patch', apply)
    assert not payload(coding.code_apply_correction(json.dumps(result)))['ok']
    apply.assert_not_called()


@pytest.mark.parametrize('run_ok,verify_ok', [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize('correction_ok', [True, False])
def test_playbook_routes_and_bounds_corrections(run_ok, verify_ok, correction_ok, monkeypatch):
    playbook = json.loads((Path(__file__).parents[2] / 'agentic/playbooks/coding_agent_loop.json').read_text())
    nodes = tuple(graph.PlanNode(**{k: v for k, v in n.items() if k != 'note'}) for n in playbook['nodes'])
    plan = graph.PlanGraph('coding_agent_loop', 'Coding loop', 'fix a.py', nodes,
                           _extras={'$state:target_file': 'a.py', 'max_workers': 1})
    calls = []

    def run(node, prompt, results, *args):
        resolved = graph._substitute(node.args, prompt, results, plan._extras)
        calls.append(node.id)
        ok = True
        if node.id == 'run_code':
            ok = run_ok
        elif node.id == 'verify':
            ok = verify_ok
        elif node.id.startswith('correct_'):
            ok = correction_ok
            assert 'summary' in resolved['correction']
        elif node.id.startswith(('rerun_', 'reverify_')):
            ok = False  # Persistent failures must stop after one correction.
        elif node.id.startswith('triage_'):
            assert '"ok": false' in resolved['context']
            return graph.NodeResult(node.id, node.tool, True, json_block('needle_subagent', {
                'ok': True, 'summary': '{"relative_path":"a.py","old_text":"a","new_text":"b"}',
            }))
        return graph.NodeResult(node.id, node.tool, True, json_block(node.tool, {'ok': ok}))

    monkeypatch.setattr(graph, '_run_node', run)
    monkeypatch.setattr(graph, '_synthesize_without_llm', lambda *args: 'done')
    result = graph._execute_graph_inner(plan)
    assert all(r.error_type != 'dependency_error' for r in result.results)
    base = ['init_run', 'explore', 'plan', 'write_code', 'run_code', 'verify']
    if run_ok and verify_ok:
        assert calls == base
    else:
        suffix = 'failures' if not run_ok else 'verify_failures'
        expected = base + ['triage_' + suffix, 'correct_' + suffix]
        if correction_ok:
            expected += ['rerun_' + suffix, 'reverify_' + suffix]
        assert calls == expected
