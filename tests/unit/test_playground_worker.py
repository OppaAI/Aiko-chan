"""Exercise actual worker writes/execution and request assembly at 8K."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agentic import agentic as ag
from agentic.toolkit import coding, self_improve
from system.playground import playground_work_path, worker_skill_text


def payload(text):
    return json.loads(text.split(']', 1)[1])


@pytest.fixture
def playground(tmp_path, monkeypatch):
    root = tmp_path / 'Playground'
    (root / 'sandbox').mkdir(parents=True)
    (root / 'GOALS.md').write_text('# Goals\nBuild a greeting.\n')
    # A real subprocess wrapper with the Playground runner's JSON contract.
    # The marker proves scripts are never executed directly by sandbox_run.
    (root / 'sandbox/run.py').write_text('''import json, pathlib, subprocess, sys
pathlib.Path("runner-used").touch()
sep = sys.argv.index("--")
r = subprocess.run([sys.executable, sys.argv[1], *sys.argv[sep+1:]],
                   cwd=pathlib.Path(sys.argv[1]).parent, capture_output=True, text=True)
print(json.dumps(dict(ok=r.returncode == 0, returncode=r.returncode,
                      stdout=r.stdout, stderr=r.stderr)))
''')
    monkeypatch.setenv('AIKO_PLAYGROUND_DIR', str(root))
    monkeypatch.setenv('USER_SPACE_ROOT', str(tmp_path / 'state'))
    monkeypatch.setattr(ag, '_append_step_trace', lambda *a, **k: None)
    monkeypatch.setattr(ag, '_preference_requires_approval', lambda name: False)
    return root


def context(root):
    return ag.AgentContext(user_id='worker-test', run_id='test', worker_root=root)


def execute(root, name, args):
    return ag.execute_tool_with_policy(name, args, ag.TaskState(goal='build'), ctx=context(root), guards=[])


def test_configured_checkout_reads_and_searches_absolute_paths(playground):
    assert 'Build a greeting' in self_improve.repo_read_file(str(playground / 'GOALS.md'))
    found = payload(self_improve.repo_search_text('greeting', prefix=str(playground)))
    assert found['count'] == 1
    assert found['matches'][0]['file'] == str(playground / 'GOALS.md')
    tree = payload(self_improve.repo_file_tree(prefix=str(playground)))
    assert str(playground / 'GOALS.md') in tree['files']


def test_worker_create_patch_and_execute_without_interactive_approval(playground, monkeypatch):
    gate = Mock(return_value=None)
    monkeypatch.setattr(ag, '_gate_tool_call', gate)
    path = 'work/greeting/main.py'
    created = execute(playground, 'repo_write_file', {'relative_path': path, 'content': 'print("hello")\n'})
    assert created.ok, created.content
    changed = execute(playground, 'code_apply_patch', {'relative_path': str(playground / path),
                      'old_text': 'hello', 'new_text': 'hi'})
    assert changed.ok, changed.content
    ran = execute(playground, 'sandbox_run', {'relative_path': path})
    assert ran.ok, ran.content
    assert payload(ran.content)['stdout'] == 'hi\n'
    assert (playground / 'runner-used').exists()
    assert gate.call_count == 3  # Scoped authorization does not bypass conscience.


@pytest.mark.parametrize('path', ['GOALS.md', 'sandbox/run.py', 'work/file.py',
    '../outside.py', 'work/.git/file.py', 'work/greeting/.git/config.py'])
def test_worker_rejects_writes_outside_goal_work(playground, monkeypatch, path):
    handler = Mock()
    monkeypatch.setattr(ag, '_dispatch_tool_impl', handler)
    result = execute(playground, 'repo_write_file', {'relative_path': path, 'content': 'bad'})
    assert result.error_type == 'worker_scope'
    handler.assert_not_called()


def test_worker_rejects_symlink_escape_and_kill_switch(playground, tmp_path, monkeypatch):
    (playground / 'work').mkdir()
    (playground / 'work/escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        playground_work_path('work/escape/out.py')
    (playground / 'loop').mkdir()
    (playground / 'loop/disabled').touch()
    result = execute(playground, 'repo_write_file', {'relative_path': 'work/g/main.py', 'content': 'x=1'})
    assert result.error_type == 'worker_scope'
    assert not (playground / 'work/g/main.py').exists()


def test_worker_cannot_invoke_shell_or_bypass_conscience(playground, monkeypatch):
    assert execute(playground, 'shell_run', {'command': 'pwd'}).error_type == 'worker_scope'
    monkeypatch.setattr(ag, '_gate_tool_call', lambda *a, **k: {'decision': 'refuse'})
    result = execute(playground, 'repo_write_file', {'relative_path': 'work/g/main.py', 'content': 'x=1'})
    assert result.error_type == 'conscience_refuse'
    assert not (playground / 'work/g/main.py').exists()


def test_interactive_code_writes_still_require_approval(playground, monkeypatch):
    monkeypatch.setattr(ag, '_persist_pending_approval', Mock())
    result = ag.execute_tool_with_policy('repo_write_file', {
        'relative_path': str(playground / 'work/g/main.py'), 'content': 'x=1',
    }, ag.TaskState(goal='build'), ctx=ag.AgentContext(user_id='test'), guards=[])
    assert result.error_type == 'needs_approval'


def test_sandbox_reports_script_failure_even_when_wrapper_exits_zero(playground):
    path = playground / 'work/g/fail.py'
    path.parent.mkdir(parents=True)
    path.write_text('raise RuntimeError("failed goal test")\n')
    result = payload(coding.sandbox_run(str(path)))
    assert result['ok'] is False
    assert result['returncode'] != 0
    assert 'failed goal test' in result['stderr']


def test_worker_loop_fits_8k_without_persona_memory_or_chat_history(playground, monkeypatch):
    monkeypatch.setattr(ag, 'LLM_CTX_SIZE', 8192)
    monkeypatch.setattr(ag, 'AGENT_REACT_BACKEND', 'openai')
    monkeypatch.setattr(ag, '_effective_max_tokens_safe', lambda: 512)
    monkeypatch.setattr(ag, '_gate_tool_call', lambda *a, **k: None)
    monkeypatch.setattr(ag.experience, 'record_experience', lambda *a, **k: None)
    monkeypatch.setattr(ag.skill_learning, 'propose_skill_from_run', lambda *a, **k: None)
    owner = NS(_client=None, _llm_model='test', user_id='worker-test', _memorize=None,
               _history=[{'role': 'user', 'content': 'PERSONAL_HISTORY' * 2000}],
               _history_lock=threading.Lock(), _emit=Mock(), _store_async=Mock(),
               _persona_core=Mock(side_effect=AssertionError('persona fetched')),
               _fetch_memory_and_knowledge=Mock(side_effect=AssertionError('memory fetched')))
    steps = iter([
        ('repo_read_file', {'relative_path': str(playground / 'GOALS.md')}),
        ('repo_write_file', {'relative_path': 'work/g/PLAN.md', 'content': 'Print hi, run and check output.'}),
        ('repo_write_file', {'relative_path': 'work/g/main.py', 'content': 'print("hi")\n'}),
        ('sandbox_run', {'relative_path': 'work/g/main.py'}),
        ('repo_write_file', {'relative_path': 'work/g/CHECKPOINT.md', 'content': 'status: done\nverified: hi'}),
        ('final_answer', {'answer': 'Built and ran greeting.'}),
    ])
    requests = []

    def stream(_owner, messages, tools, max_tokens):
        requests.append(json.loads(json.dumps(messages)))
        assert ag._estimate_messages_tokens(messages, tools) + max_tokens <= 8192
        assert 'PERSONAL_HISTORY' not in json.dumps(messages)
        assert messages[0]['content'] == ag.WORKER_SYSTEM_PROMPT
        assert {s['function']['name'] for s in tools} == ag.WORKER_TOOLS
        name, args = next(steps)
        call = NS(index=0, id=f'call-{len(requests)}', function=NS(name=name, arguments=json.dumps(args)))
        return [NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[call]))])]

    monkeypatch.setattr(ag, '_create_agent_stream', stream)
    result = ag.run_agentic_chat(owner, worker_skill_text(), worker_mode=True, lean_context=True)
    assert result == 'Built and ran greeting.'
    assert len(requests) == 6
    assert (playground / 'work/g/main.py').read_text() == 'print("hi")\n'
    assert (playground / 'work/g/CHECKPOINT.md').exists()
    owner._persona_core.assert_not_called()
    owner._fetch_memory_and_knowledge.assert_not_called()
    observations = [m['content'] for m in requests[-1] if m['role'] == 'tool']
    assert any('hi' in s and 'stdout' in s for s in observations)
    assert all('needs_approval' not in s for s in observations)
