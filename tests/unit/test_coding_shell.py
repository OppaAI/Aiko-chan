"""Shell output stays bounded while both pipes are drained."""
from __future__ import annotations

import json
import shlex
import sys
import time
import tracemalloc

import pytest

from agentic.toolkit import coding


def python_command(source):
    return f"exec {shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def payload(text):
    return json.loads(text.split(']', 1)[1])


@pytest.mark.parametrize('stream', ['stdout', 'stderr', 'both'])
def test_shell_large_output_tails(stream, monkeypatch):
    monkeypatch.setattr(coding, 'SHELL_OUTPUT_CHARS', 37)
    source = "import sys\n"
    for name in ('stdout', 'stderr'):
        if stream in (name, 'both'):
            source += f"sys.{name}.write('x' * 200000 + '{name}-end'); sys.{name}.flush()\n"
    source += "sys.exit(7)"

    result = payload(coding.shell_run(python_command(source)))

    expected = {
        name: ('x' * 37 + f'{name}-end')[-37:] if stream in (name, 'both') else ''
        for name in ('stdout', 'stderr')
    }
    assert result == {'ok': False, 'returncode': 7, 'readonly': False, **expected}


def test_shell_empty_output_and_readonly():
    assert payload(coding.shell_run('echo -n')) == {
        'ok': True, 'returncode': 0, 'readonly': True, 'stdout': '', 'stderr': '',
    }


def test_shell_unicode_and_newline_tails(monkeypatch):
    monkeypatch.setattr(coding, 'SHELL_OUTPUT_CHARS', 37)
    source = (
        "import os\n"
        "for fd in (1, 2):\n"
        "    os.write(fd, ('界' * 5000 + '\\r\\nfin\\r').encode('utf-8'))\n"
    )
    result = payload(coding.shell_run(python_command(source)))
    expected = ('界' * 5000 + '\nfin\n')[-37:]
    assert result['ok']
    assert result['stdout'] == result['stderr'] == expected


def test_shell_output_memory_is_bounded():
    source = (
        "import os\n"
        "chunk = b'x' * 65536\n"
        "for _ in range(256):\n"
        "    os.write(1, chunk)\n"
        "    os.write(2, chunk)\n"
    )
    tracemalloc.start()
    try:
        result = payload(coding.shell_run(python_command(source)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result['ok']
    assert result['stdout'] == result['stderr'] == 'x' * coding.SHELL_OUTPUT_CHARS
    assert peak < 2_000_000


@pytest.mark.parametrize('close_pipes', [False, True])
def test_shell_timeout_reaps_process_and_closes_pipes(close_pipes, monkeypatch):
    processes = []
    real_popen = coding.subprocess.Popen

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(coding.subprocess, 'Popen', popen)
    source = 'import os, time; '
    if close_pipes:
        source += 'os.close(1); os.close(2); '
    source += 'time.sleep(60)'
    start = time.monotonic()
    result = payload(coding.shell_run(python_command(source), timeout=1))

    assert result == {'ok': False, 'error': 'timeout after 5s'}
    assert time.monotonic() - start < 10
    assert len(processes) == 1
    proc = processes[0]
    assert proc.returncode is not None
    assert proc.stdout.closed and proc.stderr.closed
