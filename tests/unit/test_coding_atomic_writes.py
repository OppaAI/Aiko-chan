"""Repository writes must not follow predictable staging or backup aliases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic.toolkit import coding, self_improve_write


def payload(result: str) -> dict:
    return json.loads(result.split(']', 1)[1])


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(coding, 'REPO_ROOT', root)
    monkeypatch.setattr(self_improve_write, 'REPO_ROOT', root)
    return root


def test_patch_does_not_follow_predictable_staging_symlink(repo, tmp_path):
    source = repo / 'a.py'
    source.write_text('x = 1\n')
    outside = tmp_path / 'outside.py'
    outside.write_text('outside stays intact\n')
    staging = repo / 'a.py.tmp'
    staging.symlink_to(outside)

    result = payload(coding.code_apply_patch('a.py', 'x = 1', 'x = 2'))

    assert outside.read_text() == 'outside stays intact\n'
    assert result['ok'], result
    assert source.read_text() == 'x = 2\n'
    assert staging.is_symlink()
    assert (repo / 'a.py.bak').read_text() == 'x = 1\n'


def test_patch_rejects_backup_symlink_before_modifying_source(repo, tmp_path):
    source = repo / 'a.py'
    source.write_text('x = 1\n')
    outside = tmp_path / 'outside.py'
    outside.write_text('outside stays intact\n')
    (repo / 'a.py.bak').symlink_to(outside)

    result = payload(coding.code_apply_patch('a.py', 'x = 1', 'x = 2'))

    assert not result['ok']
    assert outside.read_text() == 'outside stays intact\n'
    assert source.read_text() == 'x = 1\n'
    assert (repo / 'a.py.bak').is_symlink()


def test_patch_stops_when_backup_cannot_be_created(repo):
    source = repo / 'a.py'
    source.write_text('x = 1\n')
    (repo / 'a.py.bak').mkdir()

    result = payload(coding.code_apply_patch('a.py', 'x = 1', 'x = 2'))

    assert not result['ok']
    assert source.read_text() == 'x = 1\n'


@pytest.mark.parametrize('writer', ['repo_write_file', 'repo_replace_text'])
def test_repo_write_does_not_modify_external_hardlink(repo, tmp_path, writer):
    outside = tmp_path / 'outside.py'
    outside.write_text('x = 1\n')
    source = repo / 'a.py'
    source.hardlink_to(outside)
    if writer == 'repo_write_file':
        result = self_improve_write.repo_write_file('a.py', 'x = 2\n')
    else:
        result = self_improve_write.repo_replace_text('a.py', 'x = 1', 'x = 2')

    assert '"status": "success"' in result, result
    assert source.read_text() == 'x = 2\n'
    assert outside.read_text() == 'x = 1\n'


def test_patch_preserves_executable_mode(repo):
    source = repo / 'run.py'
    source.write_text('print(1)\n')
    source.chmod(0o750)

    result = payload(coding.code_apply_patch('run.py', 'print(1)', 'print(2)'))

    assert result['ok'], result
    assert source.stat().st_mode & 0o777 == 0o750
    assert (repo / 'run.py.bak').read_text() == 'print(1)\n'


def test_patch_rejects_invalid_utf8_without_replacing_unrelated_bytes(repo):
    source = repo / 'a.py'
    original = b'# retained byte: \xff\nx = 1\n'
    source.write_bytes(original)

    result = payload(coding.code_apply_patch('a.py', 'x = 1', 'x = 2'))

    assert not result['ok']
    assert source.read_bytes() == original
    assert not (repo / 'a.py.bak').exists()


def test_failed_atomic_replace_leaves_source_and_cleans_staging(repo, monkeypatch):
    source = repo / 'a.py'
    source.write_text('x = 1\n')
    real_replace = coding.os.replace

    def reject_source(staging, destination):
        if Path(destination) == source:
            raise OSError('simulated destination failure')
        return real_replace(staging, destination)

    monkeypatch.setattr(coding.os, 'replace', reject_source)
    result = payload(coding.code_apply_patch('a.py', 'x = 1', 'x = 2'))

    assert not result['ok']
    assert source.read_text() == 'x = 1\n'
    assert (repo / 'a.py.bak').read_text() == 'x = 1\n'
    assert not list(repo.glob('.a.py*.tmp'))


@pytest.mark.parametrize('writer', ['code_apply_patch', 'repo_write_file', 'repo_replace_text'])
def test_resolved_alias_cannot_edit_protected_file(repo, writer):
    protected = repo / 'credentials.py'
    protected.write_text('x = 1\n')
    (repo / 'ordinary.py').symlink_to(protected)

    if writer == 'code_apply_patch':
        result = coding.code_apply_patch('ordinary.py', 'x = 1', 'x = 2')
        assert not payload(result)['ok']
    elif writer == 'repo_write_file':
        result = self_improve_write.repo_write_file('ordinary.py', 'x = 2\n')
        assert result.startswith('[repo write failed:')
    else:
        result = self_improve_write.repo_replace_text('ordinary.py', 'x = 1', 'x = 2')
        assert result.startswith('[repo replace failed:')
    assert protected.read_text() == 'x = 1\n'


@pytest.mark.parametrize('directory', ['.git', '.venv', '__pycache__', '.cert'])
def test_replace_text_rejects_restricted_directory(repo, directory):
    target = repo / directory / 'config.py'
    target.parent.mkdir()
    target.write_text('x = 1\n')

    result = self_improve_write.repo_replace_text(f'{directory}/config.py', 'x = 1', 'x = 2')

    assert result.startswith('[repo replace failed:'), result
    assert target.read_text() == 'x = 1\n'
