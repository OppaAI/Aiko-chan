"""Configuration, storage boundaries and deployment regressions; no live services."""
import base64
import importlib.util
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def config_root(tmp_path, monkeypatch):
    from system import config
    root = tmp_path / 'repo'
    (root / 'config').mkdir(parents=True)
    monkeypatch.setattr(config, '__file__', str(root / 'system' / 'config.py'))
    monkeypatch.setattr(config, '_LOADED', False)
    monkeypatch.setattr(config, '_MANAGED_ENV', {})
    monkeypatch.delenv('SOME_KEY', raising=False)
    monkeypatch.delenv('USER_SPACE_ROOT', raising=False)
    monkeypatch.delenv('ENV_AGE_PATH', raising=False)
    monkeypatch.setenv('AGE_KEY', str(tmp_path / 'missing-key'))
    (root / 'config' / 'index.yaml').write_text('configs: [a.yaml, b.yaml]')
    return config, root


def test_failed_config_load_does_not_publish_partial_environment(tmp_path, monkeypatch):
    config, root = config_root(tmp_path, monkeypatch)
    (root / 'config' / 'a.yaml').write_text('SOME_KEY: first\n')
    with pytest.raises(FileNotFoundError):
        config.load_config()
    assert 'SOME_KEY' not in os.environ and not config._MANAGED_ENV
    (root / 'config' / 'a.yaml').write_text('SOME_KEY: second\n')
    (root / 'config' / 'b.yaml').write_text('{}')
    config.load_config()
    assert os.environ['SOME_KEY'] == 'second'


@pytest.mark.parametrize('entry', ['../outside.yaml', '/outside.yaml'])
def test_config_index_cannot_escape_config_directory(entry, tmp_path, monkeypatch):
    config, root = config_root(tmp_path, monkeypatch)
    (root / 'config' / 'index.yaml').write_text('configs: [' + entry + ']')
    with pytest.raises(ValueError, match='relative paths'):
        config.load_config()


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_float_config_rejects_nonfinite(value, monkeypatch):
    from system.config import env_float
    monkeypatch.setenv('TEST_NUMBER', value)
    assert env_float('TEST_NUMBER', 2.5) == 2.5


def test_secrets_roundtrip_literal_multiline_and_backslashes(monkeypatch):
    from interface.webui.studio.settings.backend.settings_core import _dotenv_parse, _dotenv_serialize
    from dotenv import dotenv_values
    monkeypatch.setenv('SHOULD_NOT_EXPAND', 'wrong')
    values = {'TOKEN': '${SHOULD_NOT_EXPAND}', 'PATH': r'C:\users\test',
              'TEXT': 'line one\nline two\r\n"quoted" and \'apostrophe\''}
    text = _dotenv_serialize(values)
    assert _dotenv_parse(text) == values
    assert dotenv_values(stream=io.StringIO(text), interpolate=False) == values


def test_user_directory_symlink_is_rejected_before_chmod(tmp_path, monkeypatch):
    from system.userspace import user_state_dir, all_known_user_ids
    root, outside = tmp_path / 'users', tmp_path / 'outside'
    root.mkdir(); outside.mkdir(mode=0o755)
    before_mode = outside.stat().st_mode & 0o777
    (root / 'alice').symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv('USER_SPACE_ROOT', str(root))
    with pytest.raises(ValueError, match='symlink'):
        user_state_dir('alice')
    assert outside.stat().st_mode & 0o777 == before_mode
    assert 'alice' not in all_known_user_ids()


def test_display_name_does_not_grant_owner_identity(monkeypatch):
    from interface.webui import auth
    monkeypatch.setattr(auth, '_save_sessions', lambda: None)
    monkeypatch.setattr(auth, '_has_accepted_terms', lambda *_: False)
    monkeypatch.setattr(auth, 'sessions', {})
    sid = auth._create_session(user_id='stranger-id', username='OppaAI', email=None, provider='patreon')
    assert auth.sessions[sid]['user_id'] == 'patreon_stranger-id'
    owner = auth._create_session(user_id='205369547', username='renamed-owner', email=None, provider='github')
    assert auth.sessions[owner]['user_id'] == 'OppaAI'


def test_debug_prompt_dump_is_private_and_does_not_follow_symlink(tmp_path, monkeypatch):
    from cognition import think
    outside = tmp_path / 'outside'
    outside.write_text('preserve')
    target = tmp_path / 'dump'
    target.symlink_to(outside)
    monkeypatch.setenv('AIKO_DEBUG_FULL_PROMPT', '1')
    monkeypatch.setattr(think, '_DEBUG_PROMPT_DUMP_PATH', str(target))
    think._dump_full_prompt({'system_prompt': 'private', 'previous_chat_messages': []})
    assert outside.read_text() == 'preserve'
    assert not target.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o600
    assert 'private' in target.read_text()
    assert not list(tmp_path.glob('.aiko-prompt-*'))


class ModalBuilder:
    def __getattr__(self, _):
        return lambda *args, **kwargs: self


def load_modal(name, monkeypatch):
    decorator = lambda *args, **kwargs: lambda obj: obj
    modal = SimpleNamespace(Image=ModalBuilder(), Volume=ModalBuilder(), Secret=ModalBuilder(),
        App=lambda *args, **kwargs: SimpleNamespace(function=decorator, cls=decorator),
        concurrent=decorator, enter=decorator, method=decorator, asgi_app=decorator, web_server=decorator)
    monkeypatch.setitem(sys.modules, 'modal', modal)
    path = Path(__file__).resolve().parents[2] / 'backend' / 'modal' / name
    spec = importlib.util.spec_from_file_location('test_' + name.replace('-', '_').replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['krea2.py', 'flux2-klein.py'])
def test_modal_image_requests_are_bounded_before_gpu_dispatch(name, monkeypatch):
    from pydantic import ValidationError
    module = load_modal(name, monkeypatch)
    assert module.GenerateRequest(prompt='hello').width == 1024
    for invalid in [{'width': 4096}, {'width': 1025}, {'steps': 10000},
                    {'guidance_scale': float('nan')}, {'prompt': ' '}]:
        with pytest.raises(ValidationError):
            module.GenerateRequest(**({'prompt': 'hello'} | invalid))
    with pytest.raises(ValueError):
        module._decode_reference_image(base64.b64encode(b'not an image').decode())
    if name == 'krea2.py':
        with pytest.raises(ValidationError):
            module.EditRequest(instruction='edit', reference_image='invalid')
    else:
        with pytest.raises(ValidationError):
            module.GenerateRequest(prompt='hello', reference_images=['x'] * 3)


def test_osrm_build_matches_mld_server(monkeypatch):
    module = load_modal('osrm.py', monkeypatch)
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **kwargs: commands.append(cmd))
    module.build_graph()
    assert [cmd[0] for cmd in commands] == ['curl', 'osrm-extract', 'osrm-partition', 'osrm-customize']
    assert '--fail' in commands[0]


@pytest.mark.parametrize('size', [(16000000, 1), (1, 16000000), (1500, 1400)])
def test_edit_dimensions_stay_bounded_for_extreme_aspect_ratios(size, monkeypatch):
    module = load_modal('krea2.py', monkeypatch)
    width, height = module._edit_dimensions(*size, 1_000_000, 64)
    assert width <= 2048 and height <= 2048
    assert width * height <= 1_000_000
    assert width % 64 == height % 64 == 0


def test_url_sync_preserves_ampersands_and_uses_environment_precedence(tmp_path):
    import subprocess
    root = Path(__file__).resolve().parents[2]
    app = tmp_path / 'Games'
    (app / 'app').mkdir(parents=True)
    props = app / 'local.properties'
    props.write_text('sdk.dir=/keep\naikoServerUrl=old\n')
    url = 'https://example.test/path?a=1&b=2'
    env = dict(os.environ, AIKO_PUBLIC_BASE_URL=url, GAMES_DIR=str(app),
               LINGO_DIR=str(tmp_path / 'missing'), ONMYOJI_DIR=str(tmp_path / 'missing'),
               ENV_AGE_PATH=str(tmp_path / 'no-secrets'))
    result = subprocess.run(['bash', str(root / 'util/sync_app_server_url.sh')], env=env,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert props.read_text() == 'sdk.dir=/keep\naikoServerUrl=' + url + '\n'


@pytest.mark.parametrize('fail_encrypt', [False, True])
def test_secret_editor_cleans_private_temporary_files_and_preserves_failed_save(tmp_path, fail_encrypt):
    import subprocess
    root = Path(__file__).resolve().parents[2]
    bins = tmp_path / 'bin'
    bins.mkdir()
    age = bins / 'age'
    age.write_text('#!/usr/bin/env bash\nset -eu\nif [[ "$1" != -d && "${FAKE_ENCRYPT_FAIL:-0}" = 1 ]]; then exit 2; fi\ncat -- "${@: -1}"\n')
    age.chmod(0o755)
    editor = bins / 'nvim'
    editor.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
p = Path(sys.argv[-1])
assert p.stat().st_mode & 0o777 == 0o600
assert p.parent.stat().st_mode & 0o777 == 0o700
assert '-n' in sys.argv and 'NONE' in sys.argv
Path(os.environ['FAKE_EDITOR_TRACE']).write_text(str(p.parent))
p.write_text('TOKEN=new\\n')
''')
    editor.chmod(0o755)
    key, pub, enc = tmp_path / 'key', tmp_path / 'key.pub', tmp_path / 'secrets.age'
    key.write_text('test key'); pub.write_text('test recipient'); enc.write_text('TOKEN=old\n')
    trace = tmp_path / 'editor-trace'
    env = dict(os.environ, PATH=str(bins) + os.pathsep + os.environ['PATH'],
               AGE_KEY=str(key), AGE_KEY_PUB=str(pub), FAKE_EDITOR_TRACE=str(trace),
               FAKE_ENCRYPT_FAIL='1' if fail_encrypt else '0')
    result = subprocess.run(['bash', str(root / 'util/edit_dotenv.sh'), str(enc)],
                            env=env, capture_output=True, text=True, timeout=15)
    assert (result.returncode != 0) == fail_encrypt, result.stderr
    assert enc.read_text() == ('TOKEN=old\n' if fail_encrypt else 'TOKEN=new\n')
    assert not Path(trace.read_text()).exists()
    assert not list(tmp_path.glob('secrets.age.*'))


def test_decrypt_keeps_secret_substitution_literal(tmp_path, monkeypatch):
    from system import config
    enc, key = tmp_path / 'enc', tmp_path / 'key'
    enc.touch(); key.touch()
    monkeypatch.setenv('EXPAND_ME', 'wrong')
    monkeypatch.setattr(config.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=b'TOKEN="${EXPAND_ME}"\n'))
    assert config._decrypt_env(enc, key) == {'TOKEN': '${EXPAND_ME}'}


def test_old_display_name_owner_sessions_are_not_restored(tmp_path, monkeypatch):
    import json
    from interface.webui import auth
    path = tmp_path / 'sessions.json'
    when = auth.bioclock.local_now().isoformat()
    path.write_text(json.dumps({
        'wrong': dict(user_id='OppaAI', provider='patreon', provider_user_id='stranger', created_at=when),
        'right': dict(user_id='OppaAI', provider='github', provider_user_id='205369547', created_at=when)}))
    monkeypatch.setattr(auth, '_SESSIONS_PATH', path)
    monkeypatch.setattr(auth, 'sessions', {})
    auth._load_sessions()
    assert set(auth.sessions) == {'right'}


def test_decrypt_failure_does_not_publish_yaml_environment(tmp_path, monkeypatch):
    config, root = config_root(tmp_path, monkeypatch)
    state = tmp_path / 'state'
    state.mkdir()
    (state / '.env.age').write_text('dummy ciphertext')
    (root / 'config' / 'a.yaml').write_text('SOME_KEY: pending\nUSER_SPACE_ROOT: ' + str(state) + '\n')
    (root / 'config' / 'b.yaml').write_text('{}')
    with pytest.raises(FileNotFoundError, match='identity'):
        config.load_config()
    assert 'SOME_KEY' not in os.environ
    assert 'USER_SPACE_ROOT' not in os.environ
    assert not config._MANAGED_ENV
