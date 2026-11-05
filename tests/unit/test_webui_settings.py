"""Settings Studio core: allowlist validation, masked secrets, and
comment-preserving YAML writes.

The secrets tests run against PlaintextSecretsBackend — a temp plaintext
stand-in with the same contract as the age backend (the age binary is
not available in every environment, and no real key material may ever
appear in tests).
"""
from __future__ import annotations

import json

import pytest

from interface.webui.studio.settings.backend import settings_core as core


CONSCIENCE_YAML = """\
# Conscience Circuit Core tunables.
CCC:
  # Master switch. Off => permissive verdicts.
  ENABLED: 1

  # P(violation) above this refuses; inside the band asks.
  BAND_HI: 0.75
  MIN_CHARS: 12
"""

COGNITION_YAML = """\
# Cognition
LLM_MODEL: "ministral"
TEMPERATURE: "0.5"  # sampling temperature
LLM_API_KEY: "yaml-secret-key"
"""

SYSTEM_YAML = """\
AI_NAME: "Aiko"
TIMEZONE: "America/Vancouver"
USER_SPACE_ROOT: "/home/test/.aiko"
AGE_KEY: "age-key.txt"
ENV_AGE_PATH: ".env.age"
"""


@pytest.fixture()
def config_dir(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "conscience.yaml").write_text(CONSCIENCE_YAML, encoding="utf-8")
    (cfg / "cognition.yaml").write_text(COGNITION_YAML, encoding="utf-8")
    (cfg / "system.yaml").write_text(SYSTEM_YAML, encoding="utf-8")
    return cfg


@pytest.fixture()
def secrets_file(tmp_path):
    path = tmp_path / ".env.age"
    path.write_text("EXISTING_TOKEN=original-value\n", encoding="utf-8")
    return path


def _entry(data, key):
    for grp in data["groups"]:
        for s in grp["settings"]:
            if s["key"] == key:
                return s
    raise AssertionError(f"{key} not in settings payload")


# ---- validation -----------------------------------------------------------


def test_validation_rejects_wrong_types_and_ranges():
    spec = core._BY_KEY["HTTP_PORT"]
    with pytest.raises(ValueError):
        core.validate_value(spec, "not-a-port")
    with pytest.raises(ValueError):
        core.validate_value(spec, "70000")
    with pytest.raises(ValueError):
        core.validate_value(spec, "3.5")

    band = core._BY_KEY["CCC_BAND_HI"]
    assert core.validate_value(band, "0.8") == "0.8"
    with pytest.raises(ValueError):
        core.validate_value(band, "1.5")

    hitl = core._BY_KEY["CCC_HITL_DEFAULT"]
    with pytest.raises(ValueError):
        core.validate_value(hitl, "maybe")

    flag = core._BY_KEY["CCC_ENABLED"]
    assert core.validate_value(flag, "true") == "1"
    assert core.validate_value(flag, "off") == "0"
    with pytest.raises(ValueError):
        core.validate_value(flag, "perhaps")


def test_write_rejects_unknown_and_readonly_keys(config_dir):
    with pytest.raises(ValueError, match="unknown setting"):
        core.write_settings({"NOPE_KEY": "1"}, {}, config_dir=config_dir, environ={})
    with pytest.raises(ValueError, match="read-only"):
        core.write_settings({"USER_SPACE_ROOT": "/elsewhere"}, {},
                            config_dir=config_dir, environ={})


# ---- YAML writes ----------------------------------------------------------


def test_yaml_write_preserves_comments_and_unrelated_keys(config_dir):
    result = core.write_settings(
        {"CCC_BAND_HI": "0.8", "TEMPERATURE": "0.7"}, {},
        config_dir=config_dir, environ={},
    )
    assert set(result["updated"]) == {"CCC_BAND_HI", "TEMPERATURE"}

    conscience = (config_dir / "conscience.yaml").read_text(encoding="utf-8")
    assert "# Master switch. Off => permissive verdicts." in conscience
    assert "# P(violation) above this refuses; inside the band asks." in conscience
    assert "  BAND_HI: 0.8" in conscience
    assert "  MIN_CHARS: 12" in conscience  # untouched

    cognition = (config_dir / "cognition.yaml").read_text(encoding="utf-8")
    assert 'TEMPERATURE: "0.7"  # sampling temperature' in cognition
    assert 'LLM_MODEL: "ministral"' in cognition  # untouched


def test_bool_write_keeps_bare_style(config_dir):
    core.write_settings({"CCC_ENABLED": "0"}, {}, config_dir=config_dir, environ={})
    text = (config_dir / "conscience.yaml").read_text(encoding="utf-8")
    assert "  ENABLED: 0" in text


# ---- env override ---------------------------------------------------------


def test_env_overridden_yaml_key_flagged_and_write_rejected(config_dir):
    data = core.read_all(config_dir=config_dir,
                         environ={"CCC_BAND_HI": "0.99"},
                         secrets_backend=None)
    entry = _entry(data, "CCC_BAND_HI")
    assert entry["env_overridden"] is True
    assert entry["env_value"] == "0.99"
    with pytest.raises(ValueError, match="overridden by the process environment"):
        core.write_settings({"CCC_BAND_HI": "0.8"}, {},
                            config_dir=config_dir,
                            environ={"CCC_BAND_HI": "0.99"})


def test_env_overridden_secret_flagged(config_dir, secrets_file):
    backend = core.PlaintextSecretsBackend(secrets_file)
    data = core.read_all(config_dir=config_dir,
                         environ={"EXISTING_TOKEN": "from-env"},
                         secrets_backend=backend)
    keys = {k["key"]: k for k in data["secrets"]["keys"]}
    assert keys["EXISTING_TOKEN"]["env_overridden"] is True
    assert keys["EXISTING_TOKEN"]["is_set"] is True


# ---- secrets --------------------------------------------------------------


def test_secrets_never_appear_in_read_payload(config_dir, secrets_file):
    backend = core.PlaintextSecretsBackend(secrets_file)
    data = core.read_all(config_dir=config_dir, environ={}, secrets_backend=backend)
    blob = json.dumps(data)
    assert "original-value" not in blob
    assert "yaml-secret-key" not in blob  # cognition.yaml password stays masked too
    keys = {k["key"]: k for k in data["secrets"]["keys"]}
    assert keys["EXISTING_TOKEN"]["is_set"] is True
    llm_key = _entry(data, "LLM_API_KEY")
    assert llm_key["value"] == ""
    assert llm_key["is_set"] is True


def test_secret_write_roundtrip_and_empty_means_unchanged(config_dir, secrets_file):
    backend = core.PlaintextSecretsBackend(secrets_file)
    result = core.write_settings(
        {}, {"EXISTING_TOKEN": "", "NEW_TOKEN": "brand-new"},
        config_dir=config_dir, environ={}, secrets_backend=backend,
    )
    assert result["secrets_updated"] == ["NEW_TOKEN"]
    values = backend.read_values()
    assert values["EXISTING_TOKEN"] == "original-value"  # blank = unchanged
    assert values["NEW_TOKEN"] == "brand-new"


def test_secret_write_failure_leaves_original_untouched(config_dir, secrets_file):
    class BoomBackend(core.PlaintextSecretsBackend):
        def write_values(self, values):
            raise RuntimeError("encrypt failed")

    before = secrets_file.read_bytes()
    with pytest.raises(RuntimeError, match="encrypt failed"):
        core.write_settings({}, {"NEW_TOKEN": "x"}, config_dir=config_dir,
                            environ={}, secrets_backend=BoomBackend(secrets_file))
    assert secrets_file.read_bytes() == before


def test_secret_key_and_value_validation(config_dir, secrets_file):
    backend = core.PlaintextSecretsBackend(secrets_file)
    with pytest.raises(ValueError, match="invalid secret key name"):
        core.write_settings({}, {"bad key!": "x"}, config_dir=config_dir,
                            environ={}, secrets_backend=backend)
    with pytest.raises(ValueError, match="newlines"):
        core.write_settings({}, {"GOOD_KEY": "a\nb"}, config_dir=config_dir,
                            environ={}, secrets_backend=backend)


def test_dotenv_serialize_roundtrip(tmp_path):
    values = {"PLAIN": "abc-123", "SPACED": "two words", "QUOTED": "it's"}
    path = tmp_path / "env"
    backend = core.PlaintextSecretsBackend(path)
    backend.write_values(values)
    assert backend.read_values() == values


def test_age_backend_paths_and_availability(tmp_path):
    environ = {"USER_SPACE_ROOT": str(tmp_path),
               "AGE_KEY": "age-key.txt", "ENV_AGE_PATH": ".env.age"}
    backend = core.AgeSecretsBackend(environ=environ)
    assert backend.enc_path == tmp_path / ".env.age"
    assert backend.identity_path == tmp_path / "age-key.txt"
    # No identity file here (and possibly no age binary): must report
    # unavailable rather than raise, so the UI can say so.
    assert backend.available() is False
    assert backend.read_values() == {}


# ---- catalog + API surface -------------------------------------------------


def test_catalog_keys_unique_and_files_fixed():
    keys = [s["key"] for grp in core.SETTINGS_GROUPS for s in grp["settings"]]
    assert len(keys) == len(set(keys))
    allowed_files = {
        "agentic.yaml", "android_app.yaml", "cognition.yaml",
        "conscience.yaml", "interface.yaml", "memory.yaml", "sensory.yaml",
        "social.yaml", "system.yaml",
    }
    for grp in core.SETTINGS_GROUPS:
        for s in grp["settings"]:
            assert s["file"] in allowed_files


def test_api_app_constructs_with_settings_routes():
    from interface.webui.studio.settings.backend import api

    paths = {route.path for route in api.app.routes}
    assert "/api/settings" in paths
    assert "/" in paths


@pytest.mark.parametrize('raw', ['nan', 'NaN', 'inf', '-inf'])
def test_nonfinite_sampling_rejected(raw):
    with pytest.raises(ValueError, match='finite'):
        core.validate_value(core._BY_KEY['TEMPERATURE'], raw)


def test_discovery_covers_every_boot_setting_and_masks_credentials(config_dir):
    (config_dir / 'memory.yaml').write_text('CUSTOM_API_KEY: hidden-key\nNESTED:\n  LIMIT: 5\n  PATHS:\n  - one\n  - two\n')
    data = core.read_settings(config_dir, environ={})
    assert _entry(data, 'NESTED_LIMIT')['type'] == 'int'
    assert _entry(data, 'NESTED_PATHS')['type'] == 'json'
    assert _entry(data, 'CUSTOM_API_KEY')['is_set']
    assert 'hidden-key' not in json.dumps(data)
    from system.config import _flatten
    actual = {s['key'] for g in core.read_settings(environ={})['groups'] for s in g['settings']}
    for filename in core._RUNTIME_FILES:
        assert set(_flatten(core._load_yaml_file(core.DEFAULT_CONFIG_DIR / filename))) <= actual


def test_nested_list_multiline_and_ambiguous_string_roundtrip(config_dir):
    (config_dir / 'memory.yaml').write_text('NESTED:\n  PATHS:\n  - one\n  - two\n  LABEL: hello\n  TEXT: |\n    old\n    text\n# Keep this comment\nTAIL: 1\n')
    core.write_settings({'NESTED_PATHS': '["new", "with spaces"]', 'NESTED_LABEL': 'yes',
                         'NESTED_TEXT': 'new\ntext\n'}, config_dir=config_dir, environ={})
    data = core._load_yaml_file(config_dir / 'memory.yaml')
    assert data['NESTED'] == {'PATHS': ['new', 'with spaces'], 'LABEL': 'yes', 'TEXT': 'new\ntext\n'}
    assert '# Keep this comment' in (config_dir / 'memory.yaml').read_text()
    assert data['TAIL'] == 1


def test_live_sampling_and_restart_fields(config_dir):
    env = {}
    result = core.write_settings({'TEMPERATURE': '0.8', 'LLM_MODEL': 'new-model'},
                                 config_dir=config_dir, environ=env, apply_live=True)
    assert result['applied_live'] == ['TEMPERATURE']
    assert result['restart_keys'] == ['LLM_MODEL']
    assert env == {'TEMPERATURE': '0.8'}
    assert not _entry(core.read_settings(config_dir, env), 'TEMPERATURE')['restart_required']


def test_managed_pending_restart_remains_editable(config_dir, monkeypatch):
    import os
    from system import config
    monkeypatch.setattr(config, '_MANAGED_ENV', {'LLM_MODEL': 'ministral'})
    monkeypatch.setattr(config, '_LOADED', True)
    monkeypatch.setenv('LLM_MODEL', 'ministral')
    core.write_settings({'LLM_MODEL': 'first'}, config_dir=config_dir, environ=os.environ)
    entry = _entry(core.read_settings(config_dir, os.environ), 'LLM_MODEL')
    assert not entry['env_overridden']
    assert entry['pending_restart']
    core.write_settings({'LLM_MODEL': 'second'}, config_dir=config_dir, environ=os.environ)
    # An actual deployment override stays locked even when it matches the file.
    monkeypatch.setattr(config, '_MANAGED_ENV', {})
    monkeypatch.setenv('LLM_MODEL', 'second')
    with pytest.raises(ValueError, match='overridden'):
        core.write_settings({'LLM_MODEL': 'third'}, config_dir=config_dir, environ=os.environ)


def test_yaml_credential_edit_migrates_into_encrypted_backend(config_dir, secrets_file):
    backend = core.PlaintextSecretsBackend(secrets_file)
    core.write_settings({'LLM_API_KEY': 'replacement'}, config_dir=config_dir,
                        environ={}, secrets_backend=backend)
    assert core._load_yaml_file(config_dir / 'cognition.yaml')['LLM_API_KEY'] == ''
    assert backend.read_values()['LLM_API_KEY'] == 'replacement'
    payload = core.read_all(config_dir, {}, backend)
    assert _entry(payload, 'LLM_API_KEY')['is_set']
    assert 'replacement' not in json.dumps(payload)


def test_secret_failure_does_not_commit_yaml(config_dir, secrets_file):
    class Broken(core.PlaintextSecretsBackend):
        def write_values(self, values):
            raise RuntimeError('encryption failed')
    before = (config_dir / 'cognition.yaml').read_bytes()
    with pytest.raises(RuntimeError):
        core.write_settings({'TEMPERATURE': '0.8', 'LLM_API_KEY': 'replacement'},
                            config_dir=config_dir, environ={}, secrets_backend=Broken(secrets_file))
    assert (config_dir / 'cognition.yaml').read_bytes() == before
    assert secrets_file.read_text() == 'EXISTING_TOKEN=original-value\n'
    assert not list(config_dir.glob('*.settings-stage'))


def test_stale_revision_rejected(config_dir, secrets_file):
    backend = core.PlaintextSecretsBackend(secrets_file)
    revision = core.read_all(config_dir, {}, backend)['revision']
    core.write_settings({'TEMPERATURE': '0.6'}, config_dir=config_dir, environ={})
    with pytest.raises(core.SettingsConflict):
        core.write_settings({'TEMPERATURE': '0.9'}, config_dir=config_dir, environ={},
                            secrets_backend=backend, revision=revision)
    assert core._load_yaml_file(config_dir / 'cognition.yaml')['TEMPERATURE'] == '0.6'


@pytest.mark.parametrize('field', ['values', 'secret_values'])
def test_invalid_batch_shapes_rejected(config_dir, field):
    with pytest.raises(ValueError, match='objects'):
        core.write_settings(**{field: []}, **({} if field == 'values' else {'values': {}}),
                            config_dir=config_dir, environ={})


def test_settings_require_operator(monkeypatch):
    from fastapi import HTTPException
    from interface.webui.studio.settings.backend.api import require_operator, app
    import system.userspace as userspace
    monkeypatch.setattr(userspace, 'can_approve', lambda uid: False)
    with pytest.raises(HTTPException) as error:
        require_operator()
    assert error.value.status_code == 403
    monkeypatch.setattr(userspace, 'can_approve', lambda uid: True)
    require_operator()
    for route in app.routes:
        if getattr(route, 'path', '') == '/api/settings':
            assert require_operator in [dep.call for dep in route.dependant.dependencies]


def test_list_comments_and_following_settings_survive(config_dir):
    path = config_dir / 'memory.yaml'
    path.write_text('PATHS:\n  # keep above\n  - one # keep inline\n  # keep below\nTAIL: 1\n')
    core.write_settings({'PATHS': '["two"]'}, config_dir=config_dir, environ={})
    assert core._load_yaml_file(path) == {'PATHS': ['two'], 'TAIL': 1}
    for comment in ['keep above', 'keep inline', 'keep below']:
        assert comment in path.read_text()


def test_active_encrypted_secret_not_marked_pending(config_dir, secrets_file, monkeypatch):
    import os
    from system import config
    backend = core.PlaintextSecretsBackend(secrets_file)
    core.write_settings({'LLM_API_KEY': 'replacement'}, config_dir=config_dir, environ={}, secrets_backend=backend)
    monkeypatch.setattr(config, '_MANAGED_ENV', {'LLM_API_KEY': 'replacement'})
    monkeypatch.setenv('LLM_API_KEY', 'replacement')
    assert not _entry(core.read_all(config_dir, os.environ, backend), 'LLM_API_KEY')['pending_restart']
    core.write_settings({'LLM_API_KEY': 'newer'}, config_dir=config_dir, environ=os.environ, secrets_backend=backend)
    assert _entry(core.read_all(config_dir, os.environ, backend), 'LLM_API_KEY')['pending_restart']


def test_yaml_late_failure_rolls_back_earlier_replaces(config_dir, secrets_file, monkeypatch):
    before = {p: p.read_bytes() for p in config_dir.glob('*.yaml')}
    original = core.os.replace
    failed = False
    def fail_once(source, target):
        nonlocal failed
        if not failed and str(source).endswith('.settings-stage') and str(target).endswith('system.yaml'):
            failed = True
            raise OSError('disk failure')
        return original(source, target)
    monkeypatch.setattr(core.os, 'replace', fail_once)
    with pytest.raises(OSError, match='disk failure'):
        core.write_settings({'TEMPERATURE': '0.7', 'AI_NAME': 'Changed'}, {'NEW_TOKEN': 'new'},
                            config_dir=config_dir, environ={}, secrets_backend=core.PlaintextSecretsBackend(secrets_file))
    for path, data in before.items():
        assert path.read_bytes() == data
    assert secrets_file.read_text() == 'EXISTING_TOKEN=original-value\n'


def test_settings_api_auth_and_live_save(monkeypatch, config_dir, secrets_file):
    import asyncio
    async def scenario():
        import httpx
        import interface.webui.auth as auth
        import system.userspace as userspace
        from interface.webui.studio.settings.backend.api import app
        backend = core.PlaintextSecretsBackend(secrets_file)
        real_read, real_write = core.read_all, core.write_settings
        env = {}
        monkeypatch.setattr(core, 'read_all', lambda: real_read(config_dir, env, backend))
        def write(values, secrets, **kwargs):
            return real_write(values, secrets, config_dir=config_dir, environ=env, secrets_backend=backend, **kwargs)
        monkeypatch.setattr(core, 'write_settings', write)
        session = None
        async def require_session(request):
            from fastapi import HTTPException
            if session is None:
                raise HTTPException(status_code=401, detail='sign in')
            return session
        monkeypatch.setattr(auth, 'require_session', require_session)
        monkeypatch.setattr(userspace, 'can_approve', lambda uid: uid == 'owner')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.get('/api/settings')).status_code == 401
            session = {'user_id': 'visitor', 'username': 'Visitor'}
            assert (await client.get('/api/settings')).status_code == 403
            assert (await client.post('/api/settings', json={'settings': {'TEMPERATURE': '.8'}})).status_code == 403
            session = {'user_id': 'owner', 'username': 'Owner'}
            data = (await client.get('/api/settings')).json()
            assert 'yaml-secret-key' not in json.dumps(data)
            response = await client.post('/api/settings', json={'settings': {'TEMPERATURE': '.8'}, 'revision': data['revision']})
            assert response.status_code == 200, response.text
            assert response.json()['applied_live'] == ['TEMPERATURE']
            assert env['TEMPERATURE'] == '0.8'
            stale = await client.post('/api/settings', json={'settings': {'TEMPERATURE': '.9'}, 'revision': data['revision']})
            assert stale.status_code == 409
            assert (await client.post('/api/settings', json={'settings': []})).status_code == 400
    asyncio.run(scenario())
