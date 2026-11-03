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
