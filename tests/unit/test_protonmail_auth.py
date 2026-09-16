"""Unit tests for ProtonMail auth hardening (fresh-login CAPTCHA gate,
non-interactive 2FA, SRP verification check, login backoff).

No network, no credentials: the third-party client is faked and the session
path is redirected into tmp_path.
"""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ROOT = REPO_ROOT / "interface" / "mcp_server"
for p in (str(REPO_ROOT), str(MCP_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

protonmail_mod = importlib.import_module("interface.mcp_server.social.services.protonmail")


class FakeUser:
    def __init__(self, ok: bool = True):
        self._ok = ok

    def authenticated(self) -> bool:
        return self._ok


class FakeClient:
    """Stand-in for protonmail.ProtonMail with scripted login behaviour."""

    last_instance = None

    def __init__(self, *args, **kwargs):
        FakeClient.last_instance = self
        self.user = FakeUser(True)
        self.login_calls: list = []
        self.behaviour = "ok"

    def login(self, username, password, getter=None):
        self.login_calls.append((username, password, getter))
        if self.behaviour == "ok":
            return None
        raise self.behaviour

    def save_session(self, path):
        Path(path).write_bytes(b"fake-session")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(protonmail_mod, "_session_file", lambda: str(tmp_path / "protonmail_session.pickle"))
    protonmail_mod.clear_cached_client()
    monkeypatch.setattr(protonmail_mod, "_last_auth_failure_at", 0.0)
    monkeypatch.setattr(protonmail_mod, "_last_auth_failure_reason", "")
    monkeypatch.setenv("PROTONMAIL_AUTH_COOLDOWN_S", "600")
    yield
    protonmail_mod.clear_cached_client()


def _install_fake_client(monkeypatch, behaviour="ok"):
    fake_pkg = types.ModuleType("protonmail")

    def factory(*args, **kwargs):
        inst = FakeClient(*args, **kwargs)
        inst.behaviour = behaviour
        return inst

    fake_pkg.ProtonMail = factory
    monkeypatch.setitem(sys.modules, "protonmail", fake_pkg)
    return fake_pkg


def test_invalid_refresh_shapes():
    assert protonmail_mod._is_invalid_refresh_error(
        Exception("Can't update tokens, status: 422 json: {'Error': 'Invalid refresh token', 'Code': 10013}")
    )
    assert not protonmail_mod._is_invalid_refresh_error(Exception("connection reset"))


def test_captcha_abuse_shapes():
    assert protonmail_mod._is_captcha_or_abuse_error(Exception("Code 9001 captcha required"))
    assert protonmail_mod._is_captcha_or_abuse_error(
        Exception("For security reasons, please complete CAPTCHA ... appeal-abuse")
    )
    assert protonmail_mod._is_captcha_or_abuse_error(Exception("Too many recent logins (2028)"))
    assert not protonmail_mod._is_captcha_or_abuse_error(Exception("connection reset"))


def test_abuse_gate_masks_fresh_login_422():
    masked = Exception("Can't update tokens, status: 422 json: {'Error': 'Invalid refresh token', 'Code': 10013}")
    # No session file: a 422 cannot be a stale saved session -> abuse gate.
    assert protonmail_mod._is_abuse_gate(masked, session_existed=False) is True
    # With a session file it is genuinely a stale session, not a login gate.
    assert protonmail_mod._is_abuse_gate(masked, session_existed=True) is False


def test_2fa_getter_never_touches_stdin(monkeypatch):
    monkeypatch.delenv("PROTONMAIL_2FA_CODE", raising=False)
    monkeypatch.delenv("PROTONMAIL_TOTP_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="two-factor"):
        protonmail_mod._non_interactive_2fa_code()

    monkeypatch.setenv("PROTONMAIL_2FA_CODE", " 123 456 ")
    assert protonmail_mod._non_interactive_2fa_code() == "123456"


def test_password_login_rejects_unverified_user():
    client = FakeClient()
    client.user = FakeUser(False)
    with pytest.raises(RuntimeError, match="SRP verification failed"):
        protonmail_mod._password_login(client, "u", "p")


def test_fresh_login_abuse_gate_records_backoff(monkeypatch):
    monkeypatch.setenv("PROTONMAIL_USERNAME", "user@proton.me")
    monkeypatch.setenv("PROTONMAIL_PASSWORD", "secret")
    gate = Exception("Can't update tokens, status: 422 json: {'Error': 'Invalid refresh token', 'Code': 10013}")
    _install_fake_client(monkeypatch, behaviour=gate)

    client, err = protonmail_mod._get_client()
    assert client is None
    assert "human/CAPTCHA" in err["error"]
    assert protonmail_mod._auth_backoff_remaining() > 0
    assert os.path.exists(protonmail_mod._cooldown_file())


def test_backoff_blocks_further_logins_without_network(monkeypatch):
    monkeypatch.setenv("PROTONMAIL_USERNAME", "user@proton.me")
    monkeypatch.setenv("PROTONMAIL_PASSWORD", "secret")
    protonmail_mod._record_auth_failure("test gate")

    def _boom(*args, **kwargs):
        raise AssertionError("login must not be attempted during backoff")

    fake_pkg = types.ModuleType("protonmail")
    fake_pkg.ProtonMail = _boom
    monkeypatch.setitem(sys.modules, "protonmail", fake_pkg)

    client, err = protonmail_mod._get_client()
    assert client is None
    assert "paused" in err["error"]


def test_successful_login_clears_backoff(monkeypatch):
    monkeypatch.setenv("PROTONMAIL_USERNAME", "user@proton.me")
    monkeypatch.setenv("PROTONMAIL_PASSWORD", "secret")
    protonmail_mod._record_auth_failure("test gate")
    _install_fake_client(monkeypatch, behaviour="ok")
    monkeypatch.setenv("PROTONMAIL_AUTH_COOLDOWN_S", "0")  # allow the attempt

    client, err = protonmail_mod._get_client()
    assert err is None and client is not None
    assert protonmail_mod._auth_backoff_remaining() == 0
    assert not os.path.exists(protonmail_mod._cooldown_file())
