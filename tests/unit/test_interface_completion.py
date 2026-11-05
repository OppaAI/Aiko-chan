"""Interface auth, persistence, validation and bounded resource regressions."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient


def request(cookie='', body=None):
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
    return Request({'type': 'http', 'method': 'POST', 'path': '/',
                    'headers': [(b'cookie', cookie.encode())]}, receive)


@pytest.mark.parametrize('module,function', [
    ('game.games', '_require_user'), ('go.games_go', '_require_user'),
    ('shogi.games_shogi', '_require_user'), ('koikoi.games_koikoi', '_require_user'),
    ('lingo.router', 'get_lingo_session'), ('lingo.learn_api', 'require_lingo_user')])
def test_all_companion_auth_preserves_terms_rejection(monkeypatch, module, function):
    import importlib
    from interface.webui import auth
    async def session(_):
        return {'user_id': 'alice', 'accepted_terms': False}
    monkeypatch.setattr(auth, 'require_session', session)
    monkeypatch.setenv('AIKO_COMPANION_OWNER_FALLBACK', '1')
    monkeypatch.setenv('AIKO_USER_ID', 'owner')
    with pytest.raises(HTTPException) as exc:
        asyncio.run(getattr(importlib.import_module('interface.android_app.' + module), function)(request()))
    assert exc.value.status_code == 403


@pytest.mark.parametrize('studio', ['dag', 'spec', 'mcp'])
def test_remaining_studios_require_session(studio, monkeypatch):
    import importlib
    from interface.webui import auth
    async def reject(_):
        raise HTTPException(status_code=401, detail='Not authenticated')
    monkeypatch.setattr(auth, 'require_session', reject)
    app = importlib.import_module(f'interface.webui.studio.{studio}.backend.api').app
    with TestClient(app) as client:
        assert client.get('/api/playbooks').status_code == 401


@pytest.mark.parametrize('path', ['/api/validate', '/api/preview'])
def test_spec_rejects_non_object_payload(path, monkeypatch):
    from interface.webui import auth
    from interface.webui.studio.spec.backend.api import app
    async def session(_):
        return {'user_id': 'alice'}
    monkeypatch.setattr(auth, 'require_session', session)
    with TestClient(app) as client:
        assert client.post(path, json=[]).status_code == 400


def test_acceptance_and_logout_persist(monkeypatch, tmp_path):
    from interface.webui import auth
    monkeypatch.setattr(auth, '_SESSIONS_PATH', tmp_path / 'sessions.json')
    monkeypatch.setattr(auth, '_record_terms_acceptance', lambda *_: None)
    session = dict(user_id='alice', provider='github', created_at=auth.bioclock.local_now(), accepted_terms=False)
    monkeypatch.setattr(auth, 'sessions', {'test': session})
    cookie = 'session_id=' + auth.signer.dumps('test')
    asyncio.run(auth.accept_terms(request(cookie, {'accepted': True}), session))
    assert json.loads((tmp_path / 'sessions.json').read_text())['test']['accepted_terms']
    asyncio.run(auth.logout(request(cookie)))
    assert json.loads((tmp_path / 'sessions.json').read_text()) == {}
    assert not list(tmp_path.glob('.sessions-*'))


def test_log_reads_only_tail(monkeypatch, tmp_path):
    from interface.webui.studio.log.backend import api
    monkeypatch.setattr(api, '_MAX_LOG_BYTES', 150)
    path = tmp_path / 'aiko.log'
    path.write_text('x' * 500 + '\n2026-08-27 12:34:56.123  [INFO    ]  demo — latest\n')
    records = api.read_records(path)
    assert len(records) == 1 and records[0]['message'] == 'latest'


def test_slack_registers_and_acknowledges_socket_mode(monkeypatch):
    import sys
    from interface.adapter import slack
    sent, handled, listeners = [], [], []
    client = SimpleNamespace(socket_mode_request_listeners=listeners, connect=lambda: None,
                             send_socket_mode_response=sent.append, close=lambda: None)
    monkeypatch.setattr(slack, 'WebClient', lambda **_: SimpleNamespace(auth_test=lambda: {'user_id': 'bot'}))
    monkeypatch.setattr(slack, 'SocketModeClient', lambda **_: client, raising=False)
    monkeypatch.setitem(sys.modules, 'slack_sdk.socket_mode.response', SimpleNamespace(SocketModeResponse=lambda **kw: kw))
    adapter = slack.SlackAdapter()
    monkeypatch.setattr(adapter, '_read_config', lambda: {'bot_token': 'fake', 'app_token': 'fake'})
    adapter.handle_message = lambda *args: handled.append(args)
    adapter.start()
    try:
        listeners[0](client, SimpleNamespace(type='events_api', envelope_id='event1', payload={
            'event': {'type': 'message', 'text': 'hello', 'user': 'alice', 'channel': 'room'}}))
        assert sent == [{'envelope_id': 'event1'}]
        assert handled == [('room', 'alice', '<@alice>', 'hello')]
    finally:
        adapter.stop()


def test_matrix_failed_start_closes_client_and_loop(monkeypatch):
    from interface.adapter.matrix import MatrixAdapter
    adapter = MatrixAdapter()
    closed = []
    async def close():
        closed.append(True)
    async def fail():
        adapter._client = SimpleNamespace(close=close)
        raise RuntimeError('login transport failed')
    monkeypatch.setattr(adapter, '_async_main', fail)
    adapter._running = True
    adapter._run_async()
    assert closed == [True]
    assert adapter._loop is None and adapter._client is None
    assert not adapter._running


def test_approval_download_validates_redirects_and_bounds_body(monkeypatch):
    from interface.webui.studio.approval.backend import api
    from agentic.toolkit import ingest
    monkeypatch.setattr(ingest, '_check_host_ssrf', lambda host: host == 'private.test')
    calls = []
    class Response:
        status_code = 302
        headers = {'location': 'http://private.test/'}
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
    def get(url, **kwargs):
        calls.append(url)
        assert kwargs['stream'] and not kwargs['allow_redirects']
        return Response()
    monkeypatch.setattr(api.requests, 'get', get)
    with pytest.raises(ValueError, match='not allowed'):
        api._download_page('https://public.test/', {})
    assert calls == ['https://public.test/']
    Response.status_code = 200
    Response.headers = {}
    Response.raise_for_status = lambda self: None
    Response.iter_content = lambda self, _: iter([b'x' * 3_000_000, b'x' * 3_000_000])
    with pytest.raises(ValueError, match='5 MB'):
        api._download_page('https://public.test/', {})
