"""Delivery failures must reach callers; fake services never send externally."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from interface.adapter.discord import DiscordAdapter
from interface.adapter.telegram import TelegramAdapter
from interface.webui.webui import AikoWeb


@pytest.mark.parametrize('cached', [True, False])
def test_discord_chunks_cached_and_fetched_channels(cached):
    sent = []
    async def send(text):
        sent.append(text)
    channel = SimpleNamespace(send=send)
    async def fetch(cid):
        assert cid == 123
        return channel
    adapter = DiscordAdapter()
    adapter._client = SimpleNamespace(get_channel=lambda _: channel if cached else None,
                                      fetch_channel=fetch)
    asyncio.run(adapter._fetch_and_send(123, 'x' * 4200))
    assert list(map(len, sent)) == [2000, 2000, 200]


def test_discord_fetch_failure_is_not_reported_as_success():
    async def fetch(_):
        raise RuntimeError('fetch failed')
    adapter = DiscordAdapter()
    adapter._client = SimpleNamespace(get_channel=lambda _: None, fetch_channel=fetch)
    with pytest.raises(RuntimeError, match='fetch failed'):
        asyncio.run(adapter._fetch_and_send(123, 'hello'))


def test_telegram_uses_owned_loop_without_application_loop_attribute():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
    thread = threading.Thread(target=run)
    thread.start()
    assert ready.wait(2)
    sent = []
    async def send_message(**kwargs):
        sent.append(kwargs)
    adapter = TelegramAdapter()
    adapter._app = SimpleNamespace(bot=SimpleNamespace(send_message=send_message))
    adapter._loop, adapter._thread, adapter._running = loop, thread, True
    try:
        adapter.send_message('123', 'x' * 4200)
        assert [len(item['text']) for item in sent] == [4096, 104]
        assert all(item['chat_id'] == 123 for item in sent)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()


def test_websocket_send_failure_returns_false():
    async def send_text(_):
        raise ConnectionError('closed')
    assert asyncio.run(AikoWeb._safe_send(SimpleNamespace(send_text=send_text), 'hello')) is False


def test_websocket_ignores_malformed_messages_and_keeps_chat(monkeypatch):
    import json
    import queue
    from interface.webui import auth, webui
    web = AikoWeb.__new__(AikoWeb)
    web._login_event = threading.Event()
    web._fully_ready = False
    web._on_user_active = lambda _: None
    web._clients_lock = threading.Lock()
    web._clients, web._client_users = set(), {}
    web._lock = threading.Lock()
    web._active_uid = None
    web._input_q = queue.Queue()
    monkeypatch.setattr(webui, '_load_stored_display_name', lambda _: None)
    monkeypatch.setitem(auth.sessions, 'interface-test-session', {
        'user_id': 'alice', 'username': 'Alice', 'created_at': webui.bioclock.local_now()})
    class Socket:
        cookies = {'session_id': auth.signer.dumps('interface-test-session')}
        def __init__(self):
            self.messages = iter(['[]', 'null', '42', '{',
                json.dumps({'type': 'user_input', 'text': 12}),
                json.dumps({'type': 'user_input', 'text': 'hello'})])
        async def accept(self):
            pass
        async def receive(self):
            try:
                return {'text': next(self.messages)}
            except StopIteration:
                return {'type': 'websocket.disconnect'}
    asyncio.run(web._ws_handler(Socket()))
    assert web._input_q.get_nowait() == ('hello', 'alice', 'Alice')
    assert not web._clients


def test_telegram_polling_owns_loop_and_stops(monkeypatch):
    from interface.adapter import telegram
    started = threading.Event()
    calls = []
    class App:
        def add_handler(self, _):
            pass
        def run_polling(self, **kwargs):
            calls.append(kwargs)
            loop = asyncio.get_event_loop()
            loop.call_soon(started.set)
            loop.run_forever()
        def stop_running(self):
            asyncio.get_running_loop().stop()
    app = App()
    builder = SimpleNamespace(token=lambda _: SimpleNamespace(build=lambda: app))
    monkeypatch.setattr(telegram, 'Application', SimpleNamespace(builder=lambda: builder))
    adapter = TelegramAdapter({'TELEGRAM_BOT_TOKEN': 'fake'})
    monkeypatch.setattr(adapter, '_read_config', lambda: {'token': 'fake'})
    adapter.start()
    try:
        assert started.wait(2)
        original = adapter._thread
        adapter.start()
        assert adapter._thread is original
    finally:
        adapter.stop()
    assert not original.is_alive()
    assert adapter._loop is None
    assert calls == [dict(drop_pending_updates=True, stop_signals=None, close_loop=False)]
