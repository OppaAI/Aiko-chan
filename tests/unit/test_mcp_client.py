"""MCP status handling and connection ownership without external services."""
import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace as NS

import pytest

from agentic.mcp_client.client import MCPClient


@pytest.mark.parametrize('response,expected', [
    (NS(content=[NS(text='tool failed')], isError=True), {'ok': False, 'result': 'tool failed', 'error': 'tool failed'}),
    (NS(content=[NS(text='{"ok":false}')], isError=False), {'ok': False}),
    (NS(content=[NS(text='first'), NS(text='second')]), {'ok': True, 'result': 'first\nsecond'}),
    (NS(content=[], structuredContent={'value': 3}, isError=True),
     {'ok': False, 'value': 3, 'error': 'MCP tool returned an error'}),
])
def test_protocol_status_and_complete_text(response, expected):
    client = MCPClient()
    class Session:
        async def call_tool(self, name, arguments):
            return response
    client._session = Session()
    result = asyncio.run(client.call_tool('test', {}))
    assert result == expected


def test_one_connection_owner_for_concurrent_calls_and_shutdown(monkeypatch):
    owners, exits = [], []
    sdk, sessions, stdio = ModuleType('mcp'), ModuleType('mcp.client.session'), ModuleType('mcp.client.stdio')
    sdk.StdioServerParameters = lambda **kwargs: kwargs

    @asynccontextmanager
    async def transport(params):
        owner = asyncio.current_task()
        owners.append(owner)
        try:
            yield object(), object()
        finally:
            assert asyncio.current_task() is owner
            exits.append('transport')

    class Session:
        def __init__(self, *args):
            pass
        async def __aenter__(self):
            self.owner = asyncio.current_task()
            return self
        async def __aexit__(self, *args):
            assert asyncio.current_task() is self.owner
            exits.append('session')
        async def initialize(self):
            await asyncio.sleep(.01)  # expose concurrent initialization
        async def call_tool(self, name, arguments):
            return NS(content=[NS(text=json.dumps({'ok': True, **arguments}))])

    sessions.ClientSession, stdio.stdio_client = Session, transport
    monkeypatch.setitem(sys.modules, 'mcp', sdk)
    monkeypatch.setitem(sys.modules, 'mcp.client.session', sessions)
    monkeypatch.setitem(sys.modules, 'mcp.client.stdio', stdio)
    client = MCPClient()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: client.call_tool_sync('test', {'i': i}), range(4)))
        assert [result['i'] for result in results] == list(range(4))
        assert len(owners) == 1
    finally:
        thread = client._loop_thread
        client.close_sync()
    assert exits == ['session', 'transport']
    assert not thread.is_alive()


def test_sync_timeout_cancels_coroutine():
    client = MCPClient()
    cancelled = threading.Event()
    async def wait():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    try:
        with pytest.raises(TimeoutError):
            client._run_coro(wait, timeout=.05)
        assert cancelled.wait(5)
    finally:
        client.close_sync()


def test_rebuilding_tool_defs_does_not_retain_old_bridges():
    client = MCPClient()
    client._tools = {'x': {'inputSchema': {}}}
    for _ in range(5):
        client.get_bridge_tool_defs()
    assert len(client._bridges) == 1


def test_registry_catalog_accepts_callable_bridge_instances():
    from agentic.mcp_client.client import MCPToolBridge
    from agentic.registry import ToolRegistry
    registry = ToolRegistry()
    registry.register('mock', '', handler=MCPToolBridge(MCPClient(), 'mock', {}))
    exported = registry.to_catalog()['tools'][0]
    assert exported['handler'] == 'agentic.mcp_client.client:MCPToolBridge'
