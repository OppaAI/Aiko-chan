"""Unit tests for reasoning-only LLM replies in social monitors.

Thinking-model builds (ministral/granite) sometimes emit the answer as
reasoning_content with empty content. The monitors must speak that text
instead of logging 'empty reply' and staying silent (seen in production:
@oppa.ai.bot mentions matched but never answered).
"""

import importlib

import pytest

threads = importlib.import_module("interface.mcp_server.social.services.threads")
bluesky = importlib.import_module("interface.mcp_server.social.services.bluesky")
mastodon = importlib.import_module("interface.mcp_server.social.services.mastodon")


class _Msg:
    def __init__(self, content="", reasoning=None):
        self.content = content
        if reasoning is not None:
            self.reasoning_content = reasoning


@pytest.mark.parametrize("mod", [threads, bluesky, mastodon])
def test_reasoning_only_reply_is_used(mod):
    assert mod._message_text(_Msg("", "Hello! *purr* on it")) == "Hello! *purr* on it"


@pytest.mark.parametrize("mod", [threads, bluesky, mastodon])
def test_normal_content_preferred_over_reasoning(mod):
    assert mod._message_text(_Msg("Real answer", "thinking noise")) == "Real answer"


@pytest.mark.parametrize("mod", [threads, bluesky, mastodon])
def test_fully_empty_message_stays_empty(mod):
    assert mod._message_text(_Msg("", "")) == ""


@pytest.mark.parametrize("mod", [threads, bluesky, mastodon])
def test_content_block_list_is_joined(mod):
    msg = _Msg([{"text": "Hello"}, {"text": "there"}])
    assert mod._message_text(msg) == "Hello there"
