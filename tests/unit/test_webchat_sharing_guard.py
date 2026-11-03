"""Unit tests for the personal-sharing guard on the webchat path."""
from __future__ import annotations

from cognition.think import _is_personal_sharing


def test_sharing_narration_detected():
    assert _is_personal_sharing(
        "Actually it was cloudy and rainy day. We arrived there at 11am by bus, "
        "the admission fee was $7 because of open-weekend promotion."
    )
    assert _is_personal_sharing("I will tell you more about what we did inside")
    assert _is_personal_sharing("We rode the wooden coaster and ate mini donuts")
    assert _is_personal_sharing("Let me tell you what happened")


def test_questions_are_not_sharing():
    assert not _is_personal_sharing("Did we ride the coaster at PNE?")
    assert not _is_personal_sharing("what time does it open today?")


def test_explicit_internet_requests_are_never_sharing():
    assert not _is_personal_sharing("we went there, search what time it closes")
    assert not _is_personal_sharing("verify online if the fair is open today")


def test_webchat_disables_net_block_to_avoid_double_search(monkeypatch):
    """webchat() delegates to chat(web_search=True); the lighter hint-regex
    websearch_net block must be off so one turn doesn't fire two searches."""
    from cognition import think as think_module

    calls = {}

    def fake_chat(self, user_input, **kwargs):
        calls.update(kwargs)
        return "ok"

    think = object.__new__(think_module.AikoThink)
    monkeypatch.setattr(think_module.AikoThink, "chat", fake_chat)
    assert think.webchat("what is the current price of eggs? search online") == "ok"
    assert calls.get("web_search") is True
    assert calls.get("websearch_net") is False
