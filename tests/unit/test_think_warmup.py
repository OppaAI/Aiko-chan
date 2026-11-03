"""Regression tests for BUG-2 (P1): LLM warmup / client timeouts.

- The OpenAI client must carry timeout=LLM_TIMEOUT at construction, so any
  future completions call that forgets a per-call timeout degrades instead
  of hanging on the SDK's 600s default (which once wedged boot for up to
  10 minutes via the unbounded join_warmup).
- _warmup_llm must pass an explicit timeout on the warmup call itself.
"""
from __future__ import annotations

from cognition.think import AikoThink, LLM_TIMEOUT


class TestThinkClientTimeout:
    def test_client_has_timeout_backstop(self):
        think = AikoThink()
        assert think._client.timeout == LLM_TIMEOUT

    def test_warmup_call_passes_timeout(self, monkeypatch):
        think = AikoThink()
        seen = {}

        def fake_create(**kwargs):
            seen.update(kwargs)

            class _Resp:
                pass

            return _Resp()  # _warmup_llm ignores the result

        monkeypatch.setattr(think._client.chat.completions, "create", fake_create)
        think._warmup_llm()
        assert seen.get("timeout") == LLM_TIMEOUT
