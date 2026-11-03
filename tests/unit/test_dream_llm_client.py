"""P2 regression: cognition/consolidate/dream._get_llm_client() must be
thread-safe — concurrent first calls (schedule + chat threads) must share
one OpenAI client instead of each building a client + httpx pool."""

from __future__ import annotations

import threading
import time

import cognition.consolidate.dream as dream


class TestDreamLlmClientLock:
    def test_concurrent_first_calls_build_single_client(self, monkeypatch):
        constructions: list = []
        count_lock = threading.Lock()

        class FakeClient:
            def __init__(self, **kwargs):
                # Widen the check-then-set race window: without the lock,
                # 16 threads hammering this reliably build multiple clients.
                time.sleep(0.05)
                with count_lock:
                    constructions.append(kwargs)

        monkeypatch.setattr(dream, "OpenAI", FakeClient)
        monkeypatch.setattr(dream, "_LLM_CLIENT", None)

        results: list = []

        def grab():
            results.append(dream._get_llm_client())

        threads = [threading.Thread(target=grab) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert all(not t.is_alive() for t in threads)
        assert len(constructions) == 1
        assert len({id(r) for r in results}) == 1
