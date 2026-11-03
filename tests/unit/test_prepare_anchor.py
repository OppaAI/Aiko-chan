"""P2 regression: system/prepare.py's Grasp anchor block must surface (not
swallow) persona_context()/get_all() failures — a corrupt memory DB would
otherwise silently disable the post-auth relevance anchor."""

from __future__ import annotations

import logging

from system import prepare


class ExplodingMemorize:
    def switch_user(self, uid):
        return True

    def cleanup(self):
        pass

    def persona_context(self):
        raise RuntimeError("db corrupt")

    def get_all(self):
        raise RuntimeError("db corrupt")


def test_anchor_failures_are_logged_not_swallowed(caplog):
    with caplog.at_level(logging.WARNING, logger="system.prepare"):
        prepare.run_post_auth("u1", memorize=ExplodingMemorize())
    messages = [r.message for r in caplog.records]
    assert any("persona_context" in m for m in messages)
    assert any("get_all" in m for m in messages)
