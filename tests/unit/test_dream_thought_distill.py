"""Unit tests for the dream journal's inner-thought distillation.

The midnight join used to append up to 40 raw inner-speech entries as a
bullet list, making journal entries very long. Now the thoughts are
distilled to a short paragraph (LLM) with a bounded truncated fallback.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cognition.consolidate.dream import _distill_thoughts


def _thoughts(n):
    return [
        {"id": f"t{i}", "kind": "reflection", "text": f"thought number {i} " * 20}
        for i in range(n)
    ]


def test_distill_uses_llm_when_available():
    with patch(
        "cognition.consolidate.dream._llm_chat",
        return_value="A quiet undercurrent of curiosity and care.",
    ) as m:
        out = _distill_thoughts(_thoughts(30))
    assert m.called
    assert out == "A quiet undercurrent of curiosity and care."
    # Raw entries must NOT leak into the output.
    assert "thought number" not in out


def test_distill_fallback_is_bounded_on_llm_failure():
    with patch(
        "cognition.consolidate.dream._llm_chat",
        side_effect=RuntimeError("llm down"),
    ):
        out = _distill_thoughts(_thoughts(40))
    # Fallback: at most 8 fragments, each truncated — still far shorter
    # than 40 raw entries.
    assert out.startswith("Fragments:")
    assert len(out) < 2000


def test_distill_empty():
    assert _distill_thoughts([]) == ""


def test_distill_empty_llm_response_falls_back():
    with patch(
        "cognition.consolidate.dream._llm_chat", return_value="   "
    ):
        out = _distill_thoughts(_thoughts(5))
    assert out.startswith("Fragments:")
