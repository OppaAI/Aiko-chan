"""Unit tests: lorebook matcher + recall token budgets. No services needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cognition.knowledge.lorebook import load_entries, lorebook_context_for, match_entries
from cognition.recall_budget import est_tokens, fit_block


def _entries():
    return [
        {"keys": ["odessa", "weather station"], "content": "Odessa facts here."},
        {"keys": ["birthday"], "content": "Birthday facts here."},
    ]


def test_match_basic():
    hits = match_entries("tell me about ODESSA please", _entries())
    assert len(hits) == 1 and hits[0]["content"] == "Odessa facts here."


def test_match_no_substring():
    assert match_entries("odessan", _entries()) == []
    assert match_entries("nothing relevant", _entries()) == []


def test_match_multiword_key():
    hits = match_entries("the weather station is old", _entries())
    assert len(hits) == 1


def test_load_entries(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"keys": ["x"], "content": "y"}\n'
                 '{"keys": [], "content": "no-keys"}\n'
                 'not json\n'
                 '{"keys": ["z"], "content": "off", "enabled": false}\n',
                 encoding="utf-8")
    entries = load_entries(tmp_path)
    assert len(entries) == 1 and entries[0]["keys"] == ["x"]
    assert load_entries(tmp_path / "missing") == []


def test_lorebook_context_block(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"keys": ["odessa"], "content": "O."}\n', encoding="utf-8")
    block = lorebook_context_for("visit odessa?", lore_dir=tmp_path, max_chars=600)
    assert block.startswith("<lore>") and "O." in block
    assert lorebook_context_for("unrelated", lore_dir=tmp_path) == ""


def test_est_tokens():
    assert est_tokens("") == 0
    assert 5 <= est_tokens("one two three four five") <= 9


def test_fit_block_under_cap_unchanged():
    assert fit_block("short text", "memory") == "short text"
    assert fit_block("", "memory") == ""


def test_fit_block_trims_with_marker(monkeypatch):
    import cognition.recall_budget as rb
    monkeypatch.setitem(rb.RECALL_TOKEN_CAPS, "memory", 5)
    out = fit_block("word " * 100, "memory")
    assert out.endswith(rb.TRIM_MARKER)
    assert est_tokens(out) <= 12


def test_fit_block_unknown_label_untouched():
    assert fit_block("x" * 5000, "no-such-source") == "x" * 5000
