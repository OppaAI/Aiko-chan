"""Tests for recall diversity filtering (cognition/memory/diversity.py).

Exact-text duplicates are collapsed upstream; these cover near-duplicates:
same motif, different wording, filling multiple recall slots.
"""
import unittest

from cognition.memory.diversity import diversify


def _row(text, **kw):
    row = {"id": kw.pop("id", text[:12]), "memory": text}
    row.update(kw)
    return row


class TestDiversify(unittest.TestCase):
    def test_empty_and_single(self):
        self.assertEqual(diversify([]), [])
        self.assertEqual(diversify(None), [])
        one = [_row("Oppa likes black tea")]
        self.assertEqual(diversify(one), one)

    def test_near_duplicates_collapse(self):
        rows = [
            _row("Oppa bought birthday fruit tarts in Vancouver"),
            _row("Oppa purchased birthday fruit tarts while in Vancouver"),
            _row("The birthday fruit tarts Oppa got in Vancouver"),
            _row("Nobunaga built Azuchi castle on the hill"),
        ]
        kept = diversify(rows)
        texts = [r["memory"] for r in kept]
        self.assertEqual(len(kept), 2)
        self.assertIn("Oppa bought birthday fruit tarts in Vancouver", texts)
        self.assertIn("Nobunaga built Azuchi castle on the hill", texts)

    def test_distinct_rows_all_kept(self):
        rows = [
            _row("Oppa likes black tea in the morning"),
            _row("Nobunaga built Azuchi castle on the hill"),
            _row("The fly connectome maps one hundred thousand neurons"),
        ]
        self.assertEqual(len(diversify(rows)), 3)

    def test_pinned_always_survives(self):
        rows = [
            _row("Oppa bought birthday fruit tarts in Vancouver"),
            _row("Oppa purchased birthday fruit tarts while in Vancouver",
                 pinned=True),
        ]
        kept = diversify(rows)
        self.assertEqual(len(kept), 2)
        self.assertTrue(kept[1]["pinned"])

    def test_order_preserved_best_first(self):
        rows = [
            _row("zebra crossing regulations downtown"),
            _row("quantum tunneling microscopy advances"),
        ]
        kept = diversify(rows)
        self.assertEqual([r["memory"] for r in kept],
                         [r["memory"] for r in rows])

    def test_trace_key_supported(self):
        rows = [
            {"trace": "User: hi\nAiko: hello there friend"},
            {"trace": "User: hi\nAiko: hello there pal"},
        ]
        self.assertEqual(len(diversify(rows)), 1)


if __name__ == "__main__":
    unittest.main()
