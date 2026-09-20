"""Parity tests: SqliteCatalog must answer exactly like ConnectomeCatalog.

Builds one tiny synthetic graph, converts it via build_sqlite(), and checks
ids_for / ids_matching_prefixes / subgraph / summary agree on both backends.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_fly_sqlite_catalog.py -q --override-ini="addopts="
"""
from __future__ import annotations

import pytest

from cognition.fly_runtime.catalog import ConnectomeCatalog, Edge, Node
from cognition.fly_runtime.sqlite_catalog import SqliteCatalog
from cognition.fly_runtime.tools.build_catalog_sqlite import build_sqlite

_NODES = [
    Node("n1", "VS", "unknown"),
    Node("n2", "MBON01", "unknown"),
    Node("n3", "DNp01", "unknown"),
    Node("n4", "AMMC-A1", "unknown"),
    Node("n5", "ER5", "unknown"),
]
_EDGES = [
    Edge("n1", "n2", 2.0, "unknown"),
    Edge("n2", "n3", 1.5, "unknown"),
    Edge("n1", "n3", 0.5, "unknown"),
    Edge("n4", "n2", 3.0, "unknown"),
    Edge("n5", "n3", 1.0, "unknown"),
    # dangling endpoints must be dropped by both backends
    Edge("n1", "ghost", 9.0, "unknown"),
    Edge("ghost", "n2", 9.0, "unknown"),
]
_META = {"source": "test", "version": "v0", "checksum": "abc"}


@pytest.fixture()
def both(tmp_path):
    mem = ConnectomeCatalog(_NODES, _EDGES, source="test", version="v0", checksum="abc")
    dest = tmp_path / "t.db"
    stats = build_sqlite(
        [{"id": n.id, "type": n.type, "region": n.region} for n in _NODES],
        [{"source": e.source, "target": e.target, "weight": e.weight, "sign": e.sign} for e in _EDGES],
        _META, dest,
    )
    assert stats["edges_dropped"] == 2
    sql = SqliteCatalog(dest)
    yield mem, sql
    sql.close()


def _canon(graph: dict) -> dict:
    return {
        "nodes": sorted((n.id, n.type) for n in graph["nodes"]),
        "edges": sorted((e.source, e.target, e.weight) for e in graph["edges"]),
        "truncated": graph["truncated"],
        "source": graph["source"],
        "version": graph["version"],
        "checksum": graph["checksum"],
    }


def test_ids_for_parity(both):
    mem, sql = both
    assert sql.ids_for(types=("VS", "DNp01")) == mem.ids_for(types=("VS", "DNp01")) == ["n1", "n3"]
    assert sql.ids_for(types=("nope",)) == mem.ids_for(types=("nope",)) == []


def test_ids_matching_prefixes_parity(both):
    mem, sql = both
    assert sql.ids_matching_prefixes(["vs", "mb"]) == ["n1", "n2"]
    assert sql.ids_matching_prefixes([]) == []


def test_subgraph_parity(both):
    mem, sql = both
    assert _canon(sql.subgraph(["n1"], budget=100, max_hops=4)) == _canon(mem.subgraph(["n1"], budget=100, max_hops=4))
    assert _canon(sql.subgraph(["n1", "n4"], budget=2, max_hops=4)) == _canon(mem.subgraph(["n1", "n4"], budget=2, max_hops=4))
    assert _canon(sql.subgraph(["ghost"], budget=100)) == _canon(mem.subgraph(["ghost"], budget=100))
    assert _canon(sql.subgraph(["n1"], budget=0)) == _canon(mem.subgraph(["n1"], budget=0))


def test_summary_and_nodes_view(both):
    mem, sql = both
    assert sql.summary() == {"nodes": 5, "edges": 5, "source": "test", "version": "v0", "checksum": "abc"}
    assert sql.nodes["n2"].type == "MBON01"
    assert len(sql.nodes) == 5
    assert "n9" not in sql.nodes
    with pytest.raises(KeyError):
        sql.nodes["n9"]


def test_schema_guard(tmp_path):
    import sqlite3

    dest = tmp_path / "bad.db"
    conn = sqlite3.connect(str(dest))
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta(key, value) VALUES('schema', '999')")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="schema"):
        SqliteCatalog(dest)
