"""Catalog builder + loader contract (fast; the 512MB artefact is opt-in)."""
import hashlib
import json

import pytest

from cognition.fly_runtime.catalog import ConnectomeCatalog


def _write(payload: dict, path) -> None:
    body = json.dumps({k: v for k, v in payload.items() if k != "checksum"},
                      sort_keys=True, separators=(",", ":")).encode()
    payload = dict(payload)
    payload["checksum"] = hashlib.sha256(body).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["checksum"]


def test_mini_catalog_roundtrip_and_tamper_rejected(tmp_path):
    payload = {
        "source": "test", "version": "t0",
        "nodes": [{"id": "1", "type": "KC", "region": "unknown"},
                  {"id": "2", "type": "MBON01", "region": "unknown"}],
        "edges": [{"source": "1", "target": "2", "weight": 7, "sign": "unknown"}],
    }
    p = tmp_path / "mini.json"
    _write(payload, p)
    cat = ConnectomeCatalog.from_path(p)
    assert cat.summary()["nodes"] == 2
    assert cat.ids_for(types=["MBON01"]) == ["2"]
    sg = cat.subgraph(["2"], budget=10, max_hops=2)
    assert sg["truncated"] is False
    p.write_text(p.read_text(encoding="utf-8").replace('"weight": 7', '"weight": 8'),
                 encoding="utf-8")
    with pytest.raises(ValueError):
        ConnectomeCatalog.from_path(p)


@pytest.mark.integration
def test_real_catalog_if_present():
    """Opt-in: AIKO_FLY_CATALOG_PATH=... run explicitly, never by default."""
    import os
    path = (os.getenv("AIKO_FLY_CATALOG_PATH", "") or "").strip()
    if not path:
        pytest.skip("no catalog configured")
    cat = ConnectomeCatalog.from_path(path)  # checksum verified on load
    s = cat.summary()
    assert s["nodes"] > 100000 and s["edges"] > 1000000
    assert cat.ids_for(types=["MBON01", "MBON14"])
