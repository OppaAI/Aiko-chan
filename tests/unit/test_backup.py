"""Tests for system/backup.py + main.py --backup/--factory-reset wiring.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_backup.py -q --override-ini="addopts="
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

import main as main_module
from main import parse_args


@pytest.fixture()
def user_root(monkeypatch, tmp_path):
    """Isolated USER_SPACE_ROOT with one user holding a live WAL-mode DB."""
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path / ".aiko"))
    monkeypatch.setenv("AIKO_USER_ID", "test_owner")
    import system.userspace as userspace

    udir = userspace.user_state_dir("test_owner")
    (udir / "profile").mkdir(parents=True, exist_ok=True)
    (udir / "profile" / "USER.md").write_text("# owner\n", encoding="utf-8")
    (udir / "workspace").mkdir(parents=True, exist_ok=True)
    (udir / "workspace" / "note.md").write_text("hello\n", encoding="utf-8")
    db = udir / "memory" / "memory.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t (v) VALUES ('one'), ('two')")
    conn.commit()
    yield tmp_path / ".aiko"
    try:
        conn.close()
    except Exception:
        pass


@pytest.fixture()
def repo_root(tmp_path):
    repo = tmp_path / "repo"
    (repo / "agentic").mkdir(parents=True)
    (repo / "agentic" / "a.py").write_text("x=1\n", encoding="utf-8")
    (repo / "models").mkdir()
    (repo / "models" / "big.gguf").write_bytes(b"0" * 1024)
    (repo / ".venv").mkdir()
    (repo / ".venv" / "p.bin").write_bytes(b"1")
    (repo / "logs").mkdir()
    (repo / "logs" / "a.log").write_text("noise", encoding="utf-8")
    return repo


def test_settings_snapshot_quiesces_live_db(user_root):
    from system import backup as B

    staging = user_root / "staging"
    snap = B.snapshot_settings("test_owner", staging)
    rels = sorted(f.rel for f in snap.files)
    assert any(r.endswith("memory.db") for r in rels)
    # WAL sidecars + journals must never be copied raw.
    assert not any(r.endswith(("-shm", "-wal", "-journal", ".lock")) for r in rels)
    assert any(r.endswith("USER.md") for r in rels)
    dst = staging / "settings" / "memory" / "memory.db"
    conn = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchall()[0][0] == "ok"
        assert conn.execute("SELECT count(*) FROM t").fetchall()[0][0] == 2
    finally:
        conn.close()


def test_code_snapshot_excludes_weight(user_root, repo_root):
    from system import backup as B

    staging = user_root / "staging-code"
    snap = B.snapshot_code(repo_root, staging)
    rels = sorted(f.rel for f in snap.files)
    assert "code/agentic/a.py" in rels
    assert not any("models" in r or ".venv" in r or "logs" in r for r in rels)


def test_manifest_write_verify_roundtrip(user_root):
    from system import backup as B

    staging = user_root / "staging"
    snap = B.snapshot_settings("test_owner", staging)
    manifest = B.BackupManifest(backup_type="settings", user_id="test_owner", dests=["usb"])
    manifest.files = [
        {"rel": f.rel, "sha256": f.sha256, "bytes": f.bytes, "integrity": f.integrity}
        for f in snap.files
    ]
    manifest.total_bytes = sum(f.bytes for f in snap.files)
    B.verify_staging(staging, manifest)
    assert manifest.verified is True
    path = B.write_manifest(staging, manifest)
    assert path.is_file()
    assert B.read_manifest(path).verified is True


def test_verify_catches_tamper(user_root):
    from system import backup as B

    staging = user_root / "staging"
    snap = B.snapshot_settings("test_owner", staging)
    manifest = B.BackupManifest(backup_type="settings", user_id="test_owner")
    manifest.files = [
        {"rel": f.rel, "sha256": f.sha256, "bytes": f.bytes, "integrity": f.integrity}
        for f in snap.files
    ]
    (staging / "settings" / "workspace" / "note.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(B.BackupError, match="checksum mismatch"):
        B.verify_staging(staging, manifest)


def test_microsd_refuses_settings(user_root):
    from system import backup as B

    staging = user_root / "staging"
    B.snapshot_settings("test_owner", staging)
    with pytest.raises(B.BackupError, match="UNENCRYPTED"):
        B.transport_to_dest(staging, "microsd", "bid-1")


def test_run_backup_settings_to_dir(user_root):
    from system import backup as B

    dest = user_root / "dest"
    manifest = B.run_backup("settings", [f"dir:{dest}"], user_id="test_owner",
                            staging_parent=user_root / "tmp")
    assert manifest.verified is True
    copied = dest / manifest.backup_id / "settings" / "workspace" / "note.md"
    assert copied.is_file()
    # Manifest record lives outside the user dir and round-trips.
    rec = B.latest_verified_manifest("test_owner")
    assert rec is not None and rec.backup_id == manifest.backup_id


def test_factory_reset_needs_fresh_manifest(user_root):
    from system import backup as B

    with pytest.raises(B.BackupError, match="no verified backup"):
        B.perform_factory_reset("test_owner")


def test_factory_reset_wipes_after_backup(user_root):
    from system import backup as B

    dest = user_root / "dest"
    manifest = B.run_backup("settings", [f"dir:{dest}"], user_id="test_owner",
                            staging_parent=user_root / "tmp")
    report = B.perform_factory_reset("test_owner")
    assert report["backup_id"] == manifest.backup_id
    assert report["removed_files"] > 0
    # Manifest survives outside the wiped user dir.
    assert B.latest_verified_manifest("test_owner") is not None


def _argv(*args):
    return ["main.py", *args]


def test_parse_args_backup_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv("--backup", "--backup-type", "full",
                                           "--backup-dest", "usb,nas", "--dry-run"))
    ns = parse_args()
    assert ns.backup is True and ns.backup_type == "full"
    assert ns.backup_dest == "usb,nas" and ns.dry_run is True


def test_parse_args_factory_reset(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv("--factory-reset", "--user", "OppaAI"))
    ns = parse_args()
    assert ns.factory_reset is True and ns.user == "OppaAI"


def test_maintenance_group_stays_exclusive(monkeypatch):
    import pytest as _pytest

    monkeypatch.setattr(sys, "argv", _argv("--backup", "--factory-reset"))
    with _pytest.raises(SystemExit):
        parse_args()
