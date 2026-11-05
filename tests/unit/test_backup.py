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


def test_backup_preserves_path_case_and_delivers_manifest(user_root):
    from system import backup as B
    dest = user_root / 'MyBackups'
    staging = user_root / 'stage'
    first = B.run_backup('settings', [f'dir:{dest}'], 'test_owner', staging_parent=staging)
    second = B.run_backup('settings', [f'dir:{dest}'], 'test_owner', staging_parent=staging)
    assert first.backup_id != second.backup_id
    manifest_path = dest / first.backup_id / f'manifest-{first.backup_id}.json'
    assert B.read_manifest(manifest_path).verified
    assert (dest / first.backup_id / 'settings' / 'profile' / 'USER.md').is_file()
    assert not list(staging.iterdir())


def test_skipped_destination_does_not_authorize_reset(user_root, monkeypatch):
    from system import backup as B
    monkeypatch.delenv('AIKO_BACKUP_PC_REMOTE', raising=False)
    with pytest.raises(B.BackupError, match='skipped'):
        B.run_backup('settings', ['pc'], 'test_owner', staging_parent=user_root / 'stage')
    assert B.latest_verified_manifest('test_owner') is None
    assert not list((user_root / 'stage').iterdir())


def test_micro_sd_code_only_transport(user_root, monkeypatch):
    from system import backup as B
    staging = user_root / 'stage'
    staging.mkdir()
    (staging / 'code.txt').write_text('code')
    monkeypatch.setenv('AIKO_BACKUP_MICROSD_PATH', str(user_root / 'sd'))
    B.transport_to_dest(staging, 'microsd', 'test')
    assert (user_root / 'sd' / 'test' / 'code.txt').read_text() == 'code'


def test_manifest_rejects_future_timestamp_and_string_verified(tmp_path):
    from datetime import timedelta
    from system import backup as B
    m = B.BackupManifest(utc=(B._utc_now() + timedelta(days=2)).isoformat())
    assert not B.manifest_fresh_enough(m)
    path = tmp_path / 'manifest.json'
    path.write_text('{"verified": "false"}')
    assert not B.read_manifest(path).verified


def test_code_snapshot_skips_secrets_and_prunes_excluded_tree(repo_root, tmp_path, monkeypatch):
    from system import backup as B
    (repo_root / '.env').write_text('TOKEN=private')
    (repo_root / 'age-key.txt').write_text('private')
    (repo_root / '.env.example').write_text('TOKEN=example')
    original = B.os.walk
    visited = []
    def walk(*args, **kwargs):
        for item in original(*args, **kwargs):
            visited.append(Path(item[0]).name)
            yield item
    monkeypatch.setattr(B.os, 'walk', walk)
    snap = B.snapshot_code(repo_root, tmp_path / 'out')
    rels = {f.rel for f in snap.files}
    assert 'code/.env' not in rels and 'code/age-key.txt' not in rels
    assert 'code/.env.example' in rels
    assert 'models' not in visited and '.venv' not in visited


def test_staging_must_stay_outside_source(user_root):
    from system import backup as B
    from system.userspace import user_state_dir
    with pytest.raises(B.BackupError, match='outside'):
        B.snapshot_settings('test_owner', user_state_dir('test_owner') / 'stage')


def test_encrypted_snapshot_uses_same_connection_provider_for_destination(user_root, monkeypatch):
    from system import backup as B, secure
    source = user_root / 'test_owner' / 'memory' / 'memory.db'
    destination = user_root / 'snapshot.db'
    seen = []
    def connect(path, *, user_id):
        seen.append((Path(path), user_id))
        return sqlite3.connect(path)
    monkeypatch.setattr(secure, 'sqlite_encryption_enabled', lambda: True)
    monkeypatch.setattr(secure, 'connect_sqlite', connect)
    B.quiesce_sqlite(source, destination, 'test_owner')
    assert seen == [(source, 'test_owner'), (destination.with_name('snapshot.db.part'), 'test_owner')]
    assert B.sqlite_integrity_ok(destination, 'test_owner')
