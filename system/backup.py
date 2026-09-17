"""
system/backup.py

File-level backup + factory reset for the Jetson Orin Nano robot.

Two backup types (see config or --help):
    settings — <USER_SPACE_ROOT>/<user_id>/ only (~130M: memory.db,
               profile/USER.md, schedule.json, workspace, agentic state).
               Irreplaceable. Cheap enough for hourly runs.
    full     — settings plus the Aiko-chan codebase minus regenerable
               weight (models/ 22G, .venv/, build/, logs/, .git/ —
               .git restores via clone at the pinned commit in the manifest).

Destinations (comma-separated, e.g. --backup-dest usb,nas):
    usb      encrypted USB stick (LUKS) mount, default /mnt/usb.
    microsd  UNENCRYPTED microSD path — code snapshot only. Settings/DBs
             are refused here, never silently allowed.
    nas      rclone remote (default agi-nas), Cloud = agi-pdrive.
    pc       operator-defined rclone remote or path; skipped with a warning
             when unconfigured rather than failing the whole run.
    dir:<path>  explicit local path (tests, one-offs).

Safety rules (load-bearing, covered by tests/unit/test_backup.py):
  - SQLite files (*.db, *.sqlite*) are NEVER copied raw. Live DBs run in
    WAL mode (-shm/-wal siblings); an rclone-style file copy mid-write
    produces a corrupt snapshot. Every DB goes through the sqlite3 backup
    API (or the SQLCipher connection when SQLITE_ENCRYPTION=1) into the
    staging tree, then integrity-checked there.
  - rclone transport uses `copy` (additive) plus an optional --backup-dir
    for versioned deletes — never bare `sync`, which would mirror a
    damaged source onto the only good copy.
  - The manifest (sha256 per file + integrity verdicts + git commit +
    sizes + utc timestamp) lives OUTSIDE the user dir so a factory reset
    cannot eat the proof it depended on. --factory-reset refuses without
    a verified manifest younger than 24h unless forced with explicit input.

No heavy imports at module load: torch/vector-store/voice subsystems are
never touched, so --backup stays fast on 8 GB.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class BackupError(RuntimeError):
    """Raised when a backup cannot be completed or verified safely."""


# ── tunables (env-overridable, documented in --help) ──────────────────────────

BACKUP_TYPES = ("settings", "full")
BACKUP_DESTS = ("usb", "microsd", "nas", "cloud", "pc")

CODE_EXCLUDES_DIRS = {
    "models", ".venv", "build", "logs", "log", "__pycache__",
    ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    ".benchmarks", "node_modules",
}
CODE_EXCLUDES_SUFFIXES = (".pyc", ".log", ".tmp")
# Live SQLite sidecars must never be copied raw — only quiesced snapshots.
SQLITE_SIDECARS = ("-shm", "-wal", "-journal")
LOCK_SUFFIXES = (".lock",)
MANIFEST_MAX_AGE_SECONDS = 24 * 3600


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def usb_mount() -> Path:
    return Path(_env("AIKO_BACKUP_USB_MOUNT", "/mnt/usb")).expanduser()


def microsd_path() -> Path:
    return Path(_env("AIKO_BACKUP_MICROSD_PATH", "/mnt/microsd/aiko")).expanduser()


def nas_remote() -> str:
    return _env("AIKO_BACKUP_NAS_REMOTE", "agi-nas")


def cloud_remote() -> str:
    return _env("AIKO_BACKUP_CLOUD_REMOTE", "agi-pdrive")


def pc_remote() -> str:
    return _env("AIKO_BACKUP_PC_REMOTE", "")


def manifest_root() -> Path:
    """Directory holding backup manifests — OUTSIDE any user dir."""
    from system.userspace import _user_state_root_value

    return Path(_user_state_root_value()).expanduser() / "backups"


# ── small helpers ─────────────────────────────────────────────────────────────

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sqlite_file(path: Path) -> bool:
    return path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def _should_skip_settings(rel: Path) -> bool:
    name = rel.name
    if name.endswith(SQLITE_SIDECARS):
        return True
    if name.endswith(LOCK_SUFFIXES):
        return True
    if name.endswith(".tmp"):
        return True
    return False


def _should_skip_code(rel: Path) -> bool:
    parts = rel.parts
    if any(part in CODE_EXCLUDES_DIRS for part in parts):
        return True
    if rel.suffix.lower() in CODE_EXCLUDES_SUFFIXES:
        return True
    return False


# ── quiesced SQLite snapshot ──────────────────────────────────────────────────

def quiesce_sqlite(src: Path, dst: Path, user_id: str) -> None:
    """Copy one live SQLite DB consistently via the backup API.

    Never copies the file bytes: WAL-mode DBs (-shm/-wal) copied raw
    mid-write restore as corrupt. Uses the SQLCipher connection when
    encryption is enabled, else stdlib sqlite3. Raises BackupError instead
    of leaving a half-written destination.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        try:
            from system.secure import connect_sqlite, sqlite_encryption_enabled

            if sqlite_encryption_enabled():
                src_conn = connect_sqlite(src, user_id=user_id)
            else:
                src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=10.0)
        except Exception as exc:
            raise BackupError(f"cannot open {src} for snapshot: {exc}") from exc
        try:
            dst_conn = sqlite3.connect(str(tmp), timeout=10.0)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            try:
                src_conn.close()
            except Exception:
                pass
    except BackupError:
        raise
    except Exception as exc:
        raise BackupError(f"snapshot of {src} failed: {exc}") from exc
    try:
        tmp.replace(dst)
    except OSError as exc:
        raise BackupError(f"cannot finalize snapshot {dst}: {exc}") from exc


def sqlite_integrity_ok(path: Path, user_id: str) -> bool:
    """PRAGMA integrity_check on a snapshot (never on the live file)."""
    try:
        try:
            from system.secure import connect_sqlite, sqlite_encryption_enabled

            if sqlite_encryption_enabled():
                conn = connect_sqlite(path, user_id=user_id)
            else:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
        except Exception:
            return False
        try:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return bool(rows) and str(rows[0][0]).lower() == "ok"
    except Exception:
        return False


# ── snapshots ─────────────────────────────────────────────────────────────────

@dataclass
class SnapshotFile:
    rel: str            # path relative to the snapshot root
    sha256: str
    bytes: int
    integrity: str = ""  # "ok" for checked DBs, "" otherwise


@dataclass
class Snapshot:
    kind: str           # "settings" | "code"
    root: Path          # staging dir holding the snapshot tree
    files: list[SnapshotFile] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)


def snapshot_settings(user_id: str, staging: Path) -> Snapshot:
    """Quiesced copy of <USER_SPACE_ROOT>/<user_id>/ into staging/settings/."""
    from system.userspace import user_state_dir

    src_root = user_state_dir(user_id)
    if not src_root.is_dir():
        raise BackupError(f"no user state dir for {user_id!r} at {src_root}")
    dst_root = staging / "settings"
    snap = Snapshot(kind="settings", root=staging)
    for src in sorted(src_root.rglob("*")):
        if not src.is_file() or src.is_symlink():
            continue
        rel = src.relative_to(src_root)
        if _should_skip_settings(rel):
            continue
        dst = dst_root / rel
        if _is_sqlite_file(src):
            quiesce_sqlite(src, dst, user_id)
            ok = sqlite_integrity_ok(dst, user_id)
            snap.files.append(SnapshotFile(
                rel=str(Path("settings") / rel),
                sha256=sha256_file(dst),
                bytes=dst.stat().st_size,
                integrity="ok" if ok else "FAILED",
            ))
            if not ok:
                raise BackupError(f"integrity check failed for snapshotted {rel}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            snap.files.append(SnapshotFile(
                rel=str(Path("settings") / rel),
                sha256=sha256_file(dst),
                bytes=dst.stat().st_size,
            ))
    return snap


def snapshot_code(repo_root: Path, staging: Path) -> Snapshot:
    """Copy of the codebase minus regenerable weight (models/.venv/build/...)."""
    repo_root = repo_root.expanduser().resolve()
    dst_root = staging / "code"
    snap = Snapshot(kind="code", root=staging)
    for src in sorted(repo_root.rglob("*")):
        if not src.is_file() or src.is_symlink():
            continue
        rel = src.relative_to(repo_root)
        if _should_skip_code(rel):
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        snap.files.append(SnapshotFile(
            rel=str(Path("code") / rel),
            sha256=sha256_file(dst),
            bytes=dst.stat().st_size,
        ))
    return snap


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── manifest ──────────────────────────────────────────────────────────────────

@dataclass
class BackupManifest:
    version: int = 1
    backup_id: str = ""
    utc: str = ""
    backup_type: str = ""          # "settings" | "full"
    user_id: str = ""
    git_commit: str = ""
    dests: list[str] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    total_bytes: int = 0
    verified: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def write_manifest(staging: Path, manifest: BackupManifest) -> Path:
    manifest.utc = manifest.utc or _utc_now().isoformat()
    manifest.backup_id = manifest.backup_id or f"{manifest.backup_type}-{_utc_stamp()}"
    path = staging / f"manifest-{manifest.backup_id}.json"
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    final_dir = manifest_root()
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, final_dir / path.name)
    return final_dir / path.name


def read_manifest(path: Path) -> BackupManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    files = data.get("files", [])
    return BackupManifest(
        version=int(data.get("version", 1)),
        backup_id=str(data.get("backup_id", "")),
        utc=str(data.get("utc", "")),
        backup_type=str(data.get("backup_type", "")),
        user_id=str(data.get("user_id", "")),
        git_commit=str(data.get("git_commit", "")),
        dests=list(data.get("dests", [])),
        files=files,
        total_bytes=int(data.get("total_bytes", 0)),
        verified=bool(data.get("verified", False)),
    )


def verify_staging(staging: Path, manifest: BackupManifest) -> None:
    """Re-hash every staged file and re-run integrity checks. Raises."""
    by_rel = {f["rel"]: f for f in manifest.files}
    if not by_rel:
        raise BackupError("manifest lists no files — refusing to call that verified")
    for rel, entry in sorted(by_rel.items()):
        path = staging / rel
        if not path.is_file():
            raise BackupError(f"staged file missing: {rel}")
        if sha256_file(path) != entry["sha256"]:
            raise BackupError(f"checksum mismatch: {rel}")
        if entry.get("integrity") == "ok" and _is_sqlite_file(path):
            if not sqlite_integrity_ok(path, manifest.user_id):
                raise BackupError(f"integrity re-check failed: {rel}")
    manifest.verified = True


def latest_verified_manifest(user_id: str, backup_type: str = "") -> BackupManifest | None:
    """Newest verified manifest for a user, or None."""
    root = manifest_root()
    if not root.is_dir():
        return None
    best: BackupManifest | None = None
    for path in sorted(root.glob("manifest-*.json")):
        try:
            manifest = read_manifest(path)
        except (OSError, ValueError):
            continue
        if manifest.user_id != user_id or not manifest.verified:
            continue
        if backup_type and manifest.backup_type != backup_type:
            continue
        if best is None or manifest.utc > best.utc:
            best = manifest
    return best


def manifest_fresh_enough(manifest: BackupManifest, max_age_seconds: int = MANIFEST_MAX_AGE_SECONDS) -> bool:
    try:
        when = datetime.fromisoformat(manifest.utc)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (_utc_now() - when).total_seconds() <= max_age_seconds


# ── transport ─────────────────────────────────────────────────────────────────

def _rclone(*args: str, timeout: int = 600) -> None:
    binary = shutil.which("rclone")
    if not binary:
        raise BackupError("rclone not found on PATH — cannot reach NAS/Cloud/PC remotes")
    proc = subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise BackupError(f"rclone {' '.join(args[:3])} failed: {(proc.stderr or proc.stdout).strip()[:300]}")


def transport_to_dest(staging: Path, dest: str, backup_id: str, dry_run: bool = False) -> str:
    """Copy a verified staging tree to one destination. Returns human summary."""
    name = dest.strip().lower()
    if name == "usb":
        target = usb_mount() / "aiko-backups" / backup_id
        if dry_run:
            return f"usb → {target} (dry-run)"
        if not usb_mount().is_dir():
            raise BackupError(f"USB mount not present at {usb_mount()} — unlock/mount it first")
        shutil.copytree(staging, target, dirs_exist_ok=True)
        return f"usb → {target}"
    if name == "microsd":
        # Unencrypted tier: code snapshot only, never settings/DBs.
        if (staging / "settings").exists():
            raise BackupError("microsd is UNENCRYPTED — refusing settings/DB snapshot there (use usb/nas)")
        target = microsd_path() / backup_id
        if dry_run:
            return f"microsd → {target} (dry-run)"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, target, dirs_exist_ok=True)
        return f"microsd → {target}"
    if name.startswith("dir:"):
        target = Path(name[4:]).expanduser()
        if dry_run:
            return f"dir → {target} (dry-run)"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, target / backup_id, dirs_exist_ok=True)
        return f"dir → {target / backup_id}"
    if name in ("nas", "cloud", "pc"):
        remote = {"nas": nas_remote(), "cloud": cloud_remote(), "pc": pc_remote()}[name]
        if not remote:
            return f"{name} skipped (unconfigured remote)"
        timestamped = f"{remote}:aiko-backups/{backup_id}"
        versioned = f"{remote}:aiko-backups/_versions/{backup_id}"
        if dry_run:
            return f"{name} → {timestamped} (dry-run)"
        # Additive copy; deletes go to a versioned dir, never vanish.
        _rclone("copy", str(staging), timestamped,
                "--backup-dir", versioned, "--suffix", f".{_utc_stamp()}")
        return f"{name} → {timestamped}"
    raise BackupError(f"unknown dest {dest!r} (want usb|microsd|nas|cloud|pc|dir:<path>)")


# ── orchestration ─────────────────────────────────────────────────────────────

def resolve_user_id(explicit: str = "") -> str:
    uid = (explicit or "").strip()
    if uid:
        return uid
    try:
        from system.userspace import resolve_owner_user_id

        owner = resolve_owner_user_id()
        if owner:
            return owner
    except Exception:
        pass
    uid = (os.getenv("AIKO_USER_ID") or "").strip()
    if uid:
        return uid
    raise BackupError("cannot resolve user id — pass --user or set AIKO_USER_ID")


def run_backup(
    backup_type: str,
    dests: list[str],
    user_id: str = "",
    repo_root: str | Path = "",
    staging_parent: str | Path = "",
    dry_run: bool = False,
) -> BackupManifest:
    """Build snapshots, verify, transport. Returns the verified manifest."""
    if backup_type not in BACKUP_TYPES:
        raise BackupError(f"unknown backup type {backup_type!r} (want settings|full)")
    dests = [d.strip().lower() for d in dests if d.strip()]
    if not dests:
        raise BackupError("no destinations given (want usb|microsd|nas|cloud|pc|dir:<path>)")
    uid = resolve_user_id(user_id)
    repo = Path(repo_root or Path(__file__).resolve().parent.parent).expanduser()
    parent = Path(staging_parent or (Path(os.getenv("TMPDIR", "/tmp")) / "aiko-backup")).expanduser()
    staging = parent / f"{backup_type}-{_utc_stamp()}"
    staging.mkdir(parents=True, exist_ok=True)

    manifest = BackupManifest(backup_type=backup_type, user_id=uid, dests=dests)
    settings_snap = snapshot_settings(uid, staging)
    manifest.files.extend(
        {"rel": f.rel, "sha256": f.sha256, "bytes": f.bytes, "integrity": f.integrity}
        for f in settings_snap.files
    )
    if backup_type == "full":
        code_snap = snapshot_code(repo, staging)
        manifest.files.extend(
            {"rel": f.rel, "sha256": f.sha256, "bytes": f.bytes, "integrity": f.integrity}
            for f in code_snap.files
        )
        manifest.git_commit = git_commit(repo)
    manifest.total_bytes = sum(f["bytes"] for f in manifest.files)

    verify_staging(staging, manifest)
    summaries = [transport_to_dest(staging, d, manifest.backup_id or f"{backup_type}-{_utc_stamp()}", dry_run=dry_run)
                 for d in dests]
    if dry_run:
        shutil.rmtree(staging, ignore_errors=True)
        manifest.dests = [f"{s}" for s in summaries]
        return manifest
    manifest_path = write_manifest(staging, manifest)
    manifest.dests = summaries
    # Re-write with transport summaries + verified flag for the record.
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def perform_factory_reset(user_id: str = "", max_age_seconds: int = MANIFEST_MAX_AGE_SECONDS) -> dict:
    """Wipe <USER_SPACE_ROOT>/<user_id>/ after a verified fresh backup exists.

    Callers own the human gates (two prompts in main.py). This function fails
    closed: no manifest, stale manifest, or unverified manifest → BackupError,
    nothing deleted. Returns a small report dict.
    """
    from system.userspace import user_state_dir

    uid = resolve_user_id(user_id)
    manifest = latest_verified_manifest(uid)
    if manifest is None:
        raise BackupError(
            f"factory reset refused: no verified backup manifest for {uid!r} — "
            "run `python main.py --backup --backup-type full` first"
        )
    if not manifest_fresh_enough(manifest, max_age_seconds):
        raise BackupError(
            f"factory reset refused: newest verified backup {manifest.backup_id} "
            f"({manifest.utc}) is older than {max_age_seconds // 3600}h — back up first"
        )
    target = user_state_dir(uid)
    removed_files = 0
    removed_bytes = 0
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_symlink() or path.is_file():
            try:
                removed_bytes += path.stat().st_size
                path.unlink()
                removed_files += 1
            except OSError:
                pass
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return {
        "user_id": uid,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "backup_id": manifest.backup_id,
        "backup_utc": manifest.utc,
    }


__all__ = [
    "BackupError",
    "BACKUP_TYPES",
    "BACKUP_DESTS",
    "run_backup",
    "verify_staging",
    "latest_verified_manifest",
    "manifest_fresh_enough",
    "perform_factory_reset",
    "snapshot_settings",
    "snapshot_code",
    "quiesce_sqlite",
]
