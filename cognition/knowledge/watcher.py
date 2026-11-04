"""Event-driven watcher for the knowledge drop folder.

Replaces the periodic interval scan of the workspace knowledge folder with
an inotify-based watcher: zero idle CPU (blocks in the kernel on select),
wakes the instant a file lands, and triggers the knowledge ingest handler.

Linux-only (inotify). On platforms or environments without inotify, falls
back to nothing (the interval scan remains available as a safety net).
"""
from __future__ import annotations

import ctypes
import contextvars
import os
import select
import struct
import threading
from pathlib import Path

from system.log import get_logger
from system.userspace import user_workspace_root, current_user_id, set_current_user_id

log = get_logger(__name__)

# ── inotify constants (Linux ABI) ─────────────────────────────────────────────
_IN_NONBLOCK = 0o4000  # O_NONBLOCK
_IN_CLOEXEC = 0o2000000
_IN_CREATE = 0x00000100
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_IN_MOVED_FROM = 0x00000040
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ISDIR = 0x40000000
_WATCH_MASK = _IN_CREATE | _IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_MOVED_FROM | _IN_DELETE_SELF | _IN_MOVE_SELF

_INOTIFY_EVENT = struct.Struct("iIII")  # wd, mask, cookie, len


class KnowledgeFolderWatcher:
    """Watches <workspace>/<knowledge_dir> and triggers ingest on new files."""

    def __init__(
        self,
        *,
        knowledge_dir: str = "library",
        debounce_seconds: float = 1.5,
        on_files=None,
        user_id: str | None = None,
    ):
        self.user_id = user_id or current_user_id()
        self._knowledge_dir = knowledge_dir
        self._debounce = debounce_seconds
        self._on_files = on_files
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._wd: int | None = None
        self._watches: dict[int, Path] = {}

    @property
    def folder(self) -> Path:
        root = user_workspace_root(self.user_id).resolve()
        folder = (root / self._knowledge_dir).resolve()
        if folder != root and root not in folder.parents:
            raise ValueError("knowledge folder must stay inside the user workspace")
        return folder

    def _sync_watches(self) -> None:
        """Follow directory changes without following directory symlinks."""
        root = self.folder
        folders = {root}
        for parent, dirs, _files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if not name.startswith('.')
                       and not (Path(parent) / name).is_symlink()]
            folders.update(Path(parent) / name for name in dirs)
        for wd, path in list(self._watches.items()):
            if path not in folders:
                _inotify_remove_watch(self._fd, wd)
                self._watches.pop(wd, None)
        watched = set(self._watches.values())
        for path in sorted(folders - watched):
            wd = _inotify_add_watch(self._fd, str(path), _WATCH_MASK)
            self._watches[wd] = path
        self._wd = next((wd for wd, path in self._watches.items() if path == root), None)

    def start(self) -> bool:
        """Start the watcher thread. Returns False when inotify is unavailable."""
        if self._thread is not None and self._thread.is_alive():
            return True
        try:
            self._fd = _inotify_init()
        except Exception:
            log.warning("knowledge watcher: inotify unavailable — using interval scan only")
            return False
        try:
            folder = self.folder
            folder.mkdir(parents=True, exist_ok=True)
            self._watches.clear()
            self._sync_watches()
            log.info("knowledge watcher: watching %s", folder)
        except Exception as exc:
            _inotify_close(self._fd)
            self._fd = None
            self._watches.clear()
            log.warning("knowledge watcher: failed to add watch: %s", exc)
            return False
        self._stop.clear()
        context = contextvars.copy_context()
        context.run(set_current_user_id, self.user_id)
        self._thread = threading.Thread(target=context.run, args=(self._run,), name="aiko-knowledge-watch", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._fd is not None:
            _inotify_close(self._fd)
            self._fd = None
        self._watches.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Block in kernel until an inotify event or stop-wakeup window.
                r, _, _ = select.select([self._fd], [], [], 0.5)
            except OSError:
                continue
            if not r:
                continue

            pending: list[Path] = []
            while True:
                try:
                    data = os.read(self._fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                pending.extend(_parse_events(data, self._watches, overflow_root=self.folder))
                if _needs_watch_refresh(data):
                    try:
                        self._sync_watches()
                    except OSError:
                        log.exception("knowledge watcher: recursive watch refresh failed; interval scan remains available")

            # Debounce: wait for the copy to settle, then fire once.
            if self._stop.wait(self._debounce):
                break
            if not pending:
                continue

            # A populated directory may arrive in a single rename event. Its
            # existing files have no individual close-write events here.
            candidates = set()
            for path in pending:
                if path.is_dir() and not path.is_symlink():
                    candidates.update(p for p in path.rglob('*') if p.is_file())
                else:
                    candidates.add(path)
            root = self.folder
            new_files = sorted(p for p in candidates if p.is_file()
                               and root in p.resolve().parents
                               and not any(part.startswith('.') for part in p.relative_to(root).parts))
            if not new_files:
                continue
            log.info("knowledge watcher: %d new file(s) detected", len(new_files))
            if self._on_files is not None:
                try:
                    self._on_files(new_files)
                except Exception:
                    log.exception("knowledge watcher: ingest callback failed")


def _inotify_init() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(_IN_NONBLOCK | _IN_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    return fd


def _inotify_add_watch(fd: int, path: str, mask: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    wd = libc.inotify_add_watch(fd, path.encode(), ctypes.c_uint32(mask))
    if wd < 0:
        raise OSError(ctypes.get_errno(), f"inotify_add_watch failed for {path}")
    return wd


def _inotify_close(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _inotify_remove_watch(fd: int, wd: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_rm_watch(ctypes.c_int(fd), ctypes.c_int(wd))


def _parse_events(data: bytes, folder: Path | dict[int, Path], *, overflow_root: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    offset = 0
    while offset + _INOTIFY_EVENT.size <= len(data):
        wd, mask, _cookie, name_len = _INOTIFY_EVENT.unpack_from(data, offset)
        offset += _INOTIFY_EVENT.size
        if name_len > len(data) - offset:
            break
        name = data[offset : offset + name_len].rstrip(b"\x00").decode("utf-8", errors="replace")
        offset += name_len
        if mask & _IN_Q_OVERFLOW and overflow_root is not None:
            paths.append(overflow_root)
            continue
        parent = folder.get(wd) if isinstance(folder, dict) else folder
        if mask & _IN_IGNORED and isinstance(folder, dict):
            folder.pop(wd, None)
        if parent is None:
            continue
        if mask & _WATCH_MASK and name:
            paths.append(parent / name)
    return paths


def _needs_watch_refresh(data: bytes) -> bool:
    offset = 0
    while offset + _INOTIFY_EVENT.size <= len(data):
        _wd, mask, _cookie, length = _INOTIFY_EVENT.unpack_from(data, offset)
        offset += _INOTIFY_EVENT.size
        if length > len(data) - offset:
            break
        if mask & (_IN_ISDIR | _IN_IGNORED | _IN_Q_OVERFLOW | _IN_DELETE_SELF | _IN_MOVE_SELF):
            return True
        offset += length
    return False
