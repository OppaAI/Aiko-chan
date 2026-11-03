"""Small, scrolling startup display shared by WebUI boot and the plain CLI."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import tomllib


class StartupConsole:
    def __init__(self, labels, *, mode="WebUI", stream=None):
        self.labels = labels
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = bool(self.stream.isatty()) and os.getenv("TERM") != "dumb" and os.getenv("LOG_CONSOLE") != "1"
        self.color = self.interactive and "NO_COLOR" not in os.environ
        self._lock = threading.Lock()
        self._active = {}
        self._closed = False
        self._frame = 0
        self._started = time.monotonic()
        try:
            release = version("Aiko-chan")
        except PackageNotFoundError:
            try:
                with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as project:
                    release = tomllib.load(project)["project"]["version"]
            except (OSError, KeyError, ValueError):
                release = "development"
        title = f"AIKO  v{release}" if release != "development" else "AIKO  development"
        if self.color:
            title = f"\033[1;36m{title}\033[0m"
        self._write(f"\n  {title}\n  Local AI companion · {mode}\n  {'-' * 48}\n\n")

    def _write(self, text):
        self.stream.write(text)
        self.stream.flush()

    def _clear(self):
        if self.interactive:
            self._write("\r\033[2K")

    def _label(self, key):
        # Labels are display-only; keep control characters out of terminal output.
        return " ".join(str(self.labels.get(key, key)).split())

    def _render(self):
        if not self.interactive or not self._active:
            return
        frames = ("|", "/", "-", "\\")
        label = " · ".join(self._label(key) for key in self._active)
        width = max(10, shutil.get_terminal_size((80, 24)).columns - 2)
        text = f"  {frames[self._frame % len(frames)]}  {label}"
        self._frame += 1
        self._clear()
        self._write(text[:width])

    def loading(self, key):
        with self._lock:
            if self._closed or key in self._active:
                return
            self._active[key] = time.monotonic()
            if self.interactive:
                self._render()
            else:
                self._write(f"  ...  {self._label(key)}\n")

    def _complete(self, key, status):
        with self._lock:
            if self._closed:
                return
            started = self._active.pop(key, None)
            elapsed = f"  {time.monotonic() - started:.1f}s" if started is not None else ""
            self._clear()
            self._write(f"  {status:<4} {self._label(key)}{elapsed}\n")
            self._render()

    def done(self, key):
        self._complete(key, "OK")

    def skip(self, key):
        self._complete(key, "SKIP")

    def spin(self, stop):
        if not self.interactive:
            return
        while not stop.wait(0.1):
            with self._lock:
                if self._closed:
                    return
                self._render()

    def finish(self, message="Ready", *, failed=False):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._clear()
            self._active.clear()
            status = "FAILED" if failed else "READY"
            self._write(f"\n  {status}  {message}  ({time.monotonic() - self._started:.1f}s)\n\n")
