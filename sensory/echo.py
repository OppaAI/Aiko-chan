"""Bounded playback timing/text references; no microphone audio is retained."""
from __future__ import annotations

from collections import OrderedDict, deque
from difflib import SequenceMatcher
import hashlib
import math
import threading
import time
import unicodedata

from system.config import env_int


def echo_tail_seconds() -> float:
    return max(0, min(2000, env_int('ASR_ECHO_TAIL_MS', 400))) / 1000.0


def _normalize(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())[:1200]


class PlaybackEchoGuard:
    """Track actual playback and reject close transcript matches during overlap.

    Timing gates do the primary work. Text matching is a second layer for
    barge-in/AEC residuals, not an acoustic echo canceller or an identity gate.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._pending = OrderedDict()
        self._played = deque(maxlen=16)
        self._until = 0.0

    def remember(self, wav: bytes, text: str, user_id: str):
        digest = hashlib.sha256(wav).digest()
        with self._lock:
            self._pending[digest] = (_normalize(text), user_id)
            self._pending.move_to_end(digest)
            while len(self._pending) > 8:
                self._pending.popitem(last=False)

    def started(self, wav: bytes, duration: float):
        now = time.monotonic()
        duration = duration if math.isfinite(duration) else 0.0
        until = now + max(0.0, min(120.0, duration)) + echo_tail_seconds()
        with self._lock:
            self._until = max(self._until, until)
            reference = self._pending.pop(hashlib.sha256(wav).digest(), None)
            if reference:
                text, uid = reference
                self._played.append((now, until, text, uid))

    def stopped(self):
        now = time.monotonic()
        with self._lock:
            if self._until > now:
                self._until = now + echo_tail_seconds()
                self._played = deque(((start, min(end, self._until), text, uid)
                                      for start, end, text, uid in self._played), maxlen=16)

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._until - time.monotonic())

    def is_echo(self, text: str, user_id: str, started: float, stopped: float) -> bool:
        candidate = _normalize(text)
        # Never suppress short acknowledgments/commands such as "stop"/"yes".
        if len(candidate) < 8:
            return False
        now = time.monotonic()
        with self._lock:
            references = [reference for reference in self._played
                          if reference[1] >= now - 60.0]
        for begin, end, spoken, uid in references:
            if uid != user_id or started > end or stopped < begin or not spoken:
                continue
            if candidate in spoken:
                return True
            if SequenceMatcher(None, candidate, spoken, autojunk=False).ratio() >= .90:
                return True
        return False
