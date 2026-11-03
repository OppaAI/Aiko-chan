"""Freenove Robot Dog FNK0062 (ESP32-WROVER) wire protocol.

Transport: TCP socket to the dog on port 5000 (port 8000 is the camera
video stream — out of scope here). Message format is a single-char
command, '#' separators, integer params, a trailing '#', and a newline,
e.g. "F#0#0#0#5#\\n".

Protocol extracted from Freenove's open firmware
(Freenove_ESP32_Dog_Firmware, ESP-IDF): BluetoothOrders.h,
TaskCommandService.cpp, TaskMotionService.cpp.

Command set:
  F  move_any     F#x#y#rot#speed#   continuous locomotion; keeps going
                                    until stopped — F#0#0#0#5# stops.
  O  tricks       O#1..6#  1=say_hello 2=push_up 3=stretch_self
                           4=turn_around 5=sit_down 6=dancing
  A  posture      A#0/1/2#
  B  body height  B#h#
  E  twist        E#x#y#z#  (body lean)
  C  RGB LED      C#mode#r#g#b#
  D  buzzer       D#freq#
  M  auto-walk    M#0/1#
  R  move speed   R#speed#

Sense-back channels (FUTURE afferent path — not wired into flysense here):
  H  ultrasonic distance query → returns distance in cm
  I  battery voltage query → returns volts
The query builders below exist so a later phase can add a polling
afferent; nothing in this module calls them.

No-spam rule: the firmware docs warn that the dog's MCU has limited
processing — commands are fire-once. The backend therefore only
transmits when a wire string *changes* (see FreenoveBackend in body.py),
and body.py's per-actuator rate limits bound how fast intents change.
The one exception is the e-stop: safety outranks spam.

Fail-soft: DogLink never raises. Short connect timeout (~1s, env
FREENOVE_DOG_TIMEOUT). Unreachable dog → log once per host (with a
cooldown) and report unavailable; the backend falls back to null
behavior (wire strings computed, nothing sent).

Config (env):
  FREENOVE_DOG_HOST     default 192.168.4.1 (the dog's AP default)
  FREENOVE_DOG_PORT     default 5000
  FREENOVE_DOG_TIMEOUT  connect timeout seconds, default 1.0
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time

log = logging.getLogger("aiko.fly.freenove")

_DEFAULT_HOST = "192.168.4.1"
_DEFAULT_PORT = 5000
_DEFAULT_TIMEOUT = 1.0
_RECONNECT_BACKOFF_S = 1.0

# Log-once cooldown per host so an unreachable dog doesn't spam logs.
_WARN_COOLDOWN_S = 60.0
_warned_at: dict[str, float] = {}
_warn_lock = threading.Lock()


def _env(name: str, default: str) -> str:
    try:
        from system.config import env_str
        v = env_str(name, None)
        if v is not None:
            return str(v)
    except Exception:
        pass
    return os.getenv(name) or default


def host() -> str:
    return _env("FREENOVE_DOG_HOST", _DEFAULT_HOST).strip() or _DEFAULT_HOST


def port() -> int:
    try:
        return max(1, min(65535, int(_env("FREENOVE_DOG_PORT",
                                         str(_DEFAULT_PORT)))))
    except Exception:
        return _DEFAULT_PORT


def timeout_s() -> float:
    try:
        return max(0.05, min(10.0,
                             float(_env("FREENOVE_DOG_TIMEOUT",
                                        str(_DEFAULT_TIMEOUT)))))
    except Exception:
        return _DEFAULT_TIMEOUT


def encode(cmd: str, *params: int) -> str:
    """Build one wire message, e.g. encode('F', 0, 0, 0) -> 'F#0#0#0#\\n'."""
    c = (str(cmd or " ")[:1]).upper()
    parts = [c]
    for p in params:
        try:
            parts.append(str(int(p)))
        except Exception:
            parts.append("0")
    return "#".join(parts) + "#\n"


# ── command builders ────────────────────────────────────────────────

def move(x: int = 0, y: int = 0, rot: int = 0, speed: int = 0) -> str:
    """Continuous locomotion vector."""
    return encode("F", x, y, rot, speed)


STOP = encode("F", 0, 0, 0, 5)


def trick(n: int) -> str:
    """1=say_hello 2=push_up 3=stretch_self 4=turn_around 5=sit_down 6=dancing."""
    n = max(1, min(6, int(n)))
    return encode("O", n)


def posture(n: int) -> str:
    return encode("A", max(0, min(2, int(n))))


def body_height(h: int) -> str:
    return encode("B", int(h))


def twist(x: int = 0, y: int = 0, z: int = 0) -> str:
    """Body lean. Surrogate for gaze/head motion (the kit has no head servo)."""
    return encode("E", x, y, z)


def led(mode: int = 1, r: int = 0, g: int = 0, b: int = 0) -> str:
    """RGB LED. mode semantics are firmware-specific; 1 = static color."""
    return encode("C", mode,
                  max(0, min(255, int(r))),
                  max(0, min(255, int(g))),
                  max(0, min(255, int(b))))


def buzzer(freq: int) -> str:
    return encode("D", max(0, int(freq)))


def auto_walk(on: bool) -> str:
    return encode("M", 1 if on else 0)


def move_speed(speed: int) -> str:
    return encode("R", int(speed))


# Sense-back query builders — future afferent path only, unused for now.
def query_ultrasonic() -> str:
    """'H' → the dog replies with distance in cm. Not wired up yet."""
    return encode("H")


def query_battery() -> str:
    """'I' → the dog replies with battery volts. Not wired up yet."""
    return encode("I")


# ── transport ───────────────────────────────────────────────────────

class DogLink:
    """Lazy TCP link to the dog. Never raises; never blocks a turn.

    Connects on first send with a short timeout. The socket is kept open
    between sends; any failure closes it and marks the link unavailable
    until a later send succeeds. Failed connections back off briefly.
    """

    def __init__(self, host_: str | None = None, port_: int | None = None,
                 timeout: float | None = None):
        self._host = host_ or host()
        self._port = port_ if port_ is not None else port()
        self._timeout = timeout if timeout is not None else timeout_s()
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._unavailable = True
        self._next_attempt_at = 0.0

    @property
    def available(self) -> bool:
        """Last known reachability (False until a send succeeds)."""
        return not self._unavailable

    def _warn_once(self, msg: str) -> None:
        now = time.monotonic()
        with _warn_lock:
            last = _warned_at.get(self._host, 0.0)
            if now - last < _WARN_COOLDOWN_S:
                return
            _warned_at[self._host] = now
        log.warning("freenove dog %s", msg)

    def _ensure(self) -> bool:
        if self._sock is not None:
            return True
        if time.monotonic() < self._next_attempt_at:
            return False
        try:
            s = socket.create_connection(
                (self._host, self._port), timeout=self._timeout)
            s.settimeout(self._timeout)
            self._sock = s
            log.info("freenove dog connected at %s:%d",
                     self._host, self._port)
            return True
        except Exception as exc:
            self._close_locked()
            self._unavailable = True
            self._next_attempt_at = time.monotonic() + _RECONNECT_BACKOFF_S
            self._warn_once(
                f"unreachable at {self._host}:{self._port} ({exc}); "
                "backend falls back to null (no commands sent)")
            return False

    def _close_locked(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    def send(self, wire: str) -> bool:
        """Send one newline-terminated wire message. Never raises."""
        try:
            data = wire if wire.endswith("\n") else wire + "\n"
            with self._lock:
                if not self._ensure():
                    return False
                assert self._sock is not None
                self._sock.sendall(data.encode("ascii", "replace"))
                self._unavailable = False
                self._next_attempt_at = 0.0
            return True
        except Exception as exc:
            with self._lock:
                self._close_locked()
                self._unavailable = True
                self._next_attempt_at = time.monotonic() + _RECONNECT_BACKOFF_S
            self._warn_once(f"send failed ({exc}); link dropped")
            return False

    def estop(self) -> bool:
        """Hardware e-stop: F#0#0#0#5#. Bypasses change detection upstream."""
        return self.send(STOP)

    def close(self) -> None:
        try:
            with self._lock:
                self._close_locked()
        except Exception:
            pass


# Module-level shared link (the dog is one physical device, not per-user).
_link: DogLink | None = None
_link_lock = threading.Lock()


def get_link() -> DogLink:
    """Shared link, created lazily from current env (test-resettable)."""
    global _link
    with _link_lock:
        if _link is None:
            _link = DogLink()
        return _link


def reset_link() -> None:
    """Drop the shared link (tests / host change). Never raises."""
    global _link
    try:
        with _link_lock:
            if _link is not None:
                _link.close()
            _link = None
    except Exception:
        pass
