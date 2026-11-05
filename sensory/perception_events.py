"""Bounded, coalesced opportunities to speak from consented browser perception."""
from collections import OrderedDict
import threading
import time
import weakref

_lock = threading.RLock()
_pending = OrderedDict()
_watchers = OrderedDict()
_attention_calls = OrderedDict()
_wakeups = weakref.WeakSet()


def subscribe(event):
    with _lock:
        _wakeups.add(event)


def unsubscribe(event):
    with _lock:
        _wakeups.discard(event)


def is_watching(uid):
    with _lock:
        return uid in _watchers


def claim_attention(uid):
    """Bound automatic VLM calls across sockets and camera restarts."""
    now = time.monotonic()
    from system.config import env_float
    cooldown = max(15, min(3600, env_float('CAMERA_ATTENTION_COOLDOWN_S', 90)))
    with _lock:
        if uid not in _watchers or now - _attention_calls.get(uid, -float('inf')) < cooldown:
            return False
        _attention_calls[uid] = now
        _attention_calls.move_to_end(uid)
        while len(_attention_calls) > 32:
            _attention_calls.popitem(last=False)
        return True


def attention_ready(uid, reason):
    """Only initiate after the triggered frame has fresh VLM observations."""
    if reason not in {'open_palm', 'camera_gaze'}:
        return
    with _lock:
        if uid not in _watchers:
            return
        label = 'a raised open palm' if reason == 'open_palm' else 'approximate camera-directed gaze'
        _pending[uid] = (time.monotonic(), f'The user signaled attention with {label}. Fresh camera observations are available. Respond naturally to this invitation without claiming certainty about eye contact.')
        for event in list(_wakeups):
            event.set()


def watching(uid, active):
    with _lock:
        if active:
            _watchers[uid] = True
            _watchers.move_to_end(uid)
            while len(_watchers) > 32:
                victim, _ = _watchers.popitem(last=False)
                watching(victim, False)
        else:
            _watchers.pop(uid, None)
            _pending.pop(uid, None)
        for event in list(_wakeups):
            event.set()


def peek(uid):
    with _lock:
        value = _pending.get(uid)
        if value and time.monotonic() - value[0] <= 20 and uid in _watchers:
            return value
        _pending.pop(uid, None)
        return None


def consume(uid, expected=None):
    with _lock:
        if expected is None or _pending.get(uid) == expected:
            _pending.pop(uid, None)
