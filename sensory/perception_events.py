"""Bounded, coalesced opportunities to speak from consented browser perception."""
from collections import OrderedDict
import re
import threading
import time
import weakref

_lock = threading.RLock()
_pending = OrderedDict()
_stable = OrderedDict()
_watchers = OrderedDict()
_scenes = OrderedDict()
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
            for key in list(_stable):
                if key[0] == uid:
                    _stable.pop(key, None)
            _scenes.pop(uid, None)
        for event in list(_wakeups):
            event.set()


def offer(uid, signature, description, channel="cues"):
    """Require a repeated cue; retain only the latest opportunity per user."""
    now = time.monotonic()
    with _lock:
        if uid not in _watchers:
            return
        key = (uid, channel)
        previous = _stable.get(key)
        if previous is None or previous[0] != signature or now - previous[2] > (90 if channel == "scene" else 15):
            _stable[key] = (signature, 1, now, None)
        else:
            old, count, _, emitted = previous
            _stable[key] = (old, count + 1, now, emitted)
            if count >= 1 and emitted != signature:
                _stable[key] = (old, count + 1, now, signature)
                if description:
                    _pending[uid] = (now, str(description)[:500])
                    for event in list(_wakeups):
                        event.set()
        _stable.move_to_end(key)
        while len(_stable) > 64:
            victim, _ = _stable.popitem(last=False)
            _pending.pop(victim[0], None)
        while len(_pending) > 32:
            _pending.popitem(last=False)


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


def scene(uid, description):
    words = frozenset(re.findall(r"[a-z]{4,}", description.lower()))
    if len(words) < 4:
        return
    with _lock:
        if uid not in _watchers:
            return
        previous = _scenes.get(uid)
        if previous is None:
            _scenes[uid] = words
        elif len(words & previous) / max(1, len(words | previous)) < .35:
            before = _pending.get(uid)
            offer(uid, "scene_change", "The camera scene may have changed. Compare only fresh visual evidence; speak only if useful.", "scene")
            if _pending.get(uid) is not before:
                _scenes[uid] = words
        else:
            _stable.pop((uid, "scene"), None)
        _scenes.move_to_end(uid)
        while len(_scenes) > 32:
            victim, _ = _scenes.popitem(last=False)
            watching(victim, False)
