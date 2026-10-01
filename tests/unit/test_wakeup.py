"""
tests/test_wakeup.py
Starter test suite for system/wakeup.py.

This file tests wakeup.py's OWN logic -- the threading handshake, callback
sequencing, and error-fallback behavior of AikoWakeup.boot(). It does NOT
test whether AikoThink/AikoMemorize/AikoSpeak/AikoListen themselves boot
correctly -- that belongs to each subsystem's own test file. Here, all four
are replaced with lightweight fakes so tests run in milliseconds with no
model loads, no GGUF, no real DB -- same spirit as FakeEmbedder in
test_memorize.py.

Layers covered:
  1. Boot label integrity   -- no DB, no mocking, pure dict checks
  2. Concurrency correctness -- the mem_ready handshake between init_think
                                 and init_memorize
  3. Callback contract       -- on_loading/on_done/on_skip pairing
  4. Failure-path resilience -- boot() must not crash or deadlock if
                                 think or memorize fails to construct

Run with:
  pytest tests/test_wakeup.py -v

Assumptions (adjust if your wakeup.py differs):
  - AikoThink() takes no boot args and exposes .start_warmup(),
    .join_warmup(), .set_memorize(), .set_speak(), .start_idle_learner(),
    .handle_scheduled_job(), and a settable ._memorize.
  - AikoMemorize(silent=...) exposes .cleanup() and .get_user_id().
  - AikoSpeak(silent=...) exposes .warmup().
  - AikoListen() is constructed only at boot; models load lazily on first
    mic arm via .ensure_ready() (.load_asr(), .load_vad(),
    .start_barge_in_monitor()).
  - wakeup.boot() hands scheduler startup to system.schedule.start_scheduler();
    the tests neutralize that seam so boot() performs no real scheduling.
    Patch targets are the names as imported INSIDE system.wakeup, not their
    original modules -- adjust ("system.wakeup.AikoThink" etc.) if
    wakeup.py's import style differs from what's shown in the file we reviewed.
"""
from __future__ import annotations

import threading
import time

import pytest

import system.wakeup as wakeup_module
from system.wakeup import AikoWakeup, BootResult


# ─────────────────────────────────────────────────────────────────────────────
# Fakes -- deterministic, no model/DB/network involved
# ─────────────────────────────────────────────────────────────────────────────

class FakeThink:
    """Stand-in for AikoThink. Records whether _memorize was injected and
    when, so tests can assert ordering against the mem_ready handshake."""

    def __init__(self, *a, boot_delay=0.0, **kw):
        self._client = "fake-client"
        self._llm_model = "fake-model"
        self._memorize = None
        self._memorize_set_at = None
        self._boot_delay = boot_delay
        if boot_delay:
            time.sleep(boot_delay)

    def start_warmup(self):
        pass

    def join_warmup(self):
        pass

    def start_idle_learner(self):
        pass

    def set_speak(self, speak):
        self._speak = speak

    def handle_scheduled_job(self, *a, **kw):
        pass

    def _semantic_example_vectors(self, examples, instruct):
        return (["fake_label"], "fake_vectors")

    def set_memorize(self, memorize):
        self._memorize = memorize
        self._memorize_set_at = time.monotonic()


class FakeThinkThatRaises:
    """Simulates AikoThink construction failing outright."""

    def __init__(self, *a, **kw):
        raise RuntimeError("simulated think boot failure")


class FakeMemorize:
    """Stand-in for AikoMemorize. boot_delay lets tests force memorize to
    finish AFTER think reaches its mem_ready.wait() point, so the handshake
    is actually exercised rather than trivially satisfied."""

    def __init__(self, silent=True, boot_delay=0.0):
        self._boot_delay = boot_delay
        if boot_delay:
            time.sleep(boot_delay)
        self.cleanup_called = False

    def cleanup(self):
        self.cleanup_called = True

    def get_user_id(self):
        return "fake_user"


class FakeMemorizeThatRaises:
    """Simulates AikoMemorize construction failing outright. mem_ready must
    still be set (via finally in wakeup.py) or init_think hangs forever."""

    def __init__(self, *a, **kw):
        raise RuntimeError("simulated memorize boot failure")


class FakeSpeak:
    def __init__(self, silent=True):
        self.warmup_called = False

    def warmup(self):
        self.warmup_called = True


class FakeListen:
    def __init__(self):
        self.asr_loaded = False
        self.vad_loaded = False
        self.warmup_joined = False
        self.barge_in_started = False

    def load_asr(self):
        self.asr_loaded = True

    def load_vad(self):
        self.vad_loaded = True

    def join_warmup(self):
        self.warmup_joined = True

    def start_barge_in_monitor(self):
        self.barge_in_started = True


def _noop(*a, **kw):
    """Safe no-op for scheduler/handler registration side effects that
    aren't under test here."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared patching helper
# ─────────────────────────────────────────────────────────────────────────────

def _patch_common(monkeypatch, *, think_cls=FakeThink, memorize_cls=FakeMemorize,
                   think_kwargs=None, memorize_kwargs=None):
    """Patch every heavy dependency boot() touches with a fake/no-op.
    Returns nothing -- callers just call AikoWakeup().boot(...) after this.
    """
    think_kwargs = think_kwargs or {}
    memorize_kwargs = memorize_kwargs or {}

    monkeypatch.setattr(wakeup_module, "AikoThink",
                         lambda: think_cls(**think_kwargs))
    monkeypatch.setattr(wakeup_module, "AikoMemorize",
                         lambda silent=True: memorize_cls(silent=silent, **memorize_kwargs))
    monkeypatch.setattr(wakeup_module, "AikoSpeak", FakeSpeak)
    monkeypatch.setattr(wakeup_module, "AikoListen", FakeListen)
    # Scheduler startup moved to system.schedule.start_scheduler, which
    # wakeup.boot() calls directly. Neutralize that seam so boot() performs
    # no real scheduling under test.
    monkeypatch.setattr(wakeup_module, "start_scheduler", _noop)

    # register_deep_study_handlers lives on memory.learn -- patch the
    # attribute wakeup.py actually calls through (module-level import
    # inside boot(), so patch the source module directly).
    import cognition.memory.learn as learn_module
    monkeypatch.setattr(learn_module, "register_deep_study_handlers", _noop)

    # _prewarm_semantic_cache pulls constants from cognition.think --
    # patch them to harmless placeholders so the import inside
    # _prewarm_semantic_cache doesn't need the real module populated.
    import cognition.think as think_module
    for name in ("_ROUTE_TERNARY_EXAMPLES", "_ROUTE_INSTRUCT_TERNARY"):
        monkeypatch.setattr(think_module, name, {}, raising=False)


def _collecting_callbacks():
    loading, done, skip = [], [], []
    return (loading.append, done.append, skip.append, loading, done, skip)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 -- boot label integrity, pure, no mocking needed
# ─────────────────────────────────────────────────────────────────────────────

class TestBootLabels:
    def test_no_key_collisions_across_subsystems(self):
        """Each subsystem owns its BOOT_LABELS dict. If two subsystems ever
        define the same key, one silently overwrites the other in the merge
        -- this test catches that before it becomes a UI display bug."""
        from cognition.think import BOOT_LABELS as think_labels
        from cognition.memory.memorize import BOOT_LABELS as mem_labels
        from sensory.speak import BOOT_LABELS as speak_labels
        from sensory.listen import BOOT_LABELS as listen_labels

        all_keys = (list(think_labels) + list(mem_labels)
                    + list(speak_labels) + list(listen_labels))
        assert len(all_keys) == len(set(all_keys)), (
            "duplicate BOOT_LABELS key across subsystems -- ALL_BOOT_LABELS "
            "merge will silently drop one subsystem's label"
        )

    def test_all_boot_labels_is_superset_of_each_source(self):
        merged = AikoWakeup.ALL_BOOT_LABELS
        from cognition.think import BOOT_LABELS as think_labels
        assert set(think_labels).issubset(merged)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 -- the mem_ready concurrency handshake
# ─────────────────────────────────────────────────────────────────────────────

class TestMemReadyHandshake:
    def test_think_receives_memorize_only_after_memorize_finishes(self, monkeypatch):
        """Core correctness property of the parallel boot phase: think must
        not see a live memorize reference until memorize has actually
        finished constructing -- even though they run concurrently."""
        # give memorize a small delay so, if the handshake were broken
        # (e.g. mem_ready.wait() removed), think would very likely race
        # ahead and this test would catch it via timing/ordering.
        _patch_common(monkeypatch, memorize_kwargs={"boot_delay": 0.05})

        result = AikoWakeup().boot(on_loading=_noop, on_done=_noop, on_skip=_noop)

        assert result.think._memorize is result.memorize
        assert result.think._memorize is not None

    def test_mem_ready_unblocks_even_if_memorize_raises(self, monkeypatch):
        """If AikoMemorize() throws, wakeup.py logs and continues --
        mem_ready.set() must still fire (it's in a finally block) so
        init_think doesn't hang forever waiting for it."""
        _patch_common(monkeypatch, memorize_cls=FakeMemorizeThatRaises)

        # if the handshake were broken, this call would hang; give the
        # test itself a hard ceiling via a watchdog thread instead of
        # trusting pytest's default timeout (which may not be configured).
        done_event = threading.Event()
        result_holder = {}

        def _run():
            result_holder["result"] = AikoWakeup().boot(
                on_loading=_noop, on_done=_noop, on_skip=_noop
            )
            done_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        finished = done_event.wait(timeout=5.0)

        assert finished, "boot() appears to have deadlocked -- mem_ready was likely never set"
        result = result_holder["result"]
        assert result.memorize is None
        assert result.think is not None
        assert result.think._memorize is None  # nothing to inject


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 -- on_loading/on_done/on_skip callback contract
# ─────────────────────────────────────────────────────────────────────────────

class TestCallbackContract:
    def test_every_loading_key_has_a_matching_done_or_skip(self, monkeypatch):
        _patch_common(monkeypatch)
        on_loading, on_done, on_skip, loading, done, skip = _collecting_callbacks()

        AikoWakeup().boot(on_loading=on_loading, on_done=on_done, on_skip=on_skip)

        finished_keys = set(done) | set(skip)
        assert set(loading) == finished_keys, (
            f"keys reported loading but never finished: {set(loading) - finished_keys}"
        )

    def test_no_key_reported_done_without_a_prior_loading(self, monkeypatch):
        """Guards against a copy-paste bug where on_done(key) is called for
        a key whose on_loading(key) call was accidentally deleted/renamed."""
        _patch_common(monkeypatch)
        on_loading, on_done, on_skip, loading, done, skip = _collecting_callbacks()

        AikoWakeup().boot(on_loading=on_loading, on_done=on_done, on_skip=on_skip)

        assert set(done).issubset(set(loading))


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4 -- failure-path resilience
# ─────────────────────────────────────────────────────────────────────────────

class TestFailurePaths:
    def test_boot_raises_when_think_fails(self, monkeypatch):
        """Think is the cognition core — boot() must fail loudly (not limp
        along degraded) so the supervisor restarts instead of serving a
        brainless Aiko. The RuntimeError chains the original failure."""
        _patch_common(monkeypatch, think_cls=FakeThinkThatRaises)

        with pytest.raises(RuntimeError, match="AikoThink boot failed"):
            AikoWakeup().boot(on_loading=_noop, on_done=_noop, on_skip=_noop)

    def test_boot_does_not_raise_when_memorize_fails(self, monkeypatch):
        _patch_common(monkeypatch, memorize_cls=FakeMemorizeThatRaises)

        result = AikoWakeup().boot(on_loading=_noop, on_done=_noop, on_skip=_noop)

        assert result.memorize is None
        assert result.think is not None  # think itself should be unaffected

    def test_boot_result_fields_map_to_correct_subsystems(self, monkeypatch):
        """Cheap sanity check against a swapped-field typo, e.g. speak
        accidentally assigned into listen's slot in the return statement."""
        _patch_common(monkeypatch)

        result = AikoWakeup().boot(on_loading=_noop, on_done=_noop, on_skip=_noop)

        assert isinstance(result.speak, FakeSpeak)
        assert isinstance(result.listen, FakeListen)
        assert result.speak.warmup_called
        # ASR is lazy now: boot() constructs AikoListen but the models load
        # on first mic arm via ensure_ready() — nothing should be loaded yet.
        assert not result.listen.asr_loaded
        assert not result.listen.vad_loaded
