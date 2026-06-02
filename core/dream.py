"""
core/dream_scheduler.py
Schedules Aiko's nightly dream() consolidation pass at 00:00 local time.

Usage — call start() once during Aiko startup (e.g. in cli.py or main.py):

    from core.dream_scheduler import start as start_dream_scheduler
    start_dream_scheduler(memorize_instance)

The scheduler runs in a background daemon thread and fires dream() at midnight.
It does NOT block startup or conversation flow.

VRAM safety:
    dream() does zero LLM calls — only Qdrant vector ops and mem0 deletes.
    No Ollama contention. Safe to fire even if a conversation is mid-flight,
    though a _dream_lock flag is checked to avoid overlapping passes.
"""

import threading
import time
from datetime import datetime, timezone

# Prevent overlapping dream passes (e.g. if system clock jumps or scheduler
# fires twice due to a suspend/resume cycle).
_dream_lock = threading.Lock()


def _seconds_until_midnight() -> float:
    """Seconds from now until the next local 00:00:00."""
    now   = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # If it's already past midnight today, target tomorrow's midnight.
    delta = (today.replace(day=today.day + 1) - now).total_seconds()
    return max(delta, 0.0)


def _dream_loop(memorize) -> None:
    """
    Background loop: sleep until midnight, fire dream(), repeat.
    Runs as a daemon thread — exits automatically when the main process ends.
    """
    while True:
        wait = _seconds_until_midnight()
        print(f"[dream-scheduler] Next consolidation pass in {wait / 3600:.1f}h (at midnight).")
        time.sleep(wait)

        if not _dream_lock.acquire(blocking=False):
            print("[dream-scheduler] Pass already running — skipping.")
            # Sleep a bit to avoid tight loop if something is wrong
            time.sleep(60)
            continue

        try:
            print(f"[dream-scheduler] Firing dream() at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            result = memorize.dream()
            print(f"[dream-scheduler] Pass complete: {result}")
        except Exception as e:
            print(f"[dream-scheduler-error] dream() raised: {e}")
        finally:
            _dream_lock.release()

        # Sleep 90 seconds after firing so we don't re-trigger at 00:00:00
        # on the same second due to scheduler drift.
        time.sleep(90)


def start(memorize) -> threading.Thread:
    """
    Start the nightly dream scheduler as a daemon background thread.

    Args:
        memorize: An initialised AikoMemorize instance.

    Returns the Thread (rarely needed, but useful for testing).
    """
    t = threading.Thread(target=_dream_loop, args=(memorize,), daemon=True, name="dream-scheduler")
    t.start()
    print("[dream-scheduler] Started — will consolidate memories nightly at midnight.")
    return t
