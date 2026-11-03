from types import SimpleNamespace
import threading

from system.orchestrate import TypewriterSync


def test_stop_interrupts_long_pacing_wait_and_allows_clean_restart():
    emitted = []
    first_word = threading.Event()
    def token(word):
        emitted.append(word)
        first_word.set()
    ui = SimpleNamespace(stream_token=token, _draw=lambda **kwargs: None)
    writer = TypewriterSync(ui, None)
    writer.start()
    previous = writer._thread
    try:
        writer._q.put(('first', 3600))
        assert first_word.wait(2)
        writer.stop(flush=False)
        assert not previous.is_alive()
        writer.start()
        assert writer._thread is not previous
        writer.stop(flush=False)
        assert not writer._thread.is_alive()
        assert emitted == ['first ']
    finally:
        writer.stop(flush=False)


def test_start_stops_previous_worker_before_replacing_it():
    ui = SimpleNamespace(stream_token=lambda text: None, _draw=lambda **kwargs: None)
    writer = TypewriterSync(ui, None)
    writer.start()
    previous = writer._thread
    try:
        writer.start()
        assert not previous.is_alive()
        assert writer._thread is not previous
    finally:
        writer.stop(flush=False)
