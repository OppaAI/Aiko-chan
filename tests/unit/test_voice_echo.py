"""Echo timing, playback cancellation and bounded queues without audio hardware."""
import contextvars
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sensory import echo, listen as L
from sensory.speak import AikoSpeak, _SpeechQueue, _finish_queue


def test_echo_reference_is_bounded_timed_and_identity_scoped(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(echo.time, 'monotonic', lambda: now[0])
    guard = echo.PlaybackEchoGuard()
    guard.remember(b'voice', 'We can see the autumn leaves tomorrow.', 'alice')
    guard.started(b'voice', 2.0, user_id='alice')
    assert guard.is_echo('see the autumn leaves tomorrow', 'alice', 100.2, 101.0)
    assert not guard.is_echo('see the autumn leaves tomorrow', 'bob', 100.2, 101.0)
    assert not guard.is_echo('see the autumn leaves tomorrow', 'alice', 104.0, 105.0)
    assert not guard.is_echo('yes', 'alice', 100.2, 101.0)
    assert not guard.is_echo('please stop and listen to me', 'alice', 100.2, 101.0)
    guard.stopped()
    assert guard.remaining() == pytest.approx(.4)
    now[0] += .5
    assert guard.remaining() == 0
    for i in range(30):
        guard.remember(str(i).encode(), 'memory bounded reference', 'alice')
        guard.started(str(i).encode(), 1.0, user_id='alice')
    assert len(guard._pending) <= 8 and len(guard._played) == 16


def test_stop_during_synthesis_does_not_resume_old_audio(monkeypatch):
    speaker = AikoSpeak(silent=True)
    entered, release, new_played = threading.Event(), threading.Event(), threading.Event()
    played = []
    def synth(text):
        if text == 'Old response.':
            entered.set()
            assert release.wait(3)
        return text.encode()
    def play(wav):
        # Exercise the same per-session guard as the real playback entry.
        if not speaker._cancelled():
            played.append(wav)
            if wav == b'New response.':
                new_played.set()
    monkeypatch.setattr(speaker, '_playback_cancelled', lambda: False)
    monkeypatch.setattr(speaker, '_synthesize', synth)
    monkeypatch.setattr(speaker, '_play_wav_bytes', play)
    assert speaker.speak('Old response.')
    assert entered.wait(2)
    old_cancel = speaker._stop_flag
    assert speaker.speak('New response.')
    assert old_cancel.is_set() and old_cancel is not speaker._stop_flag
    assert new_played.wait(2)
    release.set()
    time.sleep(.05)
    assert played == [b'New response.']


def test_old_worker_finishing_cannot_clear_new_playing_flag():
    speaker = AikoSpeak(silent=True)
    release, entered = threading.Event(), threading.Event()
    visemes = []
    speaker.set_viseme_sink(lambda *args: visemes.append(args))
    def old():
        entered.set()
        release.wait(2)
        speaker._mark_playing(False)
        speaker._emit_viseme("A", 0)
    worker = speaker._playback_thread(old)
    worker.start()
    assert entered.wait(2)
    speaker.stop()
    speaker._reset_playback()
    release.set()
    worker.join(2)
    assert speaker.is_playing() and not visemes
    speaker._playing.clear()


def test_full_speech_queue_can_finish_or_cancel_without_blocking():
    q = _SpeechQueue()
    for i in range(q.maxsize):
        q.put_nowait(str(i))
    _finish_queue(q)
    assert q.finished.is_set() and q.qsize() == q.maxsize
    _finish_queue(q, cancel=True)
    assert q.get_nowait() is None and q.empty()


def test_backpressured_producer_exits_on_stop():
    speaker = AikoSpeak(silent=True)
    speaker._streaming_active = True
    speaker._stream_queue = _SpeechQueue()
    for _ in range(speaker._stream_queue.maxsize):
        speaker._stream_queue.put_nowait('queued sentence')
    thread = threading.Thread(target=speaker.feed_speech_stream, args=('Another sentence.',), daemon=True)
    thread.start()
    speaker.stop()
    thread.join(1)
    assert not thread.is_alive()


def test_local_audio_stop_does_not_wait_for_entire_wav(monkeypatch):
    import soundfile
    speaker = AikoSpeak(silent=True)
    active = SimpleNamespace(active=True)
    started, stopped = threading.Event(), threading.Event()
    def play(*args, **kwargs):
        started.set()
    def stop():
        active.active = False
        stopped.set()
    sd = SimpleNamespace(play=play, stop=stop, get_stream=lambda: active)
    speaker._sd = sd
    monkeypatch.setattr(soundfile, 'read', lambda *a, **kw: (np.zeros(48000), 48000))
    worker = speaker._playback_thread(speaker._play_wav_bytes, args=(b'wav',))
    worker.start()
    assert started.wait(2)
    before = time.monotonic()
    speaker.stop()
    assert stopped.is_set() and time.monotonic() - before < .5
    worker.join(2)
    assert not worker.is_alive()


def test_capture_discards_audio_if_tts_begins_mid_utterance(monkeypatch):
    listener = L.AikoListen()
    listener._vad_model = SimpleNamespace(reset=lambda: None)
    monkeypatch.setattr(L, 'barge_in_enabled', lambda: False)
    monkeypatch.setattr(L, 'MIN_SPEECH_CHUNKS', 1)
    monkeypatch.setattr(L, 'SILENCE_CHUNKS', 1)
    states = iter([False, True, False, False])
    speaker = SimpleNamespace(is_playing=lambda: next(states), echo_remaining=lambda: 0)
    audio = iter([np.full(512, value, dtype='<f4').tobytes() for value in [.1, .2, .3, 0]])
    decisions = iter([True, True, False])
    monkeypatch.setattr(listener, '_is_speech', lambda *a: next(decisions))
    pcm, _ = listener._record(chunk_source=lambda size: next(audio), speak=speaker)
    assert pcm.shape == (1024,)
    np.testing.assert_allclose(pcm[:512], .3)


def test_self_echo_transcript_is_not_sent_to_conversation(monkeypatch):
    listener = L.AikoListen()
    listener._ready = True
    listener._vad_model = object()
    monkeypatch.setattr(listener, '_record', lambda *a, **kw: (np.zeros(512), False))
    monkeypatch.setattr(listener, 'load_speaker_id', lambda: None)
    monkeypatch.setattr(listener, '_transcribe', lambda audio: 'Aiko speaking to herself')
    speaker = SimpleNamespace(is_playing=lambda: False, echo_remaining=lambda: 0,
                              is_self_echo=lambda *a: True)
    text, info = listener.listen(speak=speaker)
    assert text == '' and info['self_echo'] is True


def test_playback_threads_inherit_identity():
    actor = contextvars.ContextVar('playback_test_actor', default='guest')
    speaker = AikoSpeak(silent=True)
    seen = []
    token = actor.set('owner')
    worker = speaker._playback_thread(lambda: seen.append(actor.get()))
    actor.reset(token)
    worker.start()
    worker.join(1)
    assert seen == ['owner']


def _web_input():
    from interface.webui.webui import AikoWeb
    web = AikoWeb.__new__(AikoWeb)
    web._audio_q, web._input_q = queue.Queue(maxsize=4), queue.Queue()
    web._mic_active = threading.Event()
    web._did_barge_in = False
    web._speak = None
    web._push_vitals = lambda: None
    web._broadcast_to_current_user = lambda *a, **k: None
    return web


def test_web_voice_worker_errors_release_input_waiter():
    web = _web_input()
    web._broadcast = lambda *a, **k: None
    def fail(**kwargs):
        raise RuntimeError('ASR failed')
    text, info = web.get_voice_input(SimpleNamespace(listen=fail))
    assert text == '' and info['voice_error']
    assert not web._mic_active.is_set()


def test_cross_user_frame_does_not_truncate_current_utterance():
    from system.userspace import (set_current_user_id, reset_current_user_id,
                                 set_current_display_name, reset_current_display_name)
    web = _web_input()
    frame = bytes(2048)
    def broadcast(payload):
        if payload['type'] == 'mic':
            for item in [(frame,'alice','Alice'), (frame,'bob','Bob'), (frame,'alice','Alice'), (b'','alice','Alice')]:
                web._audio_q.put_nowait(item)
    web._broadcast = broadcast
    def listen(**kwargs):
        source = kwargs['chunk_source']
        assert source(2048) == frame
        assert source(2048) == frame
        assert source(2048) is None
        return 'complete utterance', {}
    user = set_current_user_id('alice')
    display = set_current_display_name('Alice')
    try:
        assert web.get_voice_input(SimpleNamespace(listen=listen))[0] == 'complete utterance'
    finally:
        reset_current_user_id(user)
        reset_current_display_name(display)


def test_web_ingress_checks_playback_and_tail(monkeypatch):
    web = _web_input()
    monkeypatch.setenv('BARGE_IN_ENABLED', '0')
    web._speak = SimpleNamespace(is_playing=lambda: True, echo_remaining=lambda: 0)
    assert web._audio_feedback_blocked()
    web._speak = SimpleNamespace(is_playing=lambda: False, echo_remaining=lambda: .4)
    assert web._audio_feedback_blocked()
    web._speak.echo_remaining = lambda: 0
    assert not web._audio_feedback_blocked()


def test_cli_passes_speaker_coordination_and_request_identity(monkeypatch):
    from interface.cli.cli import AikoSimpleCLI
    from system.userspace import set_current_user_id, reset_current_user_id, current_user_id
    cli = AikoSimpleCLI()
    speaker = object()
    seen = []
    def listen(speak=None, wait_fn=None):
        seen.append((speak, current_user_id()))
        return 'hello', {}
    token = set_current_user_id('alice')
    try:
        assert cli.get_voice_input(SimpleNamespace(listen=listen), speak=speaker)[0] == 'hello'
    finally:
        reset_current_user_id(token)
    assert seen == [(speaker, 'alice')]


def test_local_monitor_does_not_open_raw_mic_during_tts(monkeypatch):
    listener = L.AikoListen()
    listener._barge_in_active = True
    monkeypatch.setenv('BARGE_IN_ENABLED', '1')
    monkeypatch.setenv('BARGE_IN_LOCAL_AEC', '0')
    monkeypatch.setattr(listener, '_ensure_barge_vad', lambda: pytest.fail('unused VAD allocated'))
    monkeypatch.setattr(L.time, 'sleep', lambda _: setattr(listener, '_barge_in_active', False))
    listener._barge_in_loop()


def test_stream_cancel_halts_remote_session_for_original_user(monkeypatch):
    speaker = AikoSpeak(silent=True)
    calls = []
    class Sink:
        def audio(self, data):
            pass
        def stop_audio_playback(self, user_id=None):
            calls.append(user_id)
    sink = Sink()
    speaker._audio_sink = sink.audio
    speaker._playback_uid = 'alice'
    speaker._streaming_active = True
    speaker._stream_queue = _SpeechQueue()
    speaker._playing.set()
    speaker.stop_speech_stream(cancel=True)
    assert not speaker.is_playing()
    assert calls == ['alice']


def test_stop_interrupts_generator_consumption_without_resuming_old_speech(monkeypatch):
    speaker = AikoSpeak(silent=True)
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(speaker, '_playback_cancelled', lambda: False)
    played, displayed = [], []
    monkeypatch.setattr(speaker, '_synthesize', lambda text: played.append(text))
    def tokens():
        entered.set()
        release.wait(3)
        for word in ('cancelled', 'old', 'response'):
            displayed.append(word)
            yield word
    thread = threading.Thread(target=speaker.feed_and_play, args=(tokens(),), daemon=True)
    thread.start()
    assert entered.wait(2)
    before = time.monotonic()
    speaker.stop()
    assert time.monotonic() - before < .5
    release.set()
    thread.join(2)
    assert not thread.is_alive() and not played
    assert displayed == ['cancelled', 'old', 'response']


def test_real_stream_drains_all_queued_sentences(monkeypatch):
    speaker = AikoSpeak(silent=True)
    monkeypatch.setattr(speaker, '_playback_cancelled', lambda: False)
    monkeypatch.setattr(speaker, '_synthesize', lambda text: text.encode())
    played = []
    monkeypatch.setattr(speaker, '_play_wav_bytes', lambda wav: played.append(wav))
    speaker.start_speech_stream()
    worker = speaker._stream_thread
    for text in ['First sentence.', 'Second sentence.', 'Last sentence.']:
        speaker.feed_speech_stream(text)
    speaker.stop_speech_stream()
    worker.join(2)
    assert not worker.is_alive()
    assert played == [b'First sentence.', b'Second sentence.', b'Last sentence.']


def test_stop_during_stream_playback_does_not_deadlock_start_lock(monkeypatch):
    speaker = AikoSpeak(silent=True)
    entered = threading.Event()
    monkeypatch.setattr(speaker, '_playback_cancelled', lambda: False)
    monkeypatch.setattr(speaker, '_synthesize', lambda text: b'wav')
    def play(wav):
        with speaker._start_lock:
            entered.set()
        speaker._cancel_event().wait(3)
    monkeypatch.setattr(speaker, '_play_wav_bytes', play)
    speaker.start_speech_stream()
    worker = speaker._stream_thread
    speaker.feed_speech_stream('A sentence.')
    assert entered.wait(2)
    before = time.monotonic()
    speaker.stop_speech_stream(cancel=True)
    assert time.monotonic() - before < 1
    assert not worker.is_alive()
