"""Final #7 regressions: numeric bounds, turn timing and audio lifecycle."""
import contextvars
import io
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from cognition.flysense import FlyAL
from sensory import latency, listen as L, speak as S


def test_al_integer_empty_and_float_batches_keep_useful_dtypes():
    al = FlyAL()
    output = al.normalize(np.arange(1, 9))
    assert output.dtype == np.float32 and np.linalg.norm(output) == pytest.approx(1)
    batch = al.normalize_batch(np.ones((2, 8), dtype=np.float32))
    assert batch.dtype == np.float32 and batch.shape == (2, 8)
    assert al.normalize_batch(np.empty((0, 8), dtype=np.float32)).shape == (0, 8)


@pytest.mark.parametrize('value', [np.full(8, np.nan), np.ones((2, 4)), np.zeros(16385), np.ones(4, dtype=complex)])
def test_invalid_al_input_rejected_before_projection(value):
    al = FlyAL()
    with pytest.raises(ValueError):
        al.normalize(value)
    assert al._proj == {}


def test_al_projection_eviction_is_bounded_and_deterministic():
    al = FlyAL()
    expected = al.normalize(np.arange(1, 9))
    for dim in range(9, 19):
        al.normalize(np.arange(1, dim + 1))
    assert len(al._proj) == 4
    np.testing.assert_array_equal(al.normalize(np.arange(1, 9)), expected)
    with ThreadPoolExecutor(max_workers=4) as workers:
        outputs = list(workers.map(al.normalize, [np.arange(1, 9)] * 16))
    for output in outputs:
        np.testing.assert_array_equal(output, expected)


def test_al_no_excitation_fallback_is_still_a_unit_vector(monkeypatch):
    al = FlyAL()
    monkeypatch.setattr(al, '_matrix', lambda dim: np.ones((dim, 49)))
    output = al.normalize(np.array([-3., -4.]))
    np.testing.assert_allclose(output, [-.6, -.8])


def test_old_playback_context_cannot_mark_the_new_turn():
    latency.reset()
    old = latency.current()
    context = contextvars.copy_context()
    latency.reset()
    new = latency.current()
    context.run(latency.mark, 'first_audio')
    assert 'first_audio' in old.marks and 'first_audio' not in new.marks


def test_route_chat_and_direct_text_only_turns_have_distinct_trackers():
    @latency.chat_call
    def chat():
        latency.mark('input_received')
        return latency.current()
    @latency.routed_call
    def route():
        latency.mark('input_received')
        original = latency.current()
        assert chat() is original
        return original
    first = route()
    second = chat()
    third = chat()
    assert first is not second and second is not third


def test_latency_first_mark_is_atomic(monkeypatch):
    clock_calls = []
    def clock():
        clock_calls.append(1)
        time.sleep(.002)
        return len(clock_calls)
    monkeypatch.setattr(latency.time, 'perf_counter', clock)
    tracker = latency.TurnLatency()
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(tracker.mark, ['first_audio'] * 16))
    assert len(clock_calls) == 1


def test_monitor_restart_does_not_overlap_an_unfinished_old_worker(monkeypatch):
    listener = L.AikoListen()
    first_started, second_started, release = threading.Event(), threading.Event(), threading.Event()
    live, maximum, calls = [0], [0], []
    lock = threading.Lock()
    def loop(stop):
        with lock:
            calls.append(stop); live[0] += 1; maximum[0] = max(maximum[0], live[0])
            (first_started if len(calls) == 1 else second_started).set()
        stop.wait(3)
        release.wait(3)
        with lock:
            live[0] -= 1
    monkeypatch.setattr(listener, '_barge_in_loop', loop)
    listener.start_barge_in_monitor()
    assert first_started.wait(1)
    old = listener._barge_in_thread
    try:
        listener.stop_barge_in_monitor()
        for _ in range(4):
            listener.start_barge_in_monitor()
        assert listener._barge_in_thread is old and len(calls) == 1
        release.set()
        assert second_started.wait(2)
        assert maximum[0] == 1
    finally:
        release.set()
        listener.stop_barge_in_monitor()


def test_proactive_respects_wake_and_binds_activity_identity(monkeypatch):
    from system import orchestrate as O
    from system.userspace import current_user_id, set_current_user_id, reset_current_user_id
    monkeypatch.setattr(O, '_in_proactive_silence_window', lambda: False)
    monkeypatch.setattr(O, 'PROACTIVE_REST_AFTER_SECONDS', 0)
    monkeypatch.setattr(O, 'PROACTIVE_MAX_PER_HOUR', 10)
    monkeypatch.setattr(O, 'PROACTIVE_COOLDOWN_SECONDS', 0)
    actor = set_current_user_id('alice')
    awake = [False]
    messages = []
    ui = SimpleNamespace(add_message=lambda *a: messages.append(current_user_id()), _draw=lambda: None)
    runner = O.ProactiveIdleRunner(ui, None, lambda: False, threading.Event(), engagement_allowed_fn=lambda: awake[0])
    runner._enabled = True; runner._last_activity = 0; runner._next_checkin_after = 0
    try:
        assert runner._actor_context().run(runner._prompt_due, 1000) is None
        awake[0] = True
        assert runner._actor_context().run(runner._prompt_due, 1000) is not None
        reset_current_user_id(actor)
        actor = set_current_user_id('bob')
        runner._actor_context().run(runner._deliver_prompt, 'checkin_text', 'hello', 1000)
        assert messages == ['alice']
        runner.touch()
        assert runner._actor_context().run(current_user_id) == 'bob'
    finally:
        runner.stop(); reset_current_user_id(actor)


def test_stopping_old_proactive_runner_keeps_the_new_idle_tracker():
    from system import orchestrate as O
    args = (SimpleNamespace(), None, lambda: False, threading.Event())
    old, new = O.ProactiveIdleRunner(*args), O.ProactiveIdleRunner(*args)
    try:
        old.stop()
        assert O._idle_runner is new
    finally:
        new.stop()


def _wav():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes(bytes(32000))
    return buffer.getvalue()


def test_streaming_wav_duration_uses_real_body_not_placeholder_header():
    data = bytearray(_wav())
    data[4:8] = b'\xff' * 4
    data[40:44] = b'\xff' * 4
    assert S.AikoSpeak._wav_duration(bytes(data)) == pytest.approx(1)


def test_float_wav_duration_is_supported():
    sf = pytest.importorskip('soundfile')
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format='WAV', subtype='FLOAT')
    assert S.AikoSpeak._wav_duration(buffer.getvalue()) == pytest.approx(.1)


def test_oversized_tts_body_is_closed_without_buffering_it_all(monkeypatch):
    speaker = S.AikoSpeak(silent=True)
    closed = []
    monkeypatch.setattr(S, '_MAX_WAV_BYTES', 64)
    response = SimpleNamespace(headers={}, raise_for_status=lambda: None,
                               iter_content=lambda **kw: iter([bytes(40), bytes(40)]), close=lambda: closed.append(True))
    monkeypatch.setattr(speaker, '_http', lambda: SimpleNamespace(post=lambda *a, **kw: response))
    monkeypatch.setattr(speaker, '_push_notice', lambda *a: None)
    assert speaker._synthesize('hello') is None and closed == [True]


def test_long_deferred_speech_still_delivers_all_generator_text(monkeypatch):
    speaker = S.AikoSpeak(silent=True)
    monkeypatch.setattr(speaker, '_playback_cancelled', lambda: False)
    monkeypatch.setattr(speaker, '_push_notice', lambda *a: None)
    monkeypatch.setattr(speaker, '_playback_thread', lambda *a, **kw: pytest.fail('oversized speech was queued'))
    displayed = []
    def tokens():
        for _ in range(20):
            displayed.append(1)
            yield 'a' * 4000
    speaker.feed_and_play(tokens())
    assert len(displayed) == 20 and not speaker.is_playing()


def test_deferred_token_buffer_caps_a_single_large_token():
    speaker = S.AikoSpeak(silent=True)
    speaker.feed('a' * 20000)
    assert sum(map(len, speaker._token_buf)) == 4000


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_speech_controls_return_to_neutral(value):
    speaker = S.AikoSpeak(silent=True)
    speaker.set_expression(value, value, value)
    assert (speaker._speech_rate, speaker._speech_volume, speaker._speech_pitch) == (1, 1, 0)


def test_pcm_clipping_prevents_native_and_prosody_overflow():
    raw = np.full(512, 1e30, dtype='<f4').tobytes()
    pcm = L._pcm_chunk(raw, 512)
    assert pcm.max() == 1
    features = L._prosody_features(np.full(512, 1e30), 'hello')
    assert all(np.isfinite(value) for value in features.values())


def test_proactive_start_is_idempotent(monkeypatch):
    from system import orchestrate as O
    monkeypatch.setattr(O, 'PROACTIVE_FIRST_IDLE_MAX_SECONDS', 60)
    runner = O.ProactiveIdleRunner(SimpleNamespace(), None, lambda: False, threading.Event())
    monkeypatch.setattr(runner, '_loop', lambda: runner._stop.wait(2))
    try:
        runner.start()
        original = runner._thread
        runner.start()
        assert runner._thread is original
    finally:
        runner.stop()


def test_independent_synthesis_survives_a_previous_playback_stop(monkeypatch):
    speaker = S.AikoSpeak(silent=True)
    speaker.stop()
    data = _wav()
    response = SimpleNamespace(headers={}, raise_for_status=lambda: None,
                               iter_content=lambda **kw: iter([data]), close=lambda: None)
    monkeypatch.setattr(speaker, '_http', lambda: SimpleNamespace(post=lambda *a, **kw: response))
    assert speaker._synthesize('phone audio') == data


def test_identical_audio_reference_does_not_mix_users():
    from sensory.echo import PlaybackEchoGuard
    guard = PlaybackEchoGuard()
    guard.remember(b'wave', 'autumn leaves tomorrow', 'alice')
    guard.remember(b'wave', 'autumn leaves tomorrow', 'bob')
    now = time.monotonic()
    guard.started(b'wave', 1, user_id='alice')
    assert guard.is_echo('autumn leaves tomorrow', 'alice', now, now + .5)
    assert not guard.is_echo('autumn leaves tomorrow', 'bob', now, now + .5)


def test_phone_audio_cache_prunes_generated_files_only(monkeypatch, tmp_path):
    import importlib, os, uuid
    router = importlib.import_module('interface.android_app.lingo.router')
    monkeypatch.setattr(router, 'AUDIO_DIR', tmp_path)
    monkeypatch.setattr(router, '_AUDIO_CACHE_MAXSIZE', 2)
    monkeypatch.setattr(router, '_AUDIO_DISK_MAX_BYTES', 12)
    old = tmp_path / f'{uuid.uuid4()}_100.wav'
    recent = tmp_path / f'{uuid.uuid4()}_200.wav'
    reference = tmp_path / 'voice_reference.wav'
    for file in [old, recent, reference]:
        file.write_bytes(bytes(5))
    os.utime(old, (1, 1))
    router._prune_generated_audio(5)
    assert not old.exists() and recent.exists() and reference.exists()
    monkeypatch.setattr(router, '_audio_cache', {'text': f'https://test/lingo_audio/{old.name}'})
    monkeypatch.setattr(router, '_audio_cache_timestamps', {'text': time.time()})
    assert router._audio_cache_get('text') is None


def test_phone_oversized_incoming_audio_does_not_delete_existing_cache(monkeypatch, tmp_path):
    import importlib, uuid
    router = importlib.import_module('interface.android_app.lingo.router')
    monkeypatch.setattr(router, 'AUDIO_DIR', tmp_path)
    monkeypatch.setattr(router, '_AUDIO_DISK_MAX_BYTES', 4)
    existing = tmp_path / f'{uuid.uuid4()}_100.wav'
    existing.write_bytes(b'old')
    with pytest.raises(ValueError):
        router._prune_generated_audio(5)
    assert existing.exists()


def test_phone_tts_runs_off_event_loop_with_authenticated_identity(monkeypatch):
    import asyncio, importlib
    from system.userspace import current_user_id
    router = importlib.import_module('interface.android_app.lingo.router')
    parent_thread = threading.get_ident()
    seen = []
    monkeypatch.setattr(router, 'check_tts_rate_limit', lambda uid: True)
    monkeypatch.setattr(router, '_public_base_url', lambda request: 'http://test')
    def generate(text, base_url=None):
        seen.append((threading.get_ident(), current_user_id()))
        return 'http://test/audio.wav'
    monkeypatch.setattr(router, '_get_tts_audio_after_rate_check', generate)
    result = asyncio.run(router.get_tts('hello', request=object(), session={'user_id': 'alice'}))
    assert result['audioUrl'] == 'http://test/audio.wav'
    assert seen[0][0] != parent_thread and seen[0][1] == 'alice'


def test_repeated_warmup_has_only_one_pending_prime(monkeypatch):
    speaker = S.AikoSpeak(silent=True)
    entered, release = threading.Event(), threading.Event()
    calls = []
    monkeypatch.setattr(speaker, '_health_check', lambda: True)
    def synth(text):
        calls.append(text)
        entered.set()
        release.wait(2)
    monkeypatch.setattr(speaker, '_synthesize', synth)
    try:
        speaker.warmup()
        assert entered.wait(1)
        for _ in range(3):
            speaker.warmup()
        assert calls == ['ok']
    finally:
        release.set()
        speaker._prime_thread.join(2)
        speaker.close()


def test_close_releases_session_and_prevents_new_audio_requests():
    speaker = S.AikoSpeak(silent=True)
    closed = []
    speaker._http_session = SimpleNamespace(close=lambda: closed.append(True))
    speaker.close()
    speaker.close()
    assert closed == [True] and speaker._http_session is None
    assert not speaker.speak('closed')
    assert speaker._synthesize('closed') is None


def test_standalone_speak_loads_settings_and_waits_without_wait_flag(tmp_path):
    import subprocess, sys, textwrap
    code = textwrap.dedent('''
        import io, os, runpy, sys, time, wave
        from types import SimpleNamespace
        import numpy as np
        import requests
        from system import config
        def load():
            os.environ['MIOTTS_API_URL'] = 'http://configured.test:18089'
        config.load_config = load
        played = []
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
            output.writeframes(bytes(320))
        class Session:
            def mount(self, *args): pass
            def post(self, url, **kwargs):
                assert url == 'http://configured.test:18089/mio/tts/stream'
                time.sleep(.1)
                return SimpleNamespace(headers={}, raise_for_status=lambda: None,
                    iter_content=lambda **kw: iter([buffer.getvalue()]), close=lambda: None)
            def close(self):
                assert played, 'standalone exited before playing its audio'
                print('PLAYED_AND_CLOSED')
        requests.Session = Session
        sys.modules['soundfile'] = SimpleNamespace(read=lambda *a, **kw: (np.zeros(480), 48000))
        sys.modules['sounddevice'] = SimpleNamespace(play=lambda *a, **kw: played.append(True),
            stop=lambda: None, get_stream=lambda: SimpleNamespace(active=False))
        sys.argv = ['speak', 'hello']
        runpy.run_module('sensory.speak', run_name='__main__')
    ''')
    import os
    environ = dict(os.environ, USER_SPACE_ROOT=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20, env=environ)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PLAYED_AND_CLOSED' in result.stdout
