"""Voice-path regressions using real PCM buffers and fake devices/models."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sensory import listen as L


class FakeStream:
    def __init__(self, chunks, stop_error=False):
        self.chunks = iter(chunks)
        self.read_available = L._CHUNK_SAMPLES_VAD
        self.closed = self.stopped = False
        self.stop_error = stop_error

    def start(self):
        pass

    def read(self, count):
        return memoryview(next(self.chunks)), False

    def stop(self):
        self.stopped = True
        if self.stop_error:
            raise RuntimeError('driver stop failed')

    def close(self):
        self.closed = True


def prepared(monkeypatch):
    listener = L.AikoListen()
    listener._ready = True
    listener._model = object()
    listener._vad_model = SimpleNamespace(reset=lambda: None)
    monkeypatch.setattr(listener, 'load_speaker_id', lambda: None)
    monkeypatch.setattr(L, 'WAKE_WORD', '')
    monkeypatch.setattr(L, 'MIN_SPEECH_CHUNKS', 1)
    monkeypatch.setattr(L, 'SILENCE_CHUNKS', 1)
    return listener


def test_raw_portaudio_recording_and_cleanup(monkeypatch):
    listener = prepared(monkeypatch)
    pcm = np.full(L._CHUNK_SAMPLES_VAD, .125, dtype='<f4')
    stream = FakeStream([pcm.tobytes(), pcm.tobytes()], stop_error=True)
    listener._sd = SimpleNamespace(RawInputStream=lambda **kwargs: stream)
    decisions = iter([True, False])
    monkeypatch.setattr(listener, '_is_speech', lambda vad, chunk: next(decisions))
    audio, woke = listener._record()
    np.testing.assert_array_equal(audio, np.concatenate([pcm, pcm]))
    assert not woke
    assert stream.stopped and stream.closed
    assert not listener._recording.is_set()
    assert listener._mic_lock.acquire(blocking=False)
    listener._mic_lock.release()


def test_barge_in_reads_raw_buffers(monkeypatch):
    listener = prepared(monkeypatch)
    listener._barge_in_active = True
    listener._barge_in_armed.set()
    pcm = np.zeros(L._CHUNK_SAMPLES_VAD, dtype='<f4')
    stream = FakeStream([pcm.tobytes()])
    listener._sd = SimpleNamespace(RawInputStream=lambda **kwargs: stream)
    monkeypatch.setattr(L, 'barge_in_enabled', lambda: True)
    monkeypatch.setattr(listener, '_ensure_barge_vad', lambda: True)
    received = []
    def score(vad, chunk):
        received.append(chunk)
        listener._barge_in_active = False
        return False
    monkeypatch.setattr(listener, '_is_speech', score)
    listener._barge_in_loop()
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], pcm)
    assert stream.closed


def test_boot_runs_optional_models_before_warmup(monkeypatch):
    listener = L.AikoListen()
    stages = []
    def asr():
        stages.append('asr')
        listener._model = object()
    monkeypatch.setattr(listener, 'load_asr', asr)
    monkeypatch.setattr(listener, 'load_wakeword', lambda: stages.append('wake'))
    monkeypatch.setattr(listener, 'load_speaker_id', lambda: stages.append('speaker'))
    monkeypatch.setattr(listener, 'load_vad', lambda: stages.append('vad'))
    monkeypatch.setattr(listener, 'join_warmup', lambda: stages.append('warmup') or True)
    monkeypatch.setattr(listener, 'start_barge_in_monitor', lambda: stages.append('barge'))
    listener.ensure_ready()
    listener.ensure_ready()
    assert stages == ['asr', 'wake', 'speaker', 'vad', 'warmup', 'barge']
    assert listener._ready


def test_warmup_timeout_does_not_expose_half_ready_models(monkeypatch):
    listener = prepared(monkeypatch)
    listener._ready = False
    for stage in ('load_asr', 'load_vad', 'load_wakeword'):
        monkeypatch.setattr(listener, stage, lambda: None)
    monkeypatch.setattr(listener, 'join_warmup', lambda: False)
    listener.ensure_ready()
    assert not listener._ready and listener._voice_error
    assert listener.listen()[0] == ''


@pytest.mark.parametrize('audio,rate', [
    (np.zeros((2, 2)), 16000), (np.array([np.nan]), 16000),
    (np.array([np.inf]), 16000), (np.zeros(1), 0),
    (np.zeros(1), True), (np.zeros(1), 16000.5),
    (np.zeros(1), 1), (np.zeros(1), 1000000),
    (np.zeros(16000 * (L.MAX_RECORD_SECONDS + 1)), 16000),
])
def test_bad_pcm_rejected_before_loading_models(monkeypatch, audio, rate):
    listener = L.AikoListen()
    monkeypatch.setattr(listener, 'ensure_ready', lambda: pytest.fail('bad audio booted models'))
    with pytest.raises(ValueError):
        listener.transcribe_pcm(audio, rate)


def test_one_shot_resampling_and_failure_cache(monkeypatch):
    listener = prepared(monkeypatch)
    samples = []
    monkeypatch.setattr(listener, '_transcribe', lambda pcm: samples.append(pcm) or 'spoken')
    assert listener.transcribe_pcm(np.arange(480, dtype=np.float32), 48000) == 'spoken'
    assert samples[0].shape == (160,)
    np.testing.assert_array_equal(samples[0], np.arange(160, dtype=np.float32) * 3)
    listener._model = None
    listener._voice_error = 'runtime unavailable'
    monkeypatch.setattr(listener, 'load_asr', lambda: pytest.fail('retried a cached boot failure'))
    assert listener.transcribe_pcm(np.zeros(160), 16000) == ''


def test_speaker_verification_error_does_not_bypass_enabled_gate(monkeypatch):
    listener = prepared(monkeypatch)
    listener._speaker_extractor = object()
    listener._enrolled_embedding = np.ones(3)
    monkeypatch.setattr(listener, '_record', lambda *a, **kw: (np.zeros(512), False))
    monkeypatch.setattr(listener, '_transcribe', lambda audio: 'hello')
    monkeypatch.setattr(L, 'speaker_verify_gate', lambda: True)
    def fail(audio):
        raise RuntimeError('native verifier error')
    monkeypatch.setattr(listener, '_verify_speaker', fail)
    statuses = []
    text, info = listener.listen(status_callback=statuses.append)
    assert text == '' and info['verified'] is False
    assert statuses[-1] == '__IDLE__'


def test_asr_failure_restores_idle_status(monkeypatch):
    listener = prepared(monkeypatch)
    monkeypatch.setattr(listener, '_record', lambda *a, **kw: (np.zeros(512), False))
    def fail(audio):
        raise RuntimeError('decoder error')
    monkeypatch.setattr(listener, '_transcribe', fail)
    statuses = []
    with pytest.raises(RuntimeError, match='decoder'):
        listener.listen(status_callback=statuses.append)
    assert statuses[-1] == '__IDLE__'


def test_enrollment_switches_users_and_reuses_extractor(monkeypatch, tmp_path):
    listener = L.AikoListen()
    listener._speaker_extractor = object()  # no model rebuild on enrollment change
    model = tmp_path / 'speaker.onnx'
    model.touch()
    uid = ['alice']
    for name, emb in [('alice', [1, 0]), ('bob', [0, 1])]:
        (tmp_path / (name + '.json')).write_text(json.dumps({'embedding': emb}))
    monkeypatch.setattr(L, 'SPEAKER_VERIFY_ENABLED', True)
    monkeypatch.setattr(L, 'SPEAKER_MODEL_PATH', str(model))
    monkeypatch.setattr(L, 'current_user_id', lambda: uid[0])
    monkeypatch.setattr(listener, 'speaker_enroll_path', lambda: str(tmp_path / (uid[0] + '.json')))
    listener.load_speaker_id()
    np.testing.assert_array_equal(listener._enrolled_embedding, [1, 0])
    uid[0] = 'bob'
    listener.load_speaker_id()
    np.testing.assert_array_equal(listener._enrolled_embedding, [0, 1])
    (tmp_path / 'bob.json').write_text('{"embedding": [0.5, 0.5]}')
    listener.load_speaker_id()
    np.testing.assert_array_equal(listener._enrolled_embedding, [.5, .5])
    uid[0] = 'no-enrollment'
    listener.load_speaker_id()
    assert not listener.speaker_verify_active()


def test_listen_async_preserves_request_identity(monkeypatch):
    import contextvars
    actor = contextvars.ContextVar('listen_test_actor', default='guest')
    listener = L.AikoListen()
    monkeypatch.setattr(listener, 'listen', lambda **kwargs: (actor.get(), {}))
    results = []
    token = actor.set('owner')
    try:
        thread = listener.listen_async(lambda text, info: results.append(text))
    finally:
        actor.reset(token)
    thread.join(timeout=2)
    assert results == ['owner']


def test_stalled_capture_obeys_read_deadline(monkeypatch):
    ticks = iter([0.0, 10.0])
    monkeypatch.setattr(L.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError):
        L._read_input_chunk(SimpleNamespace(read_available=0), 512)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_prosody_does_not_manufacture_urgency(value):
    from cognition.flysense.pathways import encode_voice_prosody
    output = encode_voice_prosody({'rms': value, 'peak': value, 'words_per_second': value, 'pause_density': value})
    assert output['urgency'] == 0
    assert output['energy'] == 0
    assert output['hesitation'] == 0
    assert all(np.isfinite(v) for v in output.values())


def test_barge_monitor_releases_microphone_for_recording(monkeypatch):
    listener = prepared(monkeypatch)
    listener._barge_in_active = True
    listener._barge_in_armed.set()
    opened = threading.Event()
    streams = []
    pcm = np.zeros(512, dtype='<f4').tobytes()
    def make_stream(**kwargs):
        assert not any(not stream.closed for stream in streams), 'microphone opened twice'
        stream = FakeStream([pcm, pcm])
        if not streams:
            stream.read_available = 0  # monitor waits for audio until capture takes over
            stream.start = opened.set
        streams.append(stream)
        return stream
    listener._sd = SimpleNamespace(RawInputStream=make_stream)
    monkeypatch.setattr(L, 'barge_in_enabled', lambda: True)
    monkeypatch.setattr(L, 'barge_in_always_on', lambda: True)
    monkeypatch.setattr(listener, '_ensure_barge_vad', lambda: True)
    decisions = iter([True, False])
    def score(vad, chunk):
        speech = next(decisions)
        if not speech:
            listener._barge_in_active = False
        return speech
    monkeypatch.setattr(listener, '_is_speech', score)
    worker = threading.Thread(target=listener._barge_in_loop, daemon=True)
    worker.start()
    try:
        assert opened.wait(2)
        audio, _ = listener._record()
        assert audio is not None and audio.size == 1024
    finally:
        listener._barge_in_active = False
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(streams) == 2 and all(stream.closed for stream in streams)
