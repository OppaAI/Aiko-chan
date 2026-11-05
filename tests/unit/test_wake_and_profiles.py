"""Wake windows, streaming wake scores and safe speaker enrollment."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sensory import listen as L
from sensory.speaker_profile import save_profile, validate_embedding


class StreamingWake:
    def __init__(self):
        self.frames = []
        self.resets = 0
    def reset(self):
        self.resets += 1
    def predict(self, pcm):
        self.frames.append(pcm.size)
        assert pcm.size == 1280
        return {'aiko': .95}


def test_streaming_confirmation_counts_predictions_not_partial_input(monkeypatch):
    listener = L.AikoListen()
    listener._wake_model = StreamingWake()
    listener._wake_model_name = 'aiko'
    listener._vad_model = SimpleNamespace(reset=lambda: None)
    monkeypatch.setattr(L, 'WAKE_WORD', 'hey aiko')
    monkeypatch.setattr(L, 'WAKE_WORD_CONFIRM_FRAMES', 3)
    monkeypatch.setattr(L, 'MIN_SPEECH_CHUNKS', 1)
    monkeypatch.setattr(L, 'SILENCE_CHUNKS', 1)
    # Model evaluates 3 independent 80ms steps while Silero supplies 32ms PCM.
    buffers = iter([np.zeros(512, dtype='<f4').tobytes()] * 10)
    decisions = iter([True, False])
    monkeypatch.setattr(listener, '_is_speech', lambda *args: next(decisions))
    audio, woke = listener._record(chunk_source=lambda n: next(buffers))
    assert woke and audio.size == 1024
    assert listener._wake_model.frames == [1280] * 3
    assert listener._wake_pending.size < 1280
    assert not listener.is_active()  # candidate is committed after verification


def test_legacy_wake_classifier_receives_full_window():
    listener = L.AikoListen()
    seen = []
    def predict(audio):
        seen.append(audio.size)
        return {'aiko': .9}
    listener._wake_model = SimpleNamespace(predict=predict)
    listener._wake_model_name = 'aiko'
    for _ in range(70):
        listener._score_wake(np.zeros(512, dtype=np.float32))
    assert seen and all(size == 32000 for size in seen)
    assert listener._wake_pending.size == 32000


def test_wake_sessions_are_identity_scoped_and_bounded(monkeypatch):
    listener = L.AikoListen()
    monkeypatch.setattr(L, 'WAKE_WORD', 'hey aiko')
    listener.extend_activation('alice')
    assert listener.is_active('alice')
    assert not listener.is_active('bob')
    listener.extend_activation('bob')
    listener.sleep_now('alice')
    assert not listener.is_active('alice') and listener.is_active('bob')
    for i in range(40):
        listener.extend_activation(str(i))
    assert len(listener._active_until) == 32


def test_empty_or_rejected_utterance_does_not_extend_wake(monkeypatch):
    listener = L.AikoListen()
    monkeypatch.setattr(L, 'WAKE_WORD', 'hey aiko')
    monkeypatch.setattr(L, 'speaker_verify_gate', lambda: True)
    monkeypatch.setattr(L, 'current_user_id', lambda: 'alice')
    listener.extend_activation()
    expires = listener._active_until['alice']
    assert listener._apply_activation_gate('', True)[0] == ''
    assert listener._apply_activation_gate('hey aiko do this', False)[0] is None
    assert listener._active_until['alice'] == expires


def test_first_browser_frame_rechecks_the_accepted_identity(monkeypatch):
    listener = L.AikoListen()
    listener._wake_model = StreamingWake()
    listener._wake_model_name = 'aiko'
    listener._vad_model = SimpleNamespace(reset=lambda: None)
    monkeypatch.setattr(L, 'WAKE_WORD', 'hey aiko')
    uid = ['alice']
    monkeypatch.setattr(L, 'current_user_id', lambda: uid[0])
    listener.extend_activation('alice')
    remaining = [3]
    def source(n):
        uid[0] = 'bob'
        remaining[0] -= 1
        return bytes(n) if remaining[0] >= 0 else None
    monkeypatch.setattr(listener, '_is_speech', lambda *args: pytest.fail('Bob bypassed the wake gate'))
    assert listener._record(chunk_source=source)[0] is None
    assert not listener.is_active('bob')
    assert listener.is_active('alice')


def test_failed_speaker_cannot_activate_an_acoustic_wake(monkeypatch):
    listener = L.AikoListen()
    listener._ready = True
    listener._model = object()
    listener._vad_model = object()
    listener._wake_model = StreamingWake()
    listener._speaker_extractor = object()
    listener._enrolled_embedding = np.ones(2)
    monkeypatch.setattr(L, 'WAKE_WORD', 'hey aiko')
    monkeypatch.setattr(L, 'speaker_verify_gate', lambda: True)
    monkeypatch.setattr(listener, 'load_speaker_id', lambda: None)
    monkeypatch.setattr(listener, '_record', lambda *a, **kw: (np.zeros(512), True))
    monkeypatch.setattr(listener, '_transcribe', lambda audio: 'do this task')
    monkeypatch.setattr(listener, '_verify_speaker', lambda audio, **kwargs: (False, .1))
    text, info = listener.listen()
    assert not text and info['verified'] is False
    assert not listener.is_active()


@pytest.mark.parametrize('candidate', [np.array([np.nan, 0]), np.array([0, 0]), np.ones(3), np.array([[1, 0]])])
def test_invalid_verifier_vectors_rejected(candidate):
    listener = L.AikoListen()
    listener._enrolled_embedding = np.array([1, 0])
    listener._compute_embedding = lambda audio: candidate
    with pytest.raises(ValueError):
        listener._verify_speaker(np.zeros(160))


def test_speaker_cosine_is_finite_for_large_valid_embeddings():
    listener = L.AikoListen()
    listener._enrolled_embedding = np.full(512, 1e30, dtype=np.float32)
    listener._compute_embedding = lambda audio: np.full(512, 1e30, dtype=np.float32)
    matched, score = listener._verify_speaker(np.zeros(160))
    assert matched and score == pytest.approx(1)


def test_failed_extractor_load_does_not_cache_a_success(monkeypatch, tmp_path):
    model = tmp_path / 'speaker.onnx'
    model.touch()
    profile = tmp_path / 'voice.json'
    profile.write_text('{"embedding": [1, 0]}')
    listener = L.AikoListen()
    monkeypatch.setattr(L, 'SPEAKER_VERIFY_ENABLED', True)
    monkeypatch.setattr(L, 'SPEAKER_MODEL_PATH', str(model))
    monkeypatch.setattr(listener, 'speaker_enroll_path', lambda: str(profile))
    monkeypatch.setattr(L, '_ensure_runtime', lambda: None)
    attempts = []
    def create(config):
        attempts.append(config)
        if len(attempts) == 1:
            raise RuntimeError('transient load failure')
        return object()
    monkeypatch.setattr(L, 'sherpa_onnx', SimpleNamespace(SpeakerEmbeddingExtractorConfig=lambda **kw: kw,
                                                       SpeakerEmbeddingExtractor=create))
    with pytest.raises(RuntimeError):
        listener.load_speaker_id()
    listener.load_speaker_id()
    assert len(attempts) == 2 and listener.speaker_verify_active()


def test_profile_write_failure_preserves_previous_enrollment(monkeypatch, tmp_path):
    from sensory import speaker_profile
    path = tmp_path / 'voice.json'
    path.write_text('original profile')
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr(speaker_profile.os, 'replace', fail)
    with pytest.raises(OSError):
        save_profile(path, {'embedding': [1, 0], 'dim': 2})
    assert path.read_text() == 'original profile'
    assert not list(tmp_path.glob('*.tmp'))


def test_profile_roundtrip_private_permissions(tmp_path):
    import stat
    path = tmp_path / 'voice.json'
    save_profile(path, {'embedding': [1, 0], 'dim': 2})
    assert json.loads(path.read_text())['embedding'] == [1, 0]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for invalid in ([], [0, 0], [float('nan')]):
        with pytest.raises(ValueError):
            validate_embedding(invalid)


def test_silent_enrollment_does_not_overwrite_profile(monkeypatch, tmp_path):
    from util import enroll_speak as enroll
    model, profile = tmp_path / 'speaker.onnx', tmp_path / 'voice.json'
    model.touch(); profile.write_text('original profile')
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=lambda **kw: kw,
        SpeakerEmbeddingExtractor=lambda config: object()))
    monkeypatch.setattr(enroll, '_record_seconds', lambda seconds: np.zeros(16000, dtype=np.float32))
    args = SimpleNamespace(model=str(model), seconds=6, out=str(profile), name='Owner')
    assert enroll._enroll(args, 'owner') == 1
    assert profile.read_text() == 'original profile'


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan'), 61])
def test_invalid_enrollment_duration_rejected_before_capture(seconds):
    from util import enroll_speak as enroll
    with pytest.raises(ValueError):
        enroll._record_seconds(seconds)


def test_verifier_timeout_retains_one_worker_and_its_enrollment(monkeypatch):
    import threading
    listener = L.AikoListen()
    listener._ready = True
    listener._model = object()
    listener._vad_model = object()
    listener._speaker_extractor = object()
    listener._enrolled_embedding = np.array([1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(listener, 'load_speaker_id', lambda: None)
    monkeypatch.setattr(listener, '_record', lambda *a, **kw: (np.zeros(512), False))
    monkeypatch.setattr(listener, '_transcribe', lambda audio: 'hello')
    monkeypatch.setattr(L, 'WAKE_WORD', '')
    monkeypatch.setenv('SPEAKER_VERIFY_TIMEOUT_S', '.1')
    monkeypatch.setattr(L, 'speaker_verify_gate', lambda: False)
    release = threading.Event()
    seen = []
    def verify(audio, reference=None):
        seen.append(reference.copy())
        release.wait(3)
        return True, 1.0
    monkeypatch.setattr(listener, '_verify_speaker', verify)
    try:
        text, info = listener.listen()
        assert text == 'hello' and info['verified'] is False
        assert info['verification_error'] == 'speaker verification timed out'
        original = listener._verify_thread
        listener._enrolled_embedding = np.array([0.0, 1.0], dtype=np.float32)
        text, info = listener.listen()
        assert listener._verify_thread is original and len(seen) == 1
        assert info['verification_error'] == 'previous verification is still running'
        np.testing.assert_array_equal(seen[0], [1, 0])
    finally:
        release.set()
        listener._verify_thread.join(2)


def test_enrollment_main_writes_the_selected_owner_not_guest(monkeypatch, tmp_path):
    from util import enroll_speak as enroll
    from system import config, userspace
    model = tmp_path / 'speaker.onnx'
    model.touch()
    user = 'voice-test-owner'
    monkeypatch.setenv('USER_SPACE_ROOT', str(tmp_path / 'users'))
    monkeypatch.setattr(config, 'load_config', lambda: None)
    monkeypatch.setattr(enroll, '_parse_args', lambda: SimpleNamespace(
        user=None, model=str(model), seconds=6, out=None, name=None))
    monkeypatch.setattr(enroll, 'current_user_id', lambda: 'guest')
    monkeypatch.setattr(enroll, 'resolve_owner_user_id', lambda: user)
    monkeypatch.setattr(enroll, '_record_seconds', lambda seconds: np.full(16000, .02, dtype=np.float32))
    stream = SimpleNamespace(accept_waveform=lambda *a: None, input_finished=lambda: None)
    extractor = SimpleNamespace(create_stream=lambda: stream, compute=lambda stream: [1.0, .5])
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=lambda **kw: kw,
        SpeakerEmbeddingExtractor=lambda config: extractor))
    monkeypatch.setattr(sys, 'argv', ['enroll'])
    before = userspace.current_user_id()
    assert enroll.main() == 0
    path = userspace.user_state_path('profile/speaker_enrollment.json', user_id=user)
    assert json.loads(path.read_text())['name'] == user
    assert not (tmp_path / 'users' / 'guest' / 'profile' / 'speaker_enrollment.json').exists()
    assert userspace.current_user_id() == before


def test_cold_listen_import_does_not_import_native_voice_libraries():
    import subprocess
    import textwrap
    code = textwrap.dedent('''
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('livekit', 'onnxruntime', 'sherpa_onnx'):
                raise AssertionError('eager voice runtime import: ' + name)
            return original(name, *args, **kwargs)
        builtins.__import__ = guarded
        import sensory.listen
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_wake_sdk_sessions_use_one_cpu_thread(monkeypatch, tmp_path):
    listener = L.AikoListen()
    path = tmp_path / 'aiko.onnx'
    path.touch()
    monkeypatch.setattr(L, 'WAKE_WORD', 'hey aiko')
    monkeypatch.setattr(L, 'WAKE_WORD_MODEL_PATH', str(path))
    monkeypatch.setattr(L, '_ort', SimpleNamespace(SessionOptions=lambda: SimpleNamespace()))
    seen = []
    def model(**kwargs):
        seen.append(kwargs)
        return StreamingWake()
    monkeypatch.setattr(L, '_WakeWordModel', model)
    listener.load_wakeword()
    assert seen[0]['sess_options'].intra_op_num_threads == 1
    assert seen[0]['sess_options'].inter_op_num_threads == 1


@pytest.mark.parametrize('text', ['HEY AIKO, code tomorrow', 'Hey,Aiko code tomorrow', 'hey aiko code tomorrow'])
def test_fuzzy_wake_ignores_case_and_asr_punctuation(text):
    matched, command = L._strip_prefix_phrase(text, ['hey aiko'], 90)
    assert matched and command == 'code tomorrow'
