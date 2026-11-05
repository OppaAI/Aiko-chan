"""Phone WAV validation must use the actual sample rate and bounded reads."""
import asyncio
import importlib
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

lingo = importlib.import_module("interface.android_app.lingo.router")
from interface.webui import auth


def wav_bytes(rate, seconds, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b'\0\0' * int(rate * seconds) * channels)
    return buffer.getvalue()


def request(raw):
    upload = UploadFile(file=io.BytesIO(raw), filename='voice.wav')
    return asyncio.run(lingo.speech_to_text(audio=upload, session={'user_id': 'owner'}))


def test_high_rate_phone_audio_not_silently_truncated(monkeypatch):
    seen = []
    def transcribe(pcm, sample_rate):
        seen.append((pcm, sample_rate))
        return 'whole recording'
    monkeypatch.setattr(auth, 'aiko_web_instance', SimpleNamespace(_listen=SimpleNamespace(transcribe_pcm=transcribe)))
    assert request(wav_bytes(48000, 12, channels=1)) == {'text': 'whole recording'}
    pcm, rate = seen[0]
    assert rate == 48000 and pcm.shape == (48000 * 12,)
    assert pcm.dtype == np.float32


@pytest.mark.parametrize('raw', [wav_bytes(8000, 31), wav_bytes(1000, 1), wav_bytes(16000, 1)[:-5]])
def test_invalid_upload_rejected_before_asr(monkeypatch, raw):
    def unexpected(*args):
        pytest.fail('invalid upload reached ASR')
    monkeypatch.setattr(auth, 'aiko_web_instance', SimpleNamespace(_listen=SimpleNamespace(transcribe_pcm=unexpected)))
    with pytest.raises(HTTPException) as error:
        request(raw)
    assert error.value.status_code == 400


def test_upload_read_is_bounded():
    calls = []
    class Upload:
        async def read(self, size=-1):
            calls.append(size)
            assert size == lingo._STT_MAX_BYTES + 1
            return b'x' * size
    with pytest.raises(HTTPException) as error:
        asyncio.run(lingo.speech_to_text(audio=Upload(), session={}))
    assert error.value.status_code == 400
    assert calls == [lingo._STT_MAX_BYTES + 1]
