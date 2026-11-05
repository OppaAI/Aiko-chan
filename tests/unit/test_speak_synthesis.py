"""Exercise the real TTS request construction without HTTP or audio hardware."""
from types import SimpleNamespace

import pytest
import threading
from sensory.speak import AikoSpeak


@pytest.mark.parametrize('text,normalized', [
    ('ｱｲｺ', 'アイコ'), ('ｶﾞｯｺｳ ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ', 'ガッコウ パピプペポ'),
    ('Hello, Aiko! ①', 'Hello, Aiko! ①'), ('「こんにちは」', '「こんにちは」'),
])
def test_synthesis_normalizes_halfwidth_kana_and_sends_request(monkeypatch, text, normalized):
    speaker = AikoSpeak.__new__(AikoSpeak)
    speaker._stop_flag = threading.Event()
    speaker._speech_rate = 1.0
    speaker._speech_volume = 1.0
    speaker._speech_pitch = 0.0
    calls = []
    import io, wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes(bytes(320))
    wav = buffer.getvalue()
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(headers={}, iter_content=lambda **kw: iter([wav]), close=lambda: None, raise_for_status=lambda: None)
    monkeypatch.setattr(speaker, '_http', lambda: SimpleNamespace(post=post))
    assert speaker._synthesize(text) == wav
    assert len(calls) == 1
    assert calls[0][1]['json']['text'] == normalized
