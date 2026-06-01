"""
core/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio with energy-based VAD (auto start / stop)
  - Transcribes via faster-whisper in a background thread
  - Exposes listen() (blocking) and listen_async() (callback) for cli.py
  - Warm-up call on init loads the Whisper model into memory immediately

Dependencies:
    pip install faster-whisper sounddevice numpy
    (CUDA optional — falls back to CPU automatically)
"""

import io
import logging
import os
import queue
import threading
import warnings

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

# ── config ────────────────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL",      "base.en")   # tiny.en / base.en / small.en
WHISPER_DEVICE      = os.getenv("WHISPER_DEVICE",     "auto")       # auto | cpu | cuda
WHISPER_COMPUTE     = os.getenv("WHISPER_COMPUTE",    "default")    # default | int8 | float16

SAMPLE_RATE         = int(os.getenv("LISTEN_SAMPLE_RATE",    16000))
CHUNK_DURATION_MS   = int(os.getenv("LISTEN_CHUNK_MS",       30))    # ms per audio chunk
SILENCE_THRESHOLD   = float(os.getenv("LISTEN_SILENCE_DB",   0.015)) # RMS energy gate
SILENCE_CHUNKS      = int(os.getenv("LISTEN_SILENCE_CHUNKS", 40))    # ~1.2 s of silence → stop
MIN_SPEECH_CHUNKS   = int(os.getenv("LISTEN_MIN_CHUNKS",     10))    # ignore clips shorter than ~0.3 s
MAX_RECORD_SECONDS  = int(os.getenv("LISTEN_MAX_SECONDS",    30))    # hard ceiling

_CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
_MAX_CHUNKS    = int(MAX_RECORD_SECONDS * 1000 / CHUNK_DURATION_MS)


def _resolve_device(device_hint: str) -> tuple[str, str]:
    """Return (device, compute_type) resolving 'auto' to cuda if available."""
    if device_hint != "auto":
        return device_hint, WHISPER_COMPUTE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16" if WHISPER_COMPUTE == "default" else WHISPER_COMPUTE
    except ImportError:
        pass
    return "cpu", "int8" if WHISPER_COMPUTE == "default" else WHISPER_COMPUTE


# ── listen ────────────────────────────────────────────────────────────────────

class AikoListen:
    """
    Microphone capture + faster-whisper transcription.
    speak is not required; this module is standalone.
    Warm-up starts immediately on init (background thread).
    cli.py should call join_warmup() before first listen().
    """

    def __init__(self) -> None:
        device, compute = _resolve_device(WHISPER_DEVICE)
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute,
        )
        self._lock          = threading.Lock()   # one transcription at a time
        self._warmup_done   = threading.Event()
        self._warmup_thread = threading.Thread(target=self._warmup, daemon=True)
        self._warmup_thread.start()

    # ── public api ────────────────────────────────────────────────────────────

    def join_warmup(self) -> None:
        """Block until Whisper model is warm. Call from cli.py before first prompt."""
        self._warmup_done.wait()

    def listen(self, status_callback=None) -> str:
        """
        Block until one complete speech utterance is captured and transcribed.

        Args:
            status_callback: optional callable(str) for UI status strings
                             e.g. "__LISTENING__", "__TRANSCRIBING__", "__IDLE__"

        Returns:
            Transcribed text string, or "" if nothing intelligible was captured.
        """
        audio = self._record(status_callback)
        if audio is None:
            _cb(status_callback, "__IDLE__")
            return ""
        _cb(status_callback, "__TRANSCRIBING__")
        text = self._transcribe(audio)
        _cb(status_callback, "__IDLE__")
        return text

    def listen_async(self, on_result, status_callback=None) -> threading.Thread:
        """
        Non-blocking variant. Launches a daemon thread and calls on_result(text)
        when transcription is ready.

        Args:
            on_result:        callable(str) — receives the transcribed text
            status_callback:  optional callable(str) — same status tokens as listen()

        Returns:
            The background Thread (already started).
        """
        def _run():
            text = self.listen(status_callback=status_callback)
            on_result(text)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    # ── recording ─────────────────────────────────────────────────────────────

    def _record(self, status_callback=None) -> np.ndarray | None:
        """
        Capture mic until silence detected after speech.
        Returns float32 mono audio array at SAMPLE_RATE, or None on failure.
        """
        audio_chunks  = []
        silence_count = 0
        speech_count  = 0
        hearing_speech = False

        _cb(status_callback, "__LISTENING__")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=_CHUNK_SAMPLES,
            ) as stream:
                for _ in range(_MAX_CHUNKS):
                    chunk, _ = stream.read(_CHUNK_SAMPLES)
                    rms = float(np.sqrt(np.mean(chunk ** 2)))

                    if rms >= SILENCE_THRESHOLD:
                        # speech detected
                        hearing_speech = True
                        silence_count  = 0
                        speech_count  += 1
                        audio_chunks.append(chunk.copy())
                    else:
                        # silence
                        if hearing_speech:
                            silence_count += 1
                            audio_chunks.append(chunk.copy())   # keep trailing silence for naturalness
                            if silence_count >= SILENCE_CHUNKS:
                                break                            # utterance complete
                        # pre-speech silence — discard to avoid bloating buffer

        except sd.PortAudioError as exc:
            print(f"[listen] microphone error: {exc}")
            return None

        if speech_count < MIN_SPEECH_CHUNKS:
            return None   # too short — noise or accidental trigger

        return np.concatenate(audio_chunks, axis=0).flatten()

    # ── transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Run faster-whisper on a float32 numpy array.
        Thread-safe via self._lock — only one transcription runs at a time.
        """
        with self._lock:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,            # faster-whisper built-in VAD post-filter
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 100,
                },
                condition_on_previous_text=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

    # ── warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """
        Transcribe a 0.1 s silent buffer to force model compilation / kernel loading.
        Keeps first-utterance latency low — same pattern as think.py's LLM warmup.
        """
        try:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            self._model.transcribe(silence, language="en")
        except Exception:
            pass
        finally:
            self._warmup_done.set()


# ── helpers ───────────────────────────────────────────────────────────────────

def _cb(callback, msg: str) -> None:
    """Fire status callback safely."""
    if callback:
        try:
            callback(msg)
        except Exception:
            pass
