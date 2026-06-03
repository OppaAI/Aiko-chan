"""
core/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio with Silero VAD (neural, energy-independent)
  - Transcribes via faster-whisper in a background thread
  - Exposes listen() (blocking) and listen_async() (callback) for UI
  - Warm-up call on init loads both Whisper and Silero models immediately

Dependencies:
    pip install faster-whisper sounddevice numpy silero-vad scipy
    (CUDA optional — falls back to CPU automatically)
"""

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
import logging
from math import gcd
import numpy as np
import os
import sounddevice as sd
from scipy.signal import resample_poly
import threading
import torch
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

# ── config ────────────────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL",      "turbo")
WHISPER_DEVICE      = os.getenv("WHISPER_DEVICE",     "auto")
WHISPER_COMPUTE     = os.getenv("WHISPER_COMPUTE",    "float16")
WHISPER_LANG        = os.getenv("WHISPER_LANG",       "en")
VAD_SILENCE_MS      = int(os.getenv("LISTEN_VAD_SILENCE_MS", 300))
VAD_PAD_MS          = int(os.getenv("LISTEN_VAD_PAD_MS",     100))

SAMPLE_RATE         = 16000                                          # Whisper + Silero target
CAPTURE_RATE        = int(os.getenv("LISTEN_CAPTURE_RATE", 48000))  # device native
LISTEN_DEVICE       = os.getenv("LISTEN_DEVICE", None)              # None = default
DEVICE_INDEX        = int(LISTEN_DEVICE) if LISTEN_DEVICE else None

CHUNK_DURATION_MS   = int(os.getenv("LISTEN_CHUNK_MS",         30))  # Silero minimum
VAD_THRESHOLD       = float(os.getenv("LISTEN_VAD_THRESHOLD", 0.5))  # Silero speech prob cutoff
SILENCE_CHUNKS      = int(os.getenv("LISTEN_SILENCE_CHUNKS",   40))
MIN_SPEECH_CHUNKS   = int(os.getenv("LISTEN_MIN_CHUNKS",       10))
MAX_RECORD_SECONDS  = int(os.getenv("LISTEN_MAX_SECONDS",      30))

# Silero requires exactly 512 samples at 16 kHz (32 ms) or 256 at 8 kHz
# We capture at CAPTURE_RATE, downsample per-chunk before VAD scoring
_CHUNK_SAMPLES_CAP = int(CAPTURE_RATE * CHUNK_DURATION_MS / 1000)  # at capture rate
_CHUNK_SAMPLES_VAD = 512                                            # at 16 kHz, ~32 ms
_MAX_CHUNKS        = int(MAX_RECORD_SECONDS * 1000 / CHUNK_DURATION_MS)


def _resolve_device(device_hint: str) -> tuple[str, str]:
    """Return (device, compute_type) resolving 'auto' to cuda if available."""
    if device_hint != "auto":
        return device_hint, WHISPER_COMPUTE
    try:
        if torch.cuda.is_available():
            return "cuda", ("float16" if WHISPER_COMPUTE == "default" else WHISPER_COMPUTE)
    except Exception:
        pass
    return "cpu", "int8" if WHISPER_COMPUTE == "default" else WHISPER_COMPUTE


def _to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample a float32 mono array from src_rate → 16000 Hz."""
    if src_rate == SAMPLE_RATE:
        return audio
    g = gcd(src_rate, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, src_rate // g).astype(np.float32)


# ── listen ────────────────────────────────────────────────────────────────────

class AikoListen:
    """
    Microphone capture + faster-whisper transcription.
    Silero VAD replaces energy thresholding for robust, noise-resilient
    speech detection — critical in environments with fan or ambient noise.
    Warm-up starts immediately on init (background thread).
    UI should call join_warmup() before first listen().
    """

    def __init__(self) -> None:
        device, compute = _resolve_device(WHISPER_DEVICE)
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute,
        )
        self._vad_model = load_silero_vad()   # ~2 MB ONNX/JIT, MIT license
        self._vad_model.eval()

        self._lock          = threading.Lock()   # one transcription at a time
        self._warmup_done   = threading.Event()
        self._warmup_thread = threading.Thread(target=self._warmup, daemon=True)
        self._warmup_thread.start()

    # ── public api ────────────────────────────────────────────────────────────

    def join_warmup(self) -> None:
        """Block until both Whisper and Silero are warm. Call from UI before first prompt."""
        self._warmup_done.wait()

    def listen(self, status_callback=None, wait_fn=None) -> str:
        """
        Block until one complete speech utterance is captured and transcribed.

        Args:
            status_callback: optional callable(str) for UI status strings
                             e.g. "__LISTENING__", "__TRANSCRIBING__", "__IDLE__"

        Returns:
            Transcribed text string, or "" if nothing intelligible was captured.
        """
        if wait_fn:
            wait_fn()
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

    def _score_chunk(self, chunk_cap: np.ndarray) -> float:
        """
        Run Silero VAD on one chunk captured at CAPTURE_RATE.
        Downsamples to 16 kHz, pads/trims to exactly _CHUNK_SAMPLES_VAD frames,
        returns speech probability in [0, 1].
        """
        chunk_16k = _to_16k(chunk_cap.flatten(), CAPTURE_RATE)

        # pad or trim to the fixed window Silero expects
        if len(chunk_16k) < _CHUNK_SAMPLES_VAD:
            chunk_16k = np.pad(chunk_16k, (0, _CHUNK_SAMPLES_VAD - len(chunk_16k)))
        else:
            chunk_16k = chunk_16k[:_CHUNK_SAMPLES_VAD]

        tensor = torch.from_numpy(chunk_16k).unsqueeze(0)          # (1, 512)
        with torch.no_grad():
            prob = self._vad_model(tensor, SAMPLE_RATE).item()
        return prob

    def _record(self, status_callback=None) -> np.ndarray | None:
        """
        Capture mic until silence detected after speech (Silero VAD gating).
        Returns float32 mono audio array at SAMPLE_RATE, or None on failure.
        """
        audio_chunks   = []
        silence_count  = 0
        speech_count   = 0
        hearing_speech = False

        _cb(status_callback, "__LISTENING__")

        try:
            with sd.InputStream(
                samplerate=CAPTURE_RATE,
                channels=1,
                dtype="float32",
                blocksize=_CHUNK_SAMPLES_CAP,
                device=DEVICE_INDEX,
            ) as stream:
                for _ in range(_MAX_CHUNKS):
                    chunk, _ = stream.read(_CHUNK_SAMPLES_CAP)
                    is_speech = self._score_chunk(chunk) >= VAD_THRESHOLD

                    if is_speech:
                        # speech frame detected
                        hearing_speech = True
                        silence_count  = 0
                        speech_count  += 1
                        audio_chunks.append(chunk.copy())
                    else:
                        # non-speech frame
                        if hearing_speech:
                            silence_count += 1
                            audio_chunks.append(chunk.copy())   # keep trailing silence for naturalness
                            if silence_count >= SILENCE_CHUNKS:
                                break                            # utterance complete
                        # pre-speech silence — discard to avoid bloating buffer

        except sd.PortAudioError:
            _cb(status_callback, "__IDLE__")
            return None

        if speech_count < MIN_SPEECH_CHUNKS:
            return None   # too short — noise or accidental trigger

        audio = np.concatenate(audio_chunks, axis=0).flatten()

        # resample from CAPTURE_RATE → SAMPLE_RATE for Whisper
        if CAPTURE_RATE != SAMPLE_RATE:
            audio = _to_16k(audio, CAPTURE_RATE)

        return audio.astype(np.float32)

    # ── transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Run faster-whisper on a float32 numpy array.
        Thread-safe via self._lock — only one transcription runs at a time.
        """
        with self._lock:
            segments, _ = self._model.transcribe(
                audio,
                language=WHISPER_LANG,
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": VAD_SILENCE_MS,
                    "speech_pad_ms":           VAD_PAD_MS,
                },
                condition_on_previous_text=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

    # ── warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """
        Transcribe a silent buffer through Whisper and score a silent chunk
        through Silero to force model compilation and kernel loading.
        Keeps first-utterance latency low — same pattern as think.py's LLM warmup.
        """
        try:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            self._model.transcribe(silence, language="en")                  # warm Whisper
            tensor = torch.zeros(1, _CHUNK_SAMPLES_VAD)
            with torch.no_grad():
                self._vad_model(tensor, SAMPLE_RATE)                        # warm Silero
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
