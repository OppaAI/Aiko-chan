"""
core/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio with Silero VAD (neural, energy-independent)
  - Transcribes via faster-whisper in a background thread
  - Exposes listen() (blocking) and listen_async() (callback) for UI
  - Staged init: load_whisper() → load_vad() → join_warmup() for granular
    boot progress reporting via wakeup.py
  - Always-on barge-in VAD monitor: start_barge_in_monitor() runs a
    lightweight Silero-only daemon that sets _barge_in_event when speech is
    detected during TTS playback, enabling speak.wait_or_barge_in()

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

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'listen_whisper': 'Loading Whisper model...',
    'listen_silero':  'Loading Silero VAD...',
    'listen_warmup':  'Warming up ASR pipeline...',
    'listen_ready':   'Microphone ready',
    'listen_skip':    'ASR skipped (text mode)',
}

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
SILENCE_CHUNKS      = int(os.getenv("LISTEN_SILENCE_CHUNKS",   20))
MIN_SPEECH_CHUNKS   = int(os.getenv("LISTEN_MIN_CHUNKS",       10))
MAX_RECORD_SECONDS  = int(os.getenv("LISTEN_MAX_SECONDS",      30))

# Barge-in VAD config — separate threshold from main VAD so you can tune them
# independently. Higher threshold = less sensitive to background noise while
# Aiko is speaking. Two consecutive chunks required to avoid single-spike misfires.
BARGE_IN_THRESHOLD     = float(os.getenv("BARGE_IN_THRESHOLD",     "0.65"))
BARGE_IN_CONFIRM       = int(os.getenv("BARGE_IN_CONFIRM_CHUNKS",  "2"))
BARGE_IN_COOLDOWN_MS   = int(os.getenv("BARGE_IN_COOLDOWN_MS",     "800"))

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

    Staged init for granular boot progress reporting:
        listen = AikoListen()   # resolves device only — no heavy loading
        listen.load_whisper()   # loads WhisperModel into memory
        listen.load_vad()       # loads Silero VAD + kicks off warmup thread
        listen.join_warmup()    # blocks until warmup thread completes

    Barge-in monitor (call after join_warmup from wakeup.py):
        listen.start_barge_in_monitor()

        This launches a lightweight always-on Silero-only daemon that sets
        _barge_in_event when speech is detected. No Whisper is involved —
        VAD alone costs ~0 CPU on the Jetson while TTS plays.

        Pass speak= to listen() so it calls speak.wait_or_barge_in() instead
        of the blocking speak.wait(), enabling the user to interrupt mid-sentence.
    """

    def __init__(self) -> None:
        self._device, self._compute = _resolve_device(WHISPER_DEVICE)
        self._model:      WhisperModel | None = None   # populated by load_whisper()
        self._vad_model:  object | None       = None   # populated by load_vad()
        self._lock        = threading.Lock()            # one transcription at a time
        self._warmup_done = threading.Event()
        self._warmup_thread: threading.Thread | None = None

        # barge-in state — set externally by wakeup.py after join_warmup()
        self._barge_in_event:  threading.Event = threading.Event()
        self._barge_in_active: bool             = False
        self._barge_in_thread: threading.Thread | None = None

    # ── staged init ───────────────────────────────────────────────────────────

    def load_whisper(self) -> None:
        """
        Load the Whisper model into memory.
        Blocking — downloads on first run, then loads from cache.
        Call before load_vad() so warmup thread has both models ready.
        """
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=self._device,
            compute_type=self._compute,
        )

    def load_vad(self) -> None:
        """
        Load Silero VAD and kick off the background warmup thread.
        Requires load_whisper() to have been called first.
        """
        self._vad_model = load_silero_vad()   # ~2 MB ONNX/JIT, MIT license
        self._vad_model.eval()
        self._warmup_thread = threading.Thread(target=self._warmup, daemon=True)
        self._warmup_thread.start()

    def join_warmup(self) -> None:
        """
        Block until both Whisper and Silero are warm.
        Call after load_vad() and before first listen().
        """
        self._warmup_done.wait()

    # ── barge-in monitor ──────────────────────────────────────────────────────

    def start_barge_in_monitor(self) -> None:
        """
        Launch the always-on VAD daemon that sets _barge_in_event when speech
        is detected while Aiko is speaking.

        Uses Silero only — no Whisper — so CPU cost on the Jetson Orin Nano is
        negligible. The event is consumed by speak.wait_or_barge_in(), which
        stops playback and returns True so listen() opens the mic immediately.

        Call once from wakeup.py after join_warmup() completes. Safe to call
        more than once — second call is a no-op if already running.
        """
        if self._barge_in_active:
            return
        self._barge_in_active = True
        self._barge_in_thread = threading.Thread(
            target=self._barge_in_loop, daemon=True,
        )
        self._barge_in_thread.start()

    def stop_barge_in_monitor(self) -> None:
        """
        Graceful shutdown — sets _barge_in_active to False so the daemon loop
        exits on its next iteration. Call on process exit if needed.
        """
        self._barge_in_active = False

    def _barge_in_loop(self) -> None:
        """
        Continuously scores mic chunks through Silero VAD.

        Fires _barge_in_event when BARGE_IN_CONFIRM consecutive chunks exceed
        BARGE_IN_THRESHOLD. Auto-clears the event after BARGE_IN_COOLDOWN_MS
        so it is always ready for the next utterance — no manual reset needed
        by callers.

        Runs as a daemon so it dies cleanly with the process. If PortAudio
        throws, logs a warning and exits rather than crashing the session.
        """
        try:
            with sd.InputStream(
                samplerate=CAPTURE_RATE,
                channels=1,
                dtype="float32",
                blocksize=_CHUNK_SAMPLES_CAP,
                device=DEVICE_INDEX,
            ) as stream:
                consecutive = 0
                while self._barge_in_active:
                    chunk, _ = stream.read(_CHUNK_SAMPLES_CAP)

                    # event is set and cooling down — drain but don't re-trigger
                    if self._barge_in_event.is_set():
                        continue

                    score = self._score_chunk(chunk)
                    if score >= BARGE_IN_THRESHOLD:
                        consecutive += 1
                        if consecutive >= BARGE_IN_CONFIRM:
                            self._barge_in_event.set()
                            consecutive = 0
                            # auto-clear after cooldown so next turn starts fresh
                            threading.Timer(
                                BARGE_IN_COOLDOWN_MS / 1000.0,
                                self._barge_in_event.clear,
                            ).start()
                    else:
                        consecutive = 0

        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning(f"Barge-in monitor died: {exc}")

    # ── public api ────────────────────────────────────────────────────────────

    def listen(
        self,
        status_callback=None,
        wait_fn=None,
        speak=None,
    ) -> str:
        """
        Block until one complete speech utterance is captured and transcribed.

        Wait behaviour (mutually exclusive, speak= takes priority):
          - speak provided: calls speak.wait_or_barge_in(_barge_in_event) so
            the user can interrupt mid-sentence. Clears the barge-in event
            before opening the mic so _record() never sees a stale trigger.
          - wait_fn provided (no speak): legacy blocking wait, used in text
            mode or when TTS is toggled off at runtime.
          - neither: opens the mic immediately (Aiko is silent).

        Args:
            status_callback: optional callable(str) for UI status strings
                             e.g. "__LISTENING__", "__TRANSCRIBING__",
                             "__WAITING__", "__IDLE__"
            wait_fn:         optional callable() — legacy blocking wait kept
                             for text-mode compat; ignored when speak= is given
            speak:           optional AikoSpeak instance; when provided and
                             is_playing() is True, uses wait_or_barge_in()
                             for interruptible waiting

        Returns:
            Transcribed text string, or "" if nothing intelligible was captured.
        """
        # ── wait phase: block until it's our turn to capture ──────────────────
        if speak is not None and speak.is_playing():
            _cb(status_callback, "__WAITING__")
            interrupted = speak.wait_or_barge_in(self._barge_in_event)
            if interrupted:
                # clear the event now — _record() must not see it as stale speech
                self._barge_in_event.clear()
        elif wait_fn is not None:
            wait_fn()                          # legacy: plain blocking wait

        # ── capture + transcribe ──────────────────────────────────────────────
        _cb(status_callback, "__LISTENING__")
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
