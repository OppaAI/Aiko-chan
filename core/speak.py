"""
core/speak.py

Aiko's voice output via RealtimeTTS + Kokoro.
Voice blending supported via formula strings: "0.2*af_nicole + 0.8*jf_alpha"

Install:
    uv add "realtimetts[kokoro]"

Standalone test:
    python core/speak.py
    python core/speak.py "Hello, I'm Aiko!"
    python core/speak.py --devices
    python core/speak.py --wait "Block until done."
"""

import os
import re
import sys
import time
import argparse
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.silence import silent_stderr

# ── config ────────────────────────────────────────────────────────────────────

KOKORO_VOICE  = os.getenv("KOKORO_VOICE",  "0.2*af_nicole + 0.8*jf_alpha")
KOKORO_SPEED  = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_LANG   = os.getenv("KOKORO_LANG",   "en-us")
KOKORO_DEVICE = int(os.getenv("KOKORO_DEVICE", "-1"))

# ── text sanitization ─────────────────────────────────────────────────────────

_REPLACEMENTS = [
    (r'\*',    ''),
    (r'—',     ', '),
    (r'–',     ', '),
    (r'`',     ''),
    (r'#+ ',   ''),
    (r'\[|\]', ''),
]

_RE_REPLACEMENTS = [(re.compile(p), r) for p, r in _REPLACEMENTS]


def sanitize_for_tts(text: str) -> str:
    """Strip/replace symbols the Kokoro phonemizer cannot handle."""
    for pattern, replacement in _RE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    RealtimeTTS wrapper using Kokoro engine.
    ALSA/PyAudio C-level noise suppressed via fd-level stderr redirect.
    Printing to console is the caller's responsibility — speak.py is silent.

    Streaming usage (LLM token-by-token):
        speaker.start_listening()
        for token in llm_stream:
            speaker.feed(token)
        speaker.finish()   # flush final open sentence
        speaker.wait()     # block until audio drains

    Single-string usage:
        speaker.speak("Hello!")
        # speak() blocks internally until audio finishes — no wait() needed.
    """

    def __init__(self) -> None:
        self._stream  = None
        self._ready   = False
        self._playing = False   # manual guard — prevents double play_async
        print(f"[speak] Kokoro ready (lazy) | voice: {KOKORO_VOICE} | speed: {KOKORO_SPEED}")

    def warmup(self) -> bool:
        """Pre-load the engine — called from background thread in cli.py."""
        return self._init_engine()

    def _init_engine(self) -> bool:
        """Initialise Kokoro engine. ALSA device probe silenced via fd redirect."""
        if self._ready:
            return True
        try:
            from RealtimeTTS import TextToAudioStream, KokoroEngine
            engine = KokoroEngine(
                voice=KOKORO_VOICE,
                default_speed=KOKORO_SPEED,
            )
            with silent_stderr():
                self._stream = TextToAudioStream(
                    engine,
                    output_device_index=KOKORO_DEVICE if KOKORO_DEVICE >= 0 else None,
                    frames_per_buffer=8192,   # prevents PCM underruns on Jetson
                    playout_chunk_size=8192,
                )
            self._ready = True
            print(f"[speak] Kokoro engine loaded | voice: {KOKORO_VOICE!r}")
            return True
        except Exception as e:
            print(f"[speak] failed to load Kokoro engine: {e}")
            return False

    # ── internal ───────────────────────────────────────────────────────────────

    def _play_async(self) -> None:
        """
        Shared play_async with tuned latency params.
        fast_sentence_fragment=False  — wait for complete sentences, no missing words
        minimum_sentence_length=15    — don't fire on very short fragments
        minimum_first_fragment_length=20 — hold first chunk until solid
        buffer_threshold_seconds=0.2  — enough buffer to prevent underruns
        """
        self._playing = True
        with silent_stderr():
            self._stream.play_async(
                fast_sentence_fragment=False,
                minimum_sentence_length=10,
                minimum_first_fragment_length=15,
                buffer_threshold_seconds=0.2,
                language=KOKORO_LANG,
            )

    def _sanitized_iterator(self, iterator):
        """Sanitize tokens silently — caller handles console printing."""
        for token in iterator:
            clean = sanitize_for_tts(token)
            if clean:
                yield clean

    # ── public api ─────────────────────────────────────────────────────────────

    def speak(self, text: str, block: bool = True) -> bool:
        if not self._init_engine():
            return False
        try:
            self._stream.feed(sanitize_for_tts(text))
            self._stream.play(
                fast_sentence_fragment=False,
                minimum_sentence_length=10,
                minimum_first_fragment_length=15,
                buffer_threshold_seconds=0.2,
                language=KOKORO_LANG,
                log_synthesized_text=True,
            )
            return True
        except Exception as e:
            print(f"[speak] speak error: {e}")
            return False

    def feed(self, token: str) -> None:
        """
        Feed a single token to TTS during streaming.
        Must call start_listening() before the first feed().
        Silent — caller prints to console.
        """
        if not self._init_engine():
            return
        if not token:
            return
        clean = sanitize_for_tts(token)
        if clean:
            try:
                self._stream.feed(clean)
            except Exception as e:
                print(f"\n[speak] feed error: {e}")

    def start_listening(self) -> None:
        """
        Start play_async before tokens arrive so Kokoro synthesizes live.
        This is the correct RealtimeTTS pattern for LLM streaming.

        After all tokens are fed, call finish() then wait() to drain audio:
            speaker.start_listening()
            for tok in stream: speaker.feed(tok)
            speaker.finish()
            speaker.wait()
        """
        if not self._init_engine():
            return
        if self._playing:
            # already running — don't stack a second play_async
            return
        try:
            # NOTE: silent_stderr intentionally omitted here so engine errors
            # surface instead of being swallowed silently.
            self._playing = True
            self._stream.play_async(
                fast_sentence_fragment=False,
                minimum_sentence_length=10,
                minimum_first_fragment_length=15,
                buffer_threshold_seconds=0.2,   # was 0.0 — caused underruns
                language=KOKORO_LANG,
            )
            print(f"[debug] play_async called, is_playing: {self._stream.is_playing()}", flush=True)
        except Exception as e:
            self._playing = False
            print(f"[speak] start_listening error: {e}")

    def finish(self) -> None:
        """
        Signal end of token stream — feed tail sentinel to flush final sentence.
        Must be called after all tokens have been fed, before wait().
        """
        if not self._ready or not self._stream:
            return
        try:
            self._stream.feed("  .  ")   # closes any open sentence in the tokenizer
        except Exception as e:
            print(f"[speak] finish error: {e}")

    def feed_and_play(self, token_iterator) -> None:
        """
        Feed a token iterator, play async, and flush. Non-blocking.
        Caller should call wait() after to block until audio drains.
        """
        if not self._init_engine():
            return
        self.stop()
        self._stream.feed(self._sanitized_iterator(token_iterator))
        self._stream.feed("  .  ")   # flush final sentence
        self._play_async()

    def is_playing(self) -> bool:
        if not self._stream:
            return False
        playing = self._stream.is_playing()
        if not playing:
            self._playing = False
        return playing

    def wait(self) -> None:
        """Block until audio playback finishes."""
        while self.is_playing():
            time.sleep(0.05)
        self._playing = False

    def stop(self) -> None:
        self._playing = False
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass


# ── list audio devices ────────────────────────────────────────────────────────

def list_devices() -> None:
    import sounddevice as sd
    print("[speak] Available audio output devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            print(f"  {i:2d}: {dev['name']}")


# ── standalone test ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiko speak test")
    parser.add_argument("text", nargs="?",
        default="Hello! I'm Aiko. Nice to meet you! I run locally on your machine, so everything stays private.")
    parser.add_argument("--devices", action="store_true")
    parser.add_argument("--voice",   default=None)
    parser.add_argument("--speed",   default=None, type=float)
    parser.add_argument("--wait",    action="store_true",
        help="Deprecated — speak() now blocks by default. Kept for back-compat.")
    parser.add_argument("--no-wait", action="store_true",
        help="Return immediately without blocking until audio finishes.")
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    args = _parse_args()

    if args.devices:
        list_devices()
        sys.exit(0)

    if args.voice: os.environ["KOKORO_VOICE"] = args.voice
    if args.speed: os.environ["KOKORO_SPEED"] = str(args.speed)
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", "0.2*af_nicole + 0.8*jf_alpha")

    voice = AikoSpeak()
    block = not args.no_wait   # block=True by default
    ok    = voice.speak(args.text, block=block)
    sys.exit(0 if ok else 1)