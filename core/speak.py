"""
core/speak.py

Aiko's voice output via RealtimeTTS + Kokoro.
Voice blending supported via formula strings: "0.1*af_heart + 0.9*jf_alpha"

Install:
    uv add "realtimetts[kokoro]"

Standalone test:
    python core/speak.py
    python core/speak.py "Hello, I'm Aiko!"
    python core/speak.py --devices
    python core/speak.py --voice "0.1*af_heart + 0.9*jf_alpha" "Testing blend"
    python core/speak.py --wait "Block until done."
"""

import os
import re
import sys
import time
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── config ────────────────────────────────────────────────────────────────────

# supports single voice ("af_heart") or blend formula ("0.1*af_heart + 0.9*jf_alpha")
KOKORO_VOICE  = os.getenv("KOKORO_VOICE",  "0.1*af_heart + 0.9*jf_alpha")
KOKORO_SPEED  = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_LANG   = os.getenv("KOKORO_LANG",   "en-us")
KOKORO_DEVICE = int(os.getenv("KOKORO_DEVICE", "-1"))  # 31 = pulse, -1 = default

# ── text sanitization ─────────────────────────────────────────────────────────

_REPLACEMENTS = [
    (r'\*',    ''),       # asterisk — markdown bold/italic, drop it
    (r'—',     ', '),     # em dash — replace with pause-friendly comma
    (r'–',     ', '),     # en dash
    (r'`',     ''),       # backtick — code formatting, drop it
    (r'#+ ',   ''),       # markdown headers
    (r'\[|\]', ''),       # square brackets
]

_RE_REPLACEMENTS = [(re.compile(p), r) for p, r in _REPLACEMENTS]


def sanitize_for_tts(text: str) -> str:
    """Strip/replace symbols the Kokoro phonemizer cannot handle."""
    for pattern, replacement in _RE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r',\s*,', ',', text)   # collapse double commas left behind
    text = re.sub(r'\s{2,}', ' ', text)  # collapse extra whitespace
    return text.strip()


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    RealtimeTTS wrapper using Kokoro engine.

    Two usage patterns:

    1. Speak a complete string (non-blocking):
           aiko_speak.speak("Hello!")

    2. Stream LLM tokens (non-blocking, low latency):
           aiko_speak.feed_and_play(ollama_stream_generator)

    Never mix feed()+play_async() with speak() in the same turn —
    call stop() first if interrupting mid-playback.
    """

    def __init__(self) -> None:
        self._stream = None
        self._ready  = False
        print(f"[speak] Kokoro ready (lazy) | voice: {KOKORO_VOICE} | speed: {KOKORO_SPEED}")

    def warmup(self) -> bool:
        """Pre-load the engine at startup to avoid first-speak delay."""
        return self._init_engine()

    def _init_engine(self) -> bool:
        """Initialise RealtimeTTS Kokoro engine on first use."""
        if self._ready:
            return True
        try:
            from RealtimeTTS import TextToAudioStream, KokoroEngine
            engine = KokoroEngine(
                voice=KOKORO_VOICE,
                default_speed=KOKORO_SPEED,
            )
            self._stream = TextToAudioStream(
                engine,
                output_device_index=KOKORO_DEVICE if KOKORO_DEVICE >= 0 else None,
            )
            self._ready = True
            print(f"[speak] Kokoro engine loaded | voice: {KOKORO_VOICE!r}")
            return True
        except Exception as e:
            print(f"[speak] failed to load Kokoro engine: {e}")
            return False

    # ── internal ───────────────────────────────────────────────────────────────

    def _play_async(self) -> None:
        """Shared play_async call with consistent latency parameters."""
        self._stream.play_async(
            fast_sentence_fragment=False,       # wait for complete sentences — fixes overlap
            minimum_sentence_length=20,         # join short fragments before synthesis
            minimum_first_fragment_length=30,   # hold back until first sentence is solid
            buffer_threshold_seconds=0.3,       # smooth the seam between sentences
            language=KOKORO_LANG,
        )

    def _sanitized_iterator(self, iterator):
        """Print raw tokens to console, feed only clean text to TTS."""
        for token in iterator:
            print(token, end='', flush=True)    # raw token to console
            clean = sanitize_for_tts(token)
            if clean:
                yield clean
        print()                                 # newline after response ends

    # ── public api ─────────────────────────────────────────────────────────────

    def speak(self, text: str) -> bool:
        """
        Synthesize a complete string and play async (non-blocking).
        Returns immediately — audio plays in background.
        Call stop() before the next speak() to interrupt.
        """
        if not self._init_engine():
            return False
        try:
            self.stop()
            self._stream.feed(sanitize_for_tts(text))
            self._play_async()
            return True
        except Exception as e:
            print(f"[speak] speak error: {e}")
            return False

    def feed(self, token: str) -> None:
        """
        Feed a single LLM token into the stream.
        Call play_async() once after all tokens are fed.
        Prints the raw token to console before sanitizing.
        """
        if not self._init_engine():
            return
        if not token:
            return
        print(token, end='', flush=True)
        clean = sanitize_for_tts(token)
        if clean:
            try:
                self._stream.feed(clean)
            except Exception as e:
                print(f"\n[speak] feed error: {e}")

    def play_async(self) -> None:
        """
        Begin async playback after a manual feed() loop.
        Call this once after your token loop completes.
        """
        if not self._ready or not self._stream:
            return
        try:
            self._play_async()
        except Exception as e:
            print(f"[speak] play_async error: {e}")

    def feed_and_play(self, token_iterator) -> None:
        """
        Feed an entire LLM token iterator and start async playback.
        Non-blocking — returns as soon as playback starts.
        Prints raw tokens to console, sanitizes before TTS.

        Usage:
            aiko_speak.feed_and_play(ollama_stream_generator)
        """
        if not self._init_engine():
            return
        self.stop()
        self._stream.feed(self._sanitized_iterator(token_iterator))
        self._play_async()

    def is_playing(self) -> bool:
        """True while synthesis or playback is still active."""
        if not self._stream:
            return False
        return self._stream.is_playing()

    def wait(self) -> None:
        """Block the calling thread until playback finishes."""
        while self.is_playing():
            time.sleep(0.05)

    def stop(self) -> None:
        """
        Interrupt current playback immediately.
        Call before feeding a new response to avoid overlap.
        """
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
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello! I'm Aiko. Nice to meet you! I run locally on your machine, so everything stays private.",
        help="Text to synthesize",
    )
    parser.add_argument("--devices", action="store_true", help="List audio output devices")
    parser.add_argument("--voice",   default=None, help='Override voice e.g. "0.1*af_heart + 0.9*jf_alpha"')
    parser.add_argument("--speed",   default=None, type=float, help="Override KOKORO_SPEED")
    parser.add_argument("--wait",    action="store_true", help="Block until playback finishes")
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

    KOKORO_VOICE = os.getenv("KOKORO_VOICE", "0.1*af_heart + 0.9*jf_alpha")

    voice = AikoSpeak()
    ok    = voice.speak(args.text)

    if args.wait:
        voice.wait()                            # --wait blocks for testing

    sys.exit(0 if ok else 1)