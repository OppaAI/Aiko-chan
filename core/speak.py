"""
core/speak.py

Aiko's voice output via RealtimeTTS + Kokoro.
Voice blending supported via formula strings: "0.2*af_nicole + 0.8*jf_alpha"

Install:
    uv add "realtimetts[kokoro]"

Standalone test:
    python -m core.speak
    python -m core.speak "Hello, I'm Aiko!"
    python -m core.speak --devices
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

    Streaming usage (LLM token-by-token):
        speaker.start_listening()
        for token in llm_stream:
            speaker.feed(token)
        speaker.finish()   # flush final open sentence
        speaker.wait()     # block until audio drains

    Single-string usage:
        speaker.speak("Hello!")   # blocks until done
    """

    def __init__(self) -> None:
        self._stream = None
        self._ready  = False
        print(f"[speak] Kokoro ready (lazy) | voice: {KOKORO_VOICE} | speed: {KOKORO_SPEED}")

    def warmup(self) -> bool:
        """Pre-load the engine — called from background thread in cli.py."""
        return self._init_engine()

    def _init_engine(self) -> bool:
        """Initialise Kokoro engine."""
        if self._ready:
            return True
        try:
            from RealtimeTTS import TextToAudioStream, KokoroEngine
            engine = KokoroEngine(
                voice=KOKORO_VOICE,
                default_speed=KOKORO_SPEED,
            )
            # No silent_stderr here — swallowing errors at init causes silent failures later
            self._stream = TextToAudioStream(
                engine,
                output_device_index=KOKORO_DEVICE if KOKORO_DEVICE >= 0 else None,
                frames_per_buffer=8192,
                playout_chunk_size=8192,
            )
            self._ready = True
            print(f"[speak] Kokoro engine loaded | voice: {KOKORO_VOICE!r}")
            return True
        except Exception as e:
            print(f"[speak] failed to load Kokoro engine: {e}")
            return False

    # ── internal ───────────────────────────────────────────────────────────────

    def _play_params(self) -> dict:
        return dict(
            fast_sentence_fragment=False,
            minimum_sentence_length=15,
            minimum_first_fragment_length=30,
            buffer_threshold_seconds=0.2,
            language=KOKORO_LANG,
        )

    def _sanitized_iterator(self, iterator):
        for token in iterator:
            clean = sanitize_for_tts(token)
            if clean:
                yield clean

    # ── public api ─────────────────────────────────────────────────────────────

    def speak(self, text: str) -> bool:
        """Synthesize a complete string, blocking until audio finishes."""
        if not self._init_engine():
            return False
        try:
            self._stream.feed(sanitize_for_tts(text))
            self._stream.play(**self._play_params())
            return True
        except Exception as e:
            print(f"[speak] speak error: {e}")
            return False

    def feed(self, token: str) -> None:
        """
        Feed a single token during streaming.
        Call start_listening() before the first feed().
        """
        if not self._ready or not token:
            return
        clean = sanitize_for_tts(token)
        if clean:
            try:
                self._stream.feed(clean)
            except Exception as e:
                print(f"\n[speak] feed error: {e}")

    def start_listening(self) -> None:
        """
        Start async playback before tokens arrive so Kokoro synthesizes live.
        After all tokens: call finish() then wait().
        """
        if not self._init_engine():
            return
        try:
            self._stream.play_async(**self._play_params())
        except Exception as e:
            print(f"[speak] start_listening error: {e}")

    def finish(self) -> None:
        """
        Flush the final open sentence after all tokens have been fed.
        Call before wait().
        """
        if not self._ready or not self._stream:
            return
        try:
            self._stream.feed("  .  ")
        except Exception as e:
            print(f"[speak] finish error: {e}")

    def feed_and_play(self, token_iterator) -> None:
        """Feed a token iterator and play synchronously. Blocks until done."""
        if not self._init_engine():
            return
        self._stream.feed(self._sanitized_iterator(token_iterator))
        self._stream.feed("  .  ")
        self._stream.play(**self._play_params())

    def is_playing(self) -> bool:
        if not self._stream:
            return False
        return self._stream.is_playing()

    def wait(self) -> None:
        """Block until audio playback finishes."""
        while self.is_playing():
            time.sleep(0.05)

    def stop(self) -> None:
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

    voice = AikoSpeak()
    ok    = voice.speak(args.text)
    sys.exit(0 if ok else 1)