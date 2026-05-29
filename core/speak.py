"""
core/speak.py

Aiko's voice output via RealtimeTTS + PocketTTS.

Install:
    pip install "realtimetts[pockettts]"

Standalone test:
    python core/speak.py
    python core/speak.py "Hello, I'm Aiko!"
    python core/speak.py --devices
    python core/speak.py --list-voices
"""

import os
import sys
import argparse


# ── config ────────────────────────────────────────────────────────────────────

# Built-in voice name OR path to a .wav / .safetensors file for voice cloning.
# Built-in voices: "jessica", "lena", "emma" etc. — run --list-voices to check.
POCKET_VOICE    = os.getenv("POCKET_VOICE",    "jessica")
POCKET_LANGUAGE = os.getenv("POCKET_LANGUAGE", "english")   # "english", "japanese", etc.
POCKET_SPEED    = float(os.getenv("POCKET_SPEED", "1.0"))


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    RealtimeTTS wrapper using PocketTTS engine.
    Lazy-loads on first use to keep startup fast.
    """

    def __init__(self) -> None:
        self._stream = None
        self._ready  = False
        print(f"[speak] PocketTTS ready (lazy) | voice: {POCKET_VOICE} | lang: {POCKET_LANGUAGE}")

    def warmup(self) -> bool:
        """Pre-load the engine at startup to avoid first-speak delay."""
        return self._init_engine()

    def _init_engine(self) -> bool:
        """Initialise RealtimeTTS PocketTTS engine on first use."""
        if self._ready:
            return True
        try:
            from RealtimeTTS import TextToAudioStream, PocketTTSEngine
            engine = PocketTTSEngine(
                voice=POCKET_VOICE,
                language=POCKET_LANGUAGE,
                speed=POCKET_SPEED,
            )
            self._stream = TextToAudioStream(engine)
            self._ready  = True
            print("[speak] PocketTTS engine loaded.")
            return True
        except Exception as e:
            print(f"[speak] failed to load PocketTTS engine: {e}")
            return False

    def speak(self, text: str) -> bool:
        """
        Synthesize text and play audio (blocking).
        Returns True on success, False on failure.
        """
        if not self._init_engine():
            return False
        try:
            self._stream.feed(text)
            self._stream.play(log_synthesized_text=False)
            return True
        except Exception as e:
            print(f"[speak] playback error: {e}")
            return False

    def feed(self, token: str) -> None:
        """Feed a token into the TTS stream during LLM streaming."""
        if not self._init_engine():
            return
        try:
            self._stream.feed(token)
        except Exception as e:
            print(f"[speak] feed error: {e}")

    def play_async(self) -> None:
        """Begin async playback — call once after all tokens have been fed."""
        if self._stream:
            try:
                self._stream.play_async(log_synthesized_text=False)
            except Exception as e:
                print(f"[speak] play_async error: {e}")

    def stop(self) -> None:
        """Stop any ongoing playback."""
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


def list_voices() -> None:
    """Print available built-in PocketTTS voices."""
    try:
        from RealtimeTTS import PocketTTSEngine
        voices = PocketTTSEngine.get_voices()
        print("[speak] Available PocketTTS voices:")
        for v in voices:
            print(f"  {v}")
    except Exception as e:
        print(f"[speak] could not list voices: {e}")


# ── standalone test ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiko speak test")
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello! I'm Aiko. Nice to meet you!",
        help="Text to synthesize",
    )
    parser.add_argument("--devices",     action="store_true", help="List audio output devices")
    parser.add_argument("--list-voices", action="store_true", help="List available PocketTTS voices")
    parser.add_argument("--voice",       default=None, help="Override POCKET_VOICE")
    parser.add_argument("--language",    default=None, help="Override POCKET_LANGUAGE")
    parser.add_argument("--speed",       default=None, type=float, help="Override POCKET_SPEED")
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    args = _parse_args()

    if args.devices:
        list_devices()
        sys.exit(0)

    if args.list_voices:
        list_voices()
        sys.exit(0)

    if args.voice:    os.environ["POCKET_VOICE"]    = args.voice
    if args.language: os.environ["POCKET_LANGUAGE"] = args.language
    if args.speed:    os.environ["POCKET_SPEED"]    = str(args.speed)

    voice = AikoSpeak()
    ok    = voice.speak(args.text)
    sys.exit(0 if ok else 1)
