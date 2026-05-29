"""
core/speak.py

Aiko's voice output via RealtimeTTS + Kokoro.

Standalone test:
    python core/speak.py
    python core/speak.py "こんにちは、私はアイコちゃんです！"
    python core/speak.py --devices
"""

import os
import sys
import argparse


# ── config ────────────────────────────────────────────────────────────────────

KOKORO_VOICE  = os.getenv("KOKORO_VOICE",  "jf_alpha")   # default female voice
KOKORO_SPEED  = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_LANG   = os.getenv("KOKORO_LANG",   "en-us")


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    RealtimeTTS wrapper using Kokoro engine.
    Lazy-loads on first use to keep startup fast.
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
            self._stream = TextToAudioStream(engine)
            self._ready  = True
            print("[speak] Kokoro engine loaded.")
            return True
        except Exception as e:
            print(f"[speak] failed to load Kokoro engine: {e}")
            return False

    def speak(self, text: str) -> bool:
        """
        Synthesize text and play audio.
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


# ── standalone test ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiko speak test")
    parser.add_argument(
        "text",
        nargs="?",
        default="こんにちは！私はアイコちゃんです。よろしくね！",
        help="Text to synthesize",
    )
    parser.add_argument("--devices", action="store_true", help="List audio output devices")
    parser.add_argument("--voice",   default=None, help="Override KOKORO_VOICE")
    parser.add_argument("--speed",   default=None, type=float, help="Override KOKORO_SPEED")
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    args = _parse_args()

    if args.devices:
        list_devices()
        sys.exit(0)

    if args.voice:
        os.environ["KOKORO_VOICE"] = args.voice
    if args.speed:
        os.environ["KOKORO_SPEED"] = str(args.speed)

    voice = AikoSpeak()
    ok    = voice.speak(args.text)
    sys.exit(0 if ok else 1)