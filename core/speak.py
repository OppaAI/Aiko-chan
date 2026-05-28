"""
core/speak.py

Aiko's voice output via Style-BERT-VITS2 local API.
Expects Style-BERT-VITS2 server running at SBV2_URL (default: http://localhost:5000).

Standalone test:
    python core/speak.py
    python core/speak.py "こんにちは、私はアイコちゃんです！"
    python core/speak.py "Hello! I'm Aiko-chan." --out output.wav
    python core/speak.py --devices
"""

import os
import sys
import argparse
import requests
import sounddevice as sd
import soundfile as sf
import tempfile
from pathlib import Path


# ── config ────────────────────────────────────────────────────────────────────

SBV2_URL    = os.getenv("SBV2_URL",    "http://localhost:5000")
SBV2_MODEL  = os.getenv("SBV2_MODEL",  "")           # empty = SBV2 default model
SBV2_STYLE  = os.getenv("SBV2_STYLE",  "Neutral")
SBV2_SPEED  = float(os.getenv("SBV2_SPEED", "1.0"))
SBV2_DEVICE = os.getenv("SBV2_DEVICE", None)          # None = system default output


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    HTTP client for Style-BERT-VITS2 TTS API.
    Synthesizes text to audio and plays it via sounddevice.
    """

    def __init__(self) -> None:
        self._base   = SBV2_URL.rstrip("/")
        self._model  = SBV2_MODEL
        self._style  = SBV2_STYLE
        self._speed  = SBV2_SPEED
        self._device = int(SBV2_DEVICE) if SBV2_DEVICE else None
        print(f"[speak] Style-BERT-VITS2 at {self._base} | style: {self._style} | speed: {self._speed}")

    def speak(self, text: str, out_path: str | None = None) -> bool:
        """
        Synthesize text and play audio.
        Optionally save to out_path (.wav) instead of playing.
        Returns True on success, False on failure.
        """
        audio_bytes = self._synthesize(text)
        if not audio_bytes:
            return False

        if out_path:
            path = Path(out_path)
            path.write_bytes(audio_bytes)
            print(f"[speak] saved to {path}")
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                path = Path(f.name)

        self._play(path)

        if not out_path:
            try:
                path.unlink()
            except Exception:
                pass

        return True

    # ── internal ──────────────────────────────────────────────────────────────

    def _synthesize(self, text: str) -> bytes | None:
        """
        Call Style-BERT-VITS2 /voice endpoint and return raw WAV bytes.
        Returns None on failure.
        """
        params = {
            "text":   text,
            "style":  self._style,
            "speed":  self._speed,
            "format": "wav",
        }
        if self._model:
            params["model_name"] = self._model

        try:
            response = requests.get(
                f"{self._base}/voice",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.ConnectionError:
            print(f"[speak] could not reach Style-BERT-VITS2 at {self._base}")
        except requests.exceptions.Timeout:
            print("[speak] synthesis timed out")
        except requests.exceptions.RequestException as e:
            print(f"[speak] request error: {e}")
        return None

    def _play(self, path: Path) -> None:
        """Play a WAV file via sounddevice."""
        try:
            data, samplerate = sf.read(str(path))
            sd.play(data, samplerate, device=self._device)
            sd.wait()
        except Exception as e:
            print(f"[speak] playback error: {e}")


# ── list audio devices (debug) ────────────────────────────────────────────────

def list_devices() -> None:
    """Print all available audio output devices."""
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
    parser.add_argument("--out",     default=None,       help="Save WAV to this path instead of playing")
    parser.add_argument("--devices", action="store_true", help="List audio output devices and exit")
    parser.add_argument("--style",   default=None,       help="Override SBV2_STYLE")
    parser.add_argument("--speed",   default=None,       type=float, help="Override SBV2_SPEED")
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    args = _parse_args()

    if args.devices:
        list_devices()
        sys.exit(0)

    if args.style:
        os.environ["SBV2_STYLE"] = args.style
    if args.speed:
        os.environ["SBV2_SPEED"] = str(args.speed)

    voice = AikoSpeak()
    ok    = voice.speak(args.text, out_path=args.out)
    sys.exit(0 if ok else 1)
