"""
python -m util.enroll_speak

One-time enrollment for Aiko's speaker verification (listen.py).
Records a few seconds of your voice via parec, computes a speaker
embedding with the same sherpa-onnx model listen.py uses at runtime,
and saves it to the selected user profile (default: <USER_SPACE_ROOT>/<user_id>/profile/speaker_enrollment.json).

Usage:
    python -m util.enroll_speak
    python -m util.enroll_speak --seconds 8
    python -m util.enroll_speak --model /path/to/embedding_model.onnx

Requires the same SPEAKER_MODEL_PATH env var (or --model) as listen.py,
pointing at a sherpa-onnx speaker embedding .onnx file:
    https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
e.g. 3dspeaker_speech_eres2net_base_sv_en_voxceleb_16k.onnx (~28MB)

Saves to <USER_SPACE_ROOT>/<user_id>/profile/speaker_enrollment.json (matches listen.py's lookup) —
use --user with the identity shown in the WebUI (AIKO_USER_ID/USER_ID are also supported).

Re-running this script overwrites any existing enrollment.
"""

import argparse
import math
import os
import select
import subprocess
import sys
import time

import numpy as np
from system.userspace import (current_user_id, resolve_owner_user_id, set_current_user_id,
                              reset_current_user_id, user_state_path)
from sensory.speaker_profile import save_profile, validate_embedding

SAMPLE_RATE = 16000
_PAREC_CMD = [
    "parec",
    "--rate=16000",
    "--channels=1",
    "--format=float32le",
    "--latency-msec=30",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a voice for Aiko speaker verification")
    parser.add_argument("--model", default=os.getenv("SPEAKER_MODEL_PATH", ""),
        help="Path to sherpa-onnx speaker embedding .onnx model")
    parser.add_argument("--out", default=None,
        help="Where to save the enrollment JSON (default: <USER_SPACE_ROOT>/<user_id>/profile/speaker_enrollment.json)")
    parser.add_argument("--user", default=os.getenv("AIKO_USER_ID") or os.getenv("USER_ID"),
        help="Runtime user identity; defaults to the configured owner when no user is bound")
    parser.add_argument("--seconds", type=float, default=6.0,
        help="How long to record (seconds). 5-8s of normal speech works well.")
    parser.add_argument("--name", default=None,
        help="Label stored alongside the embedding (for your reference only)")
    return parser.parse_args()


def _record_seconds(seconds: float) -> np.ndarray:
    """Capture raw mic audio for a fixed duration via parec."""
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("recording duration must be finite and between 0 and 60 seconds")
    n_samples = int(seconds * SAMPLE_RATE)
    n_bytes   = n_samples * 4  # float32

    print(f"Recording for {seconds:.1f}s — speak naturally, a sentence or two is plenty.")
    for i in (3, 2, 1):
        print(f"  starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  recording now!          ")

    proc = subprocess.Popen(_PAREC_CMD, stdout=subprocess.PIPE)
    # Bound the whole capture: the old read-until-n_bytes loop blocked
    # forever if parec stalled with the pipe open (no data, no EOF).
    deadline = time.monotonic() + seconds + 30.0
    try:
        raw = bytearray()
        while len(raw) < n_bytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("  (stopped: mic produced no audio in time)")
                break
            rlist, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
            if not rlist:
                continue
            # os.read (not the BufferedReader's read): select() only sees the
            # OS fd, so a buffered read() could block for bytes already
            # sitting in Python's buffer or stall waiting for more.
            chunk = os.read(proc.stdout.fileno(), min(65536, n_bytes - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        # Reap the child: terminate() alone leaves a zombie if parec
        # ignores SIGTERM.
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        finally:
            proc.stdout.close()

    print("Done recording.")
    # A pipe can end part-way through its last sample; ignore the incomplete
    # trailing bytes rather than crashing after a successful partial capture.
    aligned = len(raw) - len(raw) % 4
    return np.frombuffer(raw, dtype="<f4", count=aligned // 4).copy()


def main() -> int:
    # --help should work without native voice libraries or decrypted config.
    if "--help" in sys.argv or "-h" in sys.argv:
        _parse_args()
    from system.config import load_config
    load_config()
    args = _parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        print("error: --seconds must be finite, greater than zero, and at most 60")
        return 1
    user = args.user or current_user_id()
    if user == "guest":
        user = resolve_owner_user_id()
    if not user or user == "guest":
        print("error: select the runtime identity with --user; no owner is configured")
        return 1
    token = set_current_user_id(user)
    try:
        return _enroll(args, user)
    finally:
        reset_current_user_id(token)


def _enroll(args, user: str) -> int:
    if not args.model or not os.path.isfile(args.model):
        print(f"error: --model / SPEAKER_MODEL_PATH not set or file not found: {args.model!r}")
        return 1
    try:
        import sherpa_onnx
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.model, num_threads=1, debug=False, provider="cpu",
        )
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        audio = _record_seconds(args.seconds)
        if audio.size < SAMPLE_RATE or not np.isfinite(audio).all():
            raise ValueError("record at least one second of finite speech before enrolling")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-4:
            raise ValueError("recorded audio is silent; check the microphone source")
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        ready = getattr(extractor, "is_ready", None)
        if ready is not None and not ready(stream):
            raise ValueError("not enough speech for a speaker embedding")
        embedding = validate_embedding(extractor.compute(stream))
        out_path = args.out or user_state_path("profile/speaker_enrollment.json")
        save_profile(out_path, {
            "name": args.name or user, "embedding": embedding.tolist(),
            "dim": int(embedding.size), "model": os.path.basename(args.model),
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
    except Exception as exc:
        print(f"error: enrollment failed: {exc}. Existing profile was preserved.")
        return 1
    print(f"Enrolled {user!r} ({embedding.size}-dim embedding) → {os.path.abspath(out_path)}")
    print("Set SPEAKER_VERIFY_ENABLED=1 and SPEAKER_MODEL_PATH in Settings to activate verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
