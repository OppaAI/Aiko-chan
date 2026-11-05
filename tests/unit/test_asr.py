#!/usr/bin/env python3
"""Manual SenseVoice/Silero diagnostic; no models are loaded on import.

From the repository root:
    python -m tests.unit.test_asr --stage env
    python -m tests.unit.test_asr --stage files
    python -m tests.unit.test_asr --mic

The env stage is offline and does not import torch or load native voice libraries.
Other stages may download the configured SenseVoice/Silero models to the HF cache.
Automated voice regressions live in test_listen_runtime.py.
"""
from __future__ import annotations

import argparse
import time
import traceback


def run_stage(label, fn):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    started = time.monotonic()
    result = fn()
    print(f"[OK] {label} ({time.monotonic() - started:.2f}s)")
    return result


def check_env():
    from system.config import load_config
    load_config()
    from sensory import listen as voice
    for name in ('ASR_DEVICE', 'ASR_LANGUAGE', 'ASR_NUM_THREADS', 'ASR_MODEL'):
        print(f"{name} = {getattr(voice, name)}")
    print("SenseVoice uses sherpa-onnx; torch/CUDA availability is not its provider check.")
    return voice


def check_files(voice):
    from pathlib import Path
    model, tokens = voice._resolve_sense_voice_files()
    for filename in (model, tokens):
        path = Path(filename)
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or empty SenseVoice file: {path}")
        print(f"{path.name}: {path.stat().st_size} bytes")


def check_asr(voice):
    listener = voice.AikoListen()
    listener.load_asr()
    if listener._model is None:
        raise RuntimeError("SenseVoice recognizer is unavailable")
    return listener


def check_vad(listener):
    listener.load_wakeword()
    listener.load_speaker_id()
    listener.load_vad()
    if not listener.join_warmup():
        raise RuntimeError("Voice warmup timed out")
    # Avoid reloading all models in the optional mic pass below.
    listener._ready = True
    return listener


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['env', 'files', 'mirror', 'asr', 'full'], default='full',
                        help='mirror is a compatibility alias for files')
    parser.add_argument('--mic', action='store_true', help='also record one real microphone utterance')
    args = parser.parse_args()
    listener = None
    try:
        voice = run_stage('ENV / SENSEVOICE CONFIGURATION', check_env)
        if args.stage == 'env':
            return 0
        run_stage('SENSEVOICE MODEL / TOKEN FILES', lambda: check_files(voice))
        if args.stage in ('files', 'mirror'):
            return 0
        listener = run_stage('SENSEVOICE RECOGNIZER', lambda: check_asr(voice))
        if args.stage == 'asr':
            return 0
        run_stage('SILERO / OPTIONAL MODELS / WARMUP', lambda: check_vad(listener))
        if args.mic:
            text, info = run_stage('MICROPHONE', listener.listen)
            print(f"Transcript: {text!r}; verified: {info.get('verified')}")
        print('\nAll requested stages passed.')
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if listener is not None:
            listener.stop_barge_in_monitor()


if __name__ == '__main__':
    raise SystemExit(main())
