"""Speaker-enrollment validation and atomic persistence, independent of ONNX."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np


def validate_embedding(raw) -> np.ndarray:
    embedding = np.asarray(raw, dtype=np.float32)
    if (embedding.ndim != 1 or not 0 < embedding.size <= 16384
            or not np.isfinite(embedding).all()
            or np.linalg.norm(embedding.astype(np.float64)) <= 0):
        raise ValueError('speaker embedding must be a finite nonzero vector of at most 16384 values')
    return embedding


def save_profile(path: str | Path, payload: dict) -> None:
    """Replace a profile only after valid JSON has been flushed to disk."""
    embedding = validate_embedding(payload['embedding'])
    if payload.get('dim') != embedding.size:
        raise ValueError('speaker profile dimension does not match the embedding')
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
