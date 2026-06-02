"""
core/silence.py

Context manager that suppresses C-level stderr (ALSA, ONNX, PyAudio spam)
by redirecting file descriptor 2 to /dev/null for the duration of the block.
"""

import os
from contextlib import contextmanager


@contextmanager
def silent_stderr():
    """Redirect fd 2 to /dev/null — silences C-library noise (ALSA, ONNX, PyAudio)."""
    devnull_fd      = os.open(os.devnull, os.O_WRONLY)
    real_stderr_fd  = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(real_stderr_fd, 2)
        os.close(real_stderr_fd)
        os.close(devnull_fd)