"""Offline regressions for whole-brain artifact preparation and extraction."""
from __future__ import annotations

import shutil
import threading
import urllib.request

import numpy as np
import pytest

from cognition.flymemory import fullbrain
from cognition.flymemory.tools import extract_full


def _artifact(path):
    np.savez(path, indptr=np.array([0, 1, 1]), indices=np.array([1]),
             data=np.array([1.0]), pre_sorted=np.array([0]),
             node_ids=np.array([10, 20]), node_types=np.array(["KC", "DN"]),
             node_sign=np.array([1.0, 1.0]))


def test_ensure_data_skips_invalid_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(fullbrain.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AIKO_FULLBRAIN_PATH", str(tmp_path / "bad.npz"))
    monkeypatch.setattr(fullbrain, "RELEASE_URL", "")
    (tmp_path / "bad.npz").write_bytes(b"not an npz")
    dest = tmp_path / ".aiko" / "data" / fullbrain.DATA_NAME
    dest.parent.mkdir(parents=True)
    _artifact(dest)
    assert fullbrain.ensure_data() == dest


def test_ensure_data_validates_before_atomic_publish(monkeypatch, tmp_path):
    monkeypatch.setattr(fullbrain.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("AIKO_FULLBRAIN_PATH", raising=False)
    monkeypatch.setattr(fullbrain, "RELEASE_URL", "https://example.invalid/artifact.npz")
    dest = tmp_path / ".aiko" / "data" / fullbrain.DATA_NAME
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old invalid payload")

    def invalid_download(_url, path):
        np.savez(path, indptr=np.array([0, 1]))

    monkeypatch.setattr(urllib.request, "urlretrieve", invalid_download)
    assert fullbrain.ensure_data() is None
    assert dest.read_bytes() == b"old invalid payload"
    assert list(dest.parent.glob(f".{fullbrain.DATA_NAME}.*.npz")) == []

    source = tmp_path / "source.npz"
    _artifact(source)
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda _url, path: shutil.copyfile(source, path))
    assert fullbrain.ensure_data() == dest
    assert len(fullbrain.load_fullbrain(dest)["node_ids"]) == 2


def test_get_fullbrain_does_not_block_or_hold_lock_during_retrieval(monkeypatch):
    begun = threading.Event()
    release = threading.Event()
    unlocked = []
    brain = object()

    def prepare():
        acquired = fullbrain._lock.acquire(blocking=False)
        unlocked.append(acquired)
        if acquired:
            fullbrain._lock.release()
        begun.set()
        release.wait()
        return "prepared.npz"

    monkeypatch.setattr(fullbrain, "_instance", None)
    monkeypatch.setattr(fullbrain, "_load_started", False)
    monkeypatch.setattr(fullbrain, "ensure_data", prepare)
    monkeypatch.setattr(fullbrain, "FullBrain", lambda _path: brain)
    try:
        assert fullbrain.get_fullbrain() is None
        assert begun.wait(2)
        assert unlocked == [True]
        assert fullbrain.get_fullbrain() is None
    finally:
        release.set()
    for _ in range(100):
        if fullbrain.get_fullbrain() is brain:
            break
        threading.Event().wait(0.01)
    assert fullbrain.get_fullbrain() is brain


@pytest.mark.parametrize("remote_size,raises,expected_download", [(3, False, False), (4, False, True), (3, True, True)])
def test_extractor_checks_remote_size_with_ranged_get(monkeypatch, tmp_path, remote_size, raises, expected_download):
    dest = tmp_path / "existing.feather"
    dest.write_bytes(b"old")
    downloads = []

    class Response:
        status = 206
        headers = {"Content-Range": f"bytes 0-0/{remote_size}"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    def open_range(request, timeout):
        assert request.get_header("Range") == "bytes=0-0"
        if raises:
            raise OSError("range unavailable")
        return Response()

    def download(_url, path):
        downloads.append(path)
        dest.write_bytes(b"new!")

    monkeypatch.setattr(urllib.request, "urlopen", open_range)
    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    extract_full._dl("https://example.invalid/data", str(dest))
    assert bool(downloads) is expected_download
