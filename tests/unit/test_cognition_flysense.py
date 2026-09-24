"""Unit tests for fly sensory grafts (numpy-only, offline)."""
import numpy as np
import pytest

from cognition.flysense import FlyAL, FlyDN, FlyMotion, emd_energy
from cognition.flysense.pathways import MotionPathway, _decode_image_bytes, note_visual_frame


def test_al_loads_and_normalizes():
    al = FlyAL()
    assert al.summary()["glomeruli"] == 49
    rng = np.random.default_rng(0)
    v = rng.normal(size=128).astype(np.float32)
    v /= np.linalg.norm(v)
    n = al.normalize(v)
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-5  # L2 kept for vecstore
    assert not bool((n == v).all())  # gain control actually changes geometry
    assert bool((FlyAL().normalize(v) == n).all())  # deterministic
    z = al.normalize(np.zeros(64, dtype=np.float32))
    assert bool((z == 0).all())  # zero-safe
    b = al.normalize_batch(np.stack([v.astype(np.float64), v.astype(np.float64)]))
    assert b.shape == (2, 128)


def test_emd_directional_and_still():
    rng = np.random.default_rng(1)
    f0 = rng.normal(size=(48, 48))
    e0 = emd_energy(f0, f0)
    assert e0["total"] < 1e-9 and e0["change"] < 1e-9  # identical frames are still
    e1 = emd_energy(f0, np.roll(f0, 2, axis=1))
    assert e1["h"] > e1["v"]  # horizontal shift -> horizontal energy
    e2 = emd_energy(f0, np.roll(f0, 2, axis=0))
    assert e2["v"] > e2["h"]  # vertical shift -> vertical energy


def test_motion_gate_stateful():
    rng = np.random.default_rng(2)
    fm = FlyMotion()
    f0 = rng.normal(size=(96, 96))
    moved, e = fm.moved(f0)
    assert moved and e["novel"]  # first frame arms
    moved, _ = fm.moved(f0)
    assert not moved  # identical reframe is still
    moved, _ = fm.moved(np.roll(f0, 3, axis=1))
    assert moved  # translation trips the gate
    fm.reset()
    moved, e = fm.moved(f0)
    assert moved and e["novel"]


def test_motion_pathway_keeps_a_baseline_per_source():
    rng = np.random.default_rng(3)
    frame = rng.normal(size=(48, 48))
    pathway = MotionPathway()

    assert pathway.feed_frame(frame, source="camera")["novel"] is True
    assert pathway.feed_frame(np.roll(frame, 2, axis=1), source="screen")["novel"] is True
    assert pathway.feed_frame(frame, source="camera")["novel"] is False
    assert pathway.feed_frame(frame, source="screen")["novel"] is False
    assert len(pathway._flies) == 2


def test_image_decode_bounds_float_array_before_motion():
    import io
    Image = pytest.importorskip("PIL.Image")

    image = Image.new("RGB", (2048, 1024), color=(120, 40, 10))
    data = io.BytesIO()
    image.save(data, format="JPEG")

    decoded = _decode_image_bytes(data.getvalue())

    assert decoded is not None
    assert decoded.shape == (48, 48)
    assert decoded.dtype == np.float64


def test_visual_frame_passes_source_to_motion_pathway():
    import cv2

    ok, encoded = cv2.imencode(".png", np.zeros((48, 48), dtype=np.uint8))
    assert ok
    data = encoded.tobytes()
    camera = note_visual_frame("source-isolation", data, source="camera")
    screen = note_visual_frame("source-isolation", data, source="screen")
    camera_again = note_visual_frame("source-isolation", data, source="camera")
    assert camera["ok"] and camera["novel"]
    assert screen["ok"] and screen["novel"]
    assert camera_again["ok"] and not camera_again["novel"]


def test_image_decode_cv2_fallback_is_bounded(monkeypatch):
    import cv2
    import sys

    ok, encoded = cv2.imencode(".png", np.zeros((1024, 2048), dtype=np.uint8))
    assert ok
    monkeypatch.setitem(sys.modules, "PIL", None)

    decoded = _decode_image_bytes(encoded.tobytes())

    assert decoded is not None
    assert decoded.shape == (48, 48)
    assert decoded.dtype == np.float64


def test_dn_drive_bounded():
    dn = FlyDN()
    assert dn.summary()["n_dn"] == 1342
    for e_, d, a in [(0, 0, -1), (0.5, 0.5, 0), (1, 1, 1), (9, -9, 99)]:
        r = dn.drive(energy=e_, decisiveness=d, affect=a)
        assert 0.8 < r["rate_mult"] < 1.2 and 0.8 < r["vol_mult"] < 1.2
        assert 0.0 <= r["arousal"] <= 1.0


def _subliminal():
    from cognition.subliminal import SubliminalLayer
    return SubliminalLayer()


def test_subliminal_arousal_modes(monkeypatch):
    import os
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "off")
    s = _subliminal()
    from collections import deque
    s.scan("urgent question now!", {}, deque())
    off = s._affect.arousal
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "shadow")
    s2 = _subliminal()
    s2.scan("urgent question now!", {}, deque())
    assert s2._affect.arousal == off  # shadow never changes state
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    s3 = _subliminal()
    s3.scan("urgent question now!", {}, deque())
    assert 0.0 <= s3._affect.arousal <= 1.0
    assert abs(s3._affect.arousal - off) <= 0.11  # small DN nudge only


def test_vrm_intensity_modes(monkeypatch):
    from collections import deque
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "off")
    s = _subliminal()
    s.scan("I love this wonderful day", {}, deque())
    name_off, int_off = s.vrm_expression()
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "shadow")
    s2 = _subliminal()
    s2.scan("I love this wonderful day", {}, deque())
    assert s2.vrm_expression() == (name_off, int_off)  # shadow identical
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    s3 = _subliminal()
    s3.scan("I love this wonderful day", {}, deque())
    name_live, int_live = s3.vrm_expression()
    assert name_live == name_off  # expression choice untouched, only vigor
    assert 0.0 <= int_live <= 1.0
