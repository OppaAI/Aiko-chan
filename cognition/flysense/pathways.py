"""Phase 6: real sensory → fly pathways.

Every sensory channel the turn loop already carries becomes *neural activity*
published into NeuralState — with provenance — instead of only features or
strings.

Channels
--------
voice   ASR prosody (rms, peak, voiced_fraction, pause_density,
        words_per_second — see sensory/listen.py::_prosody_features)
        → GF urgency (loud = urgent), DN arousal/vigor (speech rate =
        energy), CX decisiveness (pauses = hesitation).
motion  Camera frames → FlyMotion T4/T5 opponent energy
        (cognition/flysense/motion.py) → motion_salience, plus a GF
        urgency bump on sudden large change.
visual  Consented camera/screen frame events feed the motion pathway and
        are recorded as provenance. Image *bytes* are never retained —
        only scalar energies, matching the existing privacy posture.

Modes (AIKO_FLY_SENSE_MODE): off | shadow | live. Default shadow:
compute + record provenance, never publish. Never raises.
"""
from __future__ import annotations

import base64
import binascii
import logging
import threading
import time

log = logging.getLogger("aiko.flysense.pathways")


def _mode() -> str:
    try:
        from system.config import env_str
        return env_str("AIKO_FLY_SENSE_MODE", "shadow").strip().lower()
    except Exception:
        return "shadow"


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


# ── voice ──────────────────────────────────────────────────────────────────

def encode_voice_prosody(prosody: dict | None) -> dict | None:
    """Map ASR prosody features → neural contributions. Pure function.

    Returns None when there is no usable prosody. Mappings are deliberately
    gentle: the voice is one vote among many, never a wild swing.
    """
    if not isinstance(prosody, dict) or not prosody:
        return None
    try:
        rms = _clamp01(prosody.get("rms", 0.0))
        peak = _clamp01(prosody.get("peak", 0.0))
        try:
            wps = max(0.0, float(prosody.get("words_per_second", 0.0)))
        except Exception:
            wps = 0.0
        pause = _clamp01(prosody.get("pause_density", 0.0))

        # Loudness → urgency. Raised voice / shouting correlates with urgency;
        # capped so a loud room never fires an interrupt by itself.
        urgency = _clamp01((rms - 0.55) / 0.45) * 0.35
        if peak >= 0.95:
            urgency += 0.10
        urgency = min(0.45, urgency)

        # Speech rate → bodily energy. ~0.8 wps is near-silence, ~5 wps is
        # rapid speech; maps onto DN arousal/vigor around the 1.0 baseline.
        energy = _clamp01((wps - 0.8) / 4.2)
        arousal = 0.30 + 0.55 * energy
        vigor = 0.90 + 0.20 * energy

        # Pause density → hesitation → CX decisiveness nudge.
        hesitation = _clamp01((pause - 0.40) / 0.40)

        return {
            "urgency": round(urgency, 4),
            "energy": round(energy, 4),
            "arousal": round(arousal, 4),
            "vigor": round(vigor, 4),
            "hesitation": round(hesitation, 4),
        }
    except Exception as exc:
        log.debug("encode_voice_prosody skipped: %s", exc)
        return None


# ── motion / visual ────────────────────────────────────────────────────────

_pathways: dict[str, "MotionPathway"] = {}
_pathways_lock = threading.RLock()


def _key(user_id: str | None) -> str:
    try:
        from cognition.fly_registry import _norm_id
        return _norm_id(user_id)
    except Exception:
        return (user_id or "").strip() or "default"


class MotionPathway:
    """Stateful visual-motion → neural pathway for one identity.

    Wraps flysense.motion.FlyMotion (T4/T5 opponent Reichardt energy +
    wide-field change energy). Frames are scored on arrival; the per-turn
    readout publishes the freshest salience, then decays it so stale
    motion fades instead of haunting later turns.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._fly = None
        self._salience = 0.0
        self._sudden = False
        self._frames = 0
        self._visual_events = 0
        self._last_ts = 0.0

    def _ensure(self):
        if self._fly is None:
            from cognition.flysense.motion import FlyMotion
            self._fly = FlyMotion()
        return self._fly

    def feed_frame(self, frame) -> dict:
        """Score one frame (np.ndarray, or raw image bytes). Best-effort."""
        try:
            import numpy as np

            arr = frame
            if isinstance(frame, (bytes, bytearray)):
                arr = _decode_image_bytes(frame)
                if arr is None:
                    return {"ok": False, "reason": "decode_failed"}
            arr = np.asarray(arr)
            if arr.size == 0:
                return {"ok": False, "reason": "empty"}
            fly = self._ensure()
            e = fly.score(arr)
            total = float(e.get("total", 0.0) or 0.0)
            change = float(e.get("change", 0.0) or 0.0)
            # T4/T5 opponent energy is O(0.01–0.1); wide-field change is a
            # fraction. Map onto [0,1] salience, gentle like the voice path.
            salience = max(0.0, min(1.0, total * 6.0 + change * 1.5))
            sudden = bool(change >= 0.10)  # FlyMotion.change_threshold
            with self._lock:
                # Fresh motion wins; simultaneous feeds take the max.
                self._salience = max(self._salience, salience)
                self._sudden = self._sudden or sudden
                self._frames += 1
                self._last_ts = time.time()
            return {
                "ok": True,
                "salience": round(salience, 4),
                "sudden": sudden,
                "novel": bool(e.get("novel")),
            }
        except Exception as exc:
            log.debug("MotionPathway.feed_frame skipped: %s", exc)
            return {"ok": False, "reason": "error"}

    def note_visual_event(self) -> None:
        """A consented frame was submitted (camera/screen). Provenance only."""
        with self._lock:
            self._visual_events += 1
            self._last_ts = time.time()

    def turn_readout(self) -> dict:
        """Freshest motion salience for this turn; decays afterwards."""
        with self._lock:
            out = {
                "salience": round(self._salience, 4),
                "sudden": self._sudden,
                "frames": self._frames,
                "visual_events": self._visual_events,
            }
            # Decay so motion doesn't haunt later turns; visual-event count
            # resets each turn (it describes *this* turn's submissions).
            self._salience *= 0.5
            self._sudden = False
            self._visual_events = 0
            return out


def _decode_image_bytes(data: bytes):
    """Best-effort bytes → grayscale np.ndarray. Returns None on failure."""
    try:
        import numpy as np

        arr = np.frombuffer(data, dtype=np.uint8)
        # Prefer PIL, fall back to cv2; neither is a hard dependency.
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(data)).convert("L")
            return np.asarray(img, dtype=np.float64)
        except Exception:
            pass
        try:
            import cv2

            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return np.asarray(img, dtype=np.float64)
        except Exception:
            pass
        return None
    except Exception:
        return None


def get_motion_pathway(user_id: str | None = None) -> MotionPathway:
    key = _key(user_id)
    with _pathways_lock:
        mp = _pathways.get(key)
        if mp is None:
            mp = MotionPathway()
            _pathways[key] = mp
        return mp


def note_visual_frame(user_id: str | None, image_data_url: str | None) -> dict:
    """Feed a consented camera/screen frame into the motion pathway.

    Accepts a data-URL (as the WebUI sends) or raw bytes. Decodes to a
    thumbnail-scale array, scores motion energy, and discards the bytes —
    nothing visual is retained. Never raises.
    """
    out = {"ok": False}
    try:
        if not image_data_url:
            return out
        raw: bytes | None = None
        if isinstance(image_data_url, (bytes, bytearray)):
            raw = bytes(image_data_url)
        elif isinstance(image_data_url, str):
            s = image_data_url.strip()
            if s.startswith("data:"):
                _, _, b64 = s.partition(",")
            else:
                b64 = s
            try:
                raw = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError):
                return {**out, "reason": "bad_base64"}
        if not raw:
            return out
        # Privacy: 64px cap is plenty for motion energy; less retained, even
        # transiently, is better.
        if len(raw) > 4_000_000:
            return {**out, "reason": "too_large"}
        mp = get_motion_pathway(user_id)
        fed = mp.feed_frame(raw)
        mp.note_visual_event()
        out = {"ok": bool(fed.get("ok")), **{k: v for k, v in fed.items() if k != "ok"}}
        return out
    except Exception as exc:
        log.debug("note_visual_frame skipped: %s", exc)
        return out


# ── per-turn orchestration ─────────────────────────────────────────────────

def encode_turn_senses(
    text: str,
    *,
    user_id: str | None = None,
    prosody: dict | None = None,
) -> dict:
    """Encode this turn's sensory channels into neural activity.

    Runs inside apply_turn_priors, before the GF assessment, so fresh
    motion salience is visible to the interrupt line. In shadow mode
    (default) everything is computed and recorded but nothing is
    published. Never raises.
    """
    out: dict = {
        "mode": "off",
        "voice": None,
        "motion": None,
        "published": False,
    }
    try:
        mode = _mode()
        out["mode"] = mode
        if mode == "off":
            return out

        voice = encode_voice_prosody(prosody)
        out["voice"] = voice

        mp = get_motion_pathway(user_id)
        motion = mp.turn_readout()
        out["motion"] = motion

        try:
            from cognition.neural_state import get_neural_state

            st = get_neural_state(user_id)
            st.record_influence(
                {
                    "kind": "senses",
                    "mode": mode,
                    "voice": voice,
                    "motion": {
                        k: motion.get(k)
                        for k in ("salience", "sudden", "frames", "visual_events")
                    },
                }
            )
            if mode == "live":
                if motion and motion.get("salience", 0.0) > 0:
                    st.publish_motion(
                        float(motion["salience"]), source="pathways"
                    )
                if voice and voice.get("hesitation") is not None:
                    # Absolute per-turn mapping (not cumulative): hesitation
                    # reads out as lower decisiveness, recovers next turn.
                    decisiveness = max(
                        0.1, min(0.9, 0.55 - 0.30 * float(voice["hesitation"]))
                    )
                    st.publish_cx(
                        heading_deg=float(st.focus_heading or 0.0),
                        sharpness=float(st.focus_sharpness or 0.0),
                        decisiveness=decisiveness,
                        sleep_pressure=float(st.sleep_pressure or 0.0),
                        source="pathways",
                    )
                out["published"] = True
        except Exception as exc:
            log.debug("encode_turn_senses publish skipped: %s", exc)
        return out
    except Exception as exc:
        log.debug("encode_turn_senses skipped: %s", exc)
        return out
