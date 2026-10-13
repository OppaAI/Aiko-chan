"""Per-turn fly behavioral priors (Stage 1 closed loops).

Called once at the start of a user turn. Publishes into NeuralState:
  GF interrupt (multi-source), LH familiarity, circadian, MB valence,
  DN vigor, optional SOUL bootstrap + online teach.

Modes stay off/shadow until explicitly enabled in config.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

log = logging.getLogger("aiko.fly_behavior.turn")

# Background fullbrain stepping (latency #6): the 470ms connectome
# propagation runs off the turn's critical path. Only one step runs at
# a time (overlap guard); the turn consumes the previous step's outputs
# (one turn of staleness on slow modulators).
_fullbrain_bg_lock = threading.Lock()
_fullbrain_latest: dict = {}
_fullbrain_latest_lock = threading.Lock()


def _fullbrain_step_bg(fb, kc, kc_body_ids, user_id) -> None:
    """Run one fullbrain step in the background, then publish its outputs
    for the next turn to consume. Releases the overlap guard on exit."""
    try:
        import time as _time
        _t0 = _time.perf_counter()
        step = fb.step(kc, kc_body_ids)
        _ms = (_time.perf_counter() - _t0) * 1000.0
        dn = float(step.get("dn_drive", 0.0))
        # DN drive is O(5e-5) (measured 4.6e-5..5.4e-5); map gently
        # around the 1.0 baseline - a real neural signal, never a swing.
        latest = {
            "motor_vigor": round(max(0.8, min(1.2, 1.0 + 3000.0 * (dn - 50e-6))), 3),
            "clock_drive": round(float(step.get("clock_drive", 0.0)), 5),
            "brain_arousal": round(float(step.get("arousal", 0.0)), 5),
            "whole_brain_ms": round(_ms, 1),
        }
        with _fullbrain_latest_lock:
            _fullbrain_latest[user_id] = latest
        from cognition.flymemory.fullbrain import mark_stepped
        mark_stepped()
        try:
            from cognition.fly_behavior import metrics as _fly_metrics
            _fly_metrics.record_fullbrain(ran=True, ms=_ms, user_id=user_id)
        except Exception:
            pass
    except Exception as exc:
        log.debug("background whole-brain step failed: %s", str(exc))
    finally:
        _fullbrain_bg_lock.release()


def _circadian_phase_now() -> float:
    """0..1 fraction of local day (wall-clock until a real clock circuit exists)."""
    try:
        now = datetime.now().astimezone()
    except Exception:
        now = datetime.now(timezone.utc)
    secs = now.hour * 3600 + now.minute * 60 + now.second
    return secs / 86400.0


def apply_turn_priors(
    text: str,
    *,
    user_id: str | None = None,
    system_error: bool = False,
    priority: float = 0.0,
    prior_assistant: str | None = None,
    subliminal_urgency: float = 0.0,
    prosody: dict | None = None,
) -> dict:
    """Run Stage-1 fly priors; return a compact summary for the turn.

    Phase 6: prosody carries ASR voice features (see
    sensory/listen.py::_prosody_features); sensory pathways encode voice /
    motion / visual channels into neural activity before the GF line.

    Never raises. Safe to call on every turn.
    """
    out: dict = {
        "interrupt": False,
        "urgency": 0.0,
        "familiarity": 0.5,
        "novel": False,
        "sleep_pressure": None,
        "valence": 0.0,
        "motor_vigor": 1.0,
        "tone_bits": [],
        "gf_sources": {},
        "soul_teach": None,
        "online_teach": None,
        "senses": None,
    }
    try:
        from cognition.fly_behavior.body import discard_scored_drive
        discard_scored_drive(user_id)
    except Exception as exc:
        log.debug("body drive reset skipped: %s", exc)
    try:
        from cognition.neural_state import get_neural_state
        from cognition.fly_behavior.giant_fiber import assess_interrupt
        from cognition.fly_behavior.lateral_horn import context_prior

        st = get_neural_state(user_id)
        # Circadian is ticked after the MB / whole-brain block below so the
        # real clock-neuron drive can set the rhythm gain (Phase 4).
        _clock_wall_phase = _circadian_phase_now()

        try:
            from cognition.flymemory.soul_teach import ensure_soul_bootstrap
            out["soul_teach"] = ensure_soul_bootstrap(user_id)
        except Exception as exc:
            log.debug("soul_teach skipped: %s", exc)

        try:
            from cognition.fly_behavior.cx_topic import apply_topic_drive
            out["cx_topic"] = apply_topic_drive(text or "", user_id=user_id)
            from cognition.flymemory.online_teach import teach_from_user_text
            out["online_teach"] = teach_from_user_text(
                text or "", user_id=user_id, prior_assistant=prior_assistant
            )
        except Exception as exc:
            log.debug("online_teach skipped: %s", exc)

        try:
            from cognition.fly_behavior.stage4_hooks import after_online_teach
            after_online_teach(user_id, text or "", out)
        except Exception as exc:
            log.debug("stage4_hooks skipped: %s", exc)

        try:
            from system.config import env_str
            mb_mode = env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
        except Exception:
            mb_mode = "off"
        if mb_mode in ("shadow", "live"):
            try:
                from cognition.fly_registry import get_flymb
                from cognition.flymemory.circuit import text_features

                mb = get_flymb(user_id)
                if mb is not None:
                    kc = mb.encode(text_features(text or ""))
                    bias = float(mb.valence_bias(text_features(text or "")))
                    out["valence"] = bias
                    if mb_mode == "live":
                        st.publish_mb(bias, source="turn")
                    st.record_influence(
                        {"kind": "mb_valence", "mode": mb_mode, "bias": round(bias, 4)}
                    )
                    # Whole-brain step (Phase 4): seed slice KCs into the full
                    # connectome. Runs in the background every turn - the
                    # 470ms propagation stays off the turn's critical path.
                    # This turn consumes the PREVIOUS step's outputs (one
                    # turn of staleness on slow modulators); clock_drive
                    # below feeds the circadian rhythm gain as before.
                    try:
                        from cognition.fly_registry import get_fullbrain

                        fb = get_fullbrain()
                        if fb is not None:
                            with _fullbrain_latest_lock:
                                _latest = dict(_fullbrain_latest.get(user_id) or {})
                            if _latest:
                                out["motor_vigor"] = _latest["motor_vigor"]
                                out["clock_drive"] = _latest["clock_drive"]
                                out["brain_arousal"] = _latest["brain_arousal"]
                                out["whole_brain"] = True
                                out["whole_brain_ms"] = _latest["whole_brain_ms"]
                            # Start the next step in the background. The
                            # overlap guard (non-blocking acquire) is the
                            # throttle: if the previous step is still
                            # running, skip this turn - the next turn will
                            # try again. No EVERY_N gating: backgrounded
                            # steps cost no turn latency.
                            if _fullbrain_bg_lock.acquire(blocking=False):
                                try:
                                    _bg = threading.Thread(
                                        target=_fullbrain_step_bg,
                                        args=(fb, kc.copy(),
                                              mb.kc_body_ids.copy(),
                                              user_id),
                                        name="aiko-fullbrain-step",
                                        daemon=True,
                                    )
                                    _bg.start()
                                except Exception:
                                    _fullbrain_bg_lock.release()
                                    raise
                                out["whole_brain_bg"] = True
                            else:
                                out["whole_brain_throttled"] = True
                                try:
                                    from cognition.fly_behavior import metrics as _fly_metrics
                                    _fly_metrics.record_fullbrain(ran=False, user_id=user_id)
                                except Exception:
                                    pass
                    except Exception as exc:
                        log.debug("whole-brain step skipped: %s", str(exc))
            except Exception as exc:
                log.debug("mb valence skipped: %s", str(exc))

        # Real circadian clock circuit (Phase 4): wall clock is the zeitgeber;
        # clock-neuron drive from the whole-brain step sets the rhythm gain.
        try:
            from cognition.fly_behavior import circadian as _circ

            out["circadian"] = _circ.circadian_now(
                _clock_wall_phase,
                user_id=user_id,
                clock_drive=out.get("clock_drive"),
            )
        except Exception as exc:
            log.debug("circadian tick skipped: %s", str(exc))

        motion = 0.0
        motion_sudden = False
        voice_urgency = 0.0
        try:
            # Phase 6: encode voice / motion / visual channels into neural
            # activity before the GF line reads motion salience.
            from cognition.flysense.pathways import encode_turn_senses

            senses = encode_turn_senses(
                text or "", user_id=user_id, prosody=prosody
            )
            out["senses"] = senses
            motion = float(st.motion_salience or 0.0)
            motion_sudden = bool(
                (senses or {}).get("mode") == "live"
                and ((senses or {}).get("motion") or {}).get("sudden")
            )
            voice = ((senses or {}).get("voice") or {}) if (senses or {}).get("mode") == "live" else {}
            voice_urgency = float(voice.get("urgency") or 0.0)
        except Exception as exc:
            log.debug("sensory pathways skipped: %s", str(exc))
        gf = assess_interrupt(
            text or "",
            system_error=system_error,
            priority=priority,
            subliminal_urgency=float(subliminal_urgency or 0.0),
            motion_salience=motion,
            motion_sudden=motion_sudden,
            voice_urgency=voice_urgency,
        )
        st.publish_gf(float(gf.get("urgency") or 0.0), bool(gf.get("interrupt")), source="turn")
        out["urgency"] = gf.get("urgency", 0.0)
        out["interrupt"] = bool(gf.get("interrupt"))
        out["gf_sources"] = gf.get("sources") or {}
        if gf.get("mode") in ("shadow", "live"):
            log.debug(
                "flygf mode=%s urgency=%.2f interrupt=%s would=%s sources=%s",
                gf.get("mode"), gf.get("urgency"), gf.get("interrupt"),
                gf.get("would_interrupt"), gf.get("sources"),
            )
            st.record_influence(
                {
                    "kind": "gf",
                    "mode": gf.get("mode"),
                    "urgency": gf.get("urgency"),
                    "interrupt": gf.get("interrupt"),
                    "would_interrupt": gf.get("would_interrupt"),
                    "sources": gf.get("sources"),
                }
            )
            if gf.get("interrupt"):
                try:
                    from cognition.flymemory.online_teach import teach_interrupt_honored
                    online_teach = out.get("online_teach") or {}
                    out["interrupt_teach"] = teach_interrupt_honored(
                        user_id,
                        text or "stop abort cancel",
                        eligibility_recorded=bool(online_teach.get("eligibility_recorded")),
                    )
                except Exception:
                    pass

        lh = context_prior(text or "", user_id=user_id)
        if "familiarity" in lh:
            st.publish_lh(float(lh["familiarity"]), source="turn")
            out["familiarity"] = lh.get("familiarity", 0.5)
            out["novel"] = bool(lh.get("novel"))
            if lh.get("mode") in ("shadow", "live"):
                log.debug(
                    "flylh mode=%s familiarity=%.2f novel=%s bias=%s",
                    lh.get("mode"), lh.get("familiarity"), lh.get("novel"), lh.get("bias"),
                )
                st.record_influence(
                    {
                        "kind": "lh",
                        "mode": lh.get("mode"),
                        "familiarity": lh.get("familiarity"),
                        "novel": lh.get("novel"),
                        "bias": lh.get("bias"),
                    }
                )

        try:
            from system.config import env_str
            dn_mode = env_str("MEMORY_FLYDN_MODE", "off").strip().lower()
        except Exception:
            dn_mode = "off"
        if dn_mode in ("shadow", "live"):
            try:
                from cognition.flysense.dn import get_flydn

                energy = max(0.0, min(1.0, 1.0 - float(st.sleep_pressure or 0.0)))
                # Phase 6: voice speech-rate energy blends with rest energy —
                # a tired body can still hear an urgent voice.
                _senses = out.get("senses") or {}
                _voice = (_senses.get("voice") or {}) if _senses.get("mode") == "live" else {}
                if _voice.get("energy") is not None:
                    try:
                        energy = max(
                            0.0,
                            min(1.0, 0.6 * energy + 0.4 * float(_voice["energy"])),
                        )
                    except Exception:
                        pass
                drv = get_flydn().drive(
                    energy=energy,
                    decisiveness=float(st.decisiveness or 0.5),
                    affect=float(st.valence or 0.0),
                )
                out["motor_vigor"] = drv.get("rate_mult", 1.0)
                out["dn_drive"] = {
                    k: drv.get(k)
                    for k in (
                        "arousal", "rate_mult", "vol_mult",
                        "expression_intensity", "gesture_intensity",
                        "gaze_speed", "action_vigor",
                    )
                }
                if dn_mode == "live":
                    st.publish_dn(
                        arousal=float(drv.get("arousal") or 0.5),
                        rate_mult=float(drv.get("rate_mult", 1.0)),
                        source="turn",
                    )
                st.record_influence(
                    {
                        "kind": "dn",
                        "mode": dn_mode,
                        "arousal": drv.get("arousal"),
                        "rate_mult": drv.get("rate_mult"),
                        "vol_mult": drv.get("vol_mult"),
                        "expression": drv.get("expression_intensity"),
                        "gesture": drv.get("gesture_intensity"),
                        "gaze": drv.get("gaze_speed"),
                        "action_vigor": drv.get("action_vigor"),
                    }
                )
            except Exception as exc:
                log.debug("dn drive skipped: %s", str(exc))

        out["sleep_pressure"] = st.sleep_pressure

        # Phase 7: persistent CX temporal dynamics — heading vector, competing
        # drives, decaying urgency trace. Always stepped (shadow computes and
        # records); in live mode the trace replaces the per-turn urgency
        # spike on the bus. Never raises.
        cx_applied = False
        try:
            from cognition.centralcomplex.temporal import cx_mode, tick_cx_temporal

            cxm = cx_mode()
            if cxm in ("shadow", "live"):
                _senses = out.get("senses") or {}
                _s_voice = (_senses.get("voice") or {}) if _senses.get("mode") == "live" else {}
                try:
                    _venergy = max(0.0, min(1.0, float(_s_voice.get("energy") or 0.0)))
                except Exception:
                    _venergy = 0.0
                temporal = tick_cx_temporal(
                    text or "",
                    user_id=user_id,
                    fresh_urgency=float(out.get("urgency") or 0.0),
                    mb_valence=float(out.get("valence") or 0.0),
                    sleep_pressure=float(st.sleep_pressure or 0.0),
                    novelty=1.0 - float(out.get("familiarity") or 0.5),
                    user_energy=_cx_user_energy(text or "", _venergy),
                    voice_energy=_venergy,
                )
                out["cx_temporal"] = temporal
                cx_applied = bool(temporal.get("ok"))
                if cxm == "live" and temporal.get("ok"):
                    trace = float(temporal.get("urgency") or 0.0)
                    st.publish_gf(
                        trace, bool(out.get("interrupt")), source="turn+cx"
                    )
                    out["urgency"] = round(trace, 4)
        except Exception as exc:
            log.debug("cx temporal skipped: %s", exc)

        # Phase 11: persona explainability trace — trait → circuit gains →
        # effects, recorded into the NeuralState influence ring for the
        # future Studio panel. Computed only; shadow never modulates.
        try:
            from cognition.fly_persona import persona_mode, record_persona_trace

            if persona_mode() != "off":
                out["persona"] = record_persona_trace(
                    user_id, cx_applied=cx_applied
                )
        except Exception as exc:
            log.debug("persona trace skipped: %s", exc)

        bits: list[str] = []
        if st.interrupt or st.urgency >= 0.65:
            bits.append("user signaled urgency — answer briefly, acknowledge stop/wait")
        if st.avoidance > 0.3:
            bits.append("slightly cautious tone")
        if st.approach > 0.3 and st.context_familiarity > 0.6:
            bits.append("warm, familiar tone")
        if st.sleep_pressure > 0.6:
            bits.append("keep the reply shorter; low energy")
        phase = st.circadian_phase
        if phase < 0.25 or phase > 0.9:
            bits.append("late/early hours — softer volume of commitment")
        out["tone_bits"] = bits
    except Exception as exc:
        log.debug("fly turn priors skipped: %s", str(exc))
    return out


def _cx_user_energy(text: str, voice_energy: float) -> float:
    """Engagement-ish input for the Phase-7 engagement drive.

    Voice energy when the mic was hot, else a gentle text-length
    heuristic. Deterministic given inputs.
    """
    try:
        if voice_energy > 0.0:
            return max(0.0, min(1.0, float(voice_energy)))
    except Exception:
        pass
    try:
        return max(0.0, min(1.0, len(text or "") / 400.0))
    except Exception:
        return 0.0


def _maintenance_level_from_pressure(sleep_pressure: float) -> str:
    if sleep_pressure >= 0.75:
        return "maintenance"
    if sleep_pressure >= 0.45:
        return "reduced"
    return "normal"


def maintenance_level(user_id: str | None = None) -> str:
    """Map sleep_pressure → normal | reduced | maintenance.

    Used by agent cadence / background work. Does not itself run dreams.
    """
    try:
        from cognition.neural_state import get_neural_state
        from system.config import env_str
        mode = env_str("MEMORY_FLYSLEEP_MODE", "off").strip().lower()
        if mode not in ("shadow", "live"):
            return "normal"
        sp = float(get_neural_state(user_id).sleep_pressure or 0.0)
        level = _maintenance_level_from_pressure(sp)
        if mode == "shadow":
            log.debug("flysleep mode=shadow sleep=%.2f would_level=%s", sp, level)
            return "normal"
        return level
    except Exception:
        return "normal"
