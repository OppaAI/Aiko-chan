"""Autonomy policy: how often Aiko may act without asking, per action class.

This is the only component permitted to learn from Oppa's approvals, and the
boundary is deliberate and one-way:

    the learner may change HOW OFTEN she asks, never WHAT the conscience forbids

If approval patterns could edit the moral layer, the charter would stop being
Oppa's and become whatever he happened to tolerate -- a tired "yes" at 11pm
becomes doctrine. So nothing in this module can produce a verdict, override a
verdict, or loosen a `REFUSE`. It can only return `act`, `act_and_notify`, or
`ask`, and the conscience verdict already narrowed the space before we get here.

Design constraints, in priority order:
  1. High-stakes classes never graduate. A standing permission for
     "send_email" must not imply one for "send_irreversible_damage".
  2. Only *explicit* approvals count. Silence, ignored notices, and timeouts
     are not consent (ledger `HITL_TIMEOUT` is default-closed for the same
     reason).
  3. Approvals decay. Trust earned months ago should not be standing policy.
  4. Inspectable and reversible by hand: plain counters, a JSON file, and
     `forget()`. No gradient, no hidden state, no model.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Stakes tiers. Anything not listed is treated as HIGH: a new action class must
# opt in to autonomy, never inherit it by omission.
STAKES_LOW = "low"
STAKES_MEDIUM = "medium"
STAKES_HIGH = "high"
_STAKES_ORDER = {STAKES_LOW: 0, STAKES_MEDIUM: 1, STAKES_HIGH: 2}

# Verdicts this module may return. Mirrors `schema` action vocabulary.
ACT = "act"
ACT_NOTIFY = "act_and_notify"
ASK = "ask"


@dataclass
class TrustRecord:
    """Consecutive explicit approvals for one (user, action class) pair."""

    action_class: str
    stakes: str = STAKES_MEDIUM
    approvals: int = 0
    refusals: int = 0
    granted_at: float = 0.0          # monotonic-ish wall clock, see _now()
    revoked: bool = False
    history: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "TrustRecord":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class AutonomyPolicy:
    """Per-user learned trust, capped by stakes, decayed over time."""

    def __init__(
        self,
        user_id: str,
        *,
        threshold: int = 5,
        decay_seconds: float = 14 * 24 * 3600.0,
        notify_threshold: int = 2,
        max_history: int = 50,
        path: Path | None = None,
        clock=time.time,
    ) -> None:
        self._user_id = user_id
        self._threshold = max(1, int(threshold))
        self._decay = float(decay_seconds)
        self._notify_threshold = max(1, int(notify_threshold))
        self._max_history = max(1, int(max_history))
        self._clock = clock
        self._records: dict[str, TrustRecord] = {}
        self._lock = threading.Lock()
        self._path = path
        self.audit: list[dict] = []
        if path is not None and path.exists():
            self._load(path)

    # ── policy ──────────────────────────────────────────────────────────────
    def stakes_for(self, action_class: str) -> str:
        """Stakes tier for an action class. Unseen classes default to HIGH."""
        rec = self._records.get(action_class)
        return rec.stakes if rec else STAKES_HIGH

    def permission(self, action_class: str, *, stakes: str | None = None) -> dict:
        """Current verdict for acting on `action_class` without asking.

        Returns a dict rather than a bare string so callers can log why.
        """
        with self._lock:
            rec = self._records.get(action_class)
            tier = (stakes or (rec.stakes if rec else STAKES_HIGH)).lower()
            if tier not in _STAKES_ORDER:
                tier = STAKES_HIGH
            if tier == STAKES_HIGH:
                return self._verdict("ask", action_class, "high-stakes class never graduates")
            if tier == STAKES_LOW:
                # Low stakes is settled by the stakes tier alone. Trust history
                # is for the middle band, where silence would otherwise mean
                # "act" on something that touches other people. Demanding
                # approval history here is what turns a conscience into
                # an interrogator about the weather.
                return self._verdict("act", action_class, "low stakes, no moral signal")
            if rec is None:
                return self._verdict("ask", action_class, "no approval history")
            if rec.revoked:
                return self._verdict("ask", action_class, "permission revoked")
            self._apply_decay(rec)
            if rec.approvals >= self._threshold:
                return self._verdict(
                    ACT_NOTIFY if tier == STAKES_MEDIUM else ACT,
                    action_class,
                    f"standing permission after {rec.approvals} approvals ({tier})",
                )
            if rec.approvals >= self._notify_threshold:
                return self._verdict(
                    ACT_NOTIFY, action_class, f"partial trust ({rec.approvals}/{self._threshold})"
                )
            return self._verdict("ask", action_class, f"only {rec.approvals} approvals so far")

    # ── learning ────────────────────────────────────────────────────────────
    def record_approval(self, action_class: str, *, stakes: str | None = None) -> TrustRecord:
        """An explicit approval. Only this method increases trust."""
        with self._lock:
            rec = self._ensure(action_class, stakes)
            self._apply_decay(rec)
            rec.approvals += 1
            rec.refusals = 0
            rec.granted_at = self._clock()
            rec.revoked = False
            rec.history.append({"t": rec.granted_at, "event": "approved"})
            self._trim(rec)
            self.audit.append({"action_class": action_class, "event": "approved",
                               "approvals": rec.approvals})
            self._save()
            return rec

    def record_refusal(self, action_class: str, *, stakes: str | None = None) -> TrustRecord:
        """An explicit refusal. Resets the streak -- trust is consecutive."""
        with self._lock:
            rec = self._ensure(action_class, stakes)
            rec.approvals = 0
            rec.refusals += 1
            rec.revoked = True
            rec.history.append({"t": self._clock(), "event": "refused"})
            self._trim(rec)
            self.audit.append({"action_class": action_class, "event": "refused",
                               "refusals": rec.refusals})
            self._save()
            return rec

    def revoke(self, action_class: str) -> None:
        with self._lock:
            rec = self._records.get(action_class)
            if rec is not None:
                rec.approvals = 0
                rec.revoked = True
                rec.history.append({"t": self._clock(), "event": "revoked"})
                self._trim(rec)
                self._save()

    def forget(self, action_class: str) -> None:
        """Erase a class entirely (GDPR-style forget, and a panic button)."""
        with self._lock:
            self._records.pop(action_class, None)
            self.audit.append({"action_class": action_class, "event": "forgotten"})
            self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "user_id": self._user_id,
                "threshold": self._threshold,
                "classes": {
                    k: {
                        "stakes": v.stakes,
                        "approvals": v.approvals,
                        "refusals": v.refusals,
                        "revoked": v.revoked,
                    }
                    for k, v in sorted(self._records.items())
                },
            }

    # ── internals ───────────────────────────────────────────────────────────
    def _ensure(self, action_class: str, stakes: str | None) -> TrustRecord:
        rec = self._records.get(action_class)
        if rec is None:
            tier = (stakes or STAKES_HIGH).lower()
            rec = TrustRecord(
                action_class=action_class,
                stakes=tier if tier in _STAKES_ORDER else STAKES_HIGH,
            )
            self._records[action_class] = rec
        elif stakes:
            tier = stakes.lower()
            if tier in _STAKES_ORDER:
                rec.stakes = tier
        return rec

    def _apply_decay(self, rec: TrustRecord) -> None:
        if not rec.granted_at:
            return
        if self._decay <= 0:
            return
        if self._clock() - rec.granted_at > self._decay:
            if rec.approvals:
                log.debug("[ccc] autonomy decay: %s lost standing permission", rec.action_class)
            rec.approvals = 0

    def _trim(self, rec: TrustRecord) -> None:
        if len(rec.history) > self._max_history:
            rec.history = rec.history[-self._max_history:]

    def _verdict(self, action: str, action_class: str, why: str) -> dict:
        entry = {"action_class": action_class, "action": action, "why": why}
        self.audit.append(entry)
        return entry

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({k: v.to_json() for k, v in self._records.items()}, indent=1),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError as e:
            log.warning("[ccc] autonomy persist failed: %s", e)

    def _load(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        self._records[k] = TrustRecord.from_json({**v, "action_class": k})
        except Exception as e:  # noqa: BLE001 - corrupt file must not block startup
            log.warning("[ccc] autonomy load failed (%s); starting empty", type(e).__name__)


def policy_for(user_id: str, *, path: Path | None = None, **kwargs) -> AutonomyPolicy:
    """Per-user constructor. Pass `path` to make trust durable across restarts."""
    if path is None:
        from system.userspace import user_state_dir

        path = Path(user_state_dir(user_id)) / "conscience" / "autonomy.json"
    return AutonomyPolicy(user_id, path=path, **kwargs)


__all__ = [
    "ACT",
    "ACT_NOTIFY",
    "ASK",
    "AutonomyPolicy",
    "STAKES_HIGH",
    "STAKES_LOW",
    "STAKES_MEDIUM",
    "TrustRecord",
    "policy_for",
]