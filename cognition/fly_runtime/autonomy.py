"""Permissioned autonomous action queue: low-risk virtual actions only by default."""
from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class ActionScheduler:
    cooldown_s: float = 5.0
    allowed: set[str] = field(default_factory=lambda: {"expression", "pose", "suggest"})
    _last_at: dict[str, float] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)

    def admit(self, action: dict, *, approved: bool = False) -> bool:
        kind = str(action.get("kind", ""))
        now = time.monotonic()
        allowed = kind in self.allowed or approved
        cooled = now - self._last_at.get(kind, float("-inf")) >= self.cooldown_s
        accepted = bool(allowed and cooled)
        self.audit.append({"kind": kind, "accepted": accepted, "reason": "allowed" if accepted else ("approval_required" if not allowed else "cooldown")})
        if accepted:
            self._last_at[kind] = now
        return accepted
