# Fly-brain wiring migration (fix/fly-brain-harden-isolation)

## Goal
Harden the MaleCNS grafts already merged on 2026-09-18 before adding more circuits.

## Code changes required in call sites

### 1. cognition/memory/grasp.py
Replace module-level `_FLYMB` / `_FLY_STORE` singletons with registry:

```python
from cognition.fly_registry import get_flymb, get_fly_store, flush_all

def flymb_bias_for_turn(turn, current_turn, *, learn=False, user_id=None):
    if MEMORY_FLYMB_MODE not in ("shadow", "live"):
        return None
    mb = get_flymb(user_id)
    ...
    if learn and MEMORY_FLYMB_MODE == "live":
        ...
        store = get_fly_store(user_id)
        if store is not None:
            store.save_if_due_mb(mb)

# In GraspBuffer.fill / teaching path: pass the active user_id
# In process shutdown or identity switch: flush_all(user_id)
```

Thread `user_id` from the GraspBuffer / session context (already available as
the active OAuth identity in multi-user mode).

### 2. cognition/attention.py
Replace `_FLYCX` / `_FLY_STORE` with:

```python
from cognition.fly_registry import get_flycx, get_fly_store, flush_all

def flycx_state_for_record(state, user: str):
    if MEMORY_FLYCX_MODE not in ("shadow", "live"):
        return None
    cx = get_flycx(user)
    ...
    store = get_fly_store(user)
    if store is not None:
        store.save_if_due_cx(out["sleep_pressure"])
```

`flycx_decisiveness_for_text` should also take `user` and use `get_flycx(user)`.

### 3. agentic/needle_orchestrator.py
Delete the local `_flycx_cadence._cx = FlyCompass()` singleton.

```python
from cognition.fly_registry import get_flycx

def _flycx_cadence(task: str, user_id: str | None = None) -> str:
    mode = _flycx_mode()
    if mode not in ("shadow", "live"):
        return "parallel"
    cx = get_flycx(user_id)
    if cx is None:
        return "parallel"
    ...
```

Pass the active user identity into `complete()` if available; otherwise
`"default"` (single-user).

### 4. Shutdown hook (main.py or system/wakeup teardown)
```python
from cognition.fly_registry import flush_everything
# on clean exit / SIGTERM handler:
flush_everything()
```

### 5. config/memory.yaml
Set `MEMORY_FLYMB_MODE: "shadow"` (was `"live"`). Keep CX shadow, AL/DN off.

## Why not add giant-fiber / lateral horn / malevnc in this PR
No extracted circuit data, no evaluation harness, and isolation bugs would
propagate. Track as follow-ups after this hardens.
