# Aiko Presence — conscious stream, daydreaming subconscious, motion director

How Aiko feels like one continuous person instead of a new chatbot every
turn, and how her body answers faster than her voice.

## The mind model

```
┌─────────────────────────────────────────────────────────┐
│ CONSCIOUS  cognition/inner_voice.py  (InnerVoice)        │
│  Rolling first-person train of thought. "I noticed…",   │
│  "I feel…", "I intend…". Injected into every prompt as  │
│  <inner_voice> so replies continue the same mental      │
│  thread instead of restarting the persona each turn.    │
├─────────────────────────────────────────────────────────┤
│ SUBCONSCIOUS  cognition/subliminal.py  (SubliminalLayer) │
│  L1 pre-attentive cue scan (<0.2 ms, hot path)          │
│  L2 PAD affect + drive mix (valence/arousal/dominance)  │
│  L3 intuitions + affective tags (prompt assembly)       │
│  L4 emotion label, impulse, bias, VRM expression        │
│  DAYDREAM idle consolidation: folds recent affect into  │
│  slow lingering dispositions (warmth, heaviness,        │
│  restlessness, quiet) that tint tomorrow's mood — the   │
│  way a good or bad day colors a human morning. Also    │
│  surfaces occasional spontaneous thoughts (mind-        │
│  wandering), queued as unprompted asides.               │
└─────────────────────────────────────────────────────────┘
```

Both layers are **LLM-free, allocation-bounded, and lock-safe**:
no threads, no DB, no model calls on the hot path. Total state is a few
KiB per identity. Everything persists through the existing
`cognitive_state` SQLite row (`inner_voice` + `subliminal` keys), so she
wakes up mid-thought after a restart.

Wiring (`cognition/attention.py`, `EdgeCognitiveState`):

* `record(user, assistant)` — after the subliminal scan: updates the
  inner voice from (emotion, impulse, cues, recurring focus, lingering
  moods), then asks the motion director for one reactive gesture.
* `continuous_tick()` — idle hook: runs `daydream()`, queues any
  spontaneous thought as a future aside.
* `metacognitive_context()` — appends `<inner_voice>` to the prompt
  block think.py already injects per turn. No think.py changes needed.

## The body answers first (perceived latency)

Humans signal attention with the body in ~200 ms, long before words.
The motion director does the same:

* `interface/webui/motion_director.py` — pure function of
  (user text, emotion, intensity, impulse) → gesture name or None.
  Per-gesture (25 s) and global (8 s) cooldowns prevent flailing.
* `webui_bridge.play_gesture(name)` → `{"type": "gesture", "name"}`
  → `window.aikoPlayGesture(name)` in vrm.js.
* `get_input()` plays `leanIn` the moment she starts waiting — the
  universal "I'm listening."
* Reactive map: praise → `clap`, thanks → `bow`, apology →
  `handsClasp`, greeting/farewell → `wave`, celebration → `dance`,
  playful → `giggle`, curiosity → `curiousTilt`, gentle impulse →
  `leanIn`.

New VRM gestures (`interface/webui/static/vrm.js`): `wave`, `giggle`,
`bow`, `clap` (also in the idle pool), `dance` (reactive-only, 4.2 s
happy bounce), plus young-girl idle fidgets — `hairTwirl`, `handsBehindBack`,
`footTap`, `skirtSmooth`, `hugSelf`, `happyBounce` — picked from the idle
pool every few seconds. All follow the existing bone/blend conventions and are
validated by `KNOWN_GESTURES` on both ends.

Camera defaults frame her full body (FOV 10, pulled back, target at
mid-torso); orbit/zoom still available.

## Tuning

* `InnerVoice` bounds: 6 thoughts × 160 chars; aside cooldown 30 min.
* Daydream: at most one consolidation per 10 min; dispositions decay
  ×0.82 per run and drop below 0.05.
* Disable reactive motion: remove/None the `MotionDirector` import —
  `attention.py` degrades gracefully (guarded imports everywhere).
