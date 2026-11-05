# Visual conversation

Images and single camera/screen frames are described by the configured vision
model. The bounded description is then supplied as untrusted visual evidence to
Aiko's normal conversation engine, which produces the reply in her own persona.
Descriptions are held in memory for up to `WEBUI_VISION_CONTEXT_TTL` seconds
(default 120); image bytes and landmarks are not stored by this feature.

Use the attachment button for a JPEG/PNG/WebP image, or ask “look at me” for a
snapshot. Say “watch me” to start a visible camera session; “stop watching me”,
the Stop button, disconnecting, or hiding the page stops it. Aiko can request a
stop through her camera tool. Browsers still control camera permission. While
watching, a scene frame is sent approximately every 15 seconds after the prior
inference finishes, with one inference in flight globally. The latest observations
and current cues become context for subsequent conversation; this is not a
continuous video model or a promise of immediate recognition.

MediaPipe gesture and pose models run in the browser, on CPU, rather than being
loaded on the Jetson server. The optional browser face model enrolls one face
only after “Remember my face” is pressed. Its template stays in that browser's
local storage and can be deleted with “Forget face”; it is a fallible observation,
not authentication or approval. A model download is needed on first use and
requires network access. Gesture labels and posture are approximate; the VLM may
miss or misinterpret actions.

Real camera, browser permissions, model quality, and Jetson RAM/latency have not
been measured in the cloud environment. The configured VLM endpoint must be
running and able to process images. Frames are sent to that endpoint, which may
be remote if configured that way.

## Ongoing conversation

While Watch is active, repeated gesture/posture cues and sustained scene changes
can wake Aiko's existing proactive conversation system. These are opportunities:
her normal LLM may respond briefly or select silence. Idle check-ins remain
available independently of visual changes. `/proactive` controls initiation.

Perception opportunities are coalesced per user, expire after 20 seconds, require
a quiet gap (default 8 seconds), and are limited to one attempt per 90 seconds
and six attempts per hour. Quiet windows, resting/wake state, active turns,
queued chat, and browser speech detection take priority. The four
`PERCEPTION_*` controls are available in Settings.

Generated proactive speech is held until delivery checks finish. If user
activity arrives during generation, the stale initiation is discarded before
TTS. Only one LLM operation owns the shared inference gate at a time, while
camera capture, cues and ASR can continue waiting for input. This is an ongoing
event loop; it does not require a new user message for every initiation.

Full-duplex voice still needs confirmed acoustic echo cancellation. Raw
speaker/microphone setups retain the existing echo-suppression behavior. Real
camera, conversational timing and ASR/TTS overlap need testing on the Jetson;
JetPack 7.2 is the stated deployment target, without a ROS2 migration.

Cloud validation: the default suite passed with 2,786 tests, six skips and
34 subtests. After tightening stop-during-generation cancellation and recurring
gesture handling, the focused perception/conversation/settings regression suite
passed all 73 tests. Hardware validation remains pending as described above.
