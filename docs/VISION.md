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
watching, the browser checks attention cues approximately every two seconds.
There is no periodic VLM refresh and opening Watch does not submit a frame.
Sustained camera-directed gaze or a raised open palm requests one frame, with
one VLM inference in flight globally. Explicit snapshots and attachments still
request immediate analysis. The latest observations and current cues become
conversation context; this is not a continuous video model.

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

While Watch is active, browser MediaPipe GestureRecognizer and PoseLandmarker
check for an open palm with its wrist above shoulder level. FaceLandmarker
checks iris position, open eyes and approximate frontal face direction. This is
a heuristic estimate of looking toward the camera, not calibrated eye tracking;
looking near the camera can also trigger it. Multiple faces suppress the gaze
trigger. Ordinary fists, thumbs up, posture changes and face matches are
context only, and do not independently request VLM analysis or conversation.

Hold gaze or the raised palm for about two seconds (longer if browser detection
is slow). A trigger requests one VLM frame; only successful fresh observations
then wake Aiko's normal proactive conversation system. Her LLM may respond
briefly or choose silence. Holding a cue does not repeat requests: release it
for at least two checks, then repeat it after the default 90-second cooldown.
The server also enforces that cooldown across camera restarts. Idle check-ins
remain independent of camera attention; `/proactive` controls initiation.

Perception opportunities are coalesced per user, expire after 20 seconds, require
a quiet gap (default 8 seconds), and are limited to one attempt per 90 seconds
and six attempts per hour. Quiet windows, resting/wake state, active turns,
queued chat, and browser speech detection take priority. The four
`PERCEPTION_*` controls are available in Settings. Camera gaze/palm triggers,
automatic VLM cooldown and hold time are also in Settings (`CAMERA_*`). These
settings are marked restart-required; restart Aiko and reopen Watch after
saving. The checkboxes in the camera panel apply to the current browser session.

To enroll: log in, open Watch, keep exactly one clearly visible face in view,
and click **Enroll my face**. Five samples are averaged and stored only in this
browser. Successful enrollment enables **Require my enrolled face for triggers**.
Without that option, attention cues do not require an identity match. With it,
a missing or failed face match suppresses both triggers. **Forget face** removes
the saved template and turns matching off. Gaze detection does not require
enrollment. First use downloads browser models from their configured public CDNs.

Generated proactive speech is held until delivery checks finish. If user
activity arrives during generation, the stale initiation is discarded before
TTS. Only one LLM operation owns the shared inference gate at a time, while
camera capture, cues and ASR can continue waiting for input. This is an ongoing
event loop; it does not require a new user message for every initiation.

Full-duplex voice still needs confirmed acoustic echo cancellation. Raw
speaker/microphone setups retain the existing echo-suppression behavior. Real
camera, conversational timing and ASR/TTS overlap need testing on the Jetson;
JetPack 7.2 is the stated deployment target, without a ROS2 migration.

Cloud validation covers synthetic gaze/hand geometry, attention hold/release and
cooldown, the browser camera controller, VLM-to-conversation handoff and Settings.
Actual gaze accuracy, browser model downloads and device performance still need
validation on the deployment. The cloud proxy blocked the new gaze model URL
check; its standard MediaPipe download path has not been fetched here.

The full default regression run passed with 2,793 tests, six skips and
34 subtests. After removing the obsolete generic cue/scene initiation code,
all 105 focused perception, browser, conversation and Settings checks passed.
