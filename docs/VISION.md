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
