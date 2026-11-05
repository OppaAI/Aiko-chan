# Companion access

Mobile game and Lingo endpoints share the WebUI session and terms check. A valid
login keeps the caller's own identity. Invalid credentials and rejected terms
are rejected rather than converted into the server owner's identity.

Legacy Android clients without a cookie jar can use the explicit
`AIKO_COMPANION_OWNER_FALLBACK` setting in Settings or `config/android_app.yaml`.
It defaults to `0`. If you enable it, every client that can reach these endpoints
without a session acts as `AIKO_USER_ID`; enable it only on the trusted private
network, such as your Tailscale Serve setup. Apply the settings change and restart
Aiko as directed by Settings. Clients that send an invalid session cookie must
clear it or log in again.

Desktop builds connect to `http://localhost:8787/` by default. For Tailscale Serve,
set `AIKO_WEBUI_URL=https://your-device.your-tailnet.ts.net/` in the desktop process
environment before launching. That server supplies both the page and WebSocket.
