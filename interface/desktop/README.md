# Aiko Companion desktop shell

This Tauri shell turns the existing WebUI into a transparent, resizable,
always-on-top desktop companion. It includes a tray menu and global
`Ctrl+Shift+A` show/hide shortcut.

## Development

1. Start Aiko's server from the repository root: `python main.py`.
2. Install the [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/).
3. From `interface/desktop`, run `cargo tauri dev`.

The WebUI is intentionally loaded from `http://localhost:8787` in development.
Set `AIKO_WEBUI_URL` to select another running backend.

## Tray icon

`icons/` ships generated PNGs (`icon.png`, `32x32.png`, `tray.png`) because
Tauri requires `icons/icon.png` at compile time (`generate_context!` fails
without it). `icons/aiko-tray.svg` is the text source — to regenerate after
editing it, run `cargo tauri icon icons/aiko-tray.svg` from
`interface/desktop`. The tray PNG is embedded at compile time.

## Security notes

- `tauri.conf.json` sets `app.security.csp` to `null` deliberately: the
  wrapped WebUI uses inline scripts, a CDN import map
  (`cdn.jsdelivr.net`), Google Fonts, and `cdn.simpleicons.org` images, so a
  strict CSP would break the existing page. Before distributing a packaged
  build, set an explicit CSP covering those sources plus the WebSocket
  (`ws:`/`wss:`) and `data:`/`blob:` image URLs used for one-shot vision
  frames, and prefer bundling a local Aiko service over a remote URL.
- `capabilities/default.json` grants only `core:default` plus the
  `global-shortcut` register/unregister/is-registered permissions needed for
  the `Ctrl+Shift+A` show/hide hotkey. If hotkey registration fails (e.g. a
  conflict), the app logs a warning and keeps running — the tray menu's
  "Show / hide Aiko" item remains the fallback.

Both development and packaged builds open `http://localhost:8787/` by default.
Set `AIKO_WEBUI_URL=https://your-device.your-tailnet.ts.net/` in the desktop
process environment to use Tailscale Serve or another backend address. The
backend serves the page and WebSocket from the same origin. The tray PNG is
embedded in the executable, so launching from another working directory works.
