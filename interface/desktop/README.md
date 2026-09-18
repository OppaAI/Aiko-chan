# Aiko Companion desktop shell

This Tauri shell turns the existing WebUI into a transparent, resizable,
always-on-top desktop companion. It includes a tray menu and global
`Ctrl+Shift+A` show/hide shortcut.

## Development

1. Start Aiko's server from the repository root: `python main.py`.
2. Install the [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/).
3. From `interface/desktop`, run `cargo tauri dev`.

The WebUI is intentionally loaded from `http://localhost:8787` in development.
Production packaging can set `AIKO_WEBUI_URL` or bundle a local Aiko service.
