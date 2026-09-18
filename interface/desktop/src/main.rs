//! Native shell for the existing Aiko WebUI. Run `python main.py` first, then
//! `cargo tauri dev --config interface/desktop/tauri.conf.json` from this folder.

use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn toggle_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("companion") {
        if window.is_visible().unwrap_or(false) {
            let _ = window.hide();
        } else {
            let _ = window.show();
            let _ = window.set_focus();
        }
    }
}

fn tray_icon() -> Option<tauri::image::Image<'static>> {
    // Optional file-based icon so the repo ships no binary assets. Generate
    // `icons/tray.png` (see README) to enable it; otherwise the tray falls
    // back to a text menu and logs a warning instead of failing to start.
    for path in ["icons/tray.png", "icons/32x32.png"] {
        if let Ok(icon) = tauri::image::Image::from_path(path) {
            return Some(icon);
        }
    }
    eprintln!("aiko-companion: no tray icon found (tried icons/tray.png); see interface/desktop/README.md to generate one");
    None
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            let toggle = MenuItem::with_id(app, "toggle", "Show / hide Aiko", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit Aiko Companion", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&toggle, &quit])?;
            let handle = app.handle().clone();
            let tray = TrayIconBuilder::with_id("aiko-tray")
                .tooltip("Aiko Companion — Ctrl+Shift+A")
                .menu(&menu)
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "toggle" => toggle_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                });
            let tray = match tray_icon() {
                Some(icon) => tray.icon(icon),
                None => tray,
            };
            tray.build(&handle)?;

            let hotkey = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyA);
            // The shortcut is a convenience; a conflict must not abort startup
            // when the tray menu (Show / hide Aiko) still works.
            if let Err(err) = app.global_shortcut().on_shortcut(hotkey, |app, _, event| {
                if event.state() == ShortcutState::Pressed {
                    toggle_window(app);
                }
            }) {
                eprintln!("aiko-companion: global shortcut Ctrl+Shift+A unavailable: {err}");
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run Aiko Companion");
}
