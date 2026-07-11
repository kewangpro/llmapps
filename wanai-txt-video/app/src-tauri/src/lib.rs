// Spawns the Python FastAPI inference service (backend/service/) as a
// sidecar process on launch and terminates it on quit. See
// docs/DESIGN.md "Why a separate FastAPI service instead of calling
// ComfyUI directly from Tauri" — the Rust shell only manages the process
// lifecycle; all inference logic and HTTP calls live in the frontend
// talking directly to the local FastAPI service over localhost.
//
// This spawns the venv's python directly rather than a bundled Tauri
// "sidecar" binary — sufficient for local development (Phase 2). Bundling
// a self-contained Python runtime for distribution is Phase 5 packaging
// work (see docs/IMPLEMENT.md), not yet done.

use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

struct BackendProcess(Mutex<Option<Child>>);

fn backend_dir() -> std::path::PathBuf {
    // app/src-tauri -> ../../backend
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../backend")
        .canonicalize()
        .expect("backend/ directory not found relative to src-tauri; is the repo layout intact?")
}

fn spawn_backend() -> std::io::Result<Child> {
    let backend_dir = backend_dir();
    let python = backend_dir.join(".venv/bin/python");
    Command::new(python)
        .args([
            "-m",
            "uvicorn",
            "service.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ])
        .current_dir(&backend_dir)
        .spawn()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let child = spawn_backend().expect(
                "failed to start backend/service — check that backend/.venv exists (see README setup)",
            );
            app.manage(BackendProcess(Mutex::new(Some(child))));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<BackendProcess>();
                let taken = state.0.lock().unwrap().take();
                if let Some(mut child) = taken {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
