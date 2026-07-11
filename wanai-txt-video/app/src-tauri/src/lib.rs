// Spawns the Python FastAPI inference service (backend/service/) as a
// sidecar process on launch and terminates it on quit. See
// docs/DESIGN.md "Why a separate FastAPI service instead of calling
// ComfyUI directly from Tauri" — the Rust shell only manages the process
// lifecycle; all inference logic and HTTP calls live in the frontend
// talking directly to the local FastAPI service over localhost.
//
// This spawns the venv's python directly rather than a bundled Tauri
// "sidecar" binary — sufficient for local development (Phase 2). Bundling
// a self-contained Python runtime for distribution is Phase 4 packaging
// work (see docs/IMPLEMENT.md), not yet done.
//
// Cleanup is wired three ways, because "close the app" has to mean it in
// every case, not just the happy path (docs/IMPLEMENT.md Phase 3):
//   1. WindowEvent::Destroyed  — normal window close (red button)
//   2. RunEvent::Exit          — Cmd+Q / app-level quit
//   3. SIGTERM/SIGINT handler  — external kill (crash recovery, `kill`,
//      Activity Monitor force-quit) that bypasses Tauri's event loop
//      entirely and would otherwise orphan the backend + the ComfyUI
//      instance it owns (backend/service/comfy_process.py kills ComfyUI
//      when the backend process itself exits cleanly).
// All three converge on the same take-and-kill of the shared Child handle,
// so whichever fires first wins and the other two are no-ops.

use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

type SharedChild = Arc<Mutex<Option<Child>>>;

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

fn kill_backend(shared: &SharedChild) {
    let Some(mut child) = shared.lock().unwrap().take() else {
        return;
    };

    // SIGTERM first, not Child::kill() (which is always SIGKILL on Unix and
    // can't be caught) — the backend's own shutdown code (comfy_process.stop()
    // in backend/service/main.py's lifespan handler) is what stops ComfyUI,
    // and that code never runs if the process is killed instantly. Give it
    // a window to shut down gracefully, then force-kill only as a fallback.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            _ => break,
        }
    }
    let _ = child.kill();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend: SharedChild = Arc::new(Mutex::new(None));

    // Background thread watching for external termination signals — see
    // module doc comment, cleanup path (3).
    {
        let backend = backend.clone();
        std::thread::spawn(move || {
            let mut signals = signal_hook::iterator::Signals::new([
                signal_hook::consts::SIGTERM,
                signal_hook::consts::SIGINT,
            ])
            .expect("failed to register signal handler");
            if signals.forever().next().is_some() {
                kill_backend(&backend);
                std::process::exit(0);
            }
        });
    }

    let setup_backend = backend.clone();
    let window_event_backend = backend.clone();
    let run_event_backend = backend.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(move |_app| {
            let child = spawn_backend().expect(
                "failed to start backend/service — check that backend/.venv exists (see README setup)",
            );
            *setup_backend.lock().unwrap() = Some(child);
            Ok(())
        })
        .on_window_event(move |_window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                kill_backend(&window_event_backend);
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                kill_backend(&run_event_backend);
            }
        });
}
