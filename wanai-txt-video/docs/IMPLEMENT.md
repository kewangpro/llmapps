# Implementation Plan: Local Wan2.2 Video Generation App

Target platform: Apple Silicon Mac (MPS). Goal: a native desktop app that takes a
user prompt (and optional input image) and generates a short video locally using
the Wan2.2 model, with no cloud dependency. See [DESIGN.md](./DESIGN.md).

## Phases

### Phase 0 — Feasibility spike (1–2 days) — ✅ complete (T2V + I2V)

Results on reference machine (Apple M4, 24GB unified memory):
- Text-to-video via ComfyUI + `ComfyUI-GGUF` + MPS works end-to-end, no
  official `--mps-device` flag needed (auto-detected).
- **Quality at Q4_K_M is acceptable** at 640×384 — coherent, on-prompt output,
  no visible quantization artifacting. Confirms the GGUF/MPS path is viable,
  not just "completes without crashing."
- **Speed**: 640×384, 49 frames (~2s @ 24fps), 20 steps → **38m09s** total
  (27m26s sampling, ~82s/it). This is at *lower* resolution than the
  community 82-min/2s benchmark on a 64GB M1 Max, and this machine is not
  faster than that reference despite newer silicon (see DESIGN.md for why).
- **Memory holds on 24GB** at 640×384 — stayed within the compressor/free-RAM
  envelope throughout a real 20-step run, no OOM. Full official resolution
  (1280×704) is untested and, by rough scaling of the numbers above, likely
  multi-hour per clip on this hardware — treat as an unvalidated, expensive
  tier rather than the default.
- Correct pinned sampling params for `Wan2.2-TI2V-5B`: `shift=5.0` (via
  `ModelSamplingSD3`), `cfg=5.0`, `euler`/`simple`. (`shift` is easy to get
  wrong — an initial guess of `8.0` produced pure structured noise even at
  full step count; the working value came from the official Wan2.2 ComfyUI
  blueprints.)
- **Image-to-video validated**: same 640×384/49-frame/20-step config, seeded
  from a T2V output frame via `Wan22ImageToVideoLatent`'s `start_image`
  input — completed in 33m06s (faster than the T2V run because model/CLIP/
  VAE loaders were cached from the prior job in the same ComfyUI session).
  Output preserved the seed image's composition with visible subtle motion,
  no artifacts.

Decision carried into Phase 1: **640×384 is the pinned default resolution**,
not 1280×704. Higher resolution becomes an opt-in/settings-panel concern
(Phase 3), not something the service defaults to.

### Phase 1 — Inference service (2–4 days) — ✅ complete

`backend/service/` (FastAPI, run via `uvicorn service.main:app`):
- `POST /generate` (prompt, optional input image, resolution, frame count,
  steps/cfg/shift overrides) → job id, defaulting to the Phase 0-validated
  params (640×384, 49 frames, 20 steps, `shift=5.0`, `cfg=5.0`,
  `euler`/`simple`) via `config.py`.
- `GET /jobs/{id}` — status/progress, relayed live from ComfyUI's own
  websocket (`comfy_client.stream_progress`), not reimplemented.
- `GET /jobs/{id}/video` — streams the completed file.
- Jobs run as background `asyncio` tasks (`jobs.py`) so `/generate` returns
  immediately — required given 30-40+ minute generations.
- `comfy_process.py` starts ComfyUI as a subprocess on service startup if
  one isn't already reachable on the configured port, and only stops a
  process it started itself (reuses an already-running dev instance
  untouched).
- `workflow.py` builds the same node graph validated manually in Phase 0
  (`UnetLoaderGGUF` → `CLIPLoader` → `Wan22ImageToVideoLatent` →
  `KSampler` → `VAEDecode` → `CreateVideo` → `SaveVideo`), parameterized by
  the `/generate` request.

Validated end-to-end with a real request through the running service (not
just direct ComfyUI calls): job created, progress streamed 0→N, video served
correctly. One bug found and fixed in the process: ComfyUI's `SaveVideo`
node reports its output in the history response under the `images` key
(with a sibling `animated: [true]` flag), not `videos`/`gifs` as the node
name would suggest — `comfy_client.output_video_path` now checks `images`
first, falling back to `videos`/`gifs` defensively.

### Phase 2 — Tauri shell + packaging (2–3 days) — ✅ complete (dev mode)

`app/` (Tauri + React + TypeScript, scaffolded via `create-tauri-app`):
- `src-tauri/src/lib.rs` spawns `backend/.venv/bin/python -m uvicorn
  service.main:app` on launch and kills it when the window closes normally
  (`WindowEvent::Destroyed`). This spawns the venv's python directly, not a
  bundled Tauri sidecar binary — sufficient for development; packaging a
  self-contained Python runtime for distribution is Phase 4 work, not done.
  **Known gap**: if the Tauri process itself is force-killed/crashes rather
  than quit normally, the backend child is orphaned (observed during
  testing) — a normal window close cleans up fine, but this should be
  hardened (e.g. handle `RunEvent::Exit` too) before Phase 4.
- Frontend (`src/App.tsx`) talks directly to the backend service over
  `http://127.0.0.1:8000` (no Rust-side proxying): prompt textarea, optional
  drag-and-drop/file-picker image input, width/height/frames/steps controls
  defaulting to the Phase 0-validated config, a Generate button, a
  progress bar polling `GET /jobs/{id}` every 3s, and a video player for the
  completed output.
- History of past generations: a small JSON index in `localStorage`
  (`src/history.ts`) — prompt, job id, video URL. Satisfies the "small
  JSON index" part of the original plan; video files themselves already
  live on disk via the backend's output directory, nothing extra needed
  there.

Validated by actually running `npm run tauri dev`: window builds and
launches, the Rust setup hook spawns the FastAPI backend and it correctly
detects/reuses an already-running ComfyUI instance rather than double
spawning, and the frontend's CSS/layout was fixed live via Vite HMR during
manual review (grid overlap from a missing `box-sizing: border-box`, and
excess top spacing from an unset `body` margin).

### Phase 3 — UX for slow inference & hardening (3–5 days)
- ✅ Cancel button — `DELETE /jobs/{id}` clears ComfyUI's whole queue and
  issues a global interrupt (`ComfyClient.clear_queue` +
  `ComfyClient.interrupt_all`), rather than targeting a single prompt_id.
  This app only ever drives one job at a time (no multi-job queue UI), so a
  global stop is simpler and doesn't silently no-op like a targeted
  interrupt does when the prompt isn't the one currently executing
  (observed directly: ComfyUI logs `"Prompt X is not currently running,
  skipping interrupt"` for a merely-queued job).
- ✅ Show time estimate in progress — `JobResponse.eta_seconds` /
  `elapsed_seconds`, self-correcting: once a job has made progress it uses
  its own observed seconds/step rather than trusting the static ~85s/it
  fallback for the whole run, since actual speed varies with system load.
- ✅ Lifecycle hardening — closing the app now reliably stops ComfyUI too,
  not just the FastAPI backend (see risk #6 below for what was broken and
  how it's fixed).
- Error handling for OOM (auto-suggest lower resolution).
- First-run model download flow (fetch GGUF weights from Hugging Face, show
  download progress).
- Settings panel (quantization level, resolution presets trading speed vs
  quality).

### Phase 4 — Packaging & distribution
- `tauri build` for a signed `.app`/`.dmg`.
- Decide: bundle model weights (~5–10GB, large installer) vs download on
  first run (recommended).

## Open risks

1. **Speed**: confirmed, not hypothetical — 38 min for a 2s clip at reduced
   (640×384) resolution. Async/notify UX is required, not optional; the
   live ETA (Phase 3) is built off this measured constant, self-correcting
   from each job's own observed rate rather than the borrowed community
   number.
2. **Quantization quality tradeoff**: resolved for 640×384 — Q4_K_M output is
   on-prompt and visually clean at that resolution. Not yet validated at
   higher resolutions.
3. **No official MPS support**: relies on community ComfyUI custom nodes /
   GGUF loaders, which carries breakage risk on Wan2.2 or ComfyUI updates.
4. **Full official resolution (1280×704) unvalidated**: likely multi-hour per
   clip on 24GB unified memory based on Phase 0 scaling; needs a deliberate
   timed test (not a background assumption) before it's ever offered as a
   user-facing option.
5. ~~Image-to-video untested~~ — resolved: validated in Phase 0 at the same
   pinned config as T2V.
6. ~~Backend sidecar orphaning on force-kill~~ — resolved in Phase 3.
   Two separate bugs, both found by actually killing the running app rather
   than by inspection:
   - The Rust shell only cleaned up on a normal window close
     (`WindowEvent::Destroyed`); an external `kill`/crash bypassed Tauri's
     event loop entirely and orphaned both the backend and ComfyUI. Fixed
     with a `signal-hook`-based SIGTERM/SIGINT handler plus a
     `RunEvent::Exit` hook, alongside the existing window-close path — all
     three converge on the same cleanup so whichever fires first wins.
   - Even after adding that, ComfyUI still didn't stop: Rust's
     `Child::kill()` always sends **SIGKILL**, which can't be caught, so
     the Python backend's own shutdown code (`comfy_process.stop()`, called
     from `main.py`'s `lifespan` handler, which is what actually stops
     ComfyUI) never ran. Fixed by sending SIGTERM directly via `libc::kill`
     first, polling for graceful exit, and only falling back to
     `Child::kill()` (SIGKILL) if it doesn't exit within 10s.
   - Separately, `comfy_process.py` previously only stopped a ComfyUI
     instance it had itself spawned via `Popen` — if the app *adopted* an
     already-running instance (the dev-convenience "reuse" path), it would
     never stop it, since there was no PID to track. Fixed: on adopt, look
     up the PID via `lsof -ti tcp:{port}` and track it the same as a
     spawned process, so "close app" now unconditionally stops whatever
     ComfyUI it's using, regardless of who started it.

## Next step

Phases 0-2 are done; Phase 3's cancel/ETA/lifecycle-hardening items are
done too. Remaining Phase 3 work: OOM error handling, first-run model
download flow, and a settings panel — then Phase 4 packaging.
