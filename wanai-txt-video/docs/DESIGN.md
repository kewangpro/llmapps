# Design: Local Wan2.2 Video Generation App

This document covers the technical design and rationale behind the choices in
[IMPLEMENT.md](./IMPLEMENT.md). IMPLEMENT.md tracks phased work; this covers
*why* the system is shaped the way it is.

## Goal

A native macOS desktop app where a user enters a text prompt (and optionally
an input image) and gets a generated video back, running entirely on-device
on Apple Silicon — no cloud API calls, no data leaving the machine.

## Constraints that shaped the design

- **Apple Silicon, unified memory (24GB on the reference dev machine)** — no
  discrete VRAM, no CUDA. This rules out Wan2.2's 14B variants outright (they
  need 80GB+) and puts even the 5B variant (`Wan2.2-TI2V-5B`) under memory
  pressure, since the model, T5 text encoder, and OS all share one pool.
- **No official MPS/Metal support** in the upstream
  [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) repo — it assumes
  CUDA and FP8 checkpoints, and Metal has no `Float8_e4m3fn` kernel. The
  viable path is ComfyUI + GGUF-quantized weights (Q4/Q5) with
  `--force-fp16 --mps-device`.
- **Generation is slow** (community benchmark: ~82 min for a 2-second clip on
  a 64GB M1 Max). This is the single biggest constraint on UX: the app cannot
  pretend generation is fast. Everything about the interaction model — async
  jobs, notifications, a queue instead of a spinner — follows from this.

## Component breakdown

```
wanai-txt-video/
├── docs/            # IMPLEMENT.md (phases/tasks), DESIGN.md (this file)
├── backend/         # Python inference service
│   ├── comfyui/       # vendored ComfyUI — the actual inference engine
│   ├── service/        # FastAPI app: job queue, progress, HTTP API
│   ├── requirements.txt
│   └── .venv/
├── app/             # Tauri native desktop app
│   ├── src/            # frontend UI (prompt input, job list, video player)
│   └── src-tauri/      # Rust shell; spawns/kills the backend as a sidecar
└── models/          # downloaded GGUF weights (gitignored, multi-GB)
```

### Why ComfyUI as the inference engine, not raw diffusers

ComfyUI isn't treated as a demo/experimentation tool here — it's the
production inference engine. Reasons:
- It already has working GGUF-quantized-weight support and MPS device
  handling, contributed by the community, which we'd otherwise have to
  reimplement or patch into the upstream Wan2.2 CUDA-only codebase ourselves.
- It runs headless with an HTTP/WebSocket API, so wrapping it is a matter of
  driving that API from FastAPI, not embedding a UI we don't want.
- If Wan2.2 MPS support improves upstream later, swapping the execution
  backend is a ComfyUI config/node change, not a rewrite of our service.

Tradeoff accepted: dependency on community-maintained GGUF/MPS custom nodes,
which can break on ComfyUI or Wan2.2 updates (tracked as a risk below).

### Why a separate FastAPI service instead of calling ComfyUI directly from Tauri

Tauri's Rust side could talk to ComfyUI's API directly, but a thin FastAPI
layer in between gives us:
- A stable contract (`POST /generate`, `GET /jobs/{id}`) independent of
  ComfyUI's own API, so ComfyUI internals (or even the engine itself) can
  change without touching the frontend.
- A natural place for job queueing, progress normalization, and file
  management (writing outputs to `models/` or an app data dir) that doesn't
  belong in ComfyUI or in the Rust shell.

### Why Tauri over Electron

Smaller binary, a real native shell, and straightforward sidecar process
management (spawn the Python backend on launch, kill it on quit) without
bundling a second copy of Chromium. The frontend is still plain
web tech (React/Svelte) talking to the Rust shell over Tauri's IPC and to
the backend over localhost HTTP/WS — no inference logic lives in Rust.

### Why GGUF Q4/Q5 quantization

It's the only known way to get `Wan2.2-TI2V-5B` running on Metal at all,
since FP8 checkpoints aren't supported by MPS. This is a forced choice, not
a preference — accepted quality loss vs the NVIDIA fp16/fp8 reference output
needs to be validated in Phase 0 before committing further engineering time.

## Data flow (single generation request)

1. User fills in prompt (+ optional image) in the Tauri UI → `POST
   /generate` to the local FastAPI service.
2. FastAPI creates a job record, enqueues it, returns a job id immediately.
3. Background worker drives ComfyUI's API to run the Wan2.2 GGUF workflow,
   polling/streaming progress back into the job record.
4. UI polls or subscribes via WebSocket to `GET /jobs/{id}` for progress;
   shows a progress indicator, not a live preview (generation is too slow
   for the latter to be meaningful).
5. On completion, output video is written to disk; UI fetches it via `GET
   /jobs/{id}/video` and fires an OS notification (job may complete while
   the user is doing something else — expected, given generation times).

## Key risks (see IMPLEMENT.md for the fuller list)

- **Memory pressure at 24GB unified RAM** — below the community-recommended
  32–64GB. Phase 0 validated real headroom at 640×384 (no OOM across a full
  20-step run); full official resolution (1280×704) remains unvalidated and
  is assumed tight-to-infeasible on 24GB until measured.
- **Speed** dictates the whole interaction model — confirmed, not assumed:
  38 min for a 2s clip at 640×384 on this machine, in the same range as the
  82-min/2s community benchmark despite being at *lower* resolution on newer
  silicon. The async/notify UX is load-bearing, not a hedge.
- **Community-maintained GGUF/MPS support** is a single point of fragility
  outside our control.
- **Resolution ceiling**: 640×384 is the pinned working default (see
  IMPLEMENT.md Phase 0), well below Wan2.2's native 1280×704. This is a
  scope decision, not just a performance note — the app's baseline quality
  bar is set by what's proven to run in reasonable time, not by the model's
  best-case output.
