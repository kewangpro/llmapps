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
  discrete VRAM, no CUDA. This rules out Wan2.2's 14B variants
  (`T2V-A14B`, `I2V-A14B`, `S2V-14B`, `Animate-14B`, all from
  [Wan-AI on Hugging Face](https://huggingface.co/Wan-AI)) outright — they
  need 80GB+ — leaving `Wan2.2-TI2V-5B` as the only local-Mac candidate, and
  even that puts memory under pressure, since the model, T5 text encoder,
  and OS all share one pool. Recommended system RAM is 32–64GB for
  comfortable headroom (weights alone: ~8–12GB quantized); 24GB is workable
  but tight (validated in Phase 0 — see IMPLEMENT.md).
- **No official MPS/Metal support** in the upstream
  [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) repo — it assumes
  CUDA and FP8 checkpoints, and Metal has no `Float8_e4m3fn` kernel. The
  viable path is ComfyUI + GGUF-quantized weights (Q4/Q5) with
  `--force-fp16 --mps-device`.
- **Generation is slow** (community benchmark: ~82 min for a 2-second clip on
  a 64GB M1 Max; confirmed by our own Phase 0 measurement — see
  IMPLEMENT.md). This is the single biggest constraint on UX: the app cannot
  pretend generation is fast. Everything about the interaction model — an
  async job with a cancel button and a live progress/ETA readout instead of
  a blocking spinner — follows from this. (OS notifications and a
  multi-job queue UI were both considered and deliberately dropped from
  scope — the user is expected to check back on the window, not be paged.)
  Notably, a newer/smaller-memory Mac (M4, 24GB) wasn't faster than the
  64GB M1 Max reference at a *lower* resolution — this workload is
  memory-bandwidth-bound, not compute-bound, so raw chip generation doesn't
  buy much without more unified memory bandwidth/capacity.

## Runtime architecture

```
┌─────────────────────────────┐
│  Tauri desktop app (Rust +   │  prompt/image input, cancel button,
│  React frontend)              │  progress % + live ETA, video player
└───────────────┬───────────────┘
                │ localhost HTTP/WS
┌───────────────▼───────────────┐
│  Python inference service      │  FastAPI/uvicorn, wraps ComfyUI
│  (venv python, spawned/owned   │  headless (no diffusers path — see
│   by the Rust shell)           │  "Why ComfyUI" below)
└───────────────┬───────────────┘
                │ MPS
┌───────────────▼───────────────┐
│  Wan2.2 TI2V-5B (GGUF Q4/Q5)   │  local weights on disk (~5–10GB)
└─────────────────────────────────┘
```

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
│   ├── src/            # frontend UI (prompt input, progress/cancel, video
│   │                       player, generation history)
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
  management that doesn't belong in ComfyUI or in the Rust shell. In
  practice, output videos are served straight from ComfyUI's own
  `output/video/` directory (`GET /jobs/{id}/video`) rather than copied
  elsewhere — no separate app data dir turned out to be needed.

### Why Tauri over Electron

Smaller binary, a real native shell, and straightforward sidecar process
management (spawn the Python backend on launch, kill it on quit — including
on a force-kill or crash, not just a graceful window close, see
docs/IMPLEMENT.md Phase 3) without bundling a second copy of Chromium. The
frontend is plain web tech (React) talking to the backend directly over
localhost HTTP/WS — no inference logic lives in Rust, and no IPC round-trip
through the Rust shell either.

### Why GGUF Q4/Q5 quantization

It's the only known way to get `Wan2.2-TI2V-5B` running on Metal at all,
since FP8 checkpoints aren't supported by MPS. This is a forced choice, not
a preference — accepted quality loss vs the NVIDIA fp16/fp8 reference output.
Validated in Phase 0: Q4_K_M output is coherent and on-prompt at 640×384
(see IMPLEMENT.md for the result), so the forced choice turned out not to
cost much in practice at that resolution.

## Data flow (single generation request)

1. User fills in prompt (+ optional image) in the Tauri UI → `POST
   /generate` to the local FastAPI service.
2. FastAPI creates a job record, launches a background `asyncio` task, and
   returns a job id immediately.
3. That task drives ComfyUI's HTTP/WebSocket API to run the Wan2.2 GGUF
   workflow, relaying per-step progress (and a self-correcting live ETA)
   back into the job record.
4. UI polls `GET /jobs/{id}` every 3s for progress — a progress bar plus
   elapsed/ETA, not a live preview (generation is too slow for the latter
   to be meaningful). A cancel button clears ComfyUI's queue and interrupts
   the job at any point.
5. On completion, UI fetches the output via `GET /jobs/{id}/video` and
   shows it in a video player. No OS notification — considered and
   deliberately dropped from scope (see docs/IMPLEMENT.md Phase 3); the
   user is expected to check back on the window.

## Key risks (see IMPLEMENT.md "Open risks" for the fuller, status-tracked list)

- **Memory pressure at 24GB unified RAM** — below the community-recommended
  32–64GB. Validated as workable at the pinned 640×384 default; higher
  resolutions are an open question, not a settled no.
- **Speed** dictates the whole interaction model, confirmed by measurement,
  not just the community benchmark — the async, cancel-and-check-back UX is
  load-bearing, not a hedge.
- **Community-maintained GGUF/MPS support** is a single point of fragility
  outside our control.
- **Resolution ceiling**: 640×384 (not Wan2.2's native 1280×704) is a
  deliberate scope decision — the app's baseline quality bar is set by what's
  proven to run in reasonable time, not by the model's best-case output.
