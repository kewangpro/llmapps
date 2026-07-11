# Implementation Plan: Local Wan2.2 Video Generation App

Target platform: Apple Silicon Mac (MPS). Goal: a native desktop app that takes a
user prompt (and optional input image) and generates a short video locally using
the Wan2.2 model, with no cloud dependency.

## Background / constraints

- Model source: [Wan-AI on Hugging Face](https://huggingface.co/Wan-AI). The
  Wan2.2 family includes `TI2V-5B` (text+image→video, 5B params) and larger
  14B variants (`T2V-A14B`, `I2V-A14B`, `S2V-14B`, `Animate-14B`) that require
  80GB+ VRAM — not viable on a Mac.
- **Only `Wan2.2-TI2V-5B` is a candidate** for local Apple Silicon use.
- The official [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) repo is
  **CUDA-only**. FP8 checkpoints don't work on Metal (no `Float8_e4m3fn` kernel).
- Workaround: run via **ComfyUI with GGUF-quantized weights (Q4/Q5)** on MPS,
  using `--force-fp16 --mps-device`.
- Real-world benchmark: M1 Max 64GB took ~82 minutes for a 2-second clip in
  GGUF/fp8-fallback mode. **Generation on Mac is slow** — this shapes the UX:
  treat it as an async background job with notification on completion, not an
  interactive/live-preview tool.
- Recommended system RAM: 32–64GB for comfortable model loading + T5 text
  encoder offload. Model weights alone: ~8–12GB (5B, quantized).

## Architecture

```
┌─────────────────────────────┐
│  Tauri desktop app (Rust +   │  prompt/image input, job queue,
│  web frontend: React/Svelte) │  progress %, video preview/player
└───────────────┬───────────────┘
                │ localhost HTTP/WS
┌───────────────▼───────────────┐
│  Python inference service      │  FastAPI/uvicorn, wraps ComfyUI
│  (bundled venv, sidecar proc)  │  headless or diffusers pipeline
└───────────────┬───────────────┘
                │ MPS
┌───────────────▼───────────────┐
│  Wan2.2 TI2V-5B (GGUF Q4/Q5)   │  local weights on disk (~5–10GB)
└─────────────────────────────────┘
```

- **Tauri over Electron**: smaller binary, real native shell, easy Rust
  sidecar management for spawning/killing the Python process.
- Python backend does the actual model work (mature ML ecosystem); Tauri
  talks to it over localhost only — no need to reimplement inference in Rust.

## Phases

### Phase 0 — Feasibility spike (1–2 days) — ✅ complete (T2V), I2V deferred

Results on reference machine (Apple M4, 24GB unified memory):
- Text-to-video via ComfyUI + `ComfyUI-GGUF` + MPS works end-to-end, no
  official `--mps-device` flag needed (auto-detected).
- **Quality at Q4_K_M is acceptable** at 640×384 — coherent, on-prompt output,
  no visible quantization artifacting. Confirms the GGUF/MPS path is viable,
  not just "completes without crashing."
- **Speed**: 640×384, 49 frames (~2s @ 24fps), 20 steps → **38m09s** total
  (27m26s sampling, ~82s/it). This is at *lower* resolution than the
  community 82-min/2s benchmark on a 64GB M1 Max (DESIGN.md), so this
  machine is not faster than that reference despite being newer silicon —
  memory-bandwidth-bound, not compute-bound.
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
- **Image-to-video not yet tested** — deferred as low-risk (same model/VAE,
  same GGUF/MPS path already validated for T2V) but still an open gap before
  Phase 1 assumes I2V "just works."

Decision carried into Phase 1: **640×384 is the pinned default resolution**,
not 1280×704. Higher resolution becomes an opt-in/settings-panel concern
(Phase 4), not something the service defaults to.

### Phase 1 — Inference service (2–4 days)
- Wrap the working ComfyUI pipeline in a FastAPI service that drives
  ComfyUI's own HTTP/WebSocket API (`/prompt`, `/history/{id}`, ws progress
  events) rather than embedding inference logic directly:
  - `POST /generate` (prompt, optional input image, resolution, frame count)
    → job id. Defaults to the Phase 0-validated params (640×384, 49 frames,
    20 steps, `shift=5.0`, `cfg=5.0`, `euler`/`simple`) when the caller
    doesn't override them.
  - `GET /jobs/{id}` for status/progress
  - `GET /jobs/{id}/video` to fetch the result
- Run generation in a background worker/thread so the API stays responsive;
  stream progress via WebSocket or polling. ComfyUI already emits per-step
  progress on its own websocket — relay it, don't reimplement it.
- Pin the exact model file + quantization + sampling config (see Phase 0
  results above) in a checked-in config file, not hardcoded across call
  sites — this is what Phase 0 spent its time discovering and it should not
  need rediscovering.
- Reuse the ComfyUI-facing workflow-JSON approach validated manually in
  Phase 0 as the service's internal workflow template, parameterized by the
  `/generate` request fields.

### Phase 2 — Tauri shell + packaging (2–3 days)
- Scaffold Tauri app; bundle a Python venv (or use `pyoxidizer`/`briefcase`
  style packaging) as a sidecar binary launched on app start, killed on quit.
- Basic UI: prompt textbox, optional image drop zone, resolution/frame-count
  controls, "Generate" button, progress bar, video player for output,
  history of past generations (store as files + a small SQLite/JSON index).

### Phase 3 — UX for slow inference (1–2 days)
- Background generation with OS notification on completion.
- Ability to queue multiple prompts.
- Cancel button.
- Disk-space / time-estimate warning before starting a job.

### Phase 4 — Polish & hardening (2–3 days)
- Error handling for OOM (auto-suggest lower resolution).
- First-run model download flow (fetch GGUF weights from Hugging Face, show
  download progress).
- Settings panel (quantization level, resolution presets trading speed vs
  quality).

### Phase 5 — Packaging & distribution
- `tauri build` for a signed `.app`/`.dmg`.
- Decide: bundle model weights (~5–10GB, large installer) vs download on
  first run (recommended).

## Open risks

1. **Speed**: confirmed, not hypothetical — 38 min for a 2s clip at reduced
   (640×384) resolution. Async/notify UX is required, not optional; Phase 3
   should build the time estimate off this measured constant rather than the
   borrowed community number.
2. **Quantization quality tradeoff**: resolved for 640×384 — Q4_K_M output is
   on-prompt and visually clean at that resolution. Not yet validated at
   higher resolutions.
3. **No official MPS support**: relies on community ComfyUI custom nodes /
   GGUF loaders, which carries breakage risk on Wan2.2 or ComfyUI updates.
4. **Full official resolution (1280×704) unvalidated**: likely multi-hour per
   clip on 24GB unified memory based on Phase 0 scaling; needs a deliberate
   timed test (not a background assumption) before it's ever offered as a
   user-facing option.
5. **Image-to-video untested**: same model/pipeline as the validated T2V
   path, so risk is believed low, but Phase 1's `/generate` image input
   should not be treated as proven until it's actually run once.

## Next step

Phase 0 (text-to-video) is done — proceed to Phase 1: build the FastAPI
service around the now-validated ComfyUI workflow and pinned config. Run one
image-to-video generation manually (closing the Phase 0 gap) before or
alongside wiring the `/generate` image-input path, so Phase 1 isn't built
against an unverified assumption.
