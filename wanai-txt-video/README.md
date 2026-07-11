# wanai-txt-video

A native macOS desktop app for generating short videos locally from a text
prompt (and optional input image), using the [Wan2.2](https://huggingface.co/Wan-AI)
video generation model — fully on-device on Apple Silicon, no cloud calls.

## Status

Early build-out. See [docs/IMPLEMENT.md](docs/IMPLEMENT.md) for the phased
plan and current phase; see [docs/DESIGN.md](docs/DESIGN.md) for the
architecture and the reasoning behind it.

Currently: **Phase 0 complete (text-to-video)**, moving into **Phase 1 —
inference service**. `Wan2.2-TI2V-5B` (GGUF Q4_K_M) generates coherent,
on-prompt video on Apple Silicon at 640×384 — confirmed slow (~38 min for a
2s clip) but working. Image-to-video is untested and deferred. See
[docs/IMPLEMENT.md](docs/IMPLEMENT.md) for details.

## Why this is harder than it sounds

Wan2.2 is CUDA-only upstream, and the only model size that fits on a Mac
(`TI2V-5B`) still needs GGUF quantization and community MPS support to run
at all — and even then, generation is slow (community reports ~82 minutes
for a 2-second clip on a 64GB M1 Max). Details and tradeoffs are in
[docs/DESIGN.md](docs/DESIGN.md).

## Project layout

```
wanai-txt-video/
├── docs/            # IMPLEMENT.md (plan), DESIGN.md (architecture)
├── backend/         # Python inference service
│   ├── comfyui/       # vendored ComfyUI — the inference engine
│   ├── service/        # FastAPI app (job queue, HTTP API) — Phase 1
│   ├── requirements.txt
│   └── .venv/
├── app/             # Tauri native desktop app — Phase 2+
│   ├── src/            # frontend UI
│   └── src-tauri/      # Rust shell, backend process management
└── models/          # downloaded model weights (gitignored)
```

## Setup (backend)

`backend/comfyui/` is not committed to this repo — it's a vendored clone of
upstream ComfyUI plus the GGUF custom node, set up locally:

```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

git clone https://github.com/comfyanonymous/ComfyUI.git comfyui
git clone https://github.com/city96/ComfyUI-GGUF.git comfyui/custom_nodes/ComfyUI-GGUF

cat > comfyui/extra_model_paths.yaml <<'EOF'
wanai-txt-video:
    base_path: ../../models/
    diffusion_models: diffusion_models/
    text_encoders: text_encoders/
    vae: vae/
EOF
```

Then download the model weights into `models/` (see
[docs/IMPLEMENT.md](docs/IMPLEMENT.md) for the exact files:
`Wan2.2-TI2V-5B-Q4_K_M.gguf`, `umt5_xxl_fp8_e4m3fn_scaled.safetensors`,
`wan2.2_vae.safetensors`).

Requires Python 3.11+ (the macOS system Python is too old for ComfyUI's
dependencies).

## Requirements

- Apple Silicon Mac
- Python 3.11+
- Enough disk space for model weights (~5–10GB for the GGUF-quantized 5B
  model)
- 32GB+ unified memory recommended (24GB is workable but tight — see
  [docs/DESIGN.md](docs/DESIGN.md) for details)
