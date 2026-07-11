"""Pinned config for the Wan2.2 inference service.

Defaults come from the Phase 0 feasibility spike (see docs/IMPLEMENT.md) —
640x384 / 49 frames / 20 steps / shift=5.0 / cfg=5.0 is the only config
that's actually been measured for speed, memory, and quality on the
reference hardware. Do not change these without re-validating per
docs/IMPLEMENT.md's Phase 0 methodology.
"""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
COMFYUI_DIR = BACKEND_DIR / "comfyui"
COMFYUI_PYTHON = BACKEND_DIR / ".venv" / "bin" / "python"
COMFYUI_OUTPUT_DIR = COMFYUI_DIR / "output"

COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_BASE_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_WS_URL = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws"

UNET_NAME = "Wan2.2-TI2V-5B-Q4_K_M.gguf"
CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME = "wan2.2_vae.safetensors"

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, static, distorted"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 384
DEFAULT_LENGTH = 49  # frames, ~2s at 24fps
DEFAULT_FPS = 24.0
DEFAULT_STEPS = 20
DEFAULT_CFG = 5.0
DEFAULT_SHIFT = 5.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
