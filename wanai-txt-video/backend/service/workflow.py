"""Builds the ComfyUI API-format workflow graph for Wan2.2-TI2V-5B.

Mirrors the node graph manually validated during the Phase 0 feasibility
spike (UnetLoaderGGUF -> CLIPLoader -> Wan22ImageToVideoLatent -> KSampler
-> VAEDecode -> CreateVideo -> SaveVideo). Wan22ImageToVideoLatent handles
both T2V (start_image=None) and I2V (start_image set) since TI2V-5B is a
single model for both modes.
"""

from . import config


def build_workflow(
    *,
    prompt: str,
    negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT,
    width: int = config.DEFAULT_WIDTH,
    height: int = config.DEFAULT_HEIGHT,
    length: int = config.DEFAULT_LENGTH,
    steps: int = config.DEFAULT_STEPS,
    cfg: float = config.DEFAULT_CFG,
    shift: float = config.DEFAULT_SHIFT,
    sampler_name: str = config.DEFAULT_SAMPLER,
    scheduler: str = config.DEFAULT_SCHEDULER,
    fps: float = config.DEFAULT_FPS,
    seed: int,
    filename_prefix: str,
    start_image_name: str | None = None,
) -> dict:
    latent_inputs = {
        "vae": ["3", 0],
        "width": width,
        "height": height,
        "length": length,
        "batch_size": 1,
    }

    workflow = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": config.UNET_NAME},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": config.CLIP_NAME, "type": "wan"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": config.VAE_NAME},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        },
        "6": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": shift, "model": ["1", 0]},
        },
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": latent_inputs,
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["9", 0], "fps": fps},
        },
        "11": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["10", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }

    if start_image_name:
        workflow["12"] = {
            "class_type": "LoadImage",
            "inputs": {"image": start_image_name},
        }
        latent_inputs["start_image"] = ["12", 0]

    return workflow
