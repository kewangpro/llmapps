"""FastAPI inference service — wraps ComfyUI behind a stable HTTP contract.

See docs/DESIGN.md "Why a separate FastAPI service instead of calling
ComfyUI directly from Tauri" for the rationale, and docs/IMPLEMENT.md
Phase 1 for scope.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from .comfy_client import ComfyClient
from .comfy_process import ComfyProcessManager
from .jobs import JobStatus, JobStore

logging.basicConfig(level=logging.INFO)

comfy_client = ComfyClient()
comfy_process = ComfyProcessManager(comfy_client)
job_store = JobStore(comfy_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await comfy_process.ensure_running()
    yield
    comfy_process.stop()


app = FastAPI(title="wanai-txt-video inference service", lifespan=lifespan)

# The Tauri webview (http://localhost:1420 in dev, tauri://localhost /
# http://tauri.localhost once packaged) is a different origin than this
# service (http://127.0.0.1:8000), so the browser's CORS preflight needs
# explicit allowance — without this, every fetch from the app fails with a
# generic "Load failed" in WebKit. Everything here is localhost-only (no
# cloud calls, per docs/DESIGN.md), so a fixed local-origin allowlist is
# fine; no wildcard needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "tauri://localhost",
        "http://tauri.localhost",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    progress_step: int
    progress_total: int
    phase: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    error: Optional[str] = None
    video_url: Optional[str] = None


def _job_response(job) -> JobResponse:
    phase = None
    elapsed_seconds = None
    eta_seconds = None
    if job.status == JobStatus.RUNNING and job.started_at is not None:
        elapsed_seconds = time.time() - job.started_at

        if job.progress_total > 0 and job.progress_step >= job.progress_total:
            # Sampling (KSampler) is done, but VAE decode + video encode/save
            # still run afterward and aren't reflected in step progress — no
            # step-based ETA is meaningful here. See docs/IMPLEMENT.md risk #7.
            phase = "finishing"
        else:
            phase = "sampling"
            remaining_steps = max(job.progress_total - job.progress_step, 0)
            if job.progress_step > 0:
                # Self-correcting: use this job's own observed rate once it
                # has made progress, rather than trusting the static
                # fallback for the whole run — actual speed varies with
                # system load.
                seconds_per_step = elapsed_seconds / job.progress_step
            else:
                seconds_per_step = config.FALLBACK_SECONDS_PER_STEP
            eta_seconds = remaining_steps * seconds_per_step

    return JobResponse(
        id=job.id,
        status=job.status,
        progress_step=job.progress_step,
        progress_total=job.progress_total,
        phase=phase,
        elapsed_seconds=elapsed_seconds,
        eta_seconds=eta_seconds,
        error=job.error,
        video_url=f"/jobs/{job.id}/video" if job.status == JobStatus.COMPLETED else None,
    )


@app.post("/generate", response_model=JobResponse)
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(config.DEFAULT_NEGATIVE_PROMPT),
    width: int = Form(config.DEFAULT_WIDTH),
    height: int = Form(config.DEFAULT_HEIGHT),
    length: int = Form(config.DEFAULT_LENGTH),
    steps: int = Form(config.DEFAULT_STEPS),
    cfg: float = Form(config.DEFAULT_CFG),
    shift: float = Form(config.DEFAULT_SHIFT),
    image: Optional[UploadFile] = File(None),
):
    start_image_name = None
    if image is not None:
        content = await image.read()
        start_image_name = await comfy_client.upload_image(image.filename or "input.png", content)

    job = job_store.create(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        length=length,
        steps=steps,
        cfg=cfg,
        shift=shift,
        start_image_name=start_image_name,
    )
    return _job_response(job)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_response(job)


@app.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    cancelled = await job_store.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail=f"job is {job.status.value}, cannot cancel")
    return _job_response(job)


@app.get("/jobs/{job_id}/video")
async def get_job_video(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.COMPLETED or job.video_path is None:
        raise HTTPException(status_code=409, detail=f"job is {job.status.value}, not completed")

    video_path = config.COMFYUI_OUTPUT_DIR / job.video_path
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="video file missing on disk")
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)
