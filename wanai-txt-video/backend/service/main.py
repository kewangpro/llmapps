"""FastAPI inference service — wraps ComfyUI behind a stable HTTP contract.

See docs/DESIGN.md "Why a separate FastAPI service instead of calling
ComfyUI directly from Tauri" for the rationale, and docs/IMPLEMENT.md
Phase 1 for scope.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    progress_step: int
    progress_total: int
    error: Optional[str] = None
    video_url: Optional[str] = None


def _job_response(job) -> JobResponse:
    return JobResponse(
        id=job.id,
        status=job.status,
        progress_step=job.progress_step,
        progress_total=job.progress_total,
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
