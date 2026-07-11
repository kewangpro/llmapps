"""In-memory job store and background worker for /generate requests.

Generation takes tens of minutes (see docs/IMPLEMENT.md Phase 0 timings),
so every job runs as a background asyncio task — the API returns a job id
immediately and callers poll/watch progress separately. This is the
async/notify UX documented in docs/DESIGN.md's "Data flow" section, not an
optimization; a synchronous /generate would block for 30-40+ minutes.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from . import config, workflow as workflow_builder
from .comfy_client import ComfyClient


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    length: int
    steps: int
    created_at: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.QUEUED
    progress_step: int = 0
    progress_total: int = 0
    comfy_prompt_id: Optional[str] = None
    video_path: Optional[str] = None
    error: Optional[str] = None


class JobStore:
    def __init__(self, comfy: ComfyClient):
        self.comfy = comfy
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def create(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        length: int,
        steps: int,
        cfg: float,
        shift: float,
        start_image_name: str | None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(
            id=job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            length=length,
            steps=steps,
        )
        self._jobs[job_id] = job
        asyncio.create_task(
            self._run(
                job,
                cfg=cfg,
                shift=shift,
                start_image_name=start_image_name,
            )
        )
        return job

    async def _run(
        self,
        job: Job,
        *,
        cfg: float,
        shift: float,
        start_image_name: str | None,
    ) -> None:
        try:
            wf = workflow_builder.build_workflow(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                width=job.width,
                height=job.height,
                length=job.length,
                steps=job.steps,
                cfg=cfg,
                shift=shift,
                seed=uuid.uuid4().int & 0xFFFFFFFF,
                filename_prefix=f"video/job_{job.id}",
                start_image_name=start_image_name,
            )
            job.comfy_prompt_id = await self.comfy.submit(wf)
            job.status = JobStatus.RUNNING
            job.progress_total = job.steps

            async for event in self.comfy.stream_progress(job.comfy_prompt_id):
                if event["type"] == "progress":
                    job.progress_step = event["value"]
                    job.progress_total = event["max"]
                elif event["type"] == "execution_error":
                    job.status = JobStatus.FAILED
                    job.error = str(event["error"])
                    return

            history = await self.comfy.get_history(job.comfy_prompt_id)
            if history is None:
                raise RuntimeError("job finished but no history entry found")
            video_path = self.comfy.output_video_path(history)
            if video_path is None:
                raise RuntimeError("job finished but produced no video output")

            job.video_path = video_path
            job.status = JobStatus.COMPLETED

        except Exception as exc:  # noqa: BLE001 - surface any failure on the job record
            job.status = JobStatus.FAILED
            job.error = str(exc)
