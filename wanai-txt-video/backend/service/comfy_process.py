"""Spawns/manages the ComfyUI subprocess if one isn't already running.

If a ComfyUI instance is already listening on the configured port (e.g.
launched manually during development), we reuse it and never touch its
lifecycle — we only start/stop a process we ourselves spawned.
"""

import asyncio
import logging
import subprocess

from . import config
from .comfy_client import ComfyClient

logger = logging.getLogger("wanai_service.comfy_process")


class ComfyProcessManager:
    def __init__(self, comfy: ComfyClient):
        self.comfy = comfy
        self._process: subprocess.Popen | None = None

    async def ensure_running(self) -> None:
        try:
            await self.comfy.wait_until_ready(timeout=2.0, interval=0.5)
            logger.info("Reusing already-running ComfyUI instance at %s", self.comfy.base_url)
            return
        except TimeoutError:
            pass

        logger.info("Starting ComfyUI subprocess...")
        self._process = subprocess.Popen(
            [
                str(config.COMFYUI_PYTHON),
                "main.py",
                "--force-fp16",
                "--listen",
                config.COMFYUI_HOST,
                "--port",
                str(config.COMFYUI_PORT),
            ],
            cwd=str(config.COMFYUI_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            await self.comfy.wait_until_ready(timeout=180.0)
        except TimeoutError:
            self.stop()
            raise
        logger.info("ComfyUI ready (pid=%s)", self._process.pid)

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping ComfyUI subprocess (pid=%s)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
