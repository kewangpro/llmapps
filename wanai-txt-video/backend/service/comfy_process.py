"""Spawns/manages the ComfyUI subprocess and owns its lifecycle.

The app always takes ownership of the ComfyUI instance it uses — spawning
one if none is running, or adopting an already-running one by PID — and
always stops it when the service shuts down. This is a deliberate product
decision (docs/IMPLEMENT.md Phase 3): closing the app must stop everything
it's using, not leave an orphaned inference process consuming memory/GPU.
"""

import asyncio
import logging
import os
import signal
import subprocess
import time

from . import config
from .comfy_client import ComfyClient

logger = logging.getLogger("wanai_service.comfy_process")


def _find_pid_on_port(port: int) -> int | None:
    try:
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    pids = [p for p in out.stdout.split() if p.isdigit()]
    return int(pids[0]) if pids else None


class ComfyProcessManager:
    def __init__(self, comfy: ComfyClient):
        self.comfy = comfy
        self._process: subprocess.Popen | None = None
        self._adopted_pid: int | None = None

    async def ensure_running(self) -> None:
        try:
            await self.comfy.wait_until_ready(timeout=2.0, interval=0.5)
            self._adopted_pid = _find_pid_on_port(config.COMFYUI_PORT)
            logger.info(
                "Adopting already-running ComfyUI instance at %s (pid=%s) — will stop it on shutdown",
                self.comfy.base_url,
                self._adopted_pid,
            )
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
        if self._process is not None:
            logger.info("Stopping ComfyUI subprocess (pid=%s)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            return

        if self._adopted_pid is not None:
            logger.info("Stopping adopted ComfyUI process (pid=%s)", self._adopted_pid)
            try:
                os.kill(self._adopted_pid, signal.SIGTERM)
                deadline = time.time() + 10
                while time.time() < deadline:
                    try:
                        os.kill(self._adopted_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.5)
                else:
                    os.kill(self._adopted_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # already gone
            self._adopted_pid = None
