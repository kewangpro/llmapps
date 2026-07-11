"""Thin client for ComfyUI's own HTTP/WebSocket API.

We drive ComfyUI rather than reimplement inference logic — see
docs/DESIGN.md "Why ComfyUI as the inference engine, not raw diffusers".
"""

import asyncio
import json
import uuid
from typing import Any, AsyncIterator

import httpx
import websockets

from . import config


class ComfyClient:
    def __init__(self, base_url: str = config.COMFYUI_BASE_URL, ws_url: str = config.COMFYUI_WS_URL):
        self.base_url = base_url
        self.ws_url = ws_url
        self.client_id = uuid.uuid4().hex

    async def wait_until_ready(self, timeout: float = 120.0, interval: float = 1.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    resp = await client.get(f"{self.base_url}/system_stats", timeout=5.0)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"ComfyUI did not become ready within {timeout}s")
                await asyncio.sleep(interval)

    async def upload_image(self, filename: str, content: bytes) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/upload/image",
                files={"image": (filename, content)},
                data={"overwrite": "true"},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["name"]

    async def submit(self, workflow: dict) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": self.client_id},
                timeout=30.0,
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("node_errors"):
                raise RuntimeError(f"ComfyUI rejected workflow: {body['node_errors']}")
            return body["prompt_id"]

    async def interrupt_all(self) -> None:
        """Stops whatever prompt is currently executing (global interrupt, no prompt_id)."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/interrupt", json={}, timeout=10.0)
            resp.raise_for_status()

    async def clear_queue(self) -> None:
        """Empties the pending queue (does not affect a prompt already executing)."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/queue", json={"clear": True}, timeout=10.0)
            resp.raise_for_status()

    async def get_history(self, prompt_id: str) -> dict | None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/history/{prompt_id}", timeout=10.0)
            resp.raise_for_status()
            body = resp.json()
            return body.get(prompt_id)

    def output_video_path(self, history_entry: dict) -> str | None:
        # SaveVideo reports its output under the "images" key (with a sibling
        # "animated": [true] flag) rather than "videos" or "gifs" — confirmed
        # against a real /history response, not ComfyUI's node source.
        outputs = history_entry.get("outputs", {})
        for node_output in outputs.values():
            for entry in (
                node_output.get("images", [])
                or node_output.get("videos", [])
                or node_output.get("gifs", [])
                or []
            ):
                subfolder = entry.get("subfolder", "")
                filename = entry["filename"]
                return f"{subfolder}/{filename}" if subfolder else filename
        return None

    async def stream_progress(self, prompt_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yields progress events for `prompt_id` until it finishes or errors.

        Event shapes: {"type": "progress", "value": int, "max": int}
                      {"type": "executing", "node": str | None}  (node=None means done)
                      {"type": "execution_error", "error": dict}
        """
        uri = f"{self.ws_url}?clientId={self.client_id}"
        async with websockets.connect(uri, max_size=None) as ws:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue  # binary preview frames, not needed for progress
                msg = json.loads(raw)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "progress" and data.get("prompt_id") == prompt_id:
                    yield {"type": "progress", "value": data["value"], "max": data["max"]}

                elif msg_type == "executing" and data.get("prompt_id") == prompt_id:
                    if data.get("node") is None:
                        yield {"type": "executing", "node": None}
                        return

                elif msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                    yield {"type": "execution_error", "error": data}
                    return
