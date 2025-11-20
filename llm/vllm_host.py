"""Utilities for launching and monitoring the local vLLM server."""

from __future__ import annotations

import os
import time
import asyncio
import logging
import subprocess
from shutil import which
from typing import List, Sequence, Optional, overload

import httpx

from .config import VLLMServerConfig

logger = logging.getLogger(__name__)


def _has_nvidia_gpu() -> bool:
    """Return True if a usable NVIDIA GPU appears to be available."""

    nvidia_smi = which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi, "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
    except Exception:
        return False

    return result.returncode == 0


class VLLMServer:
    """Manage a vLLM server process lifecycle."""

    def __init__(self, config: VLLMServerConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen[str]] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def start(self) -> None:
        if self._process is not None:
            return

        if not _has_nvidia_gpu():
            raise RuntimeError(
                "vLLM requires an NVIDIA GPU but none was detected. "
                "Expose a GPU to the container before starting the server."
            )

        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model_name,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]

        if self.config.quantization:
            command.extend(["--quantization", self.config.quantization])

        if self.config.gpu_memory_utilization is not None:
            command.extend(
                [
                    "--gpu-memory-utilization",
                    str(self.config.gpu_memory_utilization),
                ]
            )

        if self.config.max_model_len is not None:
            command.extend(["--max-model-len", str(self.config.max_model_len)])

        env = os.environ.copy()
        # env.setdefault("VLLM_USE_V1", "0")
        # env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # env.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.device_id))

        self._process = subprocess.Popen(command, env=env)

        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._process is None:
            return

        self._process.terminate()

        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
        finally:
            self._process = None

    def _wait_until_ready(self) -> None:
        start = time.time()
        timeout = self.config.timeout_seconds

        with httpx.Client() as client:
            while True:
                if self._process and self._process.poll() is not None:
                    raise RuntimeError("vLLM server exited unexpectedly while starting")

                try:
                    response = client.get(f"{self.base_url}/health", timeout=10)
                    if response.status_code == 200:
                        return

                except Exception:
                    pass

                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for vLLM server to become ready")

                time.sleep(2)


class VLLMChatClient:
    def __init__(self, base_url: str, model_name: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    # --- type-level overloads для mypy/pyright ---

    @overload
    async def generate(self, prompt: str) -> str:
        ...

    @overload
    async def generate(self, prompt: Sequence[str]) -> List[str]:
        ...

    # --- реальная реализация ---

    async def generate(self, prompt: str | Sequence[str]) -> str | List[str]:
        """
        Если передан str -> вернёт str.
        Если передан список строк -> вернёт список строк той же длины.
        """
        if isinstance(prompt, str):
            return await self._generate_single(prompt)

        prompts = list(prompt)
        return await self._generate_batch(prompts)

    # --- внутренние вспомогательные методы ---

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

    def _parse_response_content(self, data: dict) -> str:
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not contain any choices")

        message = choices[0].get("message", {})
        content = message.get("content")

        if not isinstance(content, str):
            raise RuntimeError("LLM response did not include textual content")

        return content.strip()

    async def _generate_single(self, prompt: str) -> str:
        logger.info(
            "Sending prompt to LLM model %s (chars=%d)",
            self.model_name,
            len(prompt),
        )

        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(prompt)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            content = self._parse_response_content(response.json())

        logger.info("Received LLM response (%d bytes)", len(content))
        return content

    async def _generate_batch(self, prompts: Sequence[str]) -> List[str]:
        logger.info(
            "Sending batch of %d prompts to LLM model %s",
            len(prompts),
            self.model_name,
        )

        url = f"{self.base_url}/v1/chat/completions"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._post_and_parse(client, url, prompt)
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)

        logger.info(
            "Received batch LLM responses (bytes per item: %s)",
            [len(r) for r in results],
        )
        return list(results)

    async def _post_and_parse(
        self,
        client: httpx.AsyncClient,
        url: str,
        prompt: str,
    ) -> str:
        payload = self._build_payload(prompt)
        response = await client.post(url, json=payload)
        response.raise_for_status()
        content = self._parse_response_content(response.json())
        return content


__all__ = ["VLLMChatClient", "VLLMServer", "VLLMServerConfig"]
