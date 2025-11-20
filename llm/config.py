from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

MODEL_PATH_ENV_VAR = "LLM_MODEL_PATH"
MODEL_HOST_ENV_VAR = "LLM_MODEL_HOST"
SAMPLING_SIZE_ENV_VAR = "LLM_SAMPLING_SAMPLE_SIZE"
AGGREGATOR_TIMEOUT_ENV_VAR = "LLM_AGGREGATOR_TIMEOUT_SECONDS"
AGGREGATOR_HOST_ENV_VAR = "LLM_AGGREGATOR_HOST"
AGGREGATOR_PORT_ENV_VAR = "LLM_AGGREGATOR_PORT"
APP_HOST_ENV_VAR = "LLM_APP_HOST"
APP_PORT_ENV_VAR = "LLM_APP_PORT"
GPU_UTILIZATION_ENV_VAR = "LLM_GPU_UTILIZATION"
MAX_MODEL_LENGTH_ENV_VAR = "LLM_MAX_MODEL_LENGTH"


@dataclass
class ModelConfig:
    path: Path
    host: str = "0.0.0.0"
    port: int = 8001
    device_id: int = 0
    quantization: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    autostart: bool = True


@dataclass
class SamplingConfig:
    sample_size: int = 20


@dataclass
class AggregatorConfig:
    base_url: str = "http://news-aggregator:8000"
    timeout_seconds: float = 15.0


@dataclass
class VLLMServerConfig:
    model_name: str
    host: str
    port: int
    timeout_seconds: int = 900
    device_id: int = 0
    quantization: str | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None


@dataclass
class AppRuntimeConfig:
    host: str = "0.0.0.0"
    port: int = 18300


@dataclass
class AppConfig:
    docker: bool
    app: AppRuntimeConfig
    model: ModelConfig
    sampling: SamplingConfig
    aggregator: AggregatorConfig


def _parse_app_runtime_config(raw: Dict[str, Any], docker_enabled: bool) -> AppRuntimeConfig:
    host_value = str(raw.get("host", "0.0.0.0"))
    port_value = int(raw.get("port", 18300))

    if docker_enabled:
        env_host = os.environ.get(APP_HOST_ENV_VAR)
        env_port = os.environ.get(APP_PORT_ENV_VAR)
        if env_host:
            host_value = env_host
        if env_port:
            port_value = int(env_port)

    return AppRuntimeConfig(host=host_value, port=port_value)


def _parse_model_config(raw: Dict[str, Any], docker_enabled: bool) -> ModelConfig:
    path_value: str | Path | None = raw.get("path")
    if docker_enabled:
        env_path = os.environ.get(MODEL_PATH_ENV_VAR)
        if env_path:
            path_value = env_path

    if not path_value:
        raise ValueError("Model configuration must define 'path'")

    host_value = str(raw.get("host", "0.0.0.0"))
    if docker_enabled:
        env_host = os.environ.get(MODEL_HOST_ENV_VAR)
        if env_host:
            host_value = env_host

    port_value = int(raw.get("port", 8001))
    device_id_value = int(raw.get("device_id", 0))
    quantization_value = raw.get("quantization")

    gpu_utilization_value = raw.get("gpu_memory_utilization")
    if docker_enabled and os.environ.get(GPU_UTILIZATION_ENV_VAR) is not None:
        gpu_utilization_value = float(os.environ[GPU_UTILIZATION_ENV_VAR])

    max_model_len_value = raw.get("max_model_len")
    if docker_enabled and os.environ.get(MAX_MODEL_LENGTH_ENV_VAR) is not None:
        max_model_len_value = int(os.environ[MAX_MODEL_LENGTH_ENV_VAR])

    return ModelConfig(
        path=Path(path_value).expanduser().resolve(),
        host=host_value,
        port=port_value,
        device_id=device_id_value,
        quantization=quantization_value,
        gpu_memory_utilization=(
            float(gpu_utilization_value)
            if gpu_utilization_value is not None
            else None
        ),
        max_model_len=(int(max_model_len_value) if max_model_len_value is not None else None),
        autostart=bool(raw.get("autostart", True)),
    )


def _parse_sampling_config(raw: Dict[str, Any], docker_enabled: bool) -> SamplingConfig:
    size_value = raw.get("sample_size", 20)
    if docker_enabled:
        env_size = os.environ.get(SAMPLING_SIZE_ENV_VAR)
        if env_size is not None:
            size_value = env_size

    size = int(size_value)
    if size <= 0:
        raise ValueError("sample_size must be positive")
    return SamplingConfig(sample_size=size)


def _parse_aggregator_config(raw: Dict[str, Any], docker_enabled: bool) -> AggregatorConfig:
    base_url = str(raw.get("base_url", "http://news-aggregator:8000")).rstrip("/")
    timeout = float(raw.get("timeout_seconds", 15.0))

    if docker_enabled:
        host_override = os.environ.get(AGGREGATOR_HOST_ENV_VAR)
        port_override = os.environ.get(AGGREGATOR_PORT_ENV_VAR)
        if host_override or port_override:
            parsed = urlsplit(base_url)
            scheme = parsed.scheme or "http"
            host = host_override or (parsed.hostname or "localhost")
            port = port_override or (str(parsed.port) if parsed.port else "")
            netloc = host
            if port:
                netloc = f"{host}:{int(port)}"
            base_url = urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))

        timeout_override = os.environ.get(AGGREGATOR_TIMEOUT_ENV_VAR)
        if timeout_override is not None:
            timeout = float(timeout_override)

    return AggregatorConfig(base_url=base_url.rstrip("/"), timeout_seconds=timeout)


def load_app_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"LLM configuration file not found at {path}")

    raw_config: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    docker_enabled = bool(raw_config.get("docker", False))

    app_cfg = _parse_app_runtime_config(raw_config.get("app", {}), docker_enabled)
    model_cfg = _parse_model_config(raw_config.get("model", {}), docker_enabled)
    sampling_cfg = _parse_sampling_config(raw_config.get("sampling", {}), docker_enabled)
    aggregator_cfg = _parse_aggregator_config(raw_config.get("aggregator", {}), docker_enabled)

    return AppConfig(
        docker=docker_enabled,
        app=app_cfg,
        model=model_cfg,
        sampling=sampling_cfg,
        aggregator=aggregator_cfg,
    )


__all__ = [
    "AppConfig",
    "AppRuntimeConfig",
    "ModelConfig",
    "SamplingConfig",
    "AggregatorConfig",
    "VLLMServerConfig",
    "load_app_config",
    "DEFAULT_CONFIG_PATH",
]
