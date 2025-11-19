from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_ENV_VAR = "LLM_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

MODEL_PATH_ENV_VAR = "LLM_MODEL_PATH"
MODEL_HOST_ENV_VAR = "LLM_MODEL_HOST"
MODEL_DEVICE_ID_ENV_VAR = "LLM_MODEL_DEVICE_ID"
SAMPLING_SIZE_ENV_VAR = "LLM_SAMPLING_SAMPLE_SIZE"
AGGREGATOR_TIMEOUT_ENV_VAR = "LLM_AGGREGATOR_TIMEOUT_SECONDS"
AGGREGATOR_HOST_ENV_VAR = "LLM_AGGREGATOR_HOST"
AGGREGATOR_PORT_ENV_VAR = "LLM_AGGREGATOR_PORT"
AGGREGATOR_SCHEME_ENV_VAR = "LLM_AGGREGATOR_SCHEME"


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
class AppConfig:
    model: ModelConfig
    sampling: SamplingConfig
    aggregator: AggregatorConfig


def _parse_model_config(raw: Dict[str, Any]) -> ModelConfig:
    path_value = os.environ.get(MODEL_PATH_ENV_VAR, raw.get("path"))
    if not path_value:
        raise ValueError("Model configuration must define 'path'")

    path = Path(path_value).expanduser().resolve()
    host_value = os.environ.get(MODEL_HOST_ENV_VAR, raw.get("host", "0.0.0.0"))
    port_value = int(raw.get("port", 8001))
    device_id_value = _resolve_device_id(raw.get("device_id", 0))

    return ModelConfig(
        path=path,
        host=str(host_value),
        port=port_value,
        device_id=device_id_value,
        quantization=raw.get("quantization"),
        gpu_memory_utilization=(
            float(raw["gpu_memory_utilization"]) if raw.get("gpu_memory_utilization") is not None else None
        ),
        max_model_len=(int(raw["max_model_len"]) if raw.get("max_model_len") is not None else None),
        autostart=bool(raw.get("autostart", True)),
    )


def _parse_sampling_config(raw: Dict[str, Any]) -> SamplingConfig:
    override = os.environ.get(SAMPLING_SIZE_ENV_VAR)
    size = int(override) if override is not None else int(raw.get("sample_size", 20))
    if size <= 0:
        raise ValueError("sample_size must be positive")
    return SamplingConfig(sample_size=size)


def _parse_aggregator_config(raw: Dict[str, Any]) -> AggregatorConfig:
    base_url = str(raw.get("base_url", "http://news-aggregator:8000")).rstrip("/")
    host_override = os.environ.get(AGGREGATOR_HOST_ENV_VAR)
    port_override = os.environ.get(AGGREGATOR_PORT_ENV_VAR)
    scheme_override = os.environ.get(AGGREGATOR_SCHEME_ENV_VAR, "http")

    if host_override or port_override:
        host_value = host_override or "localhost"
        base_url = f"{scheme_override}://{host_value}"
        if port_override:
            base_url = f"{base_url}:{int(port_override)}"

    timeout_override = os.environ.get(AGGREGATOR_TIMEOUT_ENV_VAR)
    timeout = float(timeout_override) if timeout_override is not None else float(raw.get("timeout_seconds", 15.0))
    return AggregatorConfig(base_url=base_url.rstrip("/"), timeout_seconds=timeout)


def _resolve_device_id(configured: Any) -> int:
    explicit_env = os.environ.get(MODEL_DEVICE_ID_ENV_VAR)
    if explicit_env is not None:
        return int(explicit_env)

    for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue

        first = env_value.split(",")[0].strip()
        if not first or first.lower() in {"none", "n/a"}:
            continue
        if first.lower() == "all":
            return 0
        try:
            return int(first)
        except ValueError:
            continue

    return int(configured)


def load_app_config(config_path: Path | None = None) -> AppConfig:
    path = config_path
    if path is None:
        env_path = os.environ.get(CONFIG_ENV_VAR)
        if env_path:
            path = Path(env_path)
        elif DEFAULT_CONFIG_PATH.exists():
            path = DEFAULT_CONFIG_PATH
        else:
            raise FileNotFoundError("LLM configuration file was not found")

    if not path.exists():
        raise FileNotFoundError(f"LLM configuration file not found at {path}")

    raw_config: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    model_cfg = _parse_model_config(raw_config.get("model", {}))
    sampling_cfg = _parse_sampling_config(raw_config.get("sampling", {}))
    aggregator_cfg = _parse_aggregator_config(raw_config.get("aggregator", {}))

    return AppConfig(
        model=model_cfg,
        sampling=sampling_cfg,
        aggregator=aggregator_cfg,
    )


__all__ = ["AppConfig", "ModelConfig", "SamplingConfig", "AggregatorConfig", "load_app_config"]
