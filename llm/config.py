from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_ENV_VAR = "LLM_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


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
    path_value = raw.get("path")
    if not path_value:
        raise ValueError("Model configuration must define 'path'")

    path = Path(path_value).expanduser().resolve()
    return ModelConfig(
        path=path,
        host=str(raw.get("host", "0.0.0.0")),
        port=int(raw.get("port", 8001)),
        device_id=int(raw.get("device_id", 0)),
        quantization=raw.get("quantization"),
        gpu_memory_utilization=(
            float(raw["gpu_memory_utilization"]) if raw.get("gpu_memory_utilization") is not None else None
        ),
        max_model_len=(int(raw["max_model_len"]) if raw.get("max_model_len") is not None else None),
        autostart=bool(raw.get("autostart", True)),
    )


def _parse_sampling_config(raw: Dict[str, Any]) -> SamplingConfig:
    size = int(raw.get("sample_size", 20))
    if size <= 0:
        raise ValueError("sample_size must be positive")
    return SamplingConfig(sample_size=size)


def _parse_aggregator_config(raw: Dict[str, Any]) -> AggregatorConfig:
    base_url = str(raw.get("base_url", "http://news-aggregator:8000")).rstrip("/")
    timeout = float(raw.get("timeout_seconds", 15.0))
    return AggregatorConfig(base_url=base_url, timeout_seconds=timeout)


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
