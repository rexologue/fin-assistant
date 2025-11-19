import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime, timedelta

CONFIG_ENV_VAR = "NEWS_AGGREGATOR_CONFIG"

DEFAULT_PASSIVE_MODE_SECONDS = 300
DEFAULT_CACHE_DIR = Path.home() / "news_cache"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_SOURCES_PATH = Path(__file__).with_name("sources.json")

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    name: str
    id: str
    urls: List[str]

@dataclass
class RawFeed:
    source: str          
    name: str
    url: str             # конкретный RSS-URL
    xml: bytes           # сырое содержимое RSS

@dataclass
class NewsItem:
    source: str                         # имя источника
    title: str
    content: str             
    url: str                            # ссылка на реальную новость
    published_at: Optional[datetime]
    image_base64: Optional[str] = None
    guid: Optional[str] = None  


@dataclass
class AppConfig:
    cache_dir: Path = DEFAULT_CACHE_DIR
    period: str = "1d"
    passive_mode_dur: int = DEFAULT_PASSIVE_MODE_SECONDS
    sources_path: Path = Path(__file__).with_name("sources.json")

    @property
    def period_delta(self) -> timedelta:
        return parse_period(self.period)


def parse_period(period: str) -> timedelta:
    if not period:
        raise ValueError("Period string cannot be empty")

    unit = period[-1]
    try:
        value = int(period[:-1])
    except ValueError as exc:
        raise ValueError(f"Unable to parse period value from '{period}'") from exc

    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    if unit == "m":
        return timedelta(days=30 * value)

    raise ValueError(f"Unsupported period unit '{unit}' in '{period}'")



def load_app_config(config_path: Path | None = None) -> AppConfig:
    config_file = config_path

    if config_file is None:
        env_path = os.environ.get(CONFIG_ENV_VAR)

        if env_path:
            config_file = Path(env_path)
        elif DEFAULT_CONFIG_PATH.exists():
            config_file = DEFAULT_CONFIG_PATH
        else:
            raise ValueError("Config path has been set wrong!")

    cfg_dict: Dict[str, str | int] = {}
    if config_file and config_file.exists():
        logger.info("Loading application config from %s", config_file)
        cfg_dict = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    
    cache_dir = cfg_dict.get("cache_dir", DEFAULT_CACHE_DIR)

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir).expanduser().resolve()
    elif isinstance(cache_dir, Path):
        pass
    else:
        raise ValueError("Bad cache dir configuration!")
    
    cache_dir.mkdir(exist_ok=True, parents=True)
    logger.debug("Ensured cache directory exists at %s", cache_dir)

    period = str(cfg_dict.get("period", "1d"))
    logger.debug("Configured aggregation period: %s", period)
    passive = int(cfg_dict.get("passive_mode_dur", DEFAULT_PASSIVE_MODE_SECONDS))
    logger.debug("Configured passive mode duration: %s", passive)

    sources_path = cfg_dict.get("sources_path", DEFAULT_SOURCES_PATH)

    if isinstance(sources_path, str):
        sources_path = Path(sources_path).expanduser().resolve()
    elif isinstance(sources_path, Path):
        if not DEFAULT_SOURCES_PATH.exists():
            raise ValueError("Bad cache dir configuration!")
    else:
        raise ValueError("Bad cache dir configuration!")

    logger.info(
        "Loaded AppConfig(cache_dir=%s, period=%s, passive_mode_dur=%s, sources_path=%s)",
        cache_dir,
        period,
        passive,
        sources_path,
    )

    return AppConfig(
        cache_dir=cache_dir,
        period=period,
        passive_mode_dur=passive,
        sources_path=sources_path,
    )

