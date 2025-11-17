from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI

from .config import NewsItem
from .logging_config import setup_logging
from .parser import Parser

CONFIG_ENV_VAR = "NEWS_AGGREGATOR_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_CACHE_DIR = Path.home() / "news_cache"
DEFAULT_PASSIVE_MODE_SECONDS = 300


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


def _expand_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value is None:
        return None
    return Path(os.path.expanduser(path_value)).resolve()


def load_app_config(config_path: Path | None = None) -> AppConfig:
    config_file = config_path
    if config_file is None:
        env_path = os.environ.get(CONFIG_ENV_VAR)
        if env_path:
            config_file = Path(env_path)
        elif DEFAULT_CONFIG_PATH.exists():
            config_file = DEFAULT_CONFIG_PATH

    cfg_dict: Dict[str, object] = {}
    if config_file and config_file.exists():
        cfg_dict = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    cache_dir = _expand_path(str(cfg_dict.get("cache_dir"))) if cfg_dict.get("cache_dir") else DEFAULT_CACHE_DIR
    period = str(cfg_dict.get("period", "1d"))
    passive = int(cfg_dict.get("passive_mode_dur", DEFAULT_PASSIVE_MODE_SECONDS))

    sources_path_cfg = cfg_dict.get("sources_path")
    sources_path = _expand_path(str(sources_path_cfg)) if sources_path_cfg else Path(__file__).with_name("sources.json")

    return AppConfig(
        cache_dir=cache_dir,
        period=period,
        passive_mode_dur=passive,
        sources_path=sources_path,
    )


class NewsCache:
    columns = [
        "source",
        "title",
        "content",
        "url",
        "published_at",
        "image_base64",
        "guid",
    ]

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "news.parquet"
        self._lock = threading.Lock()

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.columns)

    def _read_df(self) -> pd.DataFrame:
        with self._lock:
            if not self.cache_file.exists():
                return self._empty_df()
            return pd.read_parquet(self.cache_file)

    def _write_df(self, df: pd.DataFrame) -> None:
        with self._lock:
            df.to_parquet(self.cache_file, index=False)

    def purge_before(self, cutoff: datetime) -> None:
        df = self._read_df()
        if df.empty or "published_at" not in df:
            return

        df["published_at_dt"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        filtered = df[df["published_at_dt"].isna() | (df["published_at_dt"] >= cutoff)]
        filtered = filtered.drop(columns=["published_at_dt"], errors="ignore")
        if len(filtered) != len(df):
            self._write_df(filtered)

    def append_items(self, items: Iterable[NewsItem]) -> None:
        if not items:
            return

        new_rows = [self._item_to_record(item) for item in items]
        new_df = pd.DataFrame(new_rows, columns=self.columns)
        existing_df = self._read_df()
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        self._write_df(combined)

    def _item_to_record(self, item: NewsItem) -> Dict[str, object]:
        return {
            "source": item.source,
            "title": item.title,
            "content": item.content,
            "url": item.url,
            "published_at": normalize_datetime(item.published_at),
            "image_base64": item.image_base64,
            "guid": item.guid,
        }

    def get_records(self) -> List[Dict[str, object]]:
        df = self._read_df()
        if df.empty:
            return []
        return df.to_dict(orient="records")

    def get_titles(self) -> List[str]:
        return [row["title"] for row in self.get_records()]

    def existing_identities(self) -> set[str]:
        df = self._read_df()
        if df.empty:
            return set()
        identities: set[str] = set()
        for _, row in df.iterrows():
            identities.add(
                build_identity(
                    source=str(row.get("source")),
                    title=str(row.get("title")),
                    url=_coerce_optional_str(row.get("url")),
                    guid=_coerce_optional_str(row.get("guid")),
                )
            )
        return identities


def normalize_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def build_identity(source: str, title: str, url: Optional[str], guid: Optional[str]) -> str:
    if guid:
        return guid
    if url:
        return url
    return f"{source}:{title}"


def _coerce_optional_str(value: object) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None


class AggregationService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.cache = NewsCache(config.cache_dir)
        self.parser = Parser(config.sources_path)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="news-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_active_cycle()
            except Exception:  # pragma: no cover - background thread safety
                import logging

                logging.exception("Unexpected error in aggregation worker")

            if self._stop_event.wait(self.config.passive_mode_dur):
                break

    def _run_active_cycle(self) -> None:
        cutoff = datetime.now(timezone.utc) - self.config.period_delta
        self.cache.purge_before(cutoff)

        parsed_news = self.parser.parse()
        new_items = self._flatten_news(parsed_news)
        filtered_items = self._filter_duplicates(new_items, cutoff)
        if filtered_items:
            self.cache.append_items(filtered_items)

    def _flatten_news(self, parsed: Dict[str, List[NewsItem]]) -> List[NewsItem]:
        items: List[NewsItem] = []
        for source_items in parsed.values():
            items.extend(source_items)
        return items

    def _filter_duplicates(
        self, items: List[NewsItem], cutoff: datetime
    ) -> List[NewsItem]:
        existing = self.cache.existing_identities()
        seen: set[str] = set()
        filtered: List[NewsItem] = []
        for item in items:
            if item.published_at is not None and item.published_at.tzinfo is None:
                item.published_at = item.published_at.replace(tzinfo=timezone.utc)
            if item.published_at and item.published_at < cutoff:
                continue
            identity = build_identity(item.source, item.title, item.url, item.guid)
            if identity in existing or identity in seen:
                continue
            seen.add(identity)
            filtered.append(item)
        return filtered

    def get_titles(self) -> List[str]:
        return self.cache.get_titles()

    def get_news(self) -> List[Dict[str, object]]:
        records = self.cache.get_records()
        for record in records:
            published = record.get("published_at")
            if isinstance(published, datetime):
                record["published_at"] = normalize_datetime(published)
        return records


def create_app(config_path: Path | None = None) -> FastAPI:
    config = load_app_config(config_path)
    setup_logging()

    app = FastAPI(title="News Aggregator", version="1.0.0")
    service = AggregationService(config)

    @app.on_event("startup")
    async def startup_event() -> None:  # pragma: no cover - FastAPI hook
        service.start()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:  # pragma: no cover - FastAPI hook
        service.stop()

    @app.get("/titles")
    async def get_titles() -> List[str]:
        return service.get_titles()

    @app.get("/news")
    async def get_news() -> List[Dict[str, object]]:
        return service.get_news()

    app.state.config = config
    app.state.service = service
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("news_aggregator.app:app", host="0.0.0.0", port=8000, reload=False)
