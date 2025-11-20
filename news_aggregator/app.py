from __future__ import annotations

import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Iterable, List, Optional

import pandas as pd # type: ignore
from fastapi import FastAPI

from .parser import Parser
from .logging_config import setup_logging
from .config import NewsItem, AppConfig, load_app_config


logger = logging.getLogger(__name__)


class NewsCache:
    columns = [
        "source",
        "name",
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
        logger.debug("Initialized NewsCache at %s", self.cache_file)

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.columns)

    def _read_df(self) -> pd.DataFrame:
        with self._lock:
            if not self.cache_file.exists():
                logger.debug("Cache file %s does not exist; returning empty dataframe", self.cache_file)
                return self._empty_df()

            logger.debug("Reading cache contents from %s", self.cache_file)
            return pd.read_parquet(self.cache_file)

    def _write_df(self, df: pd.DataFrame) -> None:
        with self._lock:
            df.to_parquet(self.cache_file, index=False)
            logger.debug("Persisted %d records to cache %s", len(df), self.cache_file)

    def purge_before(self, cutoff: datetime) -> None:
        df = self._read_df()
        if df.empty or "published_at" not in df:
            logger.debug("Cache purge skipped; nothing to purge")
            return

        df["published_at_dt"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        filtered = df[df["published_at_dt"].isna() | (df["published_at_dt"] >= cutoff)]
        filtered = filtered.drop(columns=["published_at_dt"], errors="ignore")

        if len(filtered) != len(df):
            logger.info(
                "Purged %d outdated items from cache", len(df) - len(filtered)
            )
            self._write_df(filtered)
        else:
            logger.debug("Cache purge resulted in no deletions")

    def append_items(self, items: Iterable[NewsItem]) -> None:
        if not items:
            logger.debug("No new items provided for cache append; skipping")
            return

        new_rows = [self._item_to_record(item) for item in items]
        new_df = pd.DataFrame(new_rows, columns=self.columns)
        existing_df = self._read_df()
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        self._write_df(combined)
        logger.info("Cached %d new items (total=%d)", len(new_rows), len(combined))

    def _item_to_record(self, item: NewsItem) -> Dict[str, object]:
        return {
            "source": item.source,
            "name": item.name,
            "title": item.title,
            "content": item.content,
            "url": item.url,
            "published_at": normalize_datetime(item.published_at),
            "image_base64": item.image_base64,
            "guid": item.guid,
        }

    def clear(self) -> None:
        with self._lock:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info("Cleared cache file at %s", self.cache_file)
            else:
                logger.debug("Cache file %s does not exist; nothing to clear", self.cache_file)

    def get_records(self) -> List[Dict[str, object]]:
        df = self._read_df()

        if df.empty:
            logger.debug("Cache is empty when requesting records")
            return []
        
        return df.to_dict(orient="records")  # type: ignore

    def get_titles(self) -> List[str]:
        return [row["title"] for row in self.get_records()] # type: ignore

    def existing_identities(self) -> set[str]:
        df = self._read_df()

        if df.empty:
            logger.debug("Cache is empty when requesting identities")
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
        self._cache_cleared = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.info("AggregationService is already running")
            return

        if self.config.update and not self._cache_cleared:
            logger.info("Update flag enabled; clearing cache before start")
            self.cache.clear()
            self._cache_cleared = True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="news-worker", daemon=True)
        self._thread.start()
        logger.info("AggregationService background worker started")

    def stop(self) -> None:
        logger.info("Stopping AggregationService background worker")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            logger.info("AggregationService background worker stopped")

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
        logger.debug("Running active cycle with cutoff %s", cutoff.isoformat())
        self.cache.purge_before(cutoff)

        parsed_news = self.parser.parse()
        new_items = self._flatten_news(parsed_news)
        logger.info("Fetched %d total items from parser", len(new_items))
        filtered_items = self._filter_duplicates(new_items, cutoff)
        logger.info("Filtered down to %d new items", len(filtered_items))

        if filtered_items:
            self.cache.append_items(filtered_items)
        else:
            logger.debug("No new items to cache after filtering")

    def _flatten_news(self, parsed: Dict[str, List[NewsItem]]) -> List[NewsItem]:
        items: List[NewsItem] = []

        for source, source_items in parsed.items():
            logger.debug("Flattening %d items from source '%s'", len(source_items), source)
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
                logger.debug(
                    "Skipping item '%s' from source '%s' due to cutoff", item.title, item.source
                )
                continue

            identity = build_identity(item.source, item.title, item.url, item.guid)
            if identity in existing or identity in seen:
                logger.debug("Skipping duplicate item '%s' (%s)", item.title, identity)
                continue

            seen.add(identity)
            filtered.append(item)

        return filtered

    def get_titles(self) -> List[str]:
        return self.cache.get_titles()

    def get_news(self) -> List[Dict[str, object]]:
        logger.debug("Fetching all news records from cache")
        records = self.cache.get_records()

        for record in records:
            published = record.get("published_at")

            if isinstance(published, datetime):
                record["published_at"] = normalize_datetime(published)

        return records


def create_app(config_path: Path | None = None) -> FastAPI:
    config = load_app_config(config_path)
    setup_logging()

    logger.info("Creating FastAPI application with cache at %s", config.cache_dir)
    service = AggregationService(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup
        logger.info("Starting application lifespan; launching service")
        service.start()
        try:
            yield
        finally:
            # shutdown
            logger.info("Stopping application lifespan; shutting down service")
            service.stop()

    app = FastAPI(
        title="News Aggregator",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.config = config
    app.state.service = service

    @app.get("/titles")
    async def get_titles() -> List[str]:
        return service.get_titles()

    @app.get("/news")
    async def get_news() -> List[Dict[str, object]]:
        return service.get_news()

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(
        "news_aggregator.app:app",
        host=app.state.config.host,
        port=app.state.config.port,
        reload=False,
    )
