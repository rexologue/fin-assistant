import json
import logging
from typing import List
from pathlib import Path

import requests

from .config import SourceConfig, RawFeed

DEFAULT_HEADERS = {
    "User-Agent": "MyNewsBot/1.0 (+contact@example.com)"
}

logger = logging.getLogger(__name__)


class Fetcher:
    def __init__(self, timeout: float = 10.0, sources_path: Path | None = None):
        self.timeout = timeout
        self.src_path = sources_path

        if self.src_path is not None:
            self.sources = self.load_sources(self.src_path)
        else:
            self.sources: List[SourceConfig] | None = None

    def load_sources(self, path: Path) -> List[SourceConfig]:
        try:
            logger.info("Loading sources from %s", path)
            data = json.loads(path.read_text(encoding="utf-8"))
            sources = [SourceConfig(**item) for item in data]

            logger.info("Successfully loaded %d sources", len(sources))
            return sources

        except Exception as e:
            logger.exception("Unable to load sources from %s", path)
            raise AssertionError(f"Unable to load sources from {path} due to {e}") from e

    def fetch_all(self, sources: List[SourceConfig] | None = None) -> List[RawFeed]:
        # Если явно не передали, берём то, что загружено из файла
        if sources is None:
            if self.sources is None:
                raise ValueError("No sources provided and self.sources is None")
            sources = self.sources

        result: List[RawFeed] = []

        for src in sources:
            logger.info("Fetching source '%s' with %d URLs", src.name, len(src.urls))

            for url in src.urls:
                logger.debug("Fetching URL %s for source '%s'", url, src.name)

                try:
                    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=self.timeout)
                    resp.raise_for_status()

                except requests.RequestException as exc:
                    logger.warning(
                        "Failed to fetch url=%s for source='%s': %s",
                        url,
                        src.name,
                        exc,
                    )
                    continue

                logger.debug(
                    "Fetched %d bytes from %s (source='%s')",
                    len(resp.content),
                    url,
                    src.name,
                )

                result.append(
                    RawFeed(
                        source=src.id,
                        name=src.name,
                        url=url,
                        xml=resp.content,
                    )
                )

        logger.info("Finished fetching. Total feeds: %d", len(result))
        return result
