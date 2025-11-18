from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging

from .fetcher import Fetcher
from .parsers import parse_rss_http_string
from .config import NewsItem


logger = logging.getLogger(__name__)


class Parser:
    """High level parser that orchestrates fetching and parsing RSS feeds."""

    def __init__(self, sources_path: Path):
        self.fetcher = Fetcher(sources_path=sources_path)

    def parse(self) -> Dict[str, List[NewsItem]]:
        logger.info("Starting parsing cycle")
        raw_feeds = self.fetcher.fetch_all()

        news: Dict[str, List[NewsItem]] = {}

        for feed in raw_feeds:
            news[feed.name] = []
            logger.debug("Parsing feed '%s' (%s)", feed.name, feed.source)

            if feed.source == "finam":
                finam_news = parse_rss_http_string(feed.xml, feed.source)

                for item in finam_news:
                    item.content = item.content.split("...")[0]
                    news[feed.name].append(item)
                logger.debug("Processed %d 'finam' items for %s", len(finam_news), feed.name)
            else:
                parsed_items = parse_rss_http_string(feed.xml, feed.source)
                news[feed.name].extend(parsed_items)
                logger.debug(
                    "Processed %d items for feed '%s'", len(parsed_items), feed.name
                )

        logger.info("Completed parsing cycle for %d feeds", len(raw_feeds))
        return news


