from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .fetcher import Fetcher
from .parsers import parse_rss_http_string
from .config import NewsItem


class Parser:
    """High level parser that orchestrates fetching and parsing RSS feeds."""

    def __init__(self, sources_path: Path):
        self.fetcher = Fetcher(sources_path=sources_path)

    def parse(self) -> Dict[str, List[NewsItem]]:
        raw_feeds = self.fetcher.fetch_all()

        news: Dict[str, List[NewsItem]] = {}

        for feed in raw_feeds:
            news[feed.name] = []

            if feed.source == "finam":
                finam_news = parse_rss_http_string(feed.xml, feed.source)

                for item in finam_news:
                    item.content = item.content.split("...")[0]
                    news[feed.name].append(item)
            else:
                news[feed.name].extend(
                    parse_rss_http_string(feed.xml, feed.source)
                )

        return news


