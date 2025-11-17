from pathlib import Path

from fetcher import Fetcher
from parsers import parse_rss_http_string
from config import RawFeed, NewsItem

class Parser:
    def __init__(self, sources_path: Path):
        self.fetcher = Fetcher(sources_path=sources_path)

    def parse(self) -> dict[str, list[NewsItem]]:
        raw_feeds = self.fetcher.fetch_all() 

        news: dict[str, list[NewsItem]] = {}

        for feed in raw_feeds:
            news[feed.name] = []

            if feed.source == "finam":
                finam_news = parse_rss_http_string(feed.xml, feed.source)

                for n in finam_news:
                    n.content = n.content.split("...")[0]
                    news[feed.name].append(n)

            else:
                news[feed.name].extend(
                    parse_rss_http_string(feed.xml, feed.source)
                )

        return news

if __name__ == "__main__":
    p = Parser(Path("sources.json"))
    
    dicts = p.parse()

    for name, news in dicts.items():
        try:
            print(news[0])
        except Exception as e:
            print(f"Skip {name} due to the {e}")
        print("\n" + "+"*10 + "\n") 