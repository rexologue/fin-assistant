from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional


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
