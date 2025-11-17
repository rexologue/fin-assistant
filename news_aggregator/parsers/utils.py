from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import re
import base64
import requests
from html import unescape
from typing import List, Optional
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from news_aggregator.config import NewsItem


def _clean_html_to_text(html_str: str) -> str:
    """
    Превращает HTML/HTML-encoded текст (description/content:encoded)
    в обычный плоский текст: снимает экранирование, выкидывает теги,
    схлопывает пробелы.
    """
    if not html_str:
        return ""

    # Сначала снимаем HTML-экранирование (&lt;...&gt; -> <...>)
    s = unescape(html_str)

    # Убираем <script> и <style>
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)

    # Выпиливаем все HTML-теги
    s = re.sub(r"(?s)<[^>]+>", " ", s)

    # Схлопываем пробелы
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _extract_image_url(item: ET.Element) -> Optional[str]:
    """
    Ищет URL изображения внутри item.
    Поддерживаются: media:content, media:thumbnail, enclosure, img в HTML.
    """

    # ----------------------------
    # 1. <media:content>
    # ----------------------------
    for tag in item.findall(".//media:content", {
        "media": "http://search.yahoo.com/mrss/"
    }):
        url = tag.get("url")
        if url and url.startswith("http"):
            return url

    # ----------------------------
    # 2. <media:thumbnail>
    # ----------------------------
    for tag in item.findall(".//media:thumbnail", {
        "media": "http://search.yahoo.com/mrss/"
    }):
        url = tag.get("url")
        if url and url.startswith("http"):
            return url

    # ----------------------------
    # 3. enclosure type="image/*"
    # ----------------------------
    for tag in item.findall("enclosure"):
        if tag.get("type", "").startswith("image"):
            url = tag.get("url")
            if url and url.startswith("http"):
                return url

    # ----------------------------
    # 4. <img src="..."> в description / content:encoded
    # ----------------------------
    html_candidates = [
        item.findtext("description") or "",
        item.findtext(
            "content:encoded",
            default="",
            namespaces={"content": "http://purl.org/rss/1.0/modules/content/"}
        ) or ""
    ]

    for html in html_candidates:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            url = m.group(1)
            if url.startswith("http"):
                return url

    return None


def _download_image_as_base64(url: str) -> Optional[str]:
    """Качает изображение по URL и возвращает base64, иначе None."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            b = r.content
            return base64.b64encode(b).decode("utf-8")
    except Exception:
        return None
    return None


def parse_rss_http_string(xml_text: bytes, source: str) -> List[NewsItem]:
    root = ET.fromstring(xml_text)

    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
    items: List[NewsItem] = []

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()

        guid_raw = item.findtext("guid")
        guid = guid_raw.strip() if isinstance(guid_raw, str) else None

        description = item.findtext("description") or ""
        content_encoded = item.findtext("content:encoded", default="", namespaces=ns)
        raw_html = content_encoded or description
        content = _clean_html_to_text(raw_html)

        pub_raw = item.findtext("pubDate")
        if pub_raw:
            try:
                published_at = parsedate_to_datetime(pub_raw.strip())
            except Exception:
                published_at = None
        else:
            published_at = None

        # ----------------------------
        # ИЗОБРАЖЕНИЯ
        # ----------------------------
        img_url = _extract_image_url(item)
        img_b64 = _download_image_as_base64(img_url) if img_url else None

        news_item = NewsItem(
            source=source,
            title=title,
            content=content,
            url=link,
            published_at=published_at,
            image_base64=img_b64,
            guid=guid,
        )

        items.append(news_item)

    return items
