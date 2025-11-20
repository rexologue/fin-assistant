from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Sequence, Tuple, Optional

NEWS_PROMPT = (
    "Ты — ассистент, который помогает выбрать наиболее полезные финансовые новости для пользователя.\n\n"
    "Тебе переданы:\n"
    "1) список новостей с индексами от 0 до N-1;\n"
    "2) направления трат пользователя;\n"
    "3) заголовки новостей, которые пользователю не понравились;\n"
    "4) число новостей, которые нужно порекомендовать.\n\n"
    "СПИСОК НОВОСТЕЙ (формат: индекс, заголовок, текст):\n{news_list}\n\n"
    "ТОП НАПРАВЛЕНИЙ ТРАТ:\n{top_spend_categories}\n\n"
    "НЕПОНРАВИВШИЕСЯ ЗАГОЛОВКИ:\n{disliked_titles}\n\n"
    "НУЖНО ПОРЕКОМЕНДОВАТЬ: {n_recommendations} новостей.\n\n"
    "Требования к выбору:\n"
    "1) Выбирай только те новости, которые тематически связаны с направлениями трат пользователя.\n"
    "2) Не выбирай новости, которые по смыслу похожи на любой из непонравившихся заголовков "
    "(учитывай не только точное совпадение текста, но и близость тематики и смысла).\n"
    "3) Не выбирай дубликаты и почти идентичные новости (одна и та же новость из разных источников).\n"
    "4) Если подходящих новостей меньше, чем {n_recommendations}, выбери столько, сколько есть, "
    "и не заполняй оставшиеся позиции.\n\n"
    "Формат ответа:\n"
    "- Верни ТОЛЬКО список индексов выбранных новостей в порядке уменьшения релевантности внутри тэгов <out> </out>.\n"
    "- Индексы должны соответствовать индексам из исходного списка новостей.\n"
    "- Не добавляй никаких комментариев, текста, пояснений или других символов.\n"
    "- Разделяй индексы ПРОБЕЛАМИ (пример: <out>0 3 5</out>).\n"
    "Если ни одна новость не подходит, верни пустую строку."
)

NEWS_TRANSLATION_PROMPT = (
    "Ты — переводчик финансовых новостей на русский язык.\n\n"
    "Тебе дан один заголовок и одно содержание новости.\n\n"
    "ЗАГОЛОВОК:\n{title}\n\n"
    "СОДЕРЖАНИЕ:\n{content}\n\n"
    "Твоя задача:\n"
    "1) Если заголовок и/или содержание уже на русском языке — НЕ переводить, а просто аккуратно переписать.\n"
    "2) Если текст не на русском — переведи его на русский без искажения смысла и без добавления новых фактов.\n"
    "3) Сохраняй числа, имена людей, компаний, валюты и другие важные сущности.\n\n"
    "Формат ответа (СТРОГО соблюдай):\n"
    "<title>здесь должен быть итоговый заголовок на русском</title>\n"
    "<content>здесь должно быть итоговое содержание новости на русском</content>\n"
    "Не добавляй никаких комментариев, пояснений или лишних символов кроме этих тегов."
)

BUDGET_ADVICE_PROMPT = (
    "Ты — финансовый ассистент. Тебе передан JSON-объект с доходом пользователя, "
    "его текущими расходами по категориям и пожеланиями по изменению бюджета.\n\n"
    "ВХОДНОЙ JSON:\n"
    "{user_data}\n\n"
    "Структура входа:\n"
    "{\n"
    '  "earnings": <число — общий месячный доход>,\n'
    '  "wastes": {\n'
    '    "<категория_1>": <сумма>,\n'
    '    "<категория_2>": <сумма>,\n'
    '    ...\n'
    "  },\n"
    '  "wishes": "<словесное пожелание пользователя по изменению бюджета>"\n'
    "}\n\n"
    "Твоя задача:\n"
    "1) Проанализировать доход (earnings), текущие траты (wastes) и пожелания (wishes).\n"
    "2) Предложить, как можно перераспределить суммы по всем существующим категориям в wastes,\n"
    "   чтобы лучше соответствовать wishes.\n"
    "3) Суммарные расходы по wastes НЕ должны превышать earnings. Допускается, что часть earnings останется нераспределённой.\n"
    "4) Не добавляй новые категории расходов и не удаляй существующие — изменяй только их числовые значения.\n"
    "5) Значения всех категорий должны быть неотрицательными числами.\n"
    "6) Поле wishes во входных данных нужно заменить на поле comment, где ты кратко объяснишь свою логику\n"
    "   и дашь совет по управлению бюджетом (на русском языке).\n\n"
    "Формат ОТВЕТА (СТРОГО):\n"
    "- Верни ТОЛЬКО один валидный JSON-объект БЕЗ какого-либо пояснительного текста.\n"
    "- Структура ответа должна быть такой же, как во входе, но с заменой wishes -> comment и\n"
    "  обновлёнными значениями в wastes. Пример структуры:\n"
    "{\n"
    '  "earnings": <то же число, что во входе>,\n'
    '  "wastes": {\n'
    '    "<категория_1>": <новая сумма>,\n'
    '    "<категория_2>": <новая сумма>,\n'
    '    ...\n'
    "  },\n"
    '  "comment": "<твой совет и объяснение на русском>"\n'
    "}\n\n"
    "Важное требование: не добавляй никаких лишних полей, комментариев, текста вне JSON."
)



def format_news_blocks(items: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, item in enumerate(items):
        block = (
            f"[НОВОСТЬ {idx}]\n"
            f"source: {item.get('source', '')}\n"
            f"title: {item.get('title', '')}\n"
            f"content: {item.get('content', '')}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def build_news_rerank_prompt(sampled: Sequence[Dict[str, Any]], n: int, top_spend: Sequence[str], disliked: Sequence[str]) -> str:
    news_list = format_news_blocks(sampled)
    top_spend_str = ", ".join(top_spend) if top_spend else "нет данных"
    disliked_str = "\n".join(disliked) if disliked else "нет"

    return NEWS_PROMPT.format(
        news_list=news_list,
        n_recommendations=n,
        top_spend_categories=top_spend_str,
        disliked_titles=disliked_str,
    )

def build_news_translate_prompt(title: str, content: str) -> str:
    return NEWS_TRANSLATION_PROMPT.format(
        title=title,
        content=content
    )

def build_advice_prompt(user_data: dict) -> str:
    pretty = json.dumps(user_data, ensure_ascii=False, indent=2)
    return BUDGET_ADVICE_PROMPT.replace("{user_data}", pretty)

def parse_news_rerank_response(output: str) -> Optional[List[int]]:
    # Находим содержимое внутри <out>...</out>
    m = re.search(r"<out>(.*?)</out>", output, flags=re.DOTALL)
    if not m:
        return None

    inner = m.group(1)
    tokens = re.findall(r"\b\d+\b", inner)

    if not tokens:
        return None

    results = [int(t) for t in tokens]

    return results if results else None

def parse_news_translate_response(output: str) -> Optional[Tuple[str, str]]:
    # Ищем строго одинарные контейнеры
    m_title = re.search(r"<title>(.*?)</title>", output, flags=re.DOTALL)
    m_content = re.search(r"<content>(.*?)</content>", output, flags=re.DOTALL)

    if not m_title or not m_content:
        return None

    title = m_title.group(1).strip()
    content = m_content.group(1).strip()

    # Пустые строки считаем невалидными
    if not title or not content:
        return None

    return title, content

__all__ = [
    "NEWS_PROMPT",
    "NEWS_TRANSLATION_PROMPT",
    "format_news_blocks",
    "build_news_rerank_prompt",
    "build_news_translate_prompt",
    "build_advice_prompt",
    "parse_news_rerank_response",
    "parse_news_translate_response"
]
