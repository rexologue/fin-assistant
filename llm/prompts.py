from __future__ import annotations

from typing import Any, Dict, List, Sequence

PROMPT_TEMPLATE = (
    "Ты — интеллектуальный помощник по выбору финансовых новостей для пользователя.\n\n"
    "Тебе передан список новостей, предпочтения пользователя по расходам и список заголовков, "
    "которые ему ранее НЕ понравились.\n\n"
    "СПИСОК НОВОСТЕЙ:\n{news_list}\n\n"
    "ТОП НАПРАВЛЕНИЙ ТРАТ:\n{top_spend_categories}\n\n"
    "ЗАГОЛОВКИ, КОТОРЫЕ НЕ НРАВЯТСЯ:\n{disliked_titles}\n\n"
    "Требования:\n"
    "1) Выбери только релевантные новости с учётом направлений трат.\n"
    "2) Игнорируй новости с заголовками из списка дизлайков.\n"
    "3) Не выводи дубликаты.\n"
    "4) Если новость уже на русском языке, не изменяй её текст.\n"
    "5) Если новость на другом языке, переведи её на русский без искажения фактов и смысла.\n"
    "6) Все ответы должны быть на русском языке.\n\n"
    "Формат ответа: последовательность тегов <srcN>, <titleN>, <contentN> без комментариев.\n"
    "Если ничего не подходит, верни пустую строку."
)


def format_news_blocks(items: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, item in enumerate(items, start=1):
        block = (
            f"[НОВОСТЬ {idx}]\n"
            f"source: {item.get('source', '')}\n"
            f"title: {item.get('title', '')}\n"
            f"content: {item.get('content', '')}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def build_prompt(sampled: Sequence[Dict[str, Any]], top_spend: Sequence[str], disliked: Sequence[str]) -> str:
    news_list = format_news_blocks(sampled)
    top_spend_str = ", ".join(top_spend) if top_spend else "нет данных"
    disliked_str = "\n".join(disliked) if disliked else "нет"
    return PROMPT_TEMPLATE.format(
        news_list=news_list,
        top_spend_categories=top_spend_str,
        disliked_titles=disliked_str,
    )


__all__ = ["PROMPT_TEMPLATE", "build_prompt", "format_news_blocks"]
