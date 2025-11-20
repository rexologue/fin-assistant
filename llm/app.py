from __future__ import annotations

import json
import logging
import random
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Sequence

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint

from .vllm_host import VLLMChatClient, VLLMServer
from .config import AppConfig, VLLMServerConfig, load_app_config
from .prompts import (
    build_news_rerank_prompt, 
    parse_news_rerank_response,
    build_news_translate_prompt,
    parse_news_translate_response,
    build_advice_prompt
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("llm_service")


class NewsRequest(BaseModel):
    n: conint(ge=1, le=10) = Field(..., description="Desired number of news items to return")
    top_spend_categories: List[str]
    disliked_titles: List[str] = Field(default_factory=list)


class AdviceRequest(BaseModel):
    earnings: float
    wastes: Dict[str, float]
    wishes: str


class AdviceResponse(BaseModel):
    earnings: float
    wastes: Dict[str, float]
    comment: str


class NewsItemResponse(BaseModel):
    source: str
    title: str
    content: str
    original_url: str
    image_base64: str | None = None


class AggregatorClient:
    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_news(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/news"
        logger.info("Requesting news from aggregator %s", url)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info("Aggregator returned %d items", len(data))
            return data


def enrich_results(
    parsed: Sequence[Dict[str, str]],
    sampled: Sequence[Dict[str, Any]],
) -> List[NewsItemResponse]:
    lookup = {
        (
            str(item.get("name", "")).strip(),
            str(item.get("title", "")).strip(),
            str(item.get("content", "")).strip(),
        ): item
        for item in sampled
    }

    enriched: List[NewsItemResponse] = []
    for entry in parsed:
        key = (entry["name"], entry["title"], entry["content"])
        original = lookup.get(key)

        if not original:
            logger.warning("LLM output entry missing in sampled data: %s", key)

        url_value = ""
        image_base64 = None

        if original:
            url_value = str(original.get("url") or "")
            image_base64 = original.get("image_base64")

        enriched.append(
            NewsItemResponse(
                source=entry["name"],
                title=entry["title"],
                content=entry["content"],
                original_url=url_value,
                image_base64=image_base64,
            )
        )

    logger.info("Enriched %d LLM entries with original metadata", len(enriched))

    return enriched


def fallback_selection(
    sampled: Sequence[Dict[str, Any]],
    limit: int,
) -> List[NewsItemResponse]:
    logger.info("Using fallback selection of up to %d sampled items", limit)
    fallback_items: List[NewsItemResponse] = []

    for item in sampled[:limit]:
        fallback_items.append(
            NewsItemResponse(
                source=str(item.get("name", "")),
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
                original_url=str(item.get("url") or ""),
                image_base64=item.get("image_base64"),
            )
        )

    return fallback_items


def sample_news(items: Sequence[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    if not items:
        return []
    
    population = list(items)

    if len(population) <= sample_size:
        logger.info(
            "Available news count (%d) is less than or equal to sample size (%d); using all",
            len(population),
            sample_size,
        )

        return population
    
    sampled = random.sample(population, sample_size)
    logger.info("Sampled %d news items out of %d", len(sampled), len(population))

    return sampled


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_app_config()

    aggregator_client = AggregatorClient(
        base_url=config.aggregator.base_url,
        timeout=config.aggregator.timeout_seconds,
    )

    vllm_config = VLLMServerConfig(
        model_name=str(config.model.path),
        host=config.model.host,
        port=config.model.port,
        device_id=config.model.device_id,
        quantization=config.model.quantization,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        max_model_len=config.model.max_model_len,
    )

    vllm_server = VLLMServer(vllm_config)
    llm_client = VLLMChatClient(vllm_server.base_url, model_name=str(config.model.path))

    logger.info(
        "Loaded configuration: model=%s, aggregator=%s, sample_size=%d",
        config.model.path,
        config.aggregator.base_url,
        config.sampling.sample_size,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.model.autostart:
            logger.info("Starting vLLM server with model %s", config.model.path)
            try:
                vllm_server.start()
            except Exception:
                logger.exception("Failed to start vLLM server")

        else:
            logger.info("vLLM autostart disabled via configuration")

        try:
            yield
        finally:
            if config.model.autostart:
                logger.info("Stopping vLLM server")
                vllm_server.stop()

    app = FastAPI(title="LLM Service", version="1.0.0", lifespan=lifespan)
    app.state.config = config
    app.state.llm_client = llm_client
    app.state.aggregator_client = aggregator_client

    @app.post("/news", response_model=List[NewsItemResponse])
    async def post_news(payload: NewsRequest) -> List[NewsItemResponse]:
        logger.info(
            "Received /news request for %d items with %d disliked titles",
            payload.n,
            len(payload.disliked_titles),
        )
        try:
            available_news = await aggregator_client.get_news()

        except httpx.HTTPError as exc:
            logger.exception("Failed to fetch news from aggregator")

            raise HTTPException(
                status_code=502,
                detail=(
                    "News aggregator unavailable at "
                    f"{config.aggregator.base_url}: {exc}"
                ),
            ) from exc

        if not available_news:
            logger.info("Aggregator returned no news; responding with empty list")
            return []

        sampled = sample_news(available_news, config.sampling.sample_size)
        logger.info("Sampled %d news items", len(sampled))

        # Start re-ranking
        prompt = build_news_rerank_prompt(sampled, payload.n, payload.top_spend_categories, payload.disliked_titles)

        llm_output: List[NewsItemResponse]
        try:
            response_text = await llm_client.generate(prompt)
            reranked_idx = parse_news_rerank_response(response_text)
            # logger.info("Parsed %d entries from LLM response", len(parsed))

            if not reranked_idx:
                logger.warning("LLM response was empty or could not be parsed; using fallback")
                raise RuntimeError

            else:
                reranked_idx = [i for i in reranked_idx if 0 <= i < config.sampling.sample_size]

                if len(reranked_idx) == 0:
                    logger.warning("LLM response was empty or could not be parsed; using fallback")
                    raise RuntimeError

                reqs = []
                for news_idx in reranked_idx:
                    item = sampled[news_idx]
                    reqs.append(
                        build_news_translate_prompt(item['title'], item['content'])
                    )

                outputs = await llm_client.generate(reqs)
                output_news = []

                for raw_output, idx in zip(outputs, reranked_idx):
                    item = sampled[idx]

                    parsed = parse_news_translate_response(raw_output)
                    if parsed is None:
                        logger.warning(
                            "Failed to parse translate response for index %d; keeping original text",
                            idx,
                        )
                        # Просто используем оригинальный item без модификации
                        output_news.append(item)
                        continue

                    title, content = parsed

                    if title:
                        item['title'] = title
                    if content:
                        item['content'] = content

                    output_news.append(item)


                enriched = enrich_results(output_news, sampled)
                llm_output = enriched[: payload.n]
                logger.info(
                    "LLM pipeline returning %d items after enrichment", len(llm_output)
                )

        except (httpx.HTTPError, RuntimeError, IndexError) as exc:
            logger.exception("LLM inference failed; returning fallback data")
            llm_output = fallback_selection(sampled, payload.n)

        return llm_output

    @app.post("/advice", response_model=AdviceResponse)
    async def post_advice(payload: AdviceRequest) -> AdviceResponse:
        logger.info(
            "Received /advice request with %d waste categories",
            len(payload.wastes),
        )

        prompt = build_advice_prompt(payload.model_dump())

        try:
            response_text = await llm_client.generate(prompt)
        except httpx.HTTPError as exc:
            logger.exception("LLM inference failed for /advice")
            raise HTTPException(
                status_code=502,
                detail="Advice model is unavailable",
            ) from exc

        try:
            parsed_json = json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.exception("LLM returned invalid JSON for /advice")
            raise HTTPException(
                status_code=502,
                detail="Advice model returned invalid JSON",
            ) from exc

        try:
            advice = AdviceResponse.model_validate(parsed_json)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Advice response validation failed")
            raise HTTPException(
                status_code=502,
                detail="Advice model returned unexpected structure",
            ) from exc

        return advice

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    runtime_config: AppConfig = app.state.config
    uvicorn.run(
        app,
        host=runtime_config.app.host,
        port=runtime_config.app.port,
        reload=False,
    )
