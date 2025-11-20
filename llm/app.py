from __future__ import annotations

import logging
import random
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Sequence

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint

from .config import AppConfig, VLLMServerConfig, load_app_config
from .prompts import build_prompt
from .vllm_host import VLLMChatClient, VLLMServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("llm_service")


class NewsRequest(BaseModel):
    n: conint(ge=1, le=10) = Field(..., description="Desired number of news items to return")
    top_spend_categories: List[str]
    disliked_titles: List[str] = Field(default_factory=list)


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


TAG_PATTERN = re.compile(
    r"<src(?P<idx>\d+)>(?P<src>.*?)</src(?P=idx)>\s*"
    r"<title(?P=idx)>(?P<title>.*?)</title(?P=idx)>\s*"
    r"<content(?P=idx)>(?P<content>.*?)</content(?P=idx)>",
    re.DOTALL,
)


def parse_llm_response(output: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for match in TAG_PATTERN.finditer(output.strip()):
        results.append(
            {
                "source": match.group("src").strip(),
                "title": match.group("title").strip(),
                "content": match.group("content").strip(),
            }
        )
    return results


def enrich_results(
    parsed: Sequence[Dict[str, str]],
    sampled: Sequence[Dict[str, Any]],
) -> List[NewsItemResponse]:
    lookup = {
        (
            str(item.get("source", "")).strip(),
            str(item.get("title", "")).strip(),
            str(item.get("content", "")).strip(),
        ): item
        for item in sampled
    }
    enriched: List[NewsItemResponse] = []
    for entry in parsed:
        key = (entry["source"], entry["title"], entry["content"])
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
                source=entry["source"],
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
                source=str(item.get("source", "")),
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
        prompt = build_prompt(sampled, payload.top_spend_categories, payload.disliked_titles)
        logger.info("Built prompt using %d sampled items", len(sampled))

        llm_output: List[NewsItemResponse]
        try:
            response_text = await llm_client.generate(prompt)
            parsed = parse_llm_response(response_text)
            logger.info("Parsed %d entries from LLM response", len(parsed))
            if not parsed:
                logger.warning("LLM response was empty or could not be parsed; using fallback")
                llm_output = fallback_selection(sampled, payload.n)
            else:
                enriched = enrich_results(parsed, sampled)
                llm_output = enriched[: payload.n]
                logger.info(
                    "LLM pipeline returning %d items after enrichment", len(llm_output)
                )
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.exception("LLM inference failed; returning fallback data")
            llm_output = fallback_selection(sampled, payload.n)

        return llm_output

    @app.get("/advice")
    async def get_advice() -> Dict[str, str]:
        return {"status": "Not implemented yet"}

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
