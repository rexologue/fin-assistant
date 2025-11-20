# Fin Assistant

Fin Assistant is a two-service platform that gathers financial news and delivers curated, LLM-enhanced results. It is designed to be the single entry point for developers and operators: this README introduces the architecture, how the services talk to each other, how to deploy them (Docker Compose first), and where to find deeper documentation for each component.

## System architecture

```
┌──────────────────┐      ┌──────────────────────────────┐      ┌──────────────────────────┐
│ External RSS/XML │ ---> │ news-aggregator (FastAPI)    │ ---> │ Parquet cache on volume   │
└──────────────────┘      │ - Fetch + parse RSS/Atom     │      └──────────────────────────┘
                          │ - Deduplicate + TTL cleanup  │
                          │ - Expose /news and /titles   │
                          └──────────────┬──────────────┘
                                         │ HTTP JSON
                                         ▼
                          ┌──────────────────────────────┐      ┌──────────────────────────┐
                          │ llm (FastAPI)                │ <--> │ vLLM server (OpenAI API) │
                          │ - Sample cached news         │      │ (autostart optional)     │
                          │ - Build + parse prompts      │      │                          │
                          │ - Expose /news and /advice   │      │                          │
                          └──────────────────────────────┘      └──────────────────────────┘
```

### How the services interact
- The **news-aggregator** fetches feeds defined in `news_aggregator/sources.json`, deduplicates items, and stores them in `news.parquet` inside a configurable cache directory.
- The **llm** service calls the aggregator’s `/news` endpoint, samples items, builds prompts, calls a vLLM server via the OpenAI-compatible API, post-processes responses, and returns curated items. When the model cannot return valid structure, it falls back to the sampled news.
- vLLM can be autostarted by the llm container or run externally. The llm service only requires the OpenAI-compatible `/v1/chat/completions` endpoint to be reachable.

## High-level functionality
- Continuously ingest RSS/Atom financial news.
- Persist news to a Parquet cache with deduplication and retention windows.
- Provide lightweight read endpoints for titles and full news items.
- Offer LLM-backed endpoints to rerank and optionally translate the news, or provide budgeting advice.

## Deploying with Docker Compose (recommended)
1. Copy example configs:
   ```bash
   cp news_aggregator/config.example.yaml news_aggregator/config.yaml
   cp llm/config.example.yaml llm/config.yaml
   ```
2. Prepare a model directory (e.g., `./models/your-model`) that vLLM can load. It will be mounted at `/model`.
3. Start the stack:
   ```bash
   docker compose -f docker-compose.example.yaml up --build
   ```
   - `news-aggregator` listens on `8000` and writes cache data to `./news_aggregator/cache` (mounted to `/cache`).
   - `llm-service` uses `network_mode: host` in the example file so it can reach the aggregator at `localhost:8000` and expose `18300`.
   - Environment variables in `docker-compose.example.yaml` override cache directory, polling periods, sampling size, GPU utilization, and model length limits.

## Basic usage
- Check cached titles:
  ```bash
  curl http://localhost:8000/titles
  ```
- Retrieve cached news records:
  ```bash
  curl http://localhost:8000/news
  ```
- Request curated articles from the LLM service:
  ```bash
  curl -X POST http://localhost:18300/news \
    -H "Content-Type: application/json" \
    -d '{
      "n": 5,
      "top_spend_categories": ["Путешествия", "Инвестиции", "Технологии"],
      "disliked_titles": ["Госдолг", "Нефть"]
    }'
  ```
- Request budgeting advice:
  ```bash
  curl -X POST http://localhost:18300/advice \
    -H "Content-Type: application/json" \
    -d '{
      "earnings": 100000,
      "wastes": {"rent": 40000, "restaurants": 15000},
      "wishes": "Собрать подушку безопасности за 6 месяцев"
    }'
  ```

## Documentation by component
- [news-aggregator README](news_aggregator/README.md) — pipeline, configuration, API, examples, persistence notes, and swagger spec.
- [llm README](llm/README.md) — configuration loading, vLLM initialization, endpoints, prompt lifecycle, examples, and swagger spec.
- Reference Docker Compose wiring: [`docker-compose.example.yaml`](docker-compose.example.yaml).

## Development notes
- Both services are FastAPI apps and can run locally via `uvicorn` (`news_aggregator.app:app` on port 8000, `llm.app:app` on port 18300).
- Configuration files live alongside each service (`config.yaml`, with `config.example.yaml` for defaults). Environment variables can override key settings for containerized deployments.

## API specifications
Each service ships with its own OpenAPI document:
- [`news_aggregator/swagger.yaml`](news_aggregator/swagger.yaml)
- [`llm/swagger.yaml`](llm/swagger.yaml)
These files describe endpoints, schemas, parameters, and example payloads to enable client generation.
