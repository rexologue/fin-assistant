# Fin Assistant Platform

Fin Assistant combines two FastAPI-based services:

- **news_aggregator** – continuously fetches and normalizes financial news from RSS feeds, caching the results locally.
- **llm** – samples cached news, builds a prompt with user preferences, calls an LLM (vLLM-compatible) to select and optionally translate the best items, and exposes them through a simplified API.

The services can run independently or together via Docker Compose. The following sections describe the architecture, configuration, and example API interactions for both apps.

## High-level architecture

```
┌─────────────────────┐       ┌────────────────────────────────┐       ┌───────────────────────────┐
│  External RSS feeds │ ====> │ news_aggregator (FastAPI)      │ ====> │ Parquet cache (news.parquet│
└─────────────────────┘       │ - Fetch + parse RSS            │       └───────────────────────────┘
                              │ - Deduplicate + TTL filtering  │
                              │ - Expose /news, /titles        │
                              └──────────────┬─────────────────┘
                                             │ HTTP JSON
                                             ▼
                              ┌────────────────────────────────┐        ┌─────────────────────────┐
                              │ llm (FastAPI)                 │ <====> │ vLLM server (OpenAI API)│
                              │ - Sample cached news          │        │ (autostart optional)    │
                              │ - Build Russian prompt        │        │                         │
                              │ - Parse + enrich LLM output   │
                              │ - Expose /news                │
                              └────────────────────────────────┘
```

### Service responsibilities

- **news_aggregator**
  - Loads RSS sources from `news_aggregator/sources.json`.
  - `Fetcher` downloads feeds, `Parser` converts them into structured `NewsItem` objects, and `AggregationService` writes them to a Parquet cache (`cache_dir/news.parquet`).
  - Items older than the configured `period` are purged every active cycle; duplicate detection uses source/title/url/guid identity keys.
  - Provides read-only endpoints `/news` (full records) and `/titles` (headlines only).

- **llm**
  - Calls the aggregator (`AggregatorClient`) to fetch cached news, samples a subset (`sample_news`), and builds a Russian-language prompt with the user’s top spending categories and disliked titles.
  - Sends the prompt to a vLLM server through the OpenAI-compatible `/v1/chat/completions` API (`VLLMChatClient`).
  - Parses structured tags from the response, enriches entries with original URLs/images, and falls back to the sampled items on failure.
  - Exposes `/news` to request curated articles; `/advice` is a placeholder.

## Running the stack

### Docker Compose (recommended)

1. Copy example configs: `cp news_aggregator/config.example.yaml news_aggregator/config.yaml` and `cp llm/config.example.yaml llm/config.yaml`.
2. Provide an LLM model directory under `./models` (mounted at `/model` for vLLM).
3. Start both services:

```bash
docker compose -f docker-compose.example.yaml up --build
```

Environment variables in `docker-compose.example.yaml` allow overriding cache location, polling periods, sampling size, and LLM memory limits.

### Local development

- **Aggregator**: `uvicorn news_aggregator.app:app --host 0.0.0.0 --port 8000`
- **LLM service**: ensure `llm/config.yaml` points to a running vLLM server (or enable `autostart`), then `uvicorn llm.app:app --host 0.0.0.0 --port 18300`.

## Configuration highlights

### Aggregator (`news_aggregator/config.yaml`)

- `cache_dir`: where `news.parquet` is written (default `~/news_cache`).
- `period`: retention window (e.g., `1d`, `12h`, `2w`).
- `passive_mode_dur`: sleep between background cycles (seconds).
- `sources_path`: optional override for the RSS list JSON.

Key env overrides: `NEWS_AGGREGATOR_CONFIG`, `NEWS_AGGREGATOR_CACHE_DIR`, `NEWS_AGGREGATOR_PERIOD`, `NEWS_AGGREGATOR_PASSIVE_MODE_DUR`.

### LLM service (`llm/config.yaml`)

- `model`: vLLM host/port, path, quantization, GPU utilization, `max_model_len`, and `autostart` toggle.
- `sampling.sample_size`: how many cached news items to include in each prompt.
- `aggregator.base_url` and `timeout_seconds`: where to reach the news service.
- `app.host`/`app.port`: FastAPI bind address.

Key env overrides (when `docker: true`): `LLM_MODEL_PATH`, `LLM_MODEL_HOST`, `LLM_GPU_UTILIZATION`, `LLM_MAX_MODEL_LENGTH`, `LLM_SAMPLING_SAMPLE_SIZE`, `LLM_AGGREGATOR_HOST`, `LLM_AGGREGATOR_PORT`, `LLM_AGGREGATOR_TIMEOUT_SECONDS`, `LLM_APP_HOST`, `LLM_APP_PORT`.

## API usage examples

### news_aggregator

- **List titles**

```bash
curl http://localhost:8000/titles
```

_Response (truncated):_
```json
[
  "ФРС США повысила ставку...",
  "ЦБ РФ опубликовал отчет..."
]
```

- **Fetch full news records**

```bash
curl http://localhost:8000/news
```

Each record includes `source`, `title`, `content`, `url`, `published_at` (ISO-8601), optional `image_base64`, and `guid`.

### llm

Request curated articles tailored to user spending preferences and disliked titles. The service samples cached news and asks the LLM to select or translate entries into Russian.

```bash
curl -X POST http://localhost:18300/news \
  -H "Content-Type: application/json" \
  -d '{
    "n": 5,
    "top_spend_categories": ["Путешествия", "Инвестиции", "Технологии"],
    "disliked_titles": ["Госдолг", "Нефть"]
  }'
```

_Response example (LLM success path):_
```json
[
  {
    "source": "Fed",
    "title": "ФРС сигнализирует о сохранении ставки",
    "content": "Регулятор подтвердил текущий диапазон и указал на снижение инфляции...",
    "original_url": "https://www.federalreserve.gov/...",
    "image_base64": null
  },
  {
    "source": "Коммерсант",
    "title": "Банки наращивают ипотечные программы",
    "content": "Рынок ожидает снижения ключевой ставки, спрос на жилье растет...",
    "original_url": "https://www.kommersant.ru/...",
    "image_base64": null
  }
]
```

If the LLM call fails or returns no structured tags, the service falls back to the sampled news items (first `n` entries) with their original content.

## Data lifecycle and deduplication

1. Background worker fetches RSS feeds every `passive_mode_dur` seconds.
2. Items older than `period` are removed from the cache.
3. New items are identified by `guid` → `url` → `source:title` (in that priority) to avoid duplicates across runs.
4. Cached data is served directly via `/news` and `/titles`, providing low-latency reads for the LLM service.

## Project layout

- `news_aggregator/` – FastAPI service for fetching, parsing, and caching RSS feeds.
- `llm/` – FastAPI service for LLM-backed news curation; starts vLLM locally when `autostart` is enabled.
- `docker-compose.example.yaml` – reference compose file wiring both services with GPU reservation for vLLM.

## Troubleshooting

- Ensure `news_aggregator/sources.json` contains valid RSS URLs; failures are logged per URL but do not abort the cycle.
- If the LLM endpoint times out, check the vLLM server logs and GPU availability; the FastAPI service will temporarily serve fallback results.
- Clear the aggregator cache by deleting `news.parquet` in the configured `cache_dir`.
