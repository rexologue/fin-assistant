# news-aggregator

The news-aggregator service ingests financial RSS/Atom feeds, normalizes entries, deduplicates them, and exposes cached results through a FastAPI application. It is designed to run continuously in the background, persisting items to a Parquet file so downstream services (such as the LLM service) can fetch fast, filtered news snapshots.

## What the service does
- Load configured feeds from `sources.json` (or a custom YAML/JSON path).
- Fetch and parse RSS/Atom XML, trimming provider-specific quirks (e.g., `finam` summaries).
- Deduplicate items using GUID → URL → `source:title` identity and drop entries older than the configured retention period.
- Persist normalized items to `news.parquet` inside a configurable cache directory.
- Expose read-only APIs for full records (`/news`) and just headlines (`/titles`).

## Data flow
1. Background worker runs on startup and then sleeps for `passive_mode_dur` seconds between cycles.
2. Fetcher downloads feeds defined in `sources.json` and returns raw XML per feed.
3. Parser converts XML into `NewsItem` objects and applies source-specific cleaning.
4. Items older than `period` are removed; duplicates are skipped based on identity keys.
5. Cache writes to `news.parquet`; reads for `/news` and `/titles` come directly from this file.

## Configuration
Configuration is read from `config.yaml` (create it from the provided example) and can be overridden with environment variables when running in Docker.

### YAML fields (config.yaml)
```yaml
host: 0.0.0.0          # Bind host for FastAPI
port: 8000             # Bind port for FastAPI
cache_dir: ~/news_cache  # Directory where news.parquet is stored
period: 1d             # Retention window (h=hours, d=days, w=weeks, m=30-day months)
passive_mode_dur: 300  # Seconds to sleep between fetch cycles
update: false          # If true, clears the cache on startup before fetching
# sources_path: /app/news_aggregator/sources.json  # Optional override for feed list
```

### Environment variable overrides (Docker-friendly)
- `NEWS_AGGREGATOR_CONFIG`: Alternate path to `config.yaml`.
- `NEWS_AGGREGATOR_CACHE_DIR`: Override cache directory (mounted volume recommended).
- `NEWS_AGGREGATOR_PERIOD`: Override retention period string.
- `NEWS_AGGREGATOR_PASSIVE_MODE_DUR`: Override sleep duration between cycles (seconds).
- `NEWS_AGGREGATOR_UPDATE`: When truthy, clear cache on startup.

## API
- `GET /news` — Return full cached records as an array of objects (`source`, `name`, `title`, `content`, `url`, `published_at`, `image_base64`, `guid`).
- `GET /titles` — Return an array of strings containing cached headlines.

See [`swagger.yaml`](./swagger.yaml) for a full OpenAPI description (schemas, examples, and error formats) suitable for client generation.

## Startup instructions

### Run locally
```bash
# Install dependencies
pip install -r news_aggregator/requirements.txt

# Copy and edit configuration
cp news_aggregator/config.example.yaml news_aggregator/config.yaml

# Start the API
uvicorn news_aggregator.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker Compose (recommended)
Add or reuse the following snippet (already present in `docker-compose.example.yaml`):
```yaml
services:
  news-aggregator:
    build:
      context: .
      dockerfile: news_aggregator/Dockerfile
    ports:
      - "8000:8000"
    environment:
      NEWS_AGGREGATOR_PERIOD: 1d
      NEWS_AGGREGATOR_PASSIVE_MODE_DUR: 300
      NEWS_AGGREGATOR_CACHE_DIR: /cache
    volumes:
      - ./news_aggregator/config.yaml:/app/news_aggregator/config.yaml:ro
      - ./news_aggregator/sources.json:/app/news_aggregator/sources.json:ro
      - ./news_aggregator/cache:/cache
```
Then launch with `docker compose -f docker-compose.example.yaml up --build`.

## Usage examples

### Sample request/response
- Fetch titles:
  ```bash
  curl http://localhost:8000/titles
  ```
  Response:
  ```json
  [
    "ФРС США повысила ставку...",
    "ЦБ РФ опубликовал отчет..."
  ]
  ```

- Fetch full news:
  ```bash
  curl http://localhost:8000/news
  ```
  Response (truncated):
  ```json
  [
    {
      "source": "finam",
      "name": "Finam",
      "title": "ФРС сигнализирует о сохранении ставки",
      "content": "Регулятор подтвердил текущий диапазон и указал на снижение инфляции...",
      "url": "https://www.finam.ru/...",
      "published_at": "2024-05-01T10:00:00+00:00",
      "image_base64": null,
      "guid": "abc-123"
    }
  ]
  ```

### Example cache contents (formatted)
```
news.parquet
├─ source       finam
├─ title        ФРС сигнализирует о сохранении ставки
├─ content      Регулятор подтвердил текущий диапазон...
├─ url          https://www.finam.ru/...
├─ published_at 2024-05-01T10:00:00Z
└─ guid         abc-123
```

## Persistence, caching, and periodic updates
- Cache lives under `cache_dir` and persists between runs; mounting it in Docker keeps state across container restarts.
- Each active cycle removes items older than `period` before inserting fresh data.
- Duplicate detection prioritizes `guid`, then `url`, then `source:title` to avoid repeated inserts across cycles.
- To force a clean rebuild of the cache on the next start, set `update: true` in `config.yaml` or `NEWS_AGGREGATOR_UPDATE=true`.

## End-to-end flow
1. Start the service with the desired configuration (local or Docker).
2. The background worker fetches feeds, normalizes, deduplicates, and persists to `news.parquet`.
3. Clients or the LLM service call `/news` or `/titles` to consume cached data.
4. On subsequent cycles, old data is purged and new items appended, keeping the cache fresh for downstream consumers.
