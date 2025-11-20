# llm

The llm service ranks and optionally translates news items using a vLLM-backed model. It consumes cached news from `news-aggregator`, builds prompts from user preferences, calls the model through an OpenAI-compatible API, post-processes the output, and exposes API endpoints for curated news and budgeting advice.

## Purpose and behavior
- Load runtime configuration from `config.yaml`, with Docker-friendly environment overrides.
- Initialize a vLLM server (autostart) or connect to an external instance via OpenAI-compatible `/v1/chat/completions`.
- Sample cached news from the aggregator, build rerank and translation prompts, and enrich the model’s response with original URLs/images.
- Fall back to sampled items if the model returns invalid output or cannot be reached.
- Provide a budgeting advice endpoint that formats the response as JSON.

## Configuration loading
Configuration is read from `config.yaml` (copy from `config.example.yaml`). When `docker: true`, several keys can be overridden by environment variables.

```yaml
docker: true
app:
  host: 0.0.0.0
  port: 18300
model:
  path: /model                  # Model directory mounted in the container
  host: 0.0.0.0                 # vLLM bind host
  port: 8001                    # vLLM bind port
  device_id: 0                  # GPU index
  quantization: null
  gpu_memory_utilization: 0.8
  max_model_len: 16384
  autostart: true               # Start vLLM when the FastAPI app starts
sampling:
  sample_size: 20               # How many news items to sample before reranking
aggregator:
  base_url: http://news-aggregator:8000
  timeout_seconds: 15
```

### Environment overrides (used when `docker: true`)
- `LLM_MODEL_PATH`, `LLM_MODEL_HOST`, `LLM_GPU_UTILIZATION`, `LLM_MAX_MODEL_LENGTH`
- `LLM_SAMPLING_SAMPLE_SIZE`
- `LLM_AGGREGATOR_HOST`, `LLM_AGGREGATOR_PORT`, `LLM_AGGREGATOR_TIMEOUT_SECONDS`
- `LLM_APP_HOST`, `LLM_APP_PORT`

## vLLM initialization
- `AppConfig` is parsed on startup; `VLLMServerConfig` uses `model.path`, `host`, `port`, `device_id`, `quantization`, `gpu_memory_utilization`, and `max_model_len`.
- When `model.autostart` is true, the service starts the vLLM server in the lifespan context; otherwise it assumes the server is reachable at the configured host/port.
- Inference calls use the OpenAI `/v1/chat/completions` endpoint via `VLLMChatClient`.

## Available endpoints
- `POST /news`
  - **Request body**: `n` (1–10 items), `top_spend_categories` (list of strings), optional `disliked_titles` (list of strings).
  - **Process**: fetch cached news → sample `sample_size` → rerank via prompt → translate titles/content → enrich with original URLs/images → return up to `n` results. Falls back to sampled data on LLM errors.
  - **Response**: array of objects `{source, title, content, original_url, image_base64}`.
- `POST /advice`
  - **Request body**: `{earnings: float, wastes: {category: amount}, wishes: string}`.
  - **Process**: build advice prompt and validate JSON returned by the model.
  - **Response**: `{earnings, wastes, comment}`.

See [`swagger.yaml`](./swagger.yaml) for complete schemas, examples, status codes, and headers.

## Prompt building and post-processing
1. **Sampling**: choose up to `sampling.sample_size` items from the aggregator’s `/news` output.
2. **Rerank prompt**: `build_news_rerank_prompt` scores items based on `top_spend_categories` and `disliked_titles` and asks for item indexes.
3. **Translation prompt**: for each reranked item, `build_news_translate_prompt` asks the model to return localized title/content.
4. **Parsing**: `parse_news_rerank_response` extracts indexes; `parse_news_translate_response` extracts translated fields.
5. **Enrichment**: `enrich_results` merges translated items with original URLs/images; `fallback_selection` returns sampled originals if parsing fails.

## Running the service

### Local
```bash
pip install -r llm/requirements.txt
cp llm/config.example.yaml llm/config.yaml
uvicorn llm.app:app --host 0.0.0.0 --port 18300
```
Ensure a vLLM server is reachable at `model.host:model.port` (or enable `autostart` and mount the model path).

### Docker Compose
This snippet is already in `docker-compose.example.yaml`:
```yaml
services:
  llm-service:
    build:
      context: .
      dockerfile: llm/Dockerfile
    depends_on:
      - news-aggregator
    network_mode: host
    environment:
      LLM_MODEL_PATH: /model
      LLM_MODEL_HOST: 0.0.0.0
      LLM_SAMPLING_SAMPLE_SIZE: 20
      LLM_AGGREGATOR_HOST: localhost
      LLM_AGGREGATOR_PORT: 8000
      LLM_AGGREGATOR_TIMEOUT_SECONDS: 15
      LLM_GPU_UTILIZATION: 0.8
      LLM_MAX_MODEL_LENGTH: 16384
    volumes:
      - ./llm/config.yaml:/app/llm/config.yaml:ro
      - ./models:/model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
Start with `docker compose -f docker-compose.example.yaml up --build` after preparing `./models`.

## Usage examples

### Request curated news
```bash
curl -X POST http://localhost:18300/news \
  -H "Content-Type: application/json" \
  -d '{
    "n": 3,
    "top_spend_categories": ["Инвестиции", "Технологии"],
    "disliked_titles": ["Дефолт", "Госдолг"]
  }'
```
Response (success path):
```json
[
  {
    "source": "Finam",
    "title": "Технологические акции растут после отчётов",
    "content": "Компании полупроводников и облачных сервисов демонстрируют рост...",
    "original_url": "https://www.finam.ru/...",
    "image_base64": null
  },
  {
    "source": "Reuters",
    "title": "Инвесторы увеличивают долю в ETF на развивающиеся рынки",
    "content": "Приток капитала связан со снижением опасений по поводу ставок...",
    "original_url": "https://reuters.com/...",
    "image_base64": null
  }
]
```
Fallback response (if model fails) simply returns up to `n` sampled entries with original text and URLs.

### Request budgeting advice
```bash
curl -X POST http://localhost:18300/advice \
  -H "Content-Type: application/json" \
  -d '{
    "earnings": 120000,
    "wastes": {"housing": 35000, "food": 20000, "transport": 8000},
    "wishes": "Хочу откладывать на подушку безопасности"
  }'
```
Response:
```json
{
  "earnings": 120000,
  "wastes": {"housing": 35000, "food": 20000, "transport": 8000},
  "comment": "Сократите расходы на рестораны и перенаправьте 10% дохода в накопления."
}
```

## End-to-end flow
1. The aggregator fetches and caches news.
2. A client POSTs to `/news` with preferences.
3. The llm service samples cached news, reranks/ translates via vLLM, enriches with URLs/images, and returns up to `n` items.
4. If vLLM is unavailable or returns invalid data, the service returns sampled items instead.
5. The `/advice` endpoint follows a simpler prompt/parse cycle to return financial advice as structured JSON.
