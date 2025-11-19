FROM python:3.11-slim

WORKDIR /app

COPY news_aggregator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY news_aggregator /app/news_aggregator

RUN cp /app/news_aggregator/config.example.yaml /app/news_aggregator/config.yaml

ENV NEWS_AGGREGATOR_CONFIG=/app/news_aggregator/config.yaml

CMD ["python3", "-m", "news_aggregator.app"]
