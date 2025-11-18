FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY llm/requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

COPY llm /app/llm

ENV LLM_CONFIG_PATH=/app/llm/config.yaml

CMD ["python3", "-m", "uvicorn", "llm.app:app", "--host", "0.0.0.0", "--port", "8001"]
