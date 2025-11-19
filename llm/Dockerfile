FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ../install.sh ./install.sh
RUN python3 -m pip install --upgrade pip \
    && bash install.sh

COPY llm /app/llm

ENV LLM_CONFIG_PATH=/app/llm/config.yaml

CMD ["python3", "-m", "uvicorn", "llm.app:app", "--host", "0.0.0.0", "--port", "8001"]
