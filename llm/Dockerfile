# 1) Сборка flash-attn
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel AS build

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /build

RUN python3 -m pip install --upgrade pip packaging ninja
RUN pip wheel flash-attn==2.8.3 --no-build-isolation -w /wheels

# 2) Основной образ для сервиса
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --from=build /wheels /wheels
RUN python3 -m pip install --upgrade pip && pip install /wheels/flash_attn-2.8.3*.whl

COPY llm/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY llm /app/llm
RUN test -f /app/llm/config.yaml

CMD ["python3", "-m", "llm.app"]
