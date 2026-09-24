FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    BIRDLY_MODEL_DIR=/app/model

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY build-manifest.json ./build-manifest.json
COPY src ./src
COPY model ./model
COPY input ./input

ENTRYPOINT ["python", "-m", "birdly.inference.cpu_benchmark"]
