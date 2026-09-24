FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    BIRDLY_API_CONFIG=/app/configs/api.yaml \
    BIRDLY_API_BACKEND=onnxruntime \
    BIRDLY_MODEL_DIR=/app/model \
    PORT=8080

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 birdly

COPY --chown=birdly:birdly api.yaml ./configs/api.yaml
COPY --chown=birdly:birdly build-manifest.json ./build-manifest.json
COPY --chown=birdly:birdly src ./src
COPY --chown=birdly:birdly model ./model

USER birdly

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.getenv('PORT') or '8080'; urllib.request.urlopen('http://127.0.0.1:' + port + '/health', timeout=3)"]

CMD ["sh", "-c", "exec uvicorn birdly.api.app:app --host 0.0.0.0 --port \"${PORT:-8080}\" --workers 1 --no-access-log"]
