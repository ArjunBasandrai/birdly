FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    BIRDLY_API_CONFIG=/app/configs/api.yaml \
    BIRDLY_MODEL_DIR=/model

WORKDIR /app

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-api.txt

RUN useradd --create-home --uid 10001 birdly \
    && mkdir -p /app/src/birdly/inference /model \
    && chown -R birdly:birdly /app /model

COPY --chown=birdly:birdly configs/api.yaml ./configs/api.yaml
COPY --chown=birdly:birdly src/birdly/__init__.py ./src/birdly/__init__.py
COPY --chown=birdly:birdly src/birdly/api ./src/birdly/api
COPY --chown=birdly:birdly src/birdly/inference/__init__.py ./src/birdly/inference/__init__.py
COPY --chown=birdly:birdly src/birdly/inference/onnx_backend.py ./src/birdly/inference/onnx_backend.py
COPY --chown=birdly:birdly src/birdly/inference/predict.py ./src/birdly/inference/predict.py
COPY --chown=birdly:birdly src/birdly/inference/taxon_names.json ./src/birdly/inference/taxon_names.json

USER birdly

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "birdly.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
