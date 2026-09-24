from __future__ import annotations

import asyncio
import io
import logging
import math
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from birdly.api.config import ApiSettings, load_api_settings
from birdly.inference.predict import (
    Prediction,
    Predictor,
    load_predictor,
    predict_image,
)

LOGGER = logging.getLogger("uvicorn.error.birdly")
LOGGER.setLevel(logging.INFO)
SUPPORTED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
READ_CHUNK_BYTES = 64 * 1024

PREDICTION_REQUESTS = Counter(
    "birdly_prediction_requests_total",
    "Birdly prediction requests by response status.",
    ("status",),
)
PREDICTION_ERRORS = Counter(
    "birdly_prediction_errors_total",
    "Birdly prediction errors by type.",
    ("error_type",),
)
REQUEST_LATENCY = Histogram(
    "birdly_request_latency_seconds",
    "Birdly HTTP request latency.",
    ("method", "endpoint"),
)
INFERENCE_LATENCY = Histogram(
    "birdly_inference_latency_seconds",
    "Birdly model inference latency.",
    ("backend",),
)
ACTIVE_INFERENCES = Gauge(
    "birdly_active_inference_requests",
    "Birdly inference calls that are currently running.",
)


class UploadTooLargeError(ValueError):
    pass


class InvalidImageError(ValueError):
    pass


class RateLimitExceededError(RuntimeError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Prediction rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class InferenceUnavailableError(RuntimeError):
    pass


class InferenceTimeoutError(RuntimeError):
    pass


class IpRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._checks = 0

    async def check(self, client_ip: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            requests = self._requests[client_ip]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self.limit:
                retry_after = max(
                    1, math.ceil(self.window_seconds - (now - requests[0]))
                )
                raise RateLimitExceededError(retry_after)
            requests.append(now)
            self._checks += 1
            if self._checks % 256 == 0:
                self._remove_expired(cutoff)

    def _remove_expired(self, cutoff: float) -> None:
        for client_ip, requests in list(self._requests.items()):
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if not requests:
                del self._requests[client_ip]


class InferenceService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.predictor: Predictor | None = None
        self.load_error: str | None = None
        self.load_count = 0
        self._semaphore = asyncio.Semaphore(settings.maximum_inference_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.maximum_inference_concurrency,
            thread_name_prefix="birdly-inference",
        )

    def load(self) -> None:
        if self.predictor is not None:
            return
        self.load_count += 1
        self.predictor = load_predictor(
            backend=self.settings.backend,
            device="cpu",
            class_mapping_path=self.settings.class_mapping_path,
            preprocessing_path=self.settings.preprocessing_path,
            onnx_model_path=self.settings.onnx_model_path,
            onnx_metadata_path=self.settings.onnx_metadata_path,
            onnx_int8_model_path=self.settings.onnx_int8_model_path,
            onnx_int8_metadata_path=self.settings.onnx_int8_metadata_path,
            cpu_threads=self.settings.cpu_threads,
        )
        self.load_error = None

    @property
    def is_ready(self) -> bool:
        return self.predictor is not None and self.load_error is None

    @property
    def backend(self) -> str | None:
        return self.predictor.backend if self.predictor is not None else None

    async def predict(self, image: Image.Image) -> list[Prediction]:
        if not self.is_ready or self.predictor is None:
            raise InferenceUnavailableError("The model is not available")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.request_timeout_seconds
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.settings.request_timeout_seconds,
            )
        except TimeoutError as error:
            raise InferenceTimeoutError("Inference capacity timed out") from error

        try:
            future = loop.run_in_executor(
                self._executor,
                self._predict_sync,
                self.predictor,
                image,
            )
        except Exception:
            self._semaphore.release()
            raise
        future.add_done_callback(self._inference_finished)

        remaining_seconds = deadline - loop.time()
        if remaining_seconds <= 0:
            raise InferenceTimeoutError("Inference timed out")
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=remaining_seconds,
            )
        except TimeoutError as error:
            raise InferenceTimeoutError("Inference timed out") from error

    def _predict_sync(
        self,
        predictor: Predictor,
        image: Image.Image,
    ) -> list[Prediction]:
        started = time.perf_counter()
        status = "error"
        ACTIVE_INFERENCES.inc()
        try:
            predictions = predict_image(predictor, image, top_k=5)
            status = "ok"
            return predictions
        finally:
            latency = time.perf_counter() - started
            ACTIVE_INFERENCES.dec()
            INFERENCE_LATENCY.labels(backend=predictor.backend).observe(latency)
            LOGGER.info(
                "inference_complete backend=%s status=%s latency_ms=%.3f",
                predictor.backend,
                status,
                latency * 1000,
            )

    def _inference_finished(self, future: asyncio.Future[Any]) -> None:
        self._semaphore.release()
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            LOGGER.error(
                "background_inference_failed backend=%s",
                self.backend,
                exc_info=(type(error), error, error.__traceback__),
            )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


async def _read_upload(upload: UploadFile, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(READ_CHUNK_BYTES):
        size += len(chunk)
        if size > maximum_bytes:
            raise UploadTooLargeError("Uploaded image exceeds the size limit")
        chunks.append(chunk)
    if size == 0:
        raise InvalidImageError("Uploaded image is empty")
    return b"".join(chunks)


def _decode_image(payload: bytes, settings: ApiSettings) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(
                io.BytesIO(payload), formats=list(SUPPORTED_IMAGE_FORMATS)
            ) as image:
                width, height = image.size
                if (
                    width > settings.maximum_image_width
                    or height > settings.maximum_image_height
                ):
                    raise InvalidImageError(
                        "Image dimensions exceed the configured limit"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImageError("Animated images are not supported")
                image.load()
                return image.copy()
    except InvalidImageError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise InvalidImageError("Uploaded data is not a supported image") from error


def _record_error(error_type: str) -> None:
    PREDICTION_ERRORS.labels(error_type=error_type).inc()
    LOGGER.warning("prediction_error type=%s", error_type)


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    selected_settings = settings or load_api_settings()
    service = InferenceService(selected_settings)
    rate_limiter = IpRateLimiter(selected_settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(service.load)
            LOGGER.info(
                "model_loaded backend=%s load_count=%d",
                service.backend,
                service.load_count,
            )
        except Exception as error:
            service.load_error = f"{type(error).__name__}: {error}"
            LOGGER.exception("model load failed")
        try:
            yield
        finally:
            await asyncio.to_thread(service.close)

    application = FastAPI(title="Birdly Inference API", lifespan=lifespan)
    application.state.inference_service = service
    application.state.rate_limiter = rate_limiter
    application.state.settings = selected_settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(selected_settings.allowed_cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.middleware("http")
    async def record_request(request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            endpoint = request.url.path
            if endpoint not in {"/predict", "/health", "/metrics"}:
                endpoint = "/other"
            latency = time.perf_counter() - started
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(latency)
            if request.url.path == "/predict":
                PREDICTION_REQUESTS.labels(status=str(status_code)).inc()
            LOGGER.info(
                "request_complete method=%s endpoint=%s status=%d latency_ms=%.3f",
                request.method,
                endpoint,
                status_code,
                latency * 1000,
            )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request,
        __: RequestValidationError,
    ) -> JSONResponse:
        _record_error("invalid_request")
        return JSONResponse(status_code=400, content={"detail": "Invalid request"})

    @application.get("/health")
    async def health() -> dict[str, Any]:
        if not service.is_ready:
            raise HTTPException(status_code=503, detail="Model is not available")
        return {
            "status": "ok",
            "model_loaded": True,
            "backend": service.backend,
        }

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.post("/predict")
    async def predict(
        request: Request,
        image: Annotated[UploadFile, File()],
    ) -> list[Prediction]:
        client_ip = request.client.host if request.client is not None else "unknown"
        try:
            await rate_limiter.check(client_ip)
        except RateLimitExceededError as error:
            _record_error("rate_limit")
            raise HTTPException(
                status_code=429,
                detail="Prediction rate limit exceeded",
                headers={"Retry-After": str(error.retry_after_seconds)},
            ) from error

        async def process_request() -> list[Prediction]:
            if image.content_type not in SUPPORTED_CONTENT_TYPES:
                raise InvalidImageError("Unsupported image content type")
            payload = await _read_upload(image, selected_settings.maximum_upload_bytes)
            decoded_image = await asyncio.to_thread(
                _decode_image,
                payload,
                selected_settings,
            )
            return await service.predict(decoded_image)

        try:
            return await asyncio.wait_for(
                process_request(),
                timeout=selected_settings.request_timeout_seconds,
            )
        except UploadTooLargeError as error:
            _record_error("upload_too_large")
            raise HTTPException(status_code=413, detail=str(error)) from error
        except InvalidImageError as error:
            _record_error("invalid_image")
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (InferenceTimeoutError, TimeoutError) as error:
            _record_error("inference_timeout")
            raise HTTPException(
                status_code=503,
                detail="Inference timed out",
            ) from error
        except (InferenceUnavailableError, RuntimeError) as error:
            _record_error("inference_unavailable")
            LOGGER.exception("inference unavailable")
            raise HTTPException(
                status_code=503,
                detail="Inference is temporarily unavailable",
            ) from error
        except HTTPException:
            raise
        except Exception as error:
            _record_error("unexpected")
            LOGGER.exception("unexpected prediction failure")
            raise HTTPException(
                status_code=500,
                detail="Unexpected server error",
            ) from error
        finally:
            await image.close()

    return application


app = create_app()
