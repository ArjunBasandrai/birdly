from __future__ import annotations

import asyncio

from fastapi import Request
from fastapi.responses import JSONResponse

from birdly.api.app import create_app

app = create_app()
app.router.routes = [
    route for route in app.router.routes if getattr(route, "path", None) != "/health"
]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    service = app.state.inference_service
    if not service.is_ready:
        return JSONResponse(
            status_code=503, content={"detail": "Model is not available"}
        )
    return JSONResponse(
        content={"status": "ok", "model_loaded": True, "backend": service.backend}
    )


admission = asyncio.Semaphore(2)


@app.middleware("http")
async def limit_prediction_queue(request: Request, call_next):
    if request.url.path != "/predict":
        return await call_next(request)
    try:
        await asyncio.wait_for(admission.acquire(), timeout=0.01)
    except TimeoutError:
        return JSONResponse(status_code=503, content={"detail": "Inference is busy"})
    try:
        return await call_next(request)
    finally:
        admission.release()
