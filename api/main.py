"""FastAPI application for Stage A (spec section 2).

Run locally:      uvicorn api.main:app --reload
Run with models:  PID_BACKEND=hf uvicorn api.main:app
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import get_settings
from .deps import build_limiter, build_service
from .observability import new_request_id
from .routes import detect as detect_routes
from .routes import meta as meta_routes

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("starting detection API v%s (backend=%s)", __version__, settings.backend)
    # Eager construction: both models are resident before the first request
    # is served, so no caller pays a cold-start penalty (spec 2.4 step 1).
    app.state.settings = settings
    app.state.limiter = build_limiter(settings)
    app.state.service = build_service(settings)
    logger.info("models ready: %s", ", ".join(app.state.service.backend.languages()))
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Prompt Injection Detection API",
    version=__version__,
    description=(
        "Bilingual (EN/RU) prompt-injection detection over the Stage 1 "
        "reference models. Implements Variant A of the project spec."
    ),
    lifespan=lifespan,
)

app.include_router(detect_routes.router)
app.include_router(meta_routes.router)


def _error_payload(code: str, message: str, extra: dict | None = None) -> dict:
    body = {
        "success": False,
        "error": {"code": code, "message": message},
        "meta": {"request_id": new_request_id()},
    }
    if extra:
        body["error"].update(extra)
    return body


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Keep failures in the same envelope as successes."""
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.pop("code", "http_error")
        message = detail.pop("message", "")
        payload = _error_payload(code, message, detail or None)
    else:
        payload = _error_payload("http_error", str(detail))
    return JSONResponse(
        status_code=exc.status_code, content=payload, headers=getattr(exc, "headers", None)
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            "validation_error",
            "Request body failed validation.",
            # pydantic v2 can put exception objects in the error context, so
            # encode rather than handing raw errors to the JSON serializer.
            {"errors": jsonable_encoder(exc.errors())},
        ),
    )


DEMO_PAGE = Path(__file__).parent / "static" / "demo.html"


@app.get("/demo", include_in_schema=False, response_class=HTMLResponse)
def demo() -> HTMLResponse:
    """Browser demo, served from the API's own origin.

    Same-origin by design: a page hosted elsewhere could not call this API
    without CORS, and the demo is meant to work unchanged behind a Colab
    proxy or a tunnel where the public URL is not known in advance.
    """
    if not DEMO_PAGE.exists():  # pragma: no cover - packaging guard
        return HTMLResponse("<h1>demo.html is missing</h1>", status_code=404)
    return HTMLResponse(DEMO_PAGE.read_text(encoding="utf-8"))


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "prompt-injection-detection",
        "version": __version__,
        "demo": "/demo",
        "docs": "/docs",
        "endpoints": ["/v1/detect", "/v1/detect/batch", "/v1/health", "/v1/languages"],
    }
