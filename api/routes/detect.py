"""POST /v1/detect and /v1/detect/batch (spec 2.3)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import Settings, get_settings
from ..deps import get_service
from ..detection.backend import UnsupportedLanguageError
from ..detection.service import DetectionService
from ..observability import log_detection, new_request_id
from ..schemas import (
    BatchDetectRequest,
    BatchDetectResponse,
    BatchMeta,
    DetectRequest,
    DetectResponse,
    Meta,
)
from ..security.auth import ClientContext, require_client

router = APIRouter(prefix="/v1", tags=["detection"])


def _validate_text(text: str, settings: Settings) -> None:
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "empty_text", "message": "text must not be empty."},
        )
    # There is no length limit by default: text longer than the model window is
    # chunked and scored in full rather than rejected. A deployment can still
    # set PID_MAX_TEXT_CHARS to bound per-request work.
    if settings.max_text_chars and len(text) > settings.max_text_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "text_too_long",
                "message": f"text exceeds {settings.max_text_chars} characters.",
            },
        )


@router.post("/detect", response_model=DetectResponse)
def detect(
    payload: DetectRequest,
    service: DetectionService = Depends(get_service),
    settings: Settings = Depends(get_settings),
    client: ClientContext = Depends(require_client),
) -> DetectResponse:
    _validate_text(payload.text, settings)

    request_id = new_request_id()
    started = time.perf_counter()
    try:
        result = service.detect(payload.text, payload.lang)
    except UnsupportedLanguageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "unsupported_language", "message": str(exc)},
        ) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    log_detection(
        request_id, [result], latency_ms, settings, endpoint="/v1/detect",
        texts=[payload.text],
    )

    return DetectResponse(
        data=result,
        meta=Meta(
            request_id=request_id,
            model_version=service.model_version([result.language_used]),
            latency_ms=latency_ms,
            credits_consumed=client.decision.consumed,
            credits_remaining=client.decision.remaining,
        ),
    )


@router.post("/detect/batch", response_model=BatchDetectResponse)
def detect_batch(
    payload: BatchDetectRequest,
    service: DetectionService = Depends(get_service),
    settings: Settings = Depends(get_settings),
    client: ClientContext = Depends(require_client),
) -> BatchDetectResponse:
    if len(payload.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "batch_too_large",
                "message": f"at most {settings.max_batch_size} texts per request.",
            },
        )
    for text in payload.texts:
        _validate_text(text, settings)

    request_id = new_request_id()
    started = time.perf_counter()
    try:
        results = service.detect_batch(payload.texts, payload.lang)
    except UnsupportedLanguageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "unsupported_language", "message": str(exc)},
        ) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    log_detection(
        request_id, results, latency_ms, settings, endpoint="/v1/detect/batch",
        texts=list(payload.texts),
    )

    return BatchDetectResponse(
        data=results,
        meta=BatchMeta(
            request_id=request_id,
            model_version=service.model_version([r.language_used for r in results]),
            latency_ms=latency_ms,
            count=len(results),
            credits_consumed=client.decision.consumed,
            credits_remaining=client.decision.remaining,
        ),
    )
