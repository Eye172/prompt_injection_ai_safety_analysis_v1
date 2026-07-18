"""Shared application state and FastAPI dependencies.

Models are built once during startup and held on ``app.state`` (spec 2.4
step 1: eager loading, so no request pays a cold-start cost).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import HTTPException, Request, status

from .config import Settings
from .detection.attack_type import AttackTypeHead
from .detection.service import DetectionService
from .security.ratelimit import FixedWindowRateLimiter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _build_backend_cached(
    kind: str, en_repo: str, ru_repo: str, token: str | None, device: str, max_length: int
):
    """Cache constructed backends by their inputs.

    Loading the two checkpoints costs tens of seconds, and the test suite
    starts the app once per test case. Keying on the settings that affect
    construction means repeated startups reuse the resident models while a
    changed setting still yields a fresh backend.
    """
    if kind == "stub":
        logger.warning(
            "starting with the STUB backend — keyword heuristic, not a detector; "
            "set PID_BACKEND=hf to serve the real models"
        )
        from .detection.stub_backend import StubBackend

        return StubBackend()

    from .detection.hf_backend import HFBackend

    return HFBackend(
        repos={"en": en_repo, "ru": ru_repo},
        token=token,
        device=device,
        max_length=max_length,
    )


def build_backend(settings: Settings):
    return _build_backend_cached(
        settings.backend,
        settings.en_model_repo,
        settings.ru_model_repo,
        settings.hf_token,
        settings.device,
        settings.max_length,
    )


def build_service(settings: Settings) -> DetectionService:
    backend = build_backend(settings)
    head = (
        AttackTypeHead.load(settings.attack_type_artifact)
        if settings.attack_type_artifact
        else None
    )
    return DetectionService(backend, settings, attack_head=head)


def build_limiter(settings: Settings) -> FixedWindowRateLimiter:
    return FixedWindowRateLimiter(
        settings.rate_limit_requests, settings.rate_limit_window_seconds
    )


def get_service(request: Request) -> DetectionService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - only if startup failed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "not_ready", "message": "Models are not loaded."},
        )
    return service


__all__ = [
    "build_backend",
    "build_service",
    "build_limiter",
    "get_service",
]
