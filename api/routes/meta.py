"""GET /v1/health and GET /v1/languages (spec 2.3)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import SUPPORTED_LANGUAGES, Settings, get_settings
from ..deps import get_service
from ..detection.service import DetectionService
from ..schemas import (
    HealthResponse,
    LanguageInfo,
    LanguagesResponse,
    LoadedModel,
)

router = APIRouter(prefix="/v1", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(
    service: DetectionService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    repos = {"en": settings.en_model_repo, "ru": settings.ru_model_repo}
    models = [
        LoadedModel(
            language=lang,
            repo=service.backend.repo_for(lang) or repos.get(lang, ""),
            version=settings.model_version_for(lang),
            loaded=service.backend.is_loaded(lang),
        )
        for lang in SUPPORTED_LANGUAGES
    ]
    all_loaded = all(m.loaded for m in models)
    return HealthResponse(
        status="ok" if all_loaded else "degraded",
        backend=service.backend.name,
        models=models,
        attack_type_head=service.has_attack_head,
    )


@router.get("/languages", response_model=LanguagesResponse)
def languages(
    service: DetectionService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> LanguagesResponse:
    """Advertises served languages.

    Shaped as a list of objects rather than bare codes so that future
    additions (kz, or an image modality) extend this response instead of
    breaking clients that parse it — spec 2.3.
    """
    served = [l for l in SUPPORTED_LANGUAGES if service.backend.is_loaded(l)]
    return LanguagesResponse(
        languages=[
            LanguageInfo(code=l, model_version=settings.model_version_for(l))
            for l in served
        ]
    )
