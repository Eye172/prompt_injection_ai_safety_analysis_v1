"""Request/response contract for /v1/detect — mirrors spec section 2.2.

The envelope (``success`` / ``data`` / ``meta``) is stable: new detection
capabilities are expected to add fields to ``data``, never to reshape it.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Several response fields are named model_version. That collides with
# pydantic's protected "model_" namespace, but the name is fixed by the spec's
# response schema, so the guard is disabled rather than the field renamed.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class Label(str, Enum):
    injection = "injection"
    benign = "benign"


class AttackType(str, Enum):
    """Taxonomy from spec 1.4 / 2.2.

    The Stage 1 corpus labels only a subset of rows, and EN and RU sources use
    different vocabularies; the Stage 2 notebook maps both onto these values
    before fitting the head.
    """

    instruction_override = "instruction_override"
    role_play = "role_play"
    fake_completion = "fake_completion"
    context_manipulation = "context_manipulation"
    encoding_obfuscation = "encoding_obfuscation"
    other = "other"


LangParam = Literal["auto", "en", "ru"]


class DetectRequest(BaseModel):
    text: str = Field(..., description="Text to classify.")
    lang: LangParam = Field(
        default="auto",
        description=(
            "'auto' runs language detection; an explicit 'en'/'ru' skips it and "
            "saves the detection latency."
        ),
    )


class BatchDetectRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    lang: LangParam = "auto"


class Span(BaseModel):
    """Word offsets of the strongest injection signal, end exclusive."""

    start_word: int
    end_word: int


class DetectionData(BaseModel):
    is_injection: bool
    score: float
    label: Label
    language_used: str
    attack_type: AttackType | None = None
    span: Span | None = None
    # Reserved for the future slow LLM-verification layer for borderline
    # scores (spec 2.2). Always null in this MVP.
    reasoning: str | None = None
    #: How many model-sized windows the text was split into. 1 for ordinary
    #: text; higher means the score is the maximum over that many windows.
    chunks_analyzed: int = 1


class Meta(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    request_id: str
    model_version: str
    latency_ms: int
    credits_consumed: int | None = None
    credits_remaining: int | None = None


class DetectResponse(BaseModel):
    success: bool = True
    data: DetectionData
    meta: Meta


class BatchMeta(Meta):
    count: int


class BatchDetectResponse(BaseModel):
    success: bool = True
    data: list[DetectionData]
    meta: BatchMeta


class LoadedModel(BaseModel):
    language: str
    repo: str
    version: str
    loaded: bool


class HealthResponse(BaseModel):
    success: bool = True
    status: Literal["ok", "degraded"]
    backend: str
    models: list[LoadedModel]
    attack_type_head: bool


class LanguageInfo(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    code: str
    model_version: str
    modalities: list[str] = ["text"]


class LanguagesResponse(BaseModel):
    success: bool = True
    languages: list[LanguageInfo]
    auto_detection: bool = True


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody
    meta: dict = {}
