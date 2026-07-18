"""Runtime configuration for the Stage A detection API.

All settings are overridable through environment variables prefixed with
``PID_`` (e.g. ``PID_BACKEND=stub``) or through a local ``.env`` file, which is
gitignored.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Languages the service can route to today. Adding a language here is the only
# change needed for /v1/languages to advertise it (spec 2.3: adding kz or the
# image modality later must not break the contract).
SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "ru")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PID_", env_file=".env", extra="ignore"
    )

    # --- model sources -------------------------------------------------
    # Stage 1 artifacts. Public repos need no credentials; a token is only
    # consulted while the repos are still private.
    en_model_repo: str = "Eye172/prompt-injection-en-deberta-v3-small"
    ru_model_repo: str = "Eye172/prompt-injection-ru-rubert-base"
    en_model_version: str = "en-deberta-v3-small-v1"
    ru_model_version: str = "ru-rubert-base-v1"

    # Optional local path to the attack-type head trained in the Stage 2
    # notebook. When absent the API still serves, with attack_type = null.
    attack_type_artifact: str | None = None

    # ``hf`` needs torch + transformers. ``stub`` is a dependency-free
    # deterministic backend so the full API surface can be exercised (and the
    # test suite run) on machines without the ML stack.
    backend: Literal["hf", "stub"] = "stub"
    hf_token: str | None = None
    device: str = "cpu"

    # --- inference -----------------------------------------------------
    # Score at or above this is reported as an injection. Kept per-language
    # because the two models are calibrated independently by design (spec 0).
    threshold_en: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold_ru: float = Field(default=0.5, ge=0.0, le=1.0)
    max_length: int = 256
    # 0 means no limit: long text is chunked rather than rejected (see
    # detection/chunking.py). Set a positive value to cap request size if the
    # deployment needs to bound per-request work.
    max_text_chars: int = 0
    max_batch_size: int = 64

    # Long-text handling. chunk_words is sized to fit comfortably inside
    # max_length tokens for both tokenizers; the overlap keeps an attack that
    # straddles a window boundary intact in at least one window.
    chunk_words: int = 180
    chunk_overlap_words: int = 40

    # Sliding-window span localisation.
    span_enabled: bool = True
    span_window_words: int = 8
    span_stride_words: int = 4
    # Below this score the text is benign enough that locating a span is noise.
    span_min_score: float = 0.5

    # --- access control ------------------------------------------------
    # Comma-separated keys. Empty (the default) disables auth, which is what
    # the local test suite and the notebook demo use.
    api_keys: str = ""
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # --- privacy -------------------------------------------------------
    # Spec 2.4 step 6: never persist request text, which may carry client PII.
    # Flipping this on is a deliberate, documented act.
    log_request_text: bool = False

    def threshold_for(self, language: str) -> float:
        return self.threshold_ru if language == "ru" else self.threshold_en

    def model_version_for(self, language: str) -> str:
        return self.ru_model_version if language == "ru" else self.en_model_version

    def parsed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
