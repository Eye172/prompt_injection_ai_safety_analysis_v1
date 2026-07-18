"""Dependency-free stand-in for the real classifiers.

This exists so the whole HTTP surface — routing, auth, limits, spans, batching
— can be run and tested on a machine without torch/transformers. It is a
keyword heuristic, NOT a detector: it is deliberately crude, and must never be
enabled in a deployment. /v1/health reports the active backend so a stub server
is always identifiable.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from .backend import ScoreResult, UnsupportedLanguageError

EMBEDDING_DIM = 32

# Weighted markers, roughly mirroring the attack families in the Stage 1
# taxonomy so that stub scores move in a plausible direction for tests.
_MARKERS: tuple[tuple[str, float], ...] = (
    (r"ignore (all|any|the)? ?(previous|prior|above)", 0.55),
    (r"disregard (all|any|the)? ?(previous|prior|above)", 0.55),
    (r"forget (all|everything|your) ", 0.45),
    (r"system prompt", 0.4),
    (r"you are now|act as|pretend to be|roleplay", 0.35),
    (r"developer mode|dan mode|jailbreak", 0.45),
    (r"(reveal|leak|print).{0,20}(instructions|prompt|secret)", 0.4),
    (r"игнорируй|проигнорируй", 0.55),
    (r"забудь (все|всё|свои)", 0.45),
    (r"новые инструкции|новая инструкция", 0.5),
    (r"ты теперь|представь, что ты|веди себя как", 0.35),
    (r"системн(ый|ые) промпт", 0.4),
    (r"покажи.{0,20}(инструкц|секрет|пароль)", 0.4),
)

_COMPILED = tuple((re.compile(p, re.IGNORECASE), w) for p, w in _MARKERS)


class StubBackend:
    """Deterministic fake backend. Same text always yields the same score."""

    name = "stub"

    def __init__(self, languages: Sequence[str] = ("en", "ru")) -> None:
        self._languages = tuple(languages)

    def languages(self) -> tuple[str, ...]:
        return self._languages

    def is_loaded(self, language: str) -> bool:
        return language in self._languages

    def repo_for(self, language: str) -> str:
        return f"stub://{language}"

    def score(
        self, texts: Sequence[str], language: str, *, with_embeddings: bool = False
    ) -> ScoreResult:
        if language not in self._languages:
            raise UnsupportedLanguageError(language, self._languages)

        scores = [self._score_one(t) for t in texts]
        embeddings = (
            np.stack([self._embed(t) for t in texts]) if with_embeddings else None
        )
        return ScoreResult(scores=scores, embeddings=embeddings)

    def _score_one(self, text: str) -> float:
        score = 0.05
        for pattern, weight in _COMPILED:
            if pattern.search(text):
                score += weight
        # A little deterministic jitter keeps scores off exact ties without
        # making the result depend on anything but the text itself.
        jitter = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:4], 16) / 65535
        score += jitter * 0.04
        return float(min(score, 0.99))

    def _embed(self, text: str) -> np.ndarray:
        # Seeded from the text so embeddings are stable across processes; a
        # generator (rather than reinterpreting hash bytes as floats) keeps the
        # values finite and well scaled.
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
        raw = np.random.default_rng(seed).normal(size=EMBEDDING_DIM).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        return raw / norm if norm else raw
