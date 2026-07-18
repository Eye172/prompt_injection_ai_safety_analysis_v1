"""Orchestration: language routing -> scoring -> span -> attack type."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..config import Settings
from ..schemas import DetectionData, Label, Span
from .attack_type import AttackTypeHead
from .backend import DetectionBackend, UnsupportedLanguageError
from .chunking import Chunk, chunk_text
from .language import LanguageDetector, ScriptRatioDetector, resolve_language
from .span import locate_span

logger = logging.getLogger(__name__)


class DetectionService:
    def __init__(
        self,
        backend: DetectionBackend,
        settings: Settings,
        *,
        detector: LanguageDetector | None = None,
        attack_head: AttackTypeHead | None = None,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.detector = detector or ScriptRatioDetector()
        self.attack_head = attack_head

    @property
    def has_attack_head(self) -> bool:
        return self.attack_head is not None

    def detect(self, text: str, lang: str) -> DetectionData:
        return self.detect_batch([text], lang)[0]

    def detect_batch(self, texts: Sequence[str], lang: str) -> list[DetectionData]:
        """Group by resolved language so each model is called once per batch.

        Text longer than the model's window is split into overlapping chunks
        and every chunk is scored; a text is as suspicious as its most
        suspicious part. Chunks from the whole batch are scored in one call, so
        long input costs more compute but not more round trips.
        """
        resolved = [resolve_language(t, lang, self.detector).language for t in texts]

        by_language: dict[str, list[int]] = {}
        for idx, language in enumerate(resolved):
            by_language.setdefault(language, []).append(idx)

        results: list[DetectionData | None] = [None] * len(texts)

        for language, indices in by_language.items():
            if not self.backend.is_loaded(language):
                raise UnsupportedLanguageError(language, self.backend.languages())

            # Only pay for hidden states when a head actually exists for this
            # language — requesting them is not free on the real models.
            want_embeddings = (
                self.attack_head is not None and self.attack_head.supports(language)
            )

            chunks: list[Chunk] = []
            owners: list[int] = []
            for global_idx in indices:
                text_chunks = chunk_text(
                    texts[global_idx],
                    chunk_words=self.settings.chunk_words,
                    overlap_words=self.settings.chunk_overlap_words,
                )
                chunks.extend(text_chunks)
                owners.extend([global_idx] * len(text_chunks))

            outcome = self.backend.score(
                [c.text for c in chunks], language, with_embeddings=want_embeddings
            )
            threshold = self.settings.threshold_for(language)

            # The chunk that carries the strongest signal represents the text:
            # it supplies the score, the span, and the attack-type embedding.
            best: dict[int, int] = {}
            counts: dict[int, int] = {}
            for flat_idx, global_idx in enumerate(owners):
                counts[global_idx] = counts.get(global_idx, 0) + 1
                current = best.get(global_idx)
                if current is None or outcome.scores[flat_idx] > outcome.scores[current]:
                    best[global_idx] = flat_idx

            attack_types: list = [None] * len(indices)
            if self.attack_head is not None and outcome.embeddings is not None:
                rows = np.stack([outcome.embeddings[best[i]] for i in indices])
                attack_types = self.attack_head.predict(rows, language)

            for position, global_idx in enumerate(indices):
                flat_idx = best[global_idx]
                chunk = chunks[flat_idx]
                score = float(outcome.scores[flat_idx])
                is_injection = score >= threshold
                results[global_idx] = DetectionData(
                    is_injection=is_injection,
                    score=round(score, 4),
                    label=Label.injection if is_injection else Label.benign,
                    language_used=language,
                    # Only meaningful for a positive detection.
                    attack_type=attack_types[position] if is_injection else None,
                    span=self._span_for(chunk, language, score) if is_injection else None,
                    chunks_analyzed=counts[global_idx],
                )

        # Every index is covered by exactly one language group; filtering
        # instead of asserting here would silently misalign results with the
        # caller's input order.
        if any(r is None for r in results):
            raise RuntimeError("internal error: not every input was scored")
        return [r for r in results if r is not None]

    def _span_for(self, chunk: Chunk, language: str, score: float) -> Span | None:
        """Locate the span inside the winning chunk, in the caller's coordinates.

        The search runs on the chunk, but the offsets returned are shifted back
        onto the original text so a client can highlight the right words even
        when the match came from page nine of a long document.
        """
        s = self.settings
        if not s.span_enabled or score < s.span_min_score:
            return None
        try:
            found = locate_span(
                chunk.text,
                language,
                self.backend,
                window_words=s.span_window_words,
                stride_words=s.span_stride_words,
                min_score=s.span_min_score,
                full_score=score,
            )
        except Exception:  # pragma: no cover - span is best-effort
            logger.exception("span localisation failed; omitting span")
            return None
        if found is None:
            return None
        return Span(
            start_word=found.start_word + chunk.word_offset,
            end_word=found.end_word + chunk.word_offset,
        )

    def model_version(self, languages: Sequence[str]) -> str:
        unique = sorted(set(languages))
        return "+".join(self.settings.model_version_for(l) for l in unique)
