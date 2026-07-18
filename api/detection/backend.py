"""Backend abstraction over the Stage 1 classifiers.

Everything above this layer (routing, spans, attack typing, the HTTP surface)
depends only on this interface, so the API can be developed and tested without
torch installed and swapped to real weights by changing one setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass
class ScoreResult:
    """Injection probabilities plus the pooled representations behind them.

    Embeddings are returned alongside the scores because the attack-type head
    (spec 2.4 step 3, option b) is a light classifier over exactly these
    vectors — recomputing them would double the forward passes.
    """

    scores: list[float]
    embeddings: np.ndarray | None = None


class DetectionBackend(Protocol):
    name: str

    def languages(self) -> tuple[str, ...]:
        """Languages this backend can currently serve."""

    def is_loaded(self, language: str) -> bool:
        ...

    def repo_for(self, language: str) -> str:
        ...

    def score(
        self, texts: Sequence[str], language: str, *, with_embeddings: bool = False
    ) -> ScoreResult:
        """Return P(injection) for each text under the model for ``language``."""


class UnsupportedLanguageError(ValueError):
    def __init__(self, language: str, supported: Sequence[str]) -> None:
        super().__init__(
            f"language {language!r} is not served; supported: {', '.join(supported)}"
        )
        self.language = language
