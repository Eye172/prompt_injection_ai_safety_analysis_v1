"""Attack-type classification (spec 2.4 step 3, option b).

A light scikit-learn head over the detector's pooled embeddings, rather than a
multi-class head on the detector itself: it can be fitted, replaced or dropped
without retraining or revalidating the Stage 1 binary models.

One head is fitted **per language**. The EN and RU detectors are separate
checkpoints, so their embedding spaces are unrelated — a single estimator
fitted over both would be mixing two incompatible coordinate systems.

The artifact is produced by notebooks/variant_a_stage2.ipynb. When it is
missing the API still serves and reports attack_type = null, which the schema
allows: an unavailable classifier must never take detection down.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..schemas import AttackType

logger = logging.getLogger(__name__)

# Source-corpus labels (EN neuralchemy / jackhhao, RU dmtrdr) folded onto the
# spec taxonomy. The notebook applies this same table before fitting, so the
# head only ever emits canonical values. Mapping canonical names to themselves
# keeps the function idempotent.
TAXONOMY_MAP: dict[str, AttackType] = {
    "instruction_override": AttackType.instruction_override,
    "ignore_prompt": AttackType.instruction_override,
    "ignore_previous_instructions": AttackType.instruction_override,
    "direct_injection": AttackType.instruction_override,
    "prompt_injection": AttackType.instruction_override,
    "role_play": AttackType.role_play,
    "roleplay": AttackType.role_play,
    "jailbreak": AttackType.role_play,
    "persona_switch": AttackType.role_play,
    "fake_completion": AttackType.fake_completion,
    "payload_splitting": AttackType.fake_completion,
    "context_manipulation": AttackType.context_manipulation,
    "context_switch": AttackType.context_manipulation,
    "social_engineering": AttackType.context_manipulation,
    "prompt_leaking": AttackType.context_manipulation,
    "encoding_obfuscation": AttackType.encoding_obfuscation,
    "obfuscation": AttackType.encoding_obfuscation,
    "encoding": AttackType.encoding_obfuscation,
    "base64": AttackType.encoding_obfuscation,
    "other": AttackType.other,
}


def canonicalize(raw_label: str | None) -> AttackType | None:
    if not raw_label:
        return None
    key = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
    return TAXONOMY_MAP.get(key, AttackType.other)


class _LanguageHead:
    def __init__(self, estimator, classes: list[str]) -> None:
        self.estimator = estimator
        self.classes = classes


class AttackTypeHead:
    """Per-language attack-type classifiers loaded from one artifact."""

    def __init__(
        self, heads: dict[str, _LanguageHead], min_confidence: float = 0.3
    ) -> None:
        self._heads = heads
        self._min_confidence = min_confidence

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(self._heads)

    @classmethod
    def load(cls, path: str | Path) -> "AttackTypeHead | None":
        p = Path(path)
        if not p.exists():
            logger.warning("attack-type artifact not found at %s; disabling", p)
            return None
        try:
            import joblib

            bundle = joblib.load(p)
            heads = {
                lang: _LanguageHead(entry["estimator"], list(entry["classes"]))
                for lang, entry in bundle["heads"].items()
            }
            head = cls(heads, float(bundle.get("min_confidence", 0.3)))
            logger.info("attack-type head loaded for: %s", ", ".join(head.languages))
            return head
        except Exception:  # pragma: no cover - depends on artifact contents
            logger.exception("failed to load attack-type head from %s; disabling", p)
            return None

    def supports(self, language: str) -> bool:
        return language in self._heads

    def predict(
        self, embeddings: np.ndarray, language: str
    ) -> list[AttackType | None]:
        """Predict for embeddings that all came from ``language``'s model."""
        if embeddings is None or len(embeddings) == 0:
            return []

        head = self._heads.get(language)
        if head is None:
            # No head trained for this language: null is the honest answer.
            return [None] * len(embeddings)

        try:
            probs = head.estimator.predict_proba(embeddings)
        except Exception:  # pragma: no cover
            logger.exception("attack-type inference failed; returning null")
            return [None] * len(embeddings)

        results: list[AttackType | None] = []
        for row in probs:
            idx = int(np.argmax(row))
            # A low-confidence guess is worse than none: callers filter on
            # attack_type, so a wrong label is actively misleading.
            if float(row[idx]) < self._min_confidence:
                results.append(None)
            else:
                results.append(canonicalize(head.classes[idx]))
        return results
