"""Language routing for lang='auto' (spec 2.1).

The spec suggests fasttext lid.176 or lingua-py. For a two-way EN/RU decision
a script-ratio test is both more reliable and far cheaper: the alphabets are
disjoint, so there is no ambiguity to model, and it costs a single pass over
the string with no model to load. ``LanguageDetector`` is kept as a seam so a
statistical detector can be dropped in when a third language (spec: kz) makes
script alone insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

CYRILLIC = (("Ѐ", "ӿ"), ("Ԁ", "ԯ"))
LATIN = (("A", "Z"), ("a", "z"))


def _in_ranges(ch: str, ranges) -> bool:
    return any(lo <= ch <= hi for lo, hi in ranges)


@dataclass
class LanguageGuess:
    language: str
    confidence: float
    detector: str


class LanguageDetector(Protocol):
    name: str

    def detect(self, text: str) -> LanguageGuess:
        ...


class ScriptRatioDetector:
    """Picks ru vs en from the Cyrillic share of alphabetic characters."""

    name = "script_ratio"

    def __init__(self, default: str = "en", cyrillic_threshold: float = 0.2) -> None:
        self._default = default
        # Set below 0.5 so that predominantly-Russian text carrying a short
        # English payload still routes to the RU model. Text that is mostly
        # English routes to EN even with some Cyrillic present — there the
        # attack itself is English, which is what the EN model was trained on.
        self._threshold = cyrillic_threshold

    def detect(self, text: str) -> LanguageGuess:
        cyrillic = latin = 0
        for ch in text:
            if _in_ranges(ch, CYRILLIC):
                cyrillic += 1
            elif _in_ranges(ch, LATIN):
                latin += 1

        total = cyrillic + latin
        if total == 0:
            # Digits, punctuation or emoji only — nothing to go on.
            return LanguageGuess(self._default, 0.0, self.name)

        ratio = cyrillic / total
        # Confidence is simply the share of evidence for the winning script.
        if ratio >= self._threshold:
            return LanguageGuess("ru", round(ratio, 4), self.name)
        return LanguageGuess("en", round(1.0 - ratio, 4), self.name)


def resolve_language(
    text: str, requested: str, detector: LanguageDetector
) -> LanguageGuess:
    """Honour an explicit lang, otherwise detect (spec 2.1: explicit skips detection)."""
    if requested != "auto":
        return LanguageGuess(requested, 1.0, "explicit")
    return detector.detect(text)
