"""Locate the words carrying the injection signal (spec 2.2, `span`).

Of the two options in the spec — attention weights or a sliding window — this
implements the sliding window. Attention mass is a well-known unreliable proxy
for attribution, and a window search asks the model the question we actually
care about ("which fragment, on its own, still scores as an attack?") using
only the public scoring interface, so it works for any backend.

Cost is bounded: the windows of one text are scored in a single batched call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .backend import DetectionBackend

_WORD_RE = re.compile(r"\S+")


@dataclass
class SpanResult:
    start_word: int
    end_word: int  # exclusive
    score: float


def locate_span(
    text: str,
    language: str,
    backend: DetectionBackend,
    *,
    window_words: int = 8,
    stride_words: int = 4,
    min_score: float = 0.5,
    full_score: float | None = None,
) -> SpanResult | None:
    """Return the highest-scoring word window, or None if none stands out.

    A window qualifies on its own merit — it must still read as an attack in
    isolation. Comparing it against the full text's score instead would almost
    never yield a span, since the complete text carries all of the evidence.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return None

    # Shorter than one window: the whole text is the span.
    if len(words) <= window_words:
        score = (
            full_score
            if full_score is not None
            else backend.score([text], language).scores[0]
        )
        return SpanResult(0, len(words), float(score))

    bounds: list[tuple[int, int]] = []
    for start in range(0, len(words), max(stride_words, 1)):
        end = min(start + window_words, len(words))
        bounds.append((start, end))
        if end == len(words):
            break

    windows = [" ".join(words[s:e]) for s, e in bounds]
    scores = backend.score(windows, language).scores

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = float(scores[best_idx])

    # No fragment is independently suspicious: the signal is spread across the
    # text rather than localised, and pointing at any one window would be
    # misleading. Report no span instead.
    if best_score < min_score:
        return None

    start, end = bounds[best_idx]
    return SpanResult(start, end, best_score)
