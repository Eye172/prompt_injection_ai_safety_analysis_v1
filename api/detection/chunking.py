"""Split long text into model-sized, overlapping windows.

The classifiers accept a bounded number of tokens (256 by default). Feeding
them a longer text does not raise — the tokenizer silently truncates it, so an
injection past the cut-off would be scored as benign. Chunking removes that
blind spot: every part of the input is scored, and the text's score is the
strongest signal found anywhere in it.

Windows overlap so that an attack straddling a boundary is still seen whole by
at least one window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    text: str
    #: Index of this chunk's first word within the original text, so spans
    #: found inside a chunk can be reported against the caller's text.
    word_offset: int


def split_words(text: str) -> list[str]:
    """Whitespace tokenisation shared with span localisation.

    Span offsets are reported in these units, so chunking and span search must
    agree on what a "word" is or reported spans would point at the wrong text.
    """
    return _WORD_RE.findall(text)


def chunk_text(
    text: str, *, chunk_words: int = 180, overlap_words: int = 40
) -> list[Chunk]:
    """Split ``text`` into overlapping windows of at most ``chunk_words``.

    Short text yields exactly one chunk covering the whole input, so callers
    need no special case for it.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    # An overlap at or above the window size would never advance the cursor.
    overlap = max(0, min(overlap_words, chunk_words - 1))

    words = split_words(text)
    if not words:
        return []
    if len(words) <= chunk_words:
        return [Chunk(text=text, word_offset=0)]

    step = chunk_words - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(Chunk(text=" ".join(words[start:end]), word_offset=start))
        if end == len(words):
            break
        start += step
    return chunks
