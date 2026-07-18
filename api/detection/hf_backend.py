"""Real backend: the two Stage 1 checkpoints served from memory.

Imports of torch/transformers are deferred to construction time so that the
rest of the API (and its test suite) can be imported on machines without the
ML stack installed.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .backend import ScoreResult, UnsupportedLanguageError

logger = logging.getLogger(__name__)


class _LoadedModel:
    def __init__(self, repo: str, tokenizer, model, injection_index: int) -> None:
        self.repo = repo
        self.tokenizer = tokenizer
        self.model = model
        self.injection_index = injection_index


def _resolve_injection_index(model) -> int:
    """Find which logit column means 'injection'.

    The two Stage 1 checkpoints were trained separately, so relying on a
    hardcoded index 1 would silently invert one model's scores if its label
    map ever differed. Read id2label and fall back to 1 only if it is absent.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower() in {"injection", "label_1", "1", "positive", "unsafe"}:
            if str(label).lower() != "label_1" or len(id2label) == 2:
                return int(idx)
    return 1


class HFBackend:
    """Loads both language models eagerly (spec 2.4 step 1: no cold start)."""

    name = "hf"

    def __init__(
        self,
        repos: dict[str, str],
        *,
        token: str | None = None,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        import torch  # noqa: F401  (import here, not at module scope)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = device
        self._max_length = max_length
        self._models: dict[str, _LoadedModel] = {}

        for language, repo in repos.items():
            logger.info("loading %s model from %s", language, repo)
            kwargs = {"token": token} if token else {}
            tokenizer = AutoTokenizer.from_pretrained(repo, **kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(repo, **kwargs)
            model.to(device).eval()
            self._models[language] = _LoadedModel(
                repo, tokenizer, model, _resolve_injection_index(model)
            )
            logger.info("loaded %s model (%s)", language, repo)

    def languages(self) -> tuple[str, ...]:
        return tuple(self._models)

    def is_loaded(self, language: str) -> bool:
        return language in self._models

    def repo_for(self, language: str) -> str:
        entry = self._models.get(language)
        return entry.repo if entry else ""

    def score(
        self, texts: Sequence[str], language: str, *, with_embeddings: bool = False
    ) -> ScoreResult:
        entry = self._models.get(language)
        if entry is None:
            raise UnsupportedLanguageError(language, tuple(self._models))
        if not texts:
            return ScoreResult(scores=[], embeddings=None)

        torch = self._torch
        enc = entry.tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        ).to(self._device)

        with torch.inference_mode():
            out = entry.model(**enc, output_hidden_states=with_embeddings)
            probs = torch.softmax(out.logits, dim=-1)[:, entry.injection_index]
            scores = probs.detach().cpu().tolist()

            embeddings = None
            if with_embeddings:
                # Mean-pool the last hidden state over real tokens only, so
                # padding in a batch cannot shift a short text's vector.
                hidden = out.hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embeddings = pooled.detach().cpu().numpy().astype(np.float32)

        return ScoreResult(scores=[float(s) for s in scores], embeddings=embeddings)
