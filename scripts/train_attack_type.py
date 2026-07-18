"""Fit the attack-type heads over the Stage 1 detectors' embeddings.

Spec 2.4 step 3, option (b): a light classifier on top of the frozen binary
detector, so attack typing can be added, retrained or dropped without touching
the Stage 1 models.

One head per language — the EN and RU detectors are independent checkpoints
whose embedding spaces are not comparable.

Usage (from the repo root, with torch/transformers available):

    python -m scripts.train_attack_type --output artifacts/attack_type_head.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from api.detection.attack_type import canonicalize

logger = logging.getLogger(__name__)

# Categories that describe a benign row rather than an attack technique.
NON_ATTACK_CATEGORIES = {"normal", "notinject_ru", "benign", "none"}

DEFAULT_DATASET_REPO = "Eye172/prompt-injection-multilingual"
DEFAULT_DATA_FILE = "consolidated_all.jsonl"


def load_labelled_attacks(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    data_file: str = DEFAULT_DATA_FILE,
    token: str | None = None,
) -> "list[dict]":
    """Return injection rows that carry a usable category label.

    Only a subset of the Stage 1 corpus is categorised (neuralchemy and
    jackhhao for EN, dmtrdr for RU); the rest is unlabelled and simply cannot
    train this head.
    """
    from datasets import load_dataset

    ds = load_dataset(
        dataset_repo, data_files=data_file, split="train", token=token or None
    )

    rows = []
    for row in ds:
        if row.get("label") != "injection":
            continue
        raw = (row.get("category") or "").strip()
        if not raw or raw.lower() in NON_ATTACK_CATEGORIES:
            continue
        canonical = canonicalize(raw)
        if canonical is None:
            continue
        rows.append(
            {
                "text": row["text"],
                "language": row.get("language", "en"),
                "target": canonical.value,
                "raw_category": raw,
            }
        )
    return rows


def embed_texts(backend, texts: Iterable[str], language: str, batch_size: int = 32):
    """Pooled embeddings from the language's detector, in input order."""
    texts = list(texts)
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = backend.score(batch, language, with_embeddings=True)
        if result.embeddings is None:
            raise RuntimeError("backend returned no embeddings")
        chunks.append(result.embeddings)
        logger.info("embedded %d/%d %s rows", min(start + batch_size, len(texts)), len(texts), language)
    return np.vstack(chunks)


def fit_language_head(
    embeddings: np.ndarray,
    targets: list[str],
    *,
    min_class_count: int = 25,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Fit one multinomial logistic regression; return (bundle, report)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    counts = Counter(targets)
    # Classes with a handful of examples cannot be learned or evaluated
    # honestly; dropping them is better than reporting a meaningless score.
    keep = {c for c, n in counts.items() if n >= min_class_count}
    if len(keep) < 2:
        raise ValueError(
            f"need >=2 classes with >={min_class_count} examples, got {dict(counts)}"
        )

    mask = np.array([t in keep for t in targets])
    X = embeddings[mask]
    y = np.array([t for t in targets if t in keep])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    estimator = LogisticRegression(
        max_iter=2000,
        # The corpus is heavily skewed toward a few techniques; without this
        # the rare classes are never predicted.
        class_weight="balanced",
        random_state=seed,
    )
    estimator.fit(X_train, y_train)

    report = classification_report(
        y_test, estimator.predict(X_test), output_dict=True, zero_division=0
    )
    text_report = classification_report(
        y_test, estimator.predict(X_test), zero_division=0
    )

    bundle = {
        "estimator": estimator,
        "classes": [str(c) for c in estimator.classes_],
    }
    return bundle, {
        "dropped_classes": {c: n for c, n in counts.items() if c not in keep},
        "kept_class_counts": {c: int(n) for c, n in counts.items() if c in keep},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "report": report,
        "text_report": text_report,
    }


def train(
    backend,
    output: str | Path,
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    data_file: str = DEFAULT_DATA_FILE,
    token: str | None = None,
    min_confidence: float = 0.3,
    min_class_count: int = 25,
    seed: int = 42,
) -> dict:
    import joblib

    rows = load_labelled_attacks(dataset_repo, data_file, token)
    if not rows:
        raise RuntimeError("no categorised injection rows found in the dataset")

    logger.info("loaded %d categorised attack rows", len(rows))

    heads: dict[str, dict] = {}
    metrics: dict[str, dict] = {}

    for language in sorted({r["language"] for r in rows}):
        subset = [r for r in rows if r["language"] == language]
        if not backend.is_loaded(language):
            logger.warning("no %s model loaded; skipping that head", language)
            continue

        logger.info("fitting %s head on %d rows", language, len(subset))
        embeddings = embed_texts(backend, [r["text"] for r in subset], language)
        try:
            bundle, report = fit_language_head(
                embeddings,
                [r["target"] for r in subset],
                min_class_count=min_class_count,
                seed=seed,
            )
        except ValueError as exc:
            logger.warning("skipping %s head: %s", language, exc)
            continue

        heads[language] = bundle
        metrics[language] = report

    if not heads:
        raise RuntimeError("no language head could be fitted")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"heads": heads, "min_confidence": min_confidence}, out_path)
    logger.info("wrote %s", out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    serialisable = {
        lang: {k: v for k, v in m.items() if k != "text_report"}
        for lang, m in metrics.items()
    }
    metrics_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/attack_type_head.joblib")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--token", default=None, help="HF token (private repos only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-class-count", type=int, default=25)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from api.config import get_settings
    from api.detection.hf_backend import HFBackend

    settings = get_settings()
    backend = HFBackend(
        repos={"en": settings.en_model_repo, "ru": settings.ru_model_repo},
        token=args.token,
        device=args.device,
        max_length=settings.max_length,
    )

    metrics = train(
        backend,
        args.output,
        dataset_repo=args.dataset_repo,
        data_file=args.data_file,
        token=args.token,
        min_confidence=args.min_confidence,
        min_class_count=args.min_class_count,
        seed=args.seed,
    )

    for language, report in metrics.items():
        print(f"\n=== {language} ===")
        print(report["text_report"])


if __name__ == "__main__":
    main()
