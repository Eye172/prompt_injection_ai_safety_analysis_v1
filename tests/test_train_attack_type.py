"""Tests for the attack-type head trainer.

The dataset download and the real embeddings need network and torch, so those
are exercised in the Stage 2 notebook. Everything here — class filtering,
fitting, the round trip through the saved artifact — runs offline against the
stub backend and synthetic embeddings.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401  (puts the repo root on sys.path)
import numpy as np

from api.detection.attack_type import AttackTypeHead
from api.schemas import AttackType
from scripts.train_attack_type import embed_texts, fit_language_head

logging.disable(logging.INFO)


def _separable_data(n_per_class: int = 60, dim: int = 16, seed: int = 0):
    """Two well-separated clusters, one per class."""
    rng = np.random.default_rng(seed)
    centres = {"role_play": np.zeros(dim), "instruction_override": np.ones(dim) * 3}
    X, y = [], []
    for label, centre in centres.items():
        X.append(rng.normal(size=(n_per_class, dim)) * 0.3 + centre)
        y.extend([label] * n_per_class)
    return np.vstack(X), y


class FitLanguageHeadTests(unittest.TestCase):
    def test_fits_and_reports_metrics(self):
        X, y = _separable_data()
        bundle, report = fit_language_head(X, y, min_class_count=10)

        self.assertEqual(
            sorted(bundle["classes"]), ["instruction_override", "role_play"]
        )
        # The clusters are separable, so a working fit must score well.
        self.assertGreater(report["report"]["accuracy"], 0.9)
        self.assertEqual(report["dropped_classes"], {})

    def test_rare_classes_are_dropped(self):
        X, y = _separable_data(n_per_class=40)
        # Add a class with too few examples to learn or evaluate honestly.
        X = np.vstack([X, np.random.default_rng(1).normal(size=(3, X.shape[1]))])
        y = y + ["encoding_obfuscation"] * 3

        bundle, report = fit_language_head(X, y, min_class_count=25)

        self.assertNotIn("encoding_obfuscation", bundle["classes"])
        self.assertEqual(report["dropped_classes"], {"encoding_obfuscation": 3})

    def test_refuses_when_fewer_than_two_usable_classes(self):
        X, y = _separable_data(n_per_class=5)
        with self.assertRaises(ValueError):
            fit_language_head(X, y, min_class_count=25)


class ArtifactRoundTripTests(unittest.TestCase):
    def test_saved_artifact_loads_and_predicts(self):
        """The trainer's output must be exactly what AttackTypeHead expects."""
        import joblib

        X, y = _separable_data()
        bundle, _ = fit_language_head(X, y, min_class_count=10)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.joblib"
            joblib.dump({"heads": {"en": bundle}, "min_confidence": 0.3}, path)

            head = AttackTypeHead.load(path)

        self.assertIsNotNone(head)
        self.assertEqual(head.languages, ("en",))

        # A point at the 'instruction_override' cluster centre must come back
        # as that canonical type.
        prediction = head.predict(np.ones((1, X.shape[1])) * 3, "en")
        self.assertEqual(prediction, [AttackType.instruction_override])


class EmbedTextsTests(unittest.TestCase):
    def test_embeddings_preserve_input_order_and_count(self):
        from api.detection.stub_backend import StubBackend

        backend = StubBackend()
        texts = [f"sample text number {i}" for i in range(10)]

        # Batch size below the input count forces the chunking path.
        embeddings = embed_texts(backend, texts, "en", batch_size=3)

        self.assertEqual(embeddings.shape[0], len(texts))
        # Row i must be the embedding of texts[i], not of some other chunk.
        for i in (0, 4, 9):
            expected = backend.score([texts[i]], "en", with_embeddings=True).embeddings
            np.testing.assert_allclose(embeddings[i], expected[0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
