"""CIDEr: consensus-based image-caption evaluation.

CIDEr compares a generated caption with the reference caption(s) through
TF-IDF weighted 1--4 grams.  It complements CLIPScore: CIDEr is
reference-aware, while CLIPScore checks image/text alignment without relying
on the reference wording.

This wrapper uses the standard MS COCO implementation from ``pycocoevalcap``.
The astronomy dataset has one reference caption per image, so the result is a
useful lexical-consensus signal, not a complete factual-correctness measure.
"""

from __future__ import annotations

from typing import Any


def _corpus_score(predictions: list[str], references: list[str]) -> tuple[float, list[float]]:
    """Score matched prediction/reference lists with the COCO CIDEr scorer."""
    from pycocoevalcap.cider.cider import Cider

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("no predictions to score")

    # The scorer expects COCO-style {image_id: [caption, ...]} mappings.
    gts = {str(i): [reference] for i, reference in enumerate(references)}
    res = {str(i): [prediction] for i, prediction in enumerate(predictions)}
    corpus, per_sample = Cider(n=4, sigma=6.0).compute_score(gts, res)
    return float(corpus), [float(value) for value in per_sample]


def compute(
    predictions: list[str],
    references: list[str],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Return corpus CIDEr plus a deterministic bootstrap 95% interval."""
    import numpy as np

    point, per_sample = _corpus_score(predictions, references)
    n = len(predictions)
    interval: list[float] | None = None

    if n >= 2 and bootstrap_samples >= 2:
        rng = np.random.default_rng(bootstrap_seed)
        values = np.asarray(per_sample, dtype=float)
        # Keep the corpus IDF fixed. Re-fitting it for every bootstrap draw can
        # make a draw with repeated image ids degenerate, which measures the
        # resampling artefact rather than uncertainty in the caption scores.
        draws = [
            float(values[rng.integers(0, n, size=n)].mean())
            for _ in range(int(bootstrap_samples))
        ]
        low, high = np.percentile(draws, (2.5, 97.5))
        interval = [round(float(low), 4), round(float(high), 4)]

    return {
        "n_samples": n,
        "n_non_empty_predictions": sum(bool(text.strip()) for text in predictions),
        "cider": round(point, 4),
        "cider_ci95": interval,
        "mean_per_sample_cider": round(float(np.mean(per_sample)), 4),
        "note": (
            "reference-aware TF-IDF n-gram consensus; this dataset has one "
            "reference caption per image, so CIDEr is not a factuality metric"
        ),
    }
