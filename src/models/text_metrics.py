"""BLEU and ROUGE, kept as an optional secondary signal.

These are not enabled by default. They compare the prediction to a reference
*string*, which for image captioning means they reward sentence shape and
punish valid paraphrase. A caption that renames every object but keeps the
reference's structure can score well; that is exactly the failure CLIPScore and
the LLM judge exist to catch.

They stay in the repo for one good reason: continuity with the original notebook
and with the wider captioning literature. Turn them on with

    make eval OVERRIDE="eval.metrics=[clipscore,llm_judge,bleu_rouge]"
"""

from __future__ import annotations

from typing import Any


def compute(
    predictions: list[str],
    references: list[str],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Corpus BLEU with a resampled interval, plus per-sample ROUGE."""
    import numpy as np
    from evaluate import load

    if not predictions:
        raise ValueError("no predictions to score")

    bleu_metric = load("bleu")
    rouge_metric = load("rouge")

    def corpus_bleu(preds: list[str], refs: list[str]) -> float:
        # BLEU is undefined without at least one non-empty hypothesis
        if not any(p.strip() for p in preds):
            return 0.0
        try:
            return float(
                bleu_metric.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
            )
        except ZeroDivisionError:
            return 0.0

    bleu_point = corpus_bleu(predictions, references)
    per_sample = rouge_metric.compute(
        predictions=predictions, references=references, use_aggregator=False
    )
    rouge_arrays = {k: np.asarray(v, dtype=float) for k, v in per_sample.items()}

    rng = np.random.default_rng(bootstrap_seed)
    n = len(predictions)
    bleu_draws: list[float] = []
    rouge_draws: dict[str, list[float]] = {k: [] for k in rouge_arrays}

    for draw in range(int(bootstrap_samples)):
        idx = rng.integers(0, n, size=n)
        bleu_draws.append(
            corpus_bleu([predictions[i] for i in idx], [references[i] for i in idx])
        )
        for key, values in rouge_arrays.items():
            rouge_draws[key].append(float(values[idx].mean()))
        if bootstrap_samples >= 200 and (draw + 1) % 200 == 0:
            print(f"  bleu bootstrap {draw + 1}/{bootstrap_samples}")

    def interval(draws: list[float]) -> list[float]:
        low, high = np.percentile(draws, (2.5, 97.5))
        return [round(float(low), 4), round(float(high), 4)]

    return {
        "n_samples": n,
        "n_non_empty_predictions": sum(1 for p in predictions if p.strip()),
        "bleu": round(bleu_point, 4),
        "bleu_ci95": interval(bleu_draws),
        **{k: round(float(v.mean()), 4) for k, v in rouge_arrays.items()},
        **{f"{k}_ci95": interval(v) for k, v in rouge_draws.items()},
        "note": "surface n-gram overlap only - blind to factual correctness",
    }
