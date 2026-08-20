"""Score a predictions file. Separate from generation on purpose.

`make eval` needs a GPU; `make score` needs a CLIP checkpoint and an API key.
Splitting them means:

* a failed or expensive judge run does not cost you the generation pass again;
* you can re-score the same outputs with a changed rubric and compare fairly;
* the judge can run on a laptop while the GPU box is busy.

    make eval    # generate + score in one go
    make score   # re-score reports/predictions_test.jsonl only

Every metric reports a bootstrap 95% interval alongside its point estimate. With
roughly 25 held-out captions the interval is the number that means something.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs

KNOWN_METRICS = ("clipscore", "llm_judge", "bleu_rouge")
PERCENTILES = (2.5, 97.5)


def variant(cfg: DictConfig) -> str:
    """Filename suffix for a base-model run.

    Derived from the config rather than passed in, so a `use_adapter=false` run
    cannot overwrite the LoRA report it is meant to be compared against.
    """
    return "" if bool(cfg.eval.get("use_adapter", True)) else "_base"


def predictions_path(cfg: DictConfig, split: str) -> Path:
    return Path(cfg.paths.reports_dir) / f"predictions_{split}{variant(cfg)}.jsonl"


def report_path(cfg: DictConfig, split: str) -> Path:
    return Path(cfg.paths.reports_dir) / f"eval_{split}{variant(cfg)}.json"


def judge_detail_path(cfg: DictConfig, split: str) -> Path:
    return Path(cfg.paths.reports_dir) / f"judge_{split}{variant(cfg)}.jsonl"


def load_predictions(cfg: DictConfig, split: str) -> list[dict]:
    path = predictions_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(
            f"cannot find {path} - run `make eval` first "
            "(it writes predictions before scoring them)"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_ci(
    values: list[float], n_samples: int = 1000, seed: int = 0
) -> list[float] | None:
    """Percentile interval for the mean of per-sample scores."""
    import numpy as np

    if len(values) < 2 or n_samples < 2:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [
        float(array[rng.integers(0, len(array), size=len(array))].mean())
        for _ in range(int(n_samples))
    ]
    low, high = np.percentile(draws, PERCENTILES)
    return [round(float(low), 4), round(float(high), 4)]


def resolve_metrics(cfg: DictConfig) -> list[str]:
    requested = [str(m) for m in cfg.eval.metrics]
    unknown = [m for m in requested if m not in KNOWN_METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; pick from {list(KNOWN_METRICS)}")
    if not requested:
        raise ValueError("eval.metrics is empty - nothing to score")
    return requested


def run_clipscore(cfg: DictConfig, rows: list[dict]) -> dict[str, Any]:
    from src.models import clip_score

    out = clip_score.score(cfg, rows)
    summary = clip_score.summarise(out, len(rows))
    samples = int(cfg.eval.bootstrap_samples)
    seed = int(cfg.eval.bootstrap_seed)
    summary["clipscore_ci95"] = bootstrap_ci(out.prediction_scores, samples, seed)
    summary["clipscore_reference_ci95"] = bootstrap_ci(out.reference_scores, samples, seed)
    summary["ref_clipscore_ci95"] = bootstrap_ci(out.ref_clipscores, samples, seed)
    return summary


def run_llm_judge(cfg: DictConfig, rows: list[dict], split: str) -> dict[str, Any]:
    from src.models import llm_judge

    judgements, meta = llm_judge.judge_rows(cfg, rows)
    summary = llm_judge.summarise(judgements, meta)

    ok = [j for j in judgements if j.ok]
    if ok:
        samples = int(cfg.eval.bootstrap_samples)
        seed = int(cfg.eval.bootstrap_seed)
        for axis in ("overall", "accuracy", "hallucination_count"):
            values = [float(j.scores[axis]) for j in ok]
            summary[f"{axis}_ci95"] = bootstrap_ci(values, samples, seed)

        detail = judge_detail_path(cfg, split)
        with detail.open("w", encoding="utf-8") as handle:
            for row in llm_judge.per_sample_rows(judgements):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["per_sample_file"] = str(detail)

    return summary


def run_bleu_rouge(cfg: DictConfig, rows: list[dict]) -> dict[str, Any]:
    from src.models import text_metrics

    return text_metrics.compute(
        [r["prediction"] for r in rows],
        [r["reference"] for r in rows],
        bootstrap_samples=int(cfg.eval.bootstrap_samples),
        bootstrap_seed=int(cfg.eval.bootstrap_seed),
    )


def score(
    cfg: DictConfig,
    rows: list[dict],
    split: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the enabled metrics over already-generated predictions."""
    ensure_dirs(cfg, "reports_dir")
    metrics = resolve_metrics(cfg)
    print(f"[score] {len(rows)} predictions from '{split}', metrics: {metrics}")

    results: dict[str, Any] = {}
    for name in metrics:
        print(f"\n--- {name} ---")
        try:
            if name == "clipscore":
                results[name] = run_clipscore(cfg, rows)
            elif name == "llm_judge":
                results[name] = run_llm_judge(cfg, rows, split)
            else:
                results[name] = run_bleu_rouge(cfg, rows)
        except Exception as error:
            # one broken metric must not throw away the others
            print(f"[score] {name} failed: {type(error).__name__}: {error}")
            results[name] = {"error": f"{type(error).__name__}: {error}"}

    report = {
        "split": split,
        "n_samples": len(rows),
        **(context or {}),
        "metrics": results,
    }
    path = report_path(cfg, split)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[score] wrote {path}")
    return report


def print_summary(report: dict[str, Any]) -> None:
    """Human-readable digest. Intervals first - the point estimates are noisy."""
    metrics = report.get("metrics", {})
    print(f"\n{'=' * 62}")
    print(f"split={report['split']}  n={report['n_samples']}")

    clip = metrics.get("clipscore")
    if clip and "error" not in clip:
        print("\nCLIPScore (image/text alignment, no reference used)")
        print(f"  model      {clip['clipscore']}   CI95 {clip.get('clipscore_ci95')}")
        print(f"  reference  {clip['clipscore_reference']}   CI95 "
              f"{clip.get('clipscore_reference_ci95')}   <- the ceiling")
        if clip["truncated_predictions"]:
            print(f"  WARNING {clip['truncated_predictions']} predictions hit CLIP's "
                  "77-token limit")

    judge = metrics.get("llm_judge")
    if judge and judge.get("skipped"):
        print(
            f"\nLLM judge SKIPPED - {judge['n_calls']} calls to {judge['model']} "
            f"would cost ~${judge['estimated_usd']}"
        )
        print("  set llm_judge.cost_estimate_only=false to run it")
    elif judge and "error" not in judge and judge.get("n_judged"):
        print(f"\nLLM judge ({judge.get('model')}, rubric {judge['rubric_version']}, "
              f"n={judge['n_judged']})")
        print(f"  overall              {judge['overall']}   CI95 {judge.get('overall_ci95')}")
        print(f"  accuracy             {judge['accuracy']}   CI95 {judge.get('accuracy_ci95')}")
        print(f"  style_match          {judge['style_match']}")
        print(f"  fluency              {judge['fluency']}")
        print(f"  hallucinations/cap   {judge['hallucination_count']}   CI95 "
              f"{judge.get('hallucination_count_ci95')}")
        print(f"  hallucination-free   {judge['hallucination_free_rate']:.0%}")
        if judge["n_failed"]:
            print(f"  WARNING {judge['n_failed']} judge calls failed")

    text = metrics.get("bleu_rouge")
    if text and "error" not in text:
        print("\nBLEU / ROUGE (surface overlap - blind to correctness)")
        print(f"  bleu     {text['bleu']}   CI95 {text.get('bleu_ci95')}")
        print(f"  rougeL   {text.get('rougeL')}   CI95 {text.get('rougeL_ci95')}")

    for name, payload in metrics.items():
        if isinstance(payload, dict) and "error" in payload:
            print(f"\n{name}: FAILED - {payload['error']}")

    print(f"\nRead the intervals, not the decimals. n={report['n_samples']} is small.")
    print("=" * 62)


def main(cfg: DictConfig) -> dict[str, Any]:
    split = str(cfg.eval.split)
    rows = load_predictions(cfg, split)
    report = score(cfg, rows, split, {"scored_from": str(predictions_path(cfg, split))})
    print_summary(report)
    return report


if __name__ == "__main__":
    main(config_from_args("score an existing predictions file"))
