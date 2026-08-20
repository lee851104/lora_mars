"""Evaluation on a held-out split.

Three things the original notebook got wrong and this module does not:

* it evaluates the `test` split loaded through load_split(..., for_eval=True),
  which refuses to hand back `train`;
* it scores only the newly generated tokens (see infer.strip_prompt), not the
  echoed prompt;
* it decodes greedily by default, so a re-run reproduces the number.

It also reports a bootstrap confidence interval, because ~25 held-out captions
cannot support a point estimate quoted to three decimals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs
from src.data.split import load_manifest, load_split

BOOTSTRAP_PERCENTILES = (2.5, 97.5)


def check_split_consistency(cfg: DictConfig) -> dict[str, Any]:
    """Warn loudly when the adapter was trained against a different split."""
    from src.models.loader import read_train_meta, resolve_adapter_source

    manifest = load_manifest(cfg)
    source, is_local = resolve_adapter_source(cfg)
    meta = read_train_meta(source if is_local else None)

    trained_hash = meta.get("split_hash")
    current_hash = manifest["split_hash"]
    consistent = trained_hash == current_hash if trained_hash else None

    if trained_hash and not consistent:
        print(
            "[eval] WARNING: adapter was trained on split_hash "
            f"{trained_hash} but the current manifest is {current_hash}. "
            "The held-out set may overlap the data this adapter saw. "
            "Re-run `make features train` before trusting these numbers."
        )
    elif trained_hash is None:
        print("[eval] note: no train_meta.json found - cannot verify the split provenance")

    return {
        "current_split_hash": current_hash,
        "adapter_split_hash": trained_hash,
        "split_consistent": consistent,
    }


def generate_predictions(cfg: DictConfig, records: list[dict], loaded: Any) -> list[dict]:
    from PIL import Image

    from src.models.infer import generate_caption

    images_root = Path(cfg.paths.raw_dir)
    instruction = cfg.eval.instruction
    rows = []

    for index, record in enumerate(records, start=1):
        with Image.open(images_root / record["image"]) as handle:
            image = handle.convert("RGB")
        generation = generate_caption(
            loaded.model,
            loaded.tokenizer,
            image,
            instruction,
            max_new_tokens=int(cfg.eval.max_new_tokens),
            do_sample=bool(cfg.eval.do_sample),
            temperature=cfg.eval.get("temperature"),
            top_p=cfg.eval.get("top_p"),
        )
        rows.append(
            {
                "image_id": record["image_id"],
                "image": record["image"],
                "reference": record["text"].strip(),
                "prediction": generation.text,
                "prompt_tokens": generation.prompt_tokens,
                "new_tokens": generation.new_tokens,
            }
        )
        if index % 5 == 0 or index == len(records):
            print(f"  generated {index}/{len(records)}")

    return rows


def compute_metrics(
    predictions: list[str],
    references: list[str],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Corpus BLEU and per-sample ROUGE, each with a bootstrap 95% interval."""
    import numpy as np
    from evaluate import load

    if not predictions:
        raise ValueError("no predictions to score")

    non_empty = sum(1 for p in predictions if p.strip())
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
    per_sample_rouge = rouge_metric.compute(
        predictions=predictions, references=references, use_aggregator=False
    )
    rouge_arrays = {key: np.asarray(values, dtype=float) for key, values in per_sample_rouge.items()}

    rng = np.random.default_rng(bootstrap_seed)
    n = len(predictions)
    bleu_draws: list[float] = []
    rouge_draws: dict[str, list[float]] = {key: [] for key in rouge_arrays}

    for draw in range(int(bootstrap_samples)):
        idx = rng.integers(0, n, size=n)
        bleu_draws.append(corpus_bleu([predictions[i] for i in idx], [references[i] for i in idx]))
        for key, values in rouge_arrays.items():
            rouge_draws[key].append(float(values[idx].mean()))
        if bootstrap_samples >= 200 and (draw + 1) % 200 == 0:
            print(f"  bootstrap {draw + 1}/{bootstrap_samples}")

    def interval(draws: list[float]) -> list[float]:
        low, high = np.percentile(draws, BOOTSTRAP_PERCENTILES)
        return [round(float(low), 4), round(float(high), 4)]

    return {
        "n_samples": n,
        "n_non_empty_predictions": non_empty,
        "bleu": round(bleu_point, 4),
        "bleu_ci95": interval(bleu_draws),
        **{key: round(float(values.mean()), 4) for key, values in rouge_arrays.items()},
        **{f"{key}_ci95": interval(draws) for key, draws in rouge_draws.items()},
        "bootstrap_samples": int(bootstrap_samples),
    }


def main(cfg: DictConfig) -> dict[str, Any]:
    from src.models.loader import load_for_inference

    ensure_dirs(cfg, "reports_dir")
    split_name = str(cfg.eval.split)
    records = load_split(cfg, split_name, for_eval=True)
    provenance = check_split_consistency(cfg)

    loaded = load_for_inference(cfg)
    print(loaded.describe())
    if not loaded.has_adapter:
        print("[eval] WARNING: scoring the BASE model - no LoRA weights were found")

    rows = generate_predictions(cfg, records, loaded)
    metrics = compute_metrics(
        [r["prediction"] for r in rows],
        [r["reference"] for r in rows],
        bootstrap_samples=int(cfg.eval.bootstrap_samples),
        bootstrap_seed=int(cfg.eval.bootstrap_seed),
    )

    report = {
        "split": split_name,
        "decoding": {
            "max_new_tokens": int(cfg.eval.max_new_tokens),
            "do_sample": bool(cfg.eval.do_sample),
            "temperature": cfg.eval.get("temperature"),
            "top_p": cfg.eval.get("top_p"),
        },
        "adapter": loaded.describe(),
        **provenance,
        "metrics": metrics,
    }

    reports_dir = Path(cfg.paths.reports_dir)
    (reports_dir / f"eval_{split_name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if bool(cfg.eval.save_predictions):
        with (reports_dir / f"predictions_{split_name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        f"\nn={metrics['n_samples']} held-out captions. Read the CI, not the point "
        "estimate - see MODEL_CARD.md."
    )
    return report


if __name__ == "__main__":
    main(config_from_args("evaluate on a held-out split"))
