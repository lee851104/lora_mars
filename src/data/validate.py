"""Data validation and EDA figures. Every figure lands in reports/figures/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs
from src.features.clean import heuristic_label


def is_valid_image(path: str | Path) -> bool:
    """Header-only check - much faster than decoding the whole image."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_records(records: list[dict], raw_dir: str | Path) -> dict[str, Any]:
    """Return a serialisable validation report. Does not mutate the input."""
    raw_dir = Path(raw_dir)
    missing_text = [r["image_id"] for r in records if not str(r.get("text", "")).strip()]
    missing_image_field = [r["image_id"] for r in records if not r.get("image")]
    broken_images = [
        r["image_id"] for r in records
        if r.get("image") and not is_valid_image(raw_dir / r["image"])
    ]

    captions = [r["text"] for r in records if r.get("text")]
    seen: dict[str, int] = {}
    for caption in captions:
        seen[caption] = seen.get(caption, 0) + 1
    duplicate_captions = sum(count - 1 for count in seen.values() if count > 1)

    lengths = [len(c) for c in captions] or [0]
    words = [len(c.split()) for c in captions] or [0]
    labels: dict[str, int] = {}
    for caption in captions:
        label = heuristic_label(caption)
        labels[label] = labels.get(label, 0) + 1

    return {
        "n_records": len(records),
        "n_unique_captions": len(seen),
        "missing_text": missing_text,
        "missing_image_field": missing_image_field,
        "broken_images": broken_images,
        "duplicate_captions": duplicate_captions,
        "caption_chars_mean": sum(lengths) / len(lengths),
        "caption_words_mean": sum(words) / len(words),
        "label_distribution": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
    }


def write_figures(records: list[dict], figures_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    captions = [r["text"] for r in records if r.get("text")]
    written = []

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist([len(c.split()) for c in captions], bins=20,
            color="skyblue", edgecolor="black")
    ax.set_title("Caption word-count distribution")
    ax.set_xlabel("words")
    ax.set_ylabel("captions")
    path = figures_dir / "caption_word_count.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    counts: dict[str, int] = {}
    for caption in captions:
        label = heuristic_label(caption)
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([k for k, _ in ordered], [v for _, v in ordered],
           color="steelblue", edgecolor="black")
    ax.set_title("Heuristic label distribution (imbalanced - see MODEL_CARD.md)")
    ax.set_ylabel("captions")
    path = figures_dir / "label_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    return written


def main(cfg: DictConfig) -> dict[str, Any]:
    from src.data.download import load_raw

    ensure_dirs(cfg, "reports_dir", "figures_dir")
    records = load_raw(cfg)
    report = validate_records(records, cfg.paths.raw_dir)
    out = Path(cfg.paths.reports_dir) / "data_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for path in write_figures(records, cfg.paths.figures_dir):
        print(f"figure: {path}")
    print(f"report: {out}")
    return report


if __name__ == "__main__":
    main(config_from_args(__doc__.splitlines()[0]))
