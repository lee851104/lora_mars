"""raw -> processed: apply cleaning, dedup, image validation, add heuristic label.

Writes data/processed/records.json.

This step deliberately does NOT split the data. Splitting lives in
src/data/split.py and always runs before training (see the Makefile
`features` target), which is what structurally prevents the test-set
leakage bug from the original notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs
from src.data.download import load_raw
from src.data.validate import is_valid_image, validate_records, write_figures
from src.features.clean import apply_clean, heuristic_label


def processed_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.processed_dir) / "records.json"


def build_records(cfg: DictConfig, records: list[dict]) -> list[dict]:
    """Pure transform: clean -> drop too-short -> dedup -> verify image -> label."""
    raw_dir = Path(cfg.paths.raw_dir)
    mode = cfg.features.clean_mode
    min_chars = int(cfg.features.min_caption_chars)

    cleaned: list[dict] = []
    for record in records:
        text = apply_clean(str(record.get("text", "")), mode)
        if len(text) < min_chars:
            continue
        cleaned.append({**record, "text": text})

    if cfg.features.drop_duplicate_captions:
        seen: set[str] = set()
        deduped = []
        for record in cleaned:
            if record["text"] in seen:
                continue
            seen.add(record["text"])
            deduped.append(record)
        cleaned = deduped

    valid = [r for r in cleaned if is_valid_image(raw_dir / r["image"])]

    return [
        {
            "image_id": r["image_id"],
            "image": r["image"],
            "text": r["text"],
            "label": heuristic_label(r["text"]),
        }
        for r in valid
    ]


def load_processed(cfg: DictConfig) -> list[dict]:
    path = processed_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"cannot find {path} - run `make features` first")
    return json.loads(path.read_text(encoding="utf-8"))


def main(cfg: DictConfig) -> list[dict]:
    ensure_dirs(cfg, "processed_dir", "reports_dir", "figures_dir")
    raw = load_raw(cfg)
    records = build_records(cfg, raw)

    path = processed_path(cfg)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "clean_mode": cfg.features.clean_mode,
        "raw": len(raw),
        "processed": len(records),
        "dropped": len(raw) - len(records),
        **validate_records(records, cfg.paths.raw_dir),
    }
    (Path(cfg.paths.reports_dir) / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_figures(records, cfg.paths.figures_dir)

    dropped = report["dropped"]
    print(f"raw {len(raw)} -> processed {len(records)} (dropped {dropped})")
    print(f"label distribution: {report['label_distribution']}")
    print(f"wrote {path}")
    return records


if __name__ == "__main__":
    main(config_from_args("raw -> processed feature build"))
