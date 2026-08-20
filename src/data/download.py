"""Materialise the Hugging Face dataset as data/raw/{data.json, images/}.

Idempotent: an existing data.json short-circuits the download.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs


def raw_paths(cfg: DictConfig) -> tuple[Path, Path]:
    raw_dir = Path(cfg.paths.raw_dir)
    return raw_dir / cfg.data.metadata_file, raw_dir / cfg.data.images_subdir


def download(cfg: DictConfig, force: bool = False) -> Path:
    from datasets import load_dataset

    ensure_dirs(cfg, "raw_dir")
    metadata_path, images_dir = raw_paths(cfg)

    if metadata_path.exists() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"{metadata_path} already exists ({len(existing)} records) - skipping download")
        return metadata_path

    images_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(cfg.data.hf_repo, split=cfg.data.hf_split)
    print(f"source columns: {dataset.column_names} | rows: {len(dataset)}")

    records = []
    for row in dataset:
        image_id = str(row["image_id"]).strip()
        if image_id.isdigit():
            image_id = image_id.zfill(3)
        relative = f"{cfg.data.images_subdir}/{image_id}.jpg"
        row["image"].convert("RGB").save(images_dir.parent / relative, quality=95)
        records.append({"image_id": image_id, "text": row["text"], "image": relative})

    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {metadata_path} ({len(records)} records)")
    return metadata_path


def load_raw(cfg: DictConfig) -> list[dict]:
    metadata_path, _ = raw_paths(cfg)
    if not metadata_path.exists():
        raise FileNotFoundError(f"cannot find {metadata_path} - run `make data` first")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    download(config_from_args("download the dataset into data/raw/"))
