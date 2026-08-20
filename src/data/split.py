"""Dataset splitting. The ONLY place in this project allowed to assign
records to train/val/test.

The original notebook's fatal bug: it trained on the full dataframe and
only split into train/test AFTERWARDS, so 100% of the "test set" had been
seen during training. BLEU/ROUGE was measuring memorisation.

Three mechanisms make that bug unwritable here:

1. Splits are materialised to data/splits/{train,val,test}.json. Training
   reads train.json only; evaluation reads test.json only, and load_split
   refuses to hand `train` to an evaluator.
2. manifest.json records the id list per split plus a split_hash. Training
   stamps that split_hash into models/<run>/train_meta.json.
3. tests/test_no_leakage.py asserts the splits are pairwise disjoint, that
   their union is complete, and that a trained model's recorded split_hash
   still matches the current manifest - so silently re-splitting after
   training fails CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs

SPLIT_NAMES = ("train", "val", "test")
MANIFEST_NAME = "manifest.json"
# Evaluation must never read `train` - that would reintroduce the bug.
EVAL_ALLOWED = ("val", "test")


def splits_dir(cfg: DictConfig) -> Path:
    return Path(cfg.paths.splits_dir)


def split_path(cfg: DictConfig, name: str) -> Path:
    if name not in SPLIT_NAMES:
        raise ValueError(f"split name must be one of {SPLIT_NAMES}, got {name!r}")
    return splits_dir(cfg) / f"{name}.json"


def manifest_path(cfg: DictConfig) -> Path:
    return splits_dir(cfg) / MANIFEST_NAME


def compute_split_hash(ids_by_split: dict[str, list[str]]) -> str:
    """Stable hash of the split assignment. Order-insensitive, content-sensitive."""
    payload = {name: sorted(ids_by_split[name]) for name in SPLIT_NAMES}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def assert_disjoint(ids_by_split: dict[str, list[str]]) -> None:
    """The three splits must be pairwise disjoint. This is the no-leakage invariant."""
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1:]:
            overlap = set(ids_by_split[left]) & set(ids_by_split[right])
            if overlap:
                sample = sorted(overlap)[:5]
                raise AssertionError(
                    f"{left} and {right} share {len(overlap)} ids ({sample} ...) "
                    "- that is data leakage"
                )
    for name in SPLIT_NAMES:
        ids = ids_by_split[name]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{name} contains duplicate ids")


def _strata(rows: list[dict], stratify_by: str | None) -> list[str] | None:
    """Stratification labels, or None when stratifying is not possible."""
    if not stratify_by:
        return None
    values = [r.get(stratify_by, "Unknown") for r in rows]
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    # sklearn needs >= 2 members per class; fall back to random rather than crash
    if not counts or min(counts.values()) < 2:
        print(f"[split] {stratify_by} has a class with < 2 members - using random split")
        return None
    return values


def make_splits(cfg: DictConfig, records: list[dict] | None = None) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    from src.data.build import load_processed

    if records is None:
        records = load_processed(cfg)
    if len(records) < 3:
        raise ValueError(f"only {len(records)} records - cannot make three splits")

    seed = int(cfg.split.seed)
    test_size = float(cfg.split.test_size)
    val_size = float(cfg.split.val_size)
    stratify_by = cfg.split.get("stratify_by")

    holdout_fraction = test_size + val_size
    if not 0 < holdout_fraction < 1:
        raise ValueError("test_size + val_size must be strictly between 0 and 1")

    train_rows, holdout_rows = train_test_split(
        records,
        test_size=holdout_fraction,
        random_state=seed,
        shuffle=True,
        stratify=_strata(records, stratify_by),
    )
    val_rows, test_rows = train_test_split(
        holdout_rows,
        test_size=test_size / holdout_fraction,
        random_state=seed,
        shuffle=True,
        stratify=_strata(holdout_rows, stratify_by),
    )

    rows_by_split = {"train": train_rows, "val": val_rows, "test": test_rows}
    ids_by_split = {
        name: [r["image_id"] for r in rows] for name, rows in rows_by_split.items()
    }
    assert_disjoint(ids_by_split)

    total = sum(len(rows) for rows in rows_by_split.values())
    if total != len(records):
        raise AssertionError(f"split produced {total} rows from {len(records)} records")

    ensure_dirs(cfg, "splits_dir")
    for name, rows in rows_by_split.items():
        split_path(cfg, name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest = {
        "split_hash": compute_split_hash(ids_by_split),
        "params": {
            "test_size": test_size,
            "val_size": val_size,
            "seed": seed,
            "stratify_by": stratify_by,
        },
        "counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "ids": {name: sorted(ids) for name, ids in ids_by_split.items()},
    }
    manifest_path(cfg).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest(cfg: DictConfig) -> dict[str, Any]:
    path = manifest_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"cannot find {path} - run `make features` first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(cfg: DictConfig, name: str, *, for_eval: bool = False) -> list[dict]:
    """Load one split. With for_eval=True, refuses to return `train`."""
    if for_eval and name not in EVAL_ALLOWED:
        raise ValueError(
            f"evaluation may only use {EVAL_ALLOWED}, got {name!r}. "
            "Scoring BLEU on the training split is meaningless."
        )
    path = split_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"cannot find {path} - run `make features` first")
    return json.loads(path.read_text(encoding="utf-8"))


def main(cfg: DictConfig) -> dict[str, Any]:
    manifest = make_splits(cfg)
    counts = manifest["counts"]
    print(
        f"split done: train {counts['train']} / val {counts['val']} / test {counts['test']}"
    )
    print(f"split_hash = {manifest['split_hash']}")
    print(f"wrote {manifest_path(cfg)}")
    return manifest


if __name__ == "__main__":
    main(config_from_args("split the dataset (the only splitting entry point)"))
