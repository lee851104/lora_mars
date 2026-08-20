"""The regression suite for the bug this repo exists to prevent.

Original notebook:

    hf_dataset = Dataset.from_list([... for _, r in df.iterrows()])   # ALL rows
    trainer.train()                                                    # trains on all
    train_df, test_df = train_test_split(df, test_size=0.1)            # splits AFTER

Every "test" record had already been trained on, so BLEU/ROUGE measured
memorisation. These tests assert the invariants that make that impossible.
"""

from __future__ import annotations

import json

import pytest

from src.data.split import (
    SPLIT_NAMES,
    assert_disjoint,
    compute_split_hash,
    load_manifest,
    load_split,
    make_splits,
)
from src.models.train import TRAIN_SPLIT, build_train_dataset


def test_splits_are_pairwise_disjoint(cfg, built_records):
    manifest = make_splits(cfg)
    ids = manifest["ids"]
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1:]:
            assert not (set(ids[left]) & set(ids[right])), f"{left} overlaps {right}"


def test_splits_cover_every_record_exactly_once(cfg, built_records):
    manifest = make_splits(cfg)
    all_ids = [i for name in SPLIT_NAMES for i in manifest["ids"][name]]
    assert len(all_ids) == len(built_records)
    assert set(all_ids) == {r["image_id"] for r in built_records}
    assert len(all_ids) == len(set(all_ids))


def test_split_sizes_match_configured_ratios(cfg, built_records):
    manifest = make_splits(cfg)
    total = len(built_records)
    counts = manifest["counts"]
    assert counts["test"] == pytest.approx(total * cfg.split.test_size, abs=2)
    assert counts["val"] == pytest.approx(total * cfg.split.val_size, abs=2)
    assert counts["train"] > counts["test"] + counts["val"]


def test_training_data_contains_no_held_out_record(cfg, built_records):
    """The exact failure mode of the original notebook."""
    manifest = make_splits(cfg)
    train_records = load_split(cfg, TRAIN_SPLIT)
    train_ids = {r["image_id"] for r in train_records}

    held_out = set(manifest["ids"]["val"]) | set(manifest["ids"]["test"])
    leaked = train_ids & held_out
    assert not leaked, f"{len(leaked)} held-out records reached the training split: {sorted(leaked)[:5]}"

    # and the dataset actually handed to the trainer is built from those records only
    dataset = build_train_dataset(cfg, train_records)
    assert len(dataset) == len(train_records)


def test_evaluation_cannot_load_the_training_split(cfg, built_records):
    make_splits(cfg)
    load_split(cfg, "test", for_eval=True)  # allowed
    load_split(cfg, "val", for_eval=True)  # allowed
    with pytest.raises(ValueError, match="evaluation may only use"):
        load_split(cfg, "train", for_eval=True)


def test_split_is_deterministic_for_a_fixed_seed(cfg, built_records):
    first = make_splits(cfg)
    second = make_splits(cfg)
    assert first["split_hash"] == second["split_hash"]
    assert first["ids"] == second["ids"]


def test_split_hash_changes_when_the_split_changes(cfg, built_records):
    baseline = make_splits(cfg)["split_hash"]
    cfg.split.seed = int(cfg.split.seed) + 1
    assert make_splits(cfg)["split_hash"] != baseline


def test_assert_disjoint_rejects_an_overlap():
    with pytest.raises(AssertionError, match="data leakage"):
        assert_disjoint({"train": ["a", "b"], "val": ["c"], "test": ["b"]})


def test_assert_disjoint_rejects_duplicates_inside_one_split():
    with pytest.raises(AssertionError, match="duplicate ids"):
        assert_disjoint({"train": ["a", "a"], "val": ["b"], "test": ["c"]})


def test_split_hash_is_order_insensitive():
    left = compute_split_hash({"train": ["a", "b"], "val": ["c"], "test": ["d"]})
    right = compute_split_hash({"train": ["b", "a"], "val": ["c"], "test": ["d"]})
    assert left == right


def test_adapter_split_hash_must_match_the_current_manifest(cfg, built_records, tmp_path):
    """A model trained before a re-split must be detected, not silently scored."""
    from src.models.loader import read_train_meta

    manifest = make_splits(cfg)
    adapter_dir = tmp_path / "models" / "lora"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "train_meta.json").write_text(
        json.dumps({"split_hash": manifest["split_hash"]}), encoding="utf-8"
    )

    assert read_train_meta(adapter_dir)["split_hash"] == load_manifest(cfg)["split_hash"]

    # re-splitting with a different seed must invalidate that stamp
    cfg.split.seed = int(cfg.split.seed) + 7
    make_splits(cfg)
    assert read_train_meta(adapter_dir)["split_hash"] != load_manifest(cfg)["split_hash"]
