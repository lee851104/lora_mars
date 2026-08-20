"""LoRA fine-tuning.

Reads data/splits/train.json and nothing else. val/test are never opened
here, so there is no code path by which held-out data can reach the
optimizer. The split_hash in use is stamped into the saved adapter's
train_meta.json for tests/test_no_leakage.py to verify later.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs
from src.data.split import load_manifest, load_split
from src.features.conversation import to_conversation

TRAIN_SPLIT = "train"


def build_train_dataset(cfg: DictConfig, records: list[dict]) -> Any:
    """Records -> HF Dataset of conversations.

    Only `messages` is kept: the vision collator reads that key, and passing
    extra columns through Arrow has bitten this pipeline before.
    """
    from datasets import Dataset

    instruction = cfg.eval.instruction
    images_root = str(cfg.paths.raw_dir)
    conversations = [
        {"messages": to_conversation(r, images_root, instruction)["messages"]}
        for r in records
    ]
    return Dataset.from_list(conversations)


def _filter_supported(config_cls: Any, desired: dict[str, Any]) -> tuple[dict, list[str]]:
    """Keep only kwargs this trl version's SFTConfig actually accepts.

    trl renamed `max_seq_length` to `max_length`; passing both lets the same
    code work across the version range pinned in pyproject.toml.
    """
    allowed = {f.name for f in dataclasses.fields(config_cls)}
    kept = {k: v for k, v in desired.items() if k in allowed and v is not None}
    dropped = sorted(set(desired) - allowed)
    return kept, dropped


def build_trainer(cfg: DictConfig, model: Any, tokenizer: Any, dataset: Any) -> Any:
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    FastVisionModel.for_training(model)

    desired = {
        "per_device_train_batch_size": int(cfg.train.per_device_train_batch_size),
        "gradient_accumulation_steps": int(cfg.train.gradient_accumulation_steps),
        "warmup_steps": int(cfg.train.warmup_steps),
        "max_steps": cfg.train.max_steps,
        "num_train_epochs": cfg.train.num_train_epochs,
        "learning_rate": float(cfg.train.learning_rate),
        "weight_decay": float(cfg.train.weight_decay),
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "logging_steps": int(cfg.train.logging_steps),
        "optim": cfg.train.optim,
        "seed": int(cfg.seed),
        "output_dir": str(cfg.paths.outputs_dir),
        "report_to": "none",
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": int(cfg.model.max_seq_length),
        "max_seq_length": int(cfg.model.max_seq_length),
    }
    args_kwargs, dropped = _filter_supported(SFTConfig, desired)
    if dropped:
        print(f"[train] this trl version does not accept: {dropped}")

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(**args_kwargs),
    )


def main(cfg: DictConfig) -> dict[str, Any]:
    import torch

    from src.models.loader import attach_lora, load_base, save_adapter

    ensure_dirs(cfg, "models_dir", "reports_dir", "outputs_dir")

    manifest = load_manifest(cfg)
    records = load_split(cfg, TRAIN_SPLIT)
    print(
        f"training on {len(records)} records from the '{TRAIN_SPLIT}' split "
        f"(split_hash {manifest['split_hash']}); val/test are not opened here"
    )

    model, tokenizer = load_base(cfg)
    model = attach_lora(model, cfg)
    dataset = build_train_dataset(cfg, records)
    trainer = build_trainer(cfg, model, tokenizer, dataset)

    start_reserved = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
    stats = trainer.train()
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0

    meta = {
        "split_hash": manifest["split_hash"],
        "split_counts": manifest["counts"],
        "train_ids": sorted(r["image_id"] for r in records),
        "n_train": len(records),
        "base_id": str(cfg.model.base_id),
        "metrics": dict(stats.metrics),
        "peak_reserved_gb": round(peak_reserved, 3),
        "train_only_reserved_gb": round(peak_reserved - start_reserved, 3),
    }

    out_dir = save_adapter(model, tokenizer, cfg.lora.local_dir, cfg, meta)
    (Path(cfg.paths.reports_dir) / "train_report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    runtime = stats.metrics.get("train_runtime", 0.0)
    print(f"runtime: {runtime:.1f}s ({runtime / 60:.2f} min)")
    print(f"peak reserved VRAM: {peak_reserved:.3f} GB")
    print(f"adapter saved to {out_dir}")
    return meta


if __name__ == "__main__":
    main(config_from_args("LoRA fine-tuning (train split only)"))
