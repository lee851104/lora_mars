"""LoRA fine-tuning.

Reads data/splits/train.json and nothing else. val/test are never opened
here, so there is no code path by which held-out data can reach the
optimizer. The split_hash in use is stamped into the saved adapter's
train_meta.json for tests/test_no_leakage.py to verify later.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs
from src.data.split import load_manifest, load_split
from src.features.conversation import to_conversation

TRAIN_SPLIT = "train"


def supports_native_bf16(capability: tuple[int, int]) -> bool:
    """Native CUDA BF16 tensor cores start at compute capability 8.0 (Ampere)."""
    return capability[0] >= 8


def apply_gemma_t4_safety(cfg: DictConfig, *, has_cuda: bool, supports_bf16: bool) -> bool:
    """Avoid the Gemma 3 vision-backprop dtype path that fails on T4-class GPUs.

    Gemma 3 checkpoints are BF16-native, but a Tesla T4 only has FP16 tensor
    cores. Some Unsloth releases therefore fall back to FP32 and currently
    hit a BF16/FP32 LayerNorm mismatch while backpropagating through SigLIP.
    Freezing the vision tower still allows image-conditioned caption training:
    the language-side LoRA learns from the image embeddings without sending a
    gradient through SigLIP.

    This is deliberately a runtime guard rather than documentation alone. A
    user can safely use the default config on T4, while BF16-capable GPUs keep
    the configured vision-LoRA setting.
    """
    is_gemma3 = "gemma-3" in str(cfg.model.base_id).lower()
    wants_vision_lora = bool(cfg.lora.finetune_vision_layers)
    if has_cuda and is_gemma3 and not supports_bf16:
        if wants_vision_lora:
            cfg.lora.finetune_vision_layers = False
        # This needs to be set before FastVisionModel imports Unsloth, whose
        # generated LayerNorm module reads it during compilation.
        os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"
        # "unsloth" uses Unsloth's asynchronous checkpointing. The current
        # Gemma 3 / T4 path fails while that implementation recomputes SigLIP;
        # native PyTorch checkpointing is slower but avoids that dtype path.
        cfg.model.use_gradient_checkpointing = True
        print(
            "[train] Gemma 3 on a CUDA GPU without native BF16: using native "
            "gradient checkpointing and FP32 LayerNorm; vision LoRA is frozen."
        )
        return True
    return False


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

    has_cuda = torch.cuda.is_available()
    # torch.cuda.is_bf16_supported() can report emulated BF16 support on a
    # T4 with recent CUDA/PyTorch builds. Gemma needs native tensor-core BF16,
    # so decide from the device architecture instead.
    capability = torch.cuda.get_device_capability() if has_cuda else (0, 0)
    apply_gemma_t4_safety(
        cfg,
        has_cuda=has_cuda,
        supports_bf16=supports_native_bf16(capability),
    )

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
