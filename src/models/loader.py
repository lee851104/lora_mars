"""Model loading. Heavy imports (torch, unsloth) are lazy so that CPU-only
CI can import this module and run the unit tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

TRAIN_META_NAME = "train_meta.json"


@dataclass
class LoadedModel:
    """What actually got loaded, so the UI and reports can be honest about it."""

    model: Any
    tokenizer: Any
    base_id: str
    adapter_source: str | None = None
    adapter_is_local: bool = False
    train_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_adapter(self) -> bool:
        return self.adapter_source is not None

    def describe(self) -> str:
        if not self.has_adapter:
            return f"base only: {self.base_id} (no LoRA weights - outputs are stock model)"
        kind = "local" if self.adapter_is_local else "hub"
        line = f"base: {self.base_id}\nLoRA ({kind}): {self.adapter_source}"
        split_hash = self.train_meta.get("split_hash")
        if split_hash:
            line += f"\ntrained on split_hash: {split_hash}"
        return line


def free_vram_gb() -> float | None:
    """Free VRAM in GiB, or None when there is no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return None
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / 1024**3


def assert_gpu_headroom(min_free_gb: float) -> None:
    """Refuse to load when a previous failed attempt is still holding VRAM.

    A failed `from_pretrained` leaves the partially loaded model alive because
    the notebook keeps the traceback (and therefore every frame local) around.
    On a 14.5 GB T4 the ~7.9 GB 4-bit model then cannot fit a second time and
    you get a confusing OOM instead of the real error. Failing fast here with
    an actionable message is much kinder than that OOM.
    """
    free = free_vram_gb()
    if free is None:
        return
    if free < min_free_gb:
        raise RuntimeError(
            f"only {free:.2f} GB VRAM free, need >= {min_free_gb:.2f} GB.\n"
            "A previous failed load is probably still resident. Restart the "
            "runtime/kernel and run again - do not simply re-run this cell, "
            "the second load will OOM."
        )


def clamp_image_resolution(tokenizer: Any, max_pixels: int | None) -> dict[str, Any]:
    """Cap the vision-token budget on dynamic-resolution processors.

    Qwen2-VL and Qwen2.5-VL size their vision input from the actual image, with
    a default ceiling around 12.8M pixels - roughly sixteen thousand vision
    tokens for one photo, which no 16 GB card survives. Clamping to ~590k pixels
    (768x768) costs some detail and buys a ~750 token budget instead.

    Architectures with fixed tiling (mllama) have no such knob; this is then a
    no-op, so the same call is safe for every model preset.
    """
    if not max_pixels:
        return {}

    image_processor = getattr(tokenizer, "image_processor", None)
    if image_processor is None:
        return {}

    applied: dict[str, Any] = {}
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = int(max_pixels)
        applied["max_pixels"] = int(max_pixels)

    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size:
        size["longest_edge"] = int(max_pixels)
        applied["size.longest_edge"] = int(max_pixels)

    return applied


def load_base(cfg: DictConfig) -> tuple[Any, Any]:
    """Load the 4-bit base vision model."""
    from unsloth import FastVisionModel

    assert_gpu_headroom(float(cfg.model.min_free_vram_gb))
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.model.base_id,
        load_in_4bit=bool(cfg.model.load_in_4bit),
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
        local_files_only=bool(cfg.model.local_files_only),
    )

    applied = clamp_image_resolution(tokenizer, cfg.model.get("image_max_pixels"))
    if applied:
        print(f"[loader] capped vision resolution: {applied}")

    return model, tokenizer


def attach_lora(model: Any, cfg: DictConfig) -> Any:
    """Attach a fresh LoRA adapter for training."""
    from unsloth import FastVisionModel

    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(cfg.lora.finetune_vision_layers),
        finetune_language_layers=bool(cfg.lora.finetune_language_layers),
        finetune_attention_modules=bool(cfg.lora.finetune_attention_modules),
        finetune_mlp_modules=bool(cfg.lora.finetune_mlp_modules),
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        bias=cfg.lora.bias,
        random_state=int(cfg.seed),
        use_rslora=bool(cfg.lora.use_rslora),
        loftq_config=None,
    )


def resolve_adapter_source(cfg: DictConfig) -> tuple[str | None, bool]:
    """Where to get LoRA weights from: local dir wins, then hub repo_id.

    Returns (source, is_local). (None, False) means no adapter is available -
    callers should say so rather than silently serving the base model as if
    it were fine-tuned.
    """
    local_dir = Path(cfg.lora.local_dir)
    if (local_dir / "adapter_config.json").exists():
        return str(local_dir), True
    repo_id = cfg.lora.get("repo_id")
    if repo_id:
        return str(repo_id), False
    return None, False


def read_train_meta(source: str | None) -> dict[str, Any]:
    if not source:
        return {}
    path = Path(source) / TRAIN_META_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_for_inference(cfg: DictConfig) -> LoadedModel:
    """Load base + LoRA (when available) and put it into inference mode."""
    from unsloth import FastVisionModel

    source, is_local = resolve_adapter_source(cfg)
    model, tokenizer = load_base(cfg)

    if source is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, source)

    FastVisionModel.for_inference(model)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        base_id=str(cfg.model.base_id),
        adapter_source=source,
        adapter_is_local=is_local,
        train_meta=read_train_meta(source if is_local else None),
    )


def save_adapter(
    model: Any,
    tokenizer: Any,
    out_dir: str | Path,
    cfg: DictConfig,
    meta: dict[str, Any],
) -> Path:
    """Save the adapter plus the provenance needed by tests and reports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    payload = {**meta, "config": OmegaConf.to_container(cfg, resolve=True)}
    (out_dir / TRAIN_META_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return out_dir
