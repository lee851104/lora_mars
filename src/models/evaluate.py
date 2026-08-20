"""Generate predictions on a held-out split, then score them.

Three things the original notebook got wrong and this module does not:

* it evaluates the `test` split loaded through load_split(..., for_eval=True),
  which refuses to hand back `train`;
* it scores only the newly generated tokens (see infer.strip_prompt), not the
  echoed prompt;
* it decodes greedily by default, so a re-run reproduces the number.

Predictions are written to disk *before* any metric runs. Scoring can fail -
an API key expires, a CLIP download times out - and when it does you should not
have to spend GPU time regenerating. `make score` picks up from the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args, ensure_dirs
from src.data.split import load_manifest, load_split
from src.models.score import predictions_path, print_summary, score


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


def write_predictions(cfg: DictConfig, rows: list[dict], split: str) -> Path:
    path = predictions_path(cfg, split)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {path} ({len(rows)} predictions)")
    return path


def release_model(loaded: Any) -> None:
    """Free the VLM before scoring so CLIP can have the GPU."""
    import gc

    import torch

    loaded.model = None
    loaded.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    context = {
        "base_id": str(cfg.model.base_id),
        "adapter": loaded.describe(),
        "decoding": {
            "max_new_tokens": int(cfg.eval.max_new_tokens),
            "do_sample": bool(cfg.eval.do_sample),
            "temperature": cfg.eval.get("temperature"),
            "top_p": cfg.eval.get("top_p"),
        },
        **provenance,
    }

    rows = generate_predictions(cfg, records, loaded)
    if bool(cfg.eval.save_predictions):
        write_predictions(cfg, rows, split_name)
    release_model(loaded)

    report = score(cfg, rows, split_name, context)
    print_summary(report)
    return report


if __name__ == "__main__":
    main(config_from_args("generate predictions on a held-out split, then score them"))
