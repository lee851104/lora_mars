"""CLIPScore - reference-free image/text alignment.

Why this instead of BLEU: BLEU and ROUGE compare the prediction to a reference
*string*. A caption that renames every object but keeps the sentence shape
scores well; a correct caption phrased differently scores badly. CLIPScore
never looks at the reference - it embeds the image and the caption into CLIP's
shared space and measures the angle between them, so it responds to whether the
caption describes *this picture*.

Definitions used here (w = clipscore.scale, default 2.5):

    clipscore            = w * max(cos(image, prediction), 0)
    clipscore_reference  = w * max(cos(image, reference), 0)
    text_similarity      =     max(cos(prediction, reference), 0)
    ref_clipscore        = harmonic_mean(clipscore, w * text_similarity)

`clipscore_reference` is the point of the whole thing: it is the score the
human-written caption gets on the same images. A model CLIPScore of 0.72 means
nothing on its own; next to a reference score of 0.78 it means something.

Caveat that must be read with the number: CLIP's text encoder truncates at 77
tokens. The report lists how many captions were cut, because a high truncation
rate means the score only describes the first half of each caption.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

MIN_CUDA_FREE_GB = 3.0


@dataclass
class ClipScoreOutput:
    prediction_scores: list[float] = field(default_factory=list)
    reference_scores: list[float] = field(default_factory=list)
    text_similarities: list[float] = field(default_factory=list)
    ref_clipscores: list[float] = field(default_factory=list)
    truncated_predictions: list[str] = field(default_factory=list)
    truncated_references: list[str] = field(default_factory=list)
    device: str = "cpu"
    model_id: str = ""


def resolve_device(requested: str) -> str:
    """`auto` picks cuda only when there is room - the VLM may still hold VRAM."""
    if requested in ("cpu", "cuda"):
        return requested

    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes / 1024**3 < MIN_CUDA_FREE_GB:
        print(
            f"[clipscore] only {free_bytes / 1024**3:.1f} GB VRAM free "
            f"(< {MIN_CUDA_FREE_GB} GB) - running CLIP on CPU"
        )
        return "cpu"
    return "cuda"


def count_truncated(tokenizer: Any, texts: list[str], limit: int) -> list[int]:
    """Indices whose token count exceeds CLIP's hard text limit."""
    encoded = tokenizer(texts, truncation=False, padding=False)["input_ids"]
    return [i for i, ids in enumerate(encoded) if len(ids) > limit]


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return 2 * left * right / (left + right)


def score(cfg: DictConfig, rows: list[dict]) -> ClipScoreOutput:
    """Score every row. Returns per-sample lists; aggregation happens in score.py."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not rows:
        raise ValueError("no rows to score")

    model_id = str(cfg.clipscore.model_id)
    device = resolve_device(str(cfg.clipscore.device))
    scale = float(cfg.clipscore.scale)
    limit = int(cfg.clipscore.max_text_tokens)
    batch_size = int(cfg.clipscore.batch_size)
    images_root = Path(cfg.paths.raw_dir)

    print(f"[clipscore] loading {model_id} on {device}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    predictions = [r["prediction"] for r in rows]
    references = [r["reference"] for r in rows]
    out = ClipScoreOutput(device=device, model_id=model_id)

    for index in count_truncated(processor.tokenizer, predictions, limit):
        out.truncated_predictions.append(rows[index]["image_id"])
    for index in count_truncated(processor.tokenizer, references, limit):
        out.truncated_references.append(rows[index]["image_id"])

    def embed_text(texts: list[str]) -> Any:
        batch = processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=limit
        ).to(device)
        features = model.get_text_features(**batch)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def embed_images(paths: list[Path]) -> Any:
        images = []
        for path in paths:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        batch = processor(images=images, return_tensors="pt").to(device)
        if dtype == torch.float16:
            batch["pixel_values"] = batch["pixel_values"].half()
        features = model.get_image_features(**batch)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            image_vectors = embed_images([images_root / r["image"] for r in chunk])
            prediction_vectors = embed_text([r["prediction"] for r in chunk])
            reference_vectors = embed_text([r["reference"] for r in chunk])

            prediction_cos = (image_vectors * prediction_vectors).sum(-1).clamp(min=0)
            reference_cos = (image_vectors * reference_vectors).sum(-1).clamp(min=0)
            text_cos = (prediction_vectors * reference_vectors).sum(-1).clamp(min=0)

            for p_cos, r_cos, t_cos in zip(
                prediction_cos.tolist(), reference_cos.tolist(), text_cos.tolist(), strict=True
            ):
                prediction_score = scale * p_cos
                out.prediction_scores.append(prediction_score)
                out.reference_scores.append(scale * r_cos)
                out.text_similarities.append(t_cos)
                out.ref_clipscores.append(_harmonic_mean(prediction_score, scale * t_cos))

            print(f"  scored {min(start + batch_size, len(rows))}/{len(rows)}")

    # drop the reference (not `del` - the closures above still bind the name)
    # so the CUDA allocation can be reclaimed before the caller does anything else
    model = None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return out


def summarise(out: ClipScoreOutput, n_rows: int) -> dict[str, Any]:
    """Point estimates plus the caveats a reader needs to interpret them."""
    from statistics import mean

    return {
        "model_id": out.model_id,
        "device": out.device,
        "clipscore": round(mean(out.prediction_scores), 4),
        "clipscore_reference": round(mean(out.reference_scores), 4),
        "text_similarity": round(mean(out.text_similarities), 4),
        "ref_clipscore": round(mean(out.ref_clipscores), 4),
        "truncated_predictions": len(out.truncated_predictions),
        "truncated_references": len(out.truncated_references),
        "truncated_prediction_ids": out.truncated_predictions,
        "note": (
            "clipscore_reference is the score the human captions get on the same "
            "images - read the model score against that ceiling, not against 0 or 1. "
            "CLIP truncates text at 77 tokens; a high truncation count means the "
            "score only covers the start of each caption."
        ),
        "n_samples": n_rows,
    }
