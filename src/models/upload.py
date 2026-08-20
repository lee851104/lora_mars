"""Upload the trained LoRA adapter to the Hugging Face Hub.

A Space cannot read models/lora/ off your laptop, so the adapter has to live on
the Hub before the demo can use it. This is also what makes `make weights`
work on a fresh clone.

    huggingface-cli login          # or export HF_TOKEN=...
    make upload OVERRIDE="upload.repo_id=<user>/<adapter-repo>"

Only the adapter is uploaded (a few tens of MB), never the base model.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from src.config import config_from_args

CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
tags:
  - lora
  - vision-language
  - astronomy
  - unsloth
license: other
---

# {repo_id}

LoRA adapter for `{base_model}`, fine-tuned for astronomy image captioning.

- rank r={r}, alpha={alpha}
- trained on {n_train} held-out-clean records (split_hash `{split_hash}`)

## Usage

```python
from peft import PeftModel
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "{base_model}", load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "{repo_id}")
FastVisionModel.for_inference(model)
```

## Limitations

Not for scientific interpretation. Trained on ~250 captions for well under one
epoch, so it learned output *style* rather than astronomy knowledge. Full
limitations, known biases and out-of-scope uses: see `MODEL_CARD.md` in the
source repository.

Source: {source_url}
"""


def build_card(repo_id: str, adapter_dir: Path, cfg: DictConfig) -> str:
    """Model card for the uploaded adapter, filled from what was actually trained."""
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    meta_path = adapter_dir / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    return CARD_TEMPLATE.format(
        repo_id=repo_id,
        base_model=adapter_config.get("base_model_name_or_path", cfg.model.base_id),
        r=adapter_config.get("r", cfg.lora.r),
        alpha=adapter_config.get("lora_alpha", cfg.lora.alpha),
        n_train=meta.get("n_train", "unknown"),
        split_hash=meta.get("split_hash", "unknown"),
        source_url=cfg.upload.get("source_url") or "(set upload.source_url in configs)",
    )


def upload(cfg: DictConfig) -> str:
    from huggingface_hub import HfApi

    from src.models.download import is_adapter_dir

    repo_id = cfg.upload.get("repo_id") or cfg.lora.get("repo_id")
    if not repo_id:
        raise ValueError(
            "no target repo. Set upload.repo_id in configs/config.yaml or pass\n"
            '  make upload OVERRIDE="upload.repo_id=<user>/<adapter-repo>"'
        )

    adapter_dir = Path(cfg.lora.local_dir)
    if not is_adapter_dir(adapter_dir):
        raise FileNotFoundError(
            f"no adapter at {adapter_dir} - run `make train` first "
            "(there is nothing to upload yet)"
        )

    private = bool(cfg.upload.private)
    files = sorted(p for p in adapter_dir.rglob("*") if p.is_file())
    total_mb = sum(p.stat().st_size for p in files) / 1024**2

    print(f"uploading {len(files)} files ({total_mb:.1f} MB)")
    print(f"  from: {adapter_dir}")
    print(f"  to  : {repo_id}  (private={private})")

    card_path = adapter_dir / "README.md"
    card_path.write_text(build_card(str(repo_id), adapter_dir, cfg), encoding="utf-8")

    api = HfApi()
    api.create_repo(repo_id=str(repo_id), private=private, exist_ok=True)
    api.upload_folder(
        repo_id=str(repo_id),
        folder_path=str(adapter_dir),
        commit_message=cfg.upload.get("commit_message") or "upload LoRA adapter",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"\ndone: {url}")
    print("Point the demo at it with:")
    print(f'  make weights OVERRIDE="lora.repo_id={repo_id}"')
    print(f"  or set the Space variable LORA_REPO_ID={repo_id}")
    return url


if __name__ == "__main__":
    upload(config_from_args("upload the LoRA adapter to the Hugging Face Hub"))
