"""Download LoRA weights from the Hugging Face Hub into models/lora/.

Weights are not in version control (see .gitignore), so this is how a fresh
clone gets a usable adapter without retraining:

    make weights OVERRIDE="lora.repo_id=<user>/<adapter-repo>"

or set lora.repo_id in configs/config.yaml once and just run `make weights`.
Everything downstream (evaluate, serve) prefers models/lora/ when it exists
and falls back to the hub repo id otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from src.config import config_from_args

ADAPTER_MARKERS = ("adapter_config.json",)
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def is_adapter_dir(path: str | Path) -> bool:
    path = Path(path)
    return all((path / marker).exists() for marker in ADAPTER_MARKERS)


def describe_local(path: str | Path) -> dict:
    path = Path(path)
    if not is_adapter_dir(path):
        return {"present": False, "path": str(path)}
    weights = [name for name in ADAPTER_WEIGHT_FILES if (path / name).exists()]
    size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    return {
        "present": True,
        "path": str(path),
        "weight_files": weights,
        "size_mb": round(size_bytes / 1024**2, 1),
        "base_model": config.get("base_model_name_or_path"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
    }


def download_weights(cfg: DictConfig, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    repo_id = cfg.lora.get("repo_id")
    local_dir = Path(cfg.lora.local_dir)

    if is_adapter_dir(local_dir) and not force:
        print(f"adapter already present at {local_dir} - nothing to download")
        print(json.dumps(describe_local(local_dir), ensure_ascii=False, indent=2))
        return local_dir

    if not repo_id:
        raise ValueError(
            "lora.repo_id is not set. Either fill it in configs/config.yaml or pass\n"
            '  make weights OVERRIDE="lora.repo_id=<user>/<adapter-repo>"'
        )

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))

    if not is_adapter_dir(local_dir):
        raise RuntimeError(
            f"{repo_id} downloaded but has no adapter_config.json. "
            "That repo does not look like a PEFT/LoRA adapter."
        )

    info = describe_local(local_dir)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return local_dir


if __name__ == "__main__":
    download_weights(config_from_args("download LoRA weights from the Hugging Face Hub"))
