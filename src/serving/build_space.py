"""Assemble a Hugging Face Space folder in build/space/.

A Space is its own git repo and needs the HF frontmatter in the root README.md,
which the project README cannot carry. Rather than duplicating the app, this
copies exactly what a Space needs and swaps in deploy/space/README.md.

    make space

Then push the staged folder to the Space (instructions are printed at the end).
Nothing here is duplicated logic - src/ is copied verbatim, so the Space always
runs the same code as `make serve`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import DictConfig

from src.config import REPO_ROOT, config_from_args

# (source, destination) relative to the repo root / staging root
FILES = (
    ("app.py", "app.py"),
    ("requirements.txt", "requirements.txt"),
    ("deploy/space/README.md", "README.md"),
    ("MODEL_CARD.md", "MODEL_CARD.md"),
)
DIRS = ("src", "configs")
SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def build_space(cfg: DictConfig, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "build" / "space"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    missing = [src for src, _ in FILES if not (REPO_ROOT / src).exists()]
    if missing:
        raise FileNotFoundError(f"missing files needed for the Space: {missing}")

    for src, dest in FILES:
        shutil.copy2(REPO_ROOT / src, out_dir / dest)

    for name in DIRS:
        shutil.copytree(
            REPO_ROOT / name,
            out_dir / name,
            ignore=shutil.ignore_patterns(*SKIP),
        )

    # The Space downloads weights from the Hub; a stale local adapter path in
    # the config would silently win over LORA_REPO_ID, so make sure it cannot.
    staged_config = out_dir / "configs" / "config.yaml"
    text = staged_config.read_text(encoding="utf-8")
    staged_config.write_text(
        text.replace("local_dir: models/lora", "local_dir: models/lora  # Space: unused"),
        encoding="utf-8",
    )

    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    total_mb = sum(p.stat().st_size for p in files) / 1024**2

    print(f"staged {len(files)} files ({total_mb:.1f} MB) in {out_dir}")
    print()
    print("Next steps:")
    print("  1. Create a Space: https://huggingface.co/new-space")
    print("     SDK = Gradio, Hardware = T4 small or better (CPU basic cannot run 11B)")
    print("  2. Clone it and copy the staged files in:")
    print("       git clone https://huggingface.co/spaces/<user>/<space>")
    print(f"       cp -r {out_dir.as_posix()}/. <space>/")
    print("       cd <space> && git add -A && git commit -m 'deploy' && git push")
    print("  3. In the Space: Settings -> Variables and secrets, add")
    print("       LORA_REPO_ID = <user>/<adapter-repo>")
    print()
    print("  Without step 3 the Space runs the BASE model and says so in the UI.")
    return out_dir


if __name__ == "__main__":
    build_space(config_from_args("stage a Hugging Face Space in build/space/"))
