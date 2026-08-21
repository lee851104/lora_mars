"""Assemble a Hugging Face Gradio or Static Space folder.

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

from src.config import REPO_ROOT, build_arg_parser, load_config

# (source, destination) relative to the repo root / staging root
GRADIO_FILES = (
    ("app.py", "app.py"),
    ("requirements.txt", "requirements.txt"),
    ("deploy/space/README.md", "README.md"),
    ("MODEL_CARD.md", "MODEL_CARD.md"),
)
GRADIO_DIRS = ("src", "configs")
STATIC_FILES = (
    ("deploy/static-space/README.md", "README.md"),
    ("deploy/static-space/index.html", "index.html"),
    ("deploy/static-space/style.css", "style.css"),
)
SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def build_space(
    cfg: DictConfig,
    out_dir: Path | None = None,
    *,
    static: bool = False,
) -> Path:
    kind = "static-space" if static else "space"
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "build" / kind
    files_to_copy = STATIC_FILES if static else GRADIO_FILES
    dirs_to_copy = () if static else GRADIO_DIRS
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    missing = [src for src, _ in files_to_copy if not (REPO_ROOT / src).exists()]
    if missing:
        raise FileNotFoundError(f"missing files needed for the Space: {missing}")

    for src, dest in files_to_copy:
        shutil.copy2(REPO_ROOT / src, out_dir / dest)

    for name in dirs_to_copy:
        shutil.copytree(
            REPO_ROOT / name,
            out_dir / name,
            ignore=shutil.ignore_patterns(*SKIP),
        )

    if not static:
        # The Space downloads weights from the Hub; a stale local adapter path
        # would silently win over LORA_REPO_ID, so mark it as unused.
        staged_config = out_dir / "configs" / "config.yaml"
        text = staged_config.read_text(encoding="utf-8")
        staged_config.write_text(
            text.replace(
                "local_dir: models/lora",
                "local_dir: models/lora  # Space: unused",
            ),
            encoding="utf-8",
        )

    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    total_mb = sum(p.stat().st_size for p in files) / 1024**2

    print(f"staged {len(files)} files ({total_mb:.1f} MB) in {out_dir}")
    print()
    print("Next steps:")
    if static:
        print("  This is the free static project page; it does not run model inference.")
        print("  Push it with --static --push. Later, PRO can replace it with Gradio.")
        return out_dir

    print("  1. Create a Space: https://huggingface.co/new-space")
    print("     SDK = Gradio, Hardware = ZeroGPU (free demo) or T4 small+ (paid, always on)")
    print("  2. Clone it and copy the staged files in:")
    print("       git clone https://huggingface.co/spaces/<user>/<space>")
    print(f"       cp -r {out_dir.as_posix()}/. <space>/")
    print("       cd <space> && git add -A && git commit -m 'deploy' && git push")
    print("  3. In the Space: Settings -> Variables and secrets, add")
    print("       LORA_REPO_ID = <user>/<adapter-repo>")
    print()
    print("  Without step 3 the Space runs the BASE model and says so in the UI.")
    return out_dir


def push_space(cfg: DictConfig, staged: Path, *, static: bool = False) -> str:
    """Upload the staged folder straight to a Space.

    Exists because the alternative - clone the Space, copy files in, git push -
    needs a local machine with git credentials. Training happens in Colab, so
    without this the deploy step would be the one thing you could not finish
    there.
    """
    from huggingface_hub import HfApi

    repo_id = cfg.space.get("repo_id")
    if not repo_id:
        raise ValueError(
            "space.repo_id is not set. Pass\n"
            '  make space-push OVERRIDE="space.repo_id=<user>/<space-name>"'
        )

    api = HfApi()
    sdk = "static" if static else str(cfg.space.sdk)
    api.create_repo(
        repo_id=str(repo_id),
        repo_type="space",
        space_sdk=sdk,
        private=bool(cfg.space.private),
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=str(repo_id),
        repo_type="space",
        folder_path=str(staged),
        commit_message=f"deploy {sdk} Space from build_space",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\npushed: {url}")
    if static:
        print("\nStatic project page deployed. It shows documentation only; no GPU is used.")
        print("After subscribing to PRO, rerun without --static to deploy Gradio + ZeroGPU.")
        return url

    print("\nStill to do in the Space UI (neither can be set from here):")
    print("  1. Settings -> Hardware: pick ZeroGPU for a free demo, or T4 small+ for paid")
    print("     always-on serving. CPU basic cannot load the model at all.")
    print("  2. Settings -> Variables: LORA_REPO_ID = <user>/<adapter-repo>")
    print("     Without it the Space serves the base model and says so.")
    return url


def main(argv: list[str] | None = None) -> Path:
    parser = build_arg_parser("stage (and optionally push) a Hugging Face Space")
    parser.add_argument(
        "--push", action="store_true", help="upload the staged folder to space.repo_id"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="stage the free static project page instead of the GPU Gradio app",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override_file, args.overrides)

    staged = build_space(cfg, static=args.static)
    if args.push:
        push_space(cfg, staged, static=args.static)
    return staged


if __name__ == "__main__":
    main()
