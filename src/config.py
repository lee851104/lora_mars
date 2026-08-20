"""Configuration loading. Every hyperparameter comes from YAML; nothing is
hard-coded in the modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

_PATH_KEYS = (
    "raw_dir",
    "processed_dir",
    "splits_dir",
    "models_dir",
    "reports_dir",
    "figures_dir",
    "outputs_dir",
)


def load_config(
    config: str | Path = DEFAULT_CONFIG,
    override_file: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load YAML, apply overrides, and resolve relative paths against the repo root.

    Precedence, later wins: main config -> override_file -> CLI dotlist.
    """
    cfg = OmegaConf.load(Path(config))
    if override_file:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(override_file)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    for key in _PATH_KEYS:
        value = cfg.paths.get(key)
        if value is not None and not Path(value).is_absolute():
            cfg.paths[key] = str((REPO_ROOT / value).resolve())

    if cfg.lora.get("local_dir") and not Path(cfg.lora.local_dir).is_absolute():
        cfg.lora.local_dir = str((REPO_ROOT / cfg.lora.local_dir).resolve())

    return cfg


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """One CLI surface shared by every __main__ entry point."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="main config file")
    parser.add_argument("--override-file", default=None, help="extra YAML override file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="dotlist overrides, e.g. train.max_steps=60 lora.r=32",
    )
    return parser


def config_from_args(description: str, argv: list[str] | None = None) -> DictConfig:
    args = build_arg_parser(description).parse_args(argv)
    return load_config(args.config, args.override_file, args.overrides)


def ensure_dirs(cfg: DictConfig, *keys: str) -> None:
    for key in keys:
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)
