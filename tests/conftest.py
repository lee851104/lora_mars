"""Shared fixtures. Everything here is synthetic and CPU-only - no model
download, no GPU, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.config import load_config

# Enough records that a 10/10/80 split still leaves >= 2 members per class.
LABEL_TEMPLATES = {
    "Earth": "A view of Earth from orbit showing cloud band number {i}.",
    "Mars": "The Martian surface near crater site {i} under a dusty sky.",
    "Mars Rover": "The Curiosity rover parked beside outcrop {i} on Mars.",
    "Milky Way": "The Milky Way arching over a dark horizon at location {i}.",
    "Hubble": "A Hubble Space Telescope image of nebula NGC {i}.",
}
PER_LABEL = 12


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A raw dataset directory: data.json plus real (tiny) JPEG files."""
    raw_dir = tmp_path / "raw"
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True)

    records = []
    counter = 0
    for template in LABEL_TEMPLATES.values():
        for index in range(PER_LABEL):
            counter += 1
            image_id = f"{counter:03d}"
            relative = f"images/{image_id}.jpg"
            Image.new("RGB", (8, 8), (index * 7 % 255, 40, 90)).save(raw_dir / relative)
            records.append(
                {
                    "image_id": image_id,
                    "text": template.format(i=index),
                    "image": relative,
                }
            )

    (raw_dir / "data.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return raw_dir


@pytest.fixture
def cfg(tmp_path: Path, dataset_root: Path):
    """Real config.yaml with every path redirected into tmp_path."""
    overrides = [
        f"paths.raw_dir={dataset_root.as_posix()}",
        f"paths.processed_dir={(tmp_path / 'processed').as_posix()}",
        f"paths.splits_dir={(tmp_path / 'splits').as_posix()}",
        f"paths.models_dir={(tmp_path / 'models').as_posix()}",
        f"paths.reports_dir={(tmp_path / 'reports').as_posix()}",
        f"paths.figures_dir={(tmp_path / 'reports' / 'figures').as_posix()}",
        f"paths.outputs_dir={(tmp_path / 'outputs').as_posix()}",
        f"lora.local_dir={(tmp_path / 'models' / 'lora').as_posix()}",
    ]
    return load_config(overrides=overrides)


@pytest.fixture
def built_records(cfg, dataset_root):
    """Processed records written to disk, ready to be split."""
    from src.data.build import build_records, processed_path
    from src.data.download import load_raw

    records = build_records(cfg, load_raw(cfg))
    path = processed_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
