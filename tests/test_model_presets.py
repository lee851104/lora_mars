"""Model presets and the cross-architecture shims they depend on.

Switching base model must be a config change, not a code change. These tests
pin the two places where that could quietly break: the vision-resolution cap
(needed by Qwen2-VL, absent on mllama) and the processor call convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import REPO_ROOT, load_config
from src.models.infer import call_processor
from src.models.loader import clamp_image_resolution

PRESET_DIR = REPO_ROOT / "configs" / "models"
PRESETS = sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def test_presets_exist():
    assert "qwen2_5_vl_7b" in PRESETS
    assert "qwen2_vl_2b" in PRESETS
    assert "llama3_2_11b_vision" in PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_loads_and_is_complete(preset):
    cfg = load_config(override_file=PRESET_DIR / f"{preset}.yaml")
    assert cfg.model.base_id
    assert cfg.model.min_free_vram_gb > 0
    assert cfg.model.license_notice, "every preset must state its licence"
    # image_max_pixels may legitimately be null, but the key must be present so
    # loader.clamp_image_resolution never sees a missing attribute
    assert "image_max_pixels" in cfg.model


def test_default_model_is_ungated_and_unrestricted_by_region():
    """The invariant is not "is Qwen" - it is that a fresh clone can just run.

    The default must not require clicking through a licence on the Hub, and must
    not carry a geographic restriction. That rules out the Llama preset, whose
    vision capabilities are barred for entities domiciled in the EU.
    """
    cfg = load_config()
    notice = cfg.model.license_notice

    assert cfg.model.base_id.startswith("unsloth/"), "use the ungated unsloth mirror"
    assert "Built with Llama" not in notice
    assert "歐盟" not in notice, "the default must not carry a region restriction"


def test_llama_preset_carries_the_required_attribution():
    cfg = load_config(override_file=PRESET_DIR / "llama3_2_11b_vision.yaml")
    assert "Built with Llama" in cfg.model.license_notice
    # mllama uses fixed tiling - there is no dynamic-resolution knob to clamp
    assert cfg.model.image_max_pixels is None


@pytest.mark.parametrize("preset", PRESETS)
def test_about_tab_renders_for_every_preset(preset):
    from src.serving.app import about_text

    cfg = load_config(override_file=PRESET_DIR / f"{preset}.yaml")
    text = about_text(cfg)
    assert cfg.model.base_id in text
    assert cfg.model.license_notice in text


class DynamicResolutionProcessor:
    """Shaped like Qwen2VLImageProcessor."""

    def __init__(self):
        self.max_pixels = 12_845_056
        self.size = {"shortest_edge": 3136, "longest_edge": 12_845_056}


class FixedTilingProcessor:
    """Shaped like MllamaImageProcessor - no resolution knob."""

    def __init__(self):
        self.size = {"height": 560, "width": 560}


class FakeTokenizer:
    def __init__(self, image_processor):
        self.image_processor = image_processor


def test_clamp_lowers_both_resolution_knobs():
    processor = DynamicResolutionProcessor()
    applied = clamp_image_resolution(FakeTokenizer(processor), 589_824)

    assert processor.max_pixels == 589_824
    assert processor.size["longest_edge"] == 589_824
    assert processor.size["shortest_edge"] == 3136, "the floor must not move"
    assert applied == {"max_pixels": 589_824, "size.longest_edge": 589_824}


def test_clamp_is_a_noop_on_fixed_tiling_architectures():
    processor = FixedTilingProcessor()
    assert clamp_image_resolution(FakeTokenizer(processor), 589_824) == {}
    assert processor.size == {"height": 560, "width": 560}


def test_clamp_is_a_noop_when_unset():
    processor = DynamicResolutionProcessor()
    assert clamp_image_resolution(FakeTokenizer(processor), None) == {}
    assert processor.max_pixels == 12_845_056


def test_clamp_tolerates_a_processor_without_an_image_processor():
    assert clamp_image_resolution(object(), 589_824) == {}


class KeywordProcessor:
    def __call__(self, images=None, text=None, **kwargs):
        return {"images": images, "text": text, "kwargs": kwargs}


class PositionalOnlyProcessor:
    """Some wrapped processors reject the keyword form."""

    def __call__(self, image, prompt, **kwargs):
        return {"images": image, "text": prompt, "kwargs": kwargs}


def test_call_processor_prefers_keywords():
    result = call_processor(KeywordProcessor(), "IMG", "TEXT")
    assert result["images"] == "IMG"
    assert result["text"] == "TEXT"
    assert result["kwargs"]["add_special_tokens"] is False


def test_call_processor_falls_back_to_positional():
    result = call_processor(PositionalOnlyProcessor(), "IMG", "TEXT")
    assert result["images"] == "IMG"
    assert result["text"] == "TEXT"


def test_preset_readme_documents_every_preset():
    readme = (PRESET_DIR / "README.md").read_text(encoding="utf-8")
    for preset in PRESETS:
        assert preset in readme, f"{preset} is undocumented"


def test_makefile_wires_the_model_variable():
    makefile = (Path(REPO_ROOT) / "Makefile").read_text(encoding="utf-8")
    assert "configs/models/$(MODEL).yaml" in makefile
