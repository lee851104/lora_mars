"""Scoring layer: metric selection, intervals, and the judge's prompt hygiene.

All CPU-only and offline - no CLIP checkpoint, no API key. The parts that need
either are exercised through their pure helpers.
"""

from __future__ import annotations

import json
from statistics import mean

import pytest
from omegaconf import OmegaConf

from src.models import clip_score, llm_judge, score


# ── metric selection ────────────────────────────────────────────────────────
def test_default_metrics_include_cider_clipscore_and_judge(cfg):
    assert list(cfg.eval.metrics) == ["cider", "clipscore", "llm_judge"]
    assert score.resolve_metrics(cfg) == ["cider", "clipscore", "llm_judge"]


def test_cider_is_available_as_a_metric(cfg):
    cfg.eval.metrics = ["cider"]
    assert score.resolve_metrics(cfg) == ["cider"]


def test_bleu_rouge_is_available_but_not_default(cfg):
    cfg.eval.metrics = ["bleu_rouge"]
    assert score.resolve_metrics(cfg) == ["bleu_rouge"]


def test_unknown_metric_is_rejected(cfg):
    cfg.eval.metrics = ["clipscore", "meteor"]
    with pytest.raises(ValueError, match="unknown metrics"):
        score.resolve_metrics(cfg)


def test_empty_metric_list_is_rejected(cfg):
    cfg.eval.metrics = []
    with pytest.raises(ValueError, match="nothing to score"):
        score.resolve_metrics(cfg)


def test_missing_predictions_file_says_what_to_run(cfg):
    with pytest.raises(FileNotFoundError, match="make eval"):
        score.load_predictions(cfg, "test")


def test_rescoring_preserves_metrics_already_in_the_report(cfg, monkeypatch):
    rows = [{"prediction": "Mars", "reference": "Mars", "image_id": "1"}]
    path = score.report_path(cfg, "test")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "split": "test",
            "n_samples": 1,
            "base_id": "example/base",
            "metrics": {"clipscore": {"clipscore": 0.7}},
        }),
        encoding="utf-8",
    )
    cfg.eval.metrics = ["cider"]
    monkeypatch.setattr(score, "run_cider", lambda _cfg, _rows: {"cider": 1.2})

    report = score.score(cfg, rows, "test", merge_existing=True)

    assert report["base_id"] == "example/base"
    assert report["metrics"]["clipscore"]["clipscore"] == 0.7
    assert report["metrics"]["cider"]["cider"] == 1.2


def test_predictions_round_trip(cfg, tmp_path):
    from src.config import ensure_dirs

    ensure_dirs(cfg, "reports_dir")
    rows = [{"image_id": "001", "image": "images/001.jpg", "reference": "a", "prediction": "b"}]
    path = score.predictions_path(cfg, "test")
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    assert score.load_predictions(cfg, "test") == rows


# ── bootstrap intervals ─────────────────────────────────────────────────────
def test_bootstrap_ci_brackets_the_mean():
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75]
    low, high = score.bootstrap_ci(values, n_samples=500, seed=0)
    assert low <= mean(values) <= high


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    values = [0.1, 0.9, 0.4, 0.6, 0.5]
    assert score.bootstrap_ci(values, 200, 7) == score.bootstrap_ci(values, 200, 7)


def test_bootstrap_ci_is_none_when_there_is_nothing_to_resample():
    assert score.bootstrap_ci([0.5], 500, 0) is None
    assert score.bootstrap_ci([], 500, 0) is None


def test_bootstrap_ci_of_a_constant_is_degenerate():
    assert score.bootstrap_ci([0.4] * 10, 200, 0) == [0.4, 0.4]


# ── CLIPScore helpers ───────────────────────────────────────────────────────
def test_harmonic_mean_matches_the_formula():
    assert clip_score._harmonic_mean(1.0, 1.0) == pytest.approx(1.0)
    assert clip_score._harmonic_mean(0.5, 1.0) == pytest.approx(2 / 3)


def test_harmonic_mean_is_zero_when_either_side_is_zero():
    assert clip_score._harmonic_mean(0.0, 0.9) == 0.0
    assert clip_score._harmonic_mean(0.9, 0.0) == 0.0


@pytest.mark.parametrize("requested", ["cpu", "cuda"])
def test_resolve_device_honours_an_explicit_choice(requested):
    """Explicit values must not import torch or second-guess the user."""
    assert clip_score.resolve_device(requested) == requested


class StubClipTokenizer:
    """Returns one id per whitespace token, like a very naive tokenizer."""

    def __call__(self, texts, truncation=False, padding=False):  # noqa: ARG002
        return {"input_ids": [list(range(len(t.split()))) for t in texts]}


def test_count_truncated_flags_only_over_length_texts():
    texts = ["a b c", " ".join(["w"] * 80), "short"]
    assert clip_score.count_truncated(StubClipTokenizer(), texts, 77) == [1]


def test_count_truncated_is_empty_when_everything_fits():
    assert clip_score.count_truncated(StubClipTokenizer(), ["a b"], 77) == []


def test_clipscore_summary_reports_the_reference_ceiling():
    out = clip_score.ClipScoreOutput(
        prediction_scores=[0.70, 0.74],
        reference_scores=[0.80, 0.82],
        text_similarities=[0.5, 0.6],
        ref_clipscores=[0.6, 0.65],
        truncated_predictions=["007"],
        device="cpu",
        model_id="openai/clip-vit-large-patch14",
    )
    summary = clip_score.summarise(out, 2)

    assert summary["clipscore"] == pytest.approx(0.72)
    assert summary["clipscore_reference"] == pytest.approx(0.81)
    assert summary["clipscore_reference"] > summary["clipscore"]
    assert summary["truncated_predictions"] == 1
    assert "ceiling" in summary["note"]


# ── LLM judge ───────────────────────────────────────────────────────────────
def test_judge_prompt_never_reveals_which_system_wrote_the_caption():
    """Position/identity bias guard: the judge must not know base from LoRA."""
    prompt = llm_judge.SYSTEM.lower()
    for leak in ("lora", "adapter", "fine-tuned model produced", "base model"):
        assert leak not in prompt, f"{leak!r} leaks system identity into the judge prompt"


def test_judge_prompt_anchors_every_scored_axis():
    prompt = llm_judge.SYSTEM
    for axis in ("accuracy", "style_match", "fluency", "hallucination_count", "overall"):
        assert axis in prompt, f"{axis} is scored but never defined in the rubric"


def test_judge_prompt_allows_true_details_the_reference_omits():
    assert "not the only one" in llm_judge.SYSTEM


def test_judge_schema_is_strict_and_complete():
    schema = llm_judge.SCHEMA
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    for axis in ("accuracy", "style_match", "fluency", "overall"):
        assert schema["properties"][axis] == {"type": "integer", "minimum": 1, "maximum": 5}
    assert schema["properties"]["hallucination_count"]["minimum"] == 0


def test_cost_estimate_scales_with_sample_count(cfg):
    one = llm_judge.estimate_cost(cfg, 1)["estimated_usd"]
    ten = llm_judge.estimate_cost(cfg, 10)["estimated_usd"]
    assert ten == pytest.approx(one * 10, rel=0.02)


def test_cost_estimate_uses_the_configured_model_price(cfg):
    opus = llm_judge.estimate_cost(cfg, 25)["estimated_usd"]
    cheap = OmegaConf.merge(cfg, {"llm_judge": {"model": "claude-haiku-4-5"}})
    assert llm_judge.estimate_cost(cheap, 25)["estimated_usd"] < opus


def test_judge_config_carries_no_temperature(cfg):
    """Opus 5 rejects temperature with a 400 - it must not be in the config."""
    assert "temperature" not in cfg.llm_judge


def _judgement(image_id, **scores):
    payload = {
        "accuracy": 4, "style_match": 4, "fluency": 5,
        "hallucination_count": 0, "overall": 4, "rationale": "clean",
    }
    payload.update(scores)
    return llm_judge.Judgement(image_id, payload)


def test_judge_summary_averages_only_successful_calls():
    judgements = [
        _judgement("001", accuracy=5, hallucination_count=0),
        _judgement("002", accuracy=3, hallucination_count=2),
        llm_judge.Judgement("003", None, "refusal (cyber)"),
    ]
    summary = llm_judge.summarise(judgements, {"model": "claude-opus-5"})

    assert summary["n_judged"] == 2
    assert summary["n_failed"] == 1
    assert summary["accuracy"] == pytest.approx(4.0)
    assert summary["hallucination_count"] == pytest.approx(1.0)
    assert summary["hallucination_free_rate"] == pytest.approx(0.5)
    assert summary["rubric_version"] == llm_judge.RUBRIC_VERSION


def test_judge_summary_reports_total_failure_without_crashing():
    summary = llm_judge.summarise(
        [llm_judge.Judgement("001", None, "AuthenticationError")], {}
    )
    assert summary["n_judged"] == 0
    assert "error" in summary


def test_per_sample_rows_keep_failures_visible():
    rows = llm_judge.per_sample_rows(
        [_judgement("001"), llm_judge.Judgement("002", None, "timeout")]
    )
    assert rows[0]["judge_error"] is None
    assert rows[1]["judge_error"] == "timeout"
    assert rows[1]["image_id"] == "002"


def test_encode_image_downscales_and_returns_jpeg(tmp_path):
    from PIL import Image

    path = tmp_path / "big.png"
    Image.new("RGB", (2000, 1000), (10, 20, 30)).save(path)

    media_type, data = llm_judge.encode_image(path, max_edge=1024)
    assert media_type == "image/jpeg"

    import base64
    import io

    with Image.open(io.BytesIO(base64.standard_b64decode(data))) as decoded:
        assert max(decoded.size) == 1024
        assert decoded.size == (1024, 512)


def test_encode_image_leaves_small_images_alone(tmp_path):
    import base64
    import io

    from PIL import Image

    path = tmp_path / "small.png"
    Image.new("RGB", (300, 200)).save(path)
    _media_type, data = llm_judge.encode_image(path, max_edge=1024)
    with Image.open(io.BytesIO(base64.standard_b64decode(data))) as decoded:
        assert decoded.size == (300, 200)

def test_skipped_judge_is_not_reported_as_a_failure():
    """cost_estimate_only is a deliberate skip, not 25 failed API calls."""
    summary = llm_judge.summarise([], {"skipped": True, "n_calls": 25, "estimated_usd": 0.75})
    assert "error" not in summary
    assert summary["n_failed"] == 0
    assert summary["skipped"] is True
    assert "cost_estimate_only" in summary["skipped_reason"]

# ── baseline variant ────────────────────────────────────────────────────────
def test_baseline_run_writes_to_separate_files(cfg):
    """A base run must not overwrite the LoRA report it is compared against."""
    lora_report = score.report_path(cfg, "test")
    lora_predictions = score.predictions_path(cfg, "test")

    cfg.eval.use_adapter = False
    base_report = score.report_path(cfg, "test")
    base_predictions = score.predictions_path(cfg, "test")

    assert score.variant(cfg) == "_base"
    assert base_report != lora_report
    assert base_predictions != lora_predictions
    assert base_report.name == "eval_test_base.json"
    assert base_predictions.name == "predictions_test_base.jsonl"
    assert score.judge_detail_path(cfg, "test").name == "judge_test_base.jsonl"


def test_adapter_run_uses_unsuffixed_names(cfg):
    assert score.variant(cfg) == ""
    assert score.report_path(cfg, "test").name == "eval_test.json"


def test_makefile_exposes_the_baseline_target():
    from src.config import REPO_ROOT

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "eval-base:" in makefile
    assert "eval.use_adapter=false" in makefile
