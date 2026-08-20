"""LLM-as-judge: a vision model scores each caption against the image.

This is the only metric here that can see the picture and therefore the only one
that can catch a hallucination. BLEU cannot tell "a rover on Mars" from "a rover
on the Moon"; CLIPScore notices the mismatch but cannot say what was wrong.

Design decisions that matter for the numbers being usable:

* The judge never learns which system wrote a caption. No "base" or "LoRA"
  label reaches the prompt, so it cannot express a preference for either.
* The rubric anchors every level explicitly. Unanchored 1-5 scales drift between
  samples and make runs incomparable.
* Structured outputs (`output_config.format`) enforce the schema at the API
  layer, so a malformed judgement is retried by the SDK rather than silently
  parsed into garbage.
* `rubric_version` goes into the report. Changing the rubric changes the numbers,
  and a report that does not say which rubric produced them is not comparable to
  anything.

It is a model's opinion, not ground truth. Report it as such.
"""

from __future__ import annotations

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

RUBRIC_VERSION = "v1"

# Cached list prices, USD per million tokens, for the cost estimate only.
# Not authoritative - check the pricing page before believing a large number.
PRICES_USD_PER_MTOK = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_PRICE = (5.0, 25.0)
# Rough per-call token estimate for the cost preview: one resized image plus the
# rubric prompt, and an adaptive-thinking answer at medium effort.
EST_INPUT_TOKENS = 2000
EST_OUTPUT_TOKENS = 800

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """You are grading image captions produced by a small vision-language \
model that was fine-tuned on an astronomy photo dataset.

You will be shown an image, one human-written reference caption, and one \
candidate caption. Grade the candidate.

The reference is ONE valid description, not the only one. Do not penalise the \
candidate for mentioning true things the reference omits. Penalise it for \
stating things the image does not support.

Score each axis on 1-5 using these anchors:

accuracy - is what the candidate says true of this image?
  5 every claim is supported by the image
  4 all main claims correct, one minor detail unsupported
  3 the main subject is right but several details are wrong
  2 the main subject is wrong, something in the scene is still recognised
  1 unrelated to the image

style_match - does it read like the reference corpus (length, register, vocabulary)?
  5 indistinguishable from a reference caption
  3 recognisably the same domain, noticeably different length or register
  1 completely different register (chatty, listy, meta-commentary)

fluency - is it grammatical, natural English?
  5 clean   3 awkward but understandable   1 broken or repetitive

hallucination_count - count the DISTINCT specific claims the image does not \
support (named objects, instruments, missions, places, numbers). A vague but \
harmless sentence has 0. Count claims, not words.

overall - your single summary judgement, 1-5.

rationale - at most two sentences. Name the specific problem, or say it is clean."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
        "style_match": {"type": "integer", "minimum": 1, "maximum": 5},
        "fluency": {"type": "integer", "minimum": 1, "maximum": 5},
        "hallucination_count": {"type": "integer", "minimum": 0},
        "overall": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string"},
    },
    "required": [
        "accuracy",
        "style_match",
        "fluency",
        "hallucination_count",
        "overall",
        "rationale",
    ],
    "additionalProperties": False,
}
NUMERIC_AXES = ("accuracy", "style_match", "fluency", "hallucination_count", "overall")


@dataclass
class Judgement:
    image_id: str
    scores: dict[str, Any] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.scores is not None


def estimate_cost(cfg: DictConfig, n_rows: int) -> dict[str, Any]:
    model = str(cfg.llm_judge.model)
    input_rate, output_rate = PRICES_USD_PER_MTOK.get(model, DEFAULT_PRICE)
    usd = n_rows * (
        EST_INPUT_TOKENS * input_rate + EST_OUTPUT_TOKENS * output_rate
    ) / 1_000_000
    return {
        "model": model,
        "n_calls": n_rows,
        "estimated_usd": round(usd, 2),
        "assumes_tokens_per_call": {
            "input": EST_INPUT_TOKENS,
            "output": EST_OUTPUT_TOKENS,
        },
        "note": "rough estimate from cached list prices; actual cost depends on "
                "image size and how much the judge thinks",
    }


def encode_image(path: Path, max_edge: int) -> tuple[str, str]:
    """Downscale and base64-encode. Returns (media_type, data).

    Resizing is a cost control: a 2000px photo buys no judging accuracy over a
    1024px one but costs several times the input tokens.
    """
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if max(image.size) > max_edge:
            scale = max_edge / max(image.size)
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)

    return "image/jpeg", base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


class AnthropicJudge:
    """Thin wrapper that also degrades gracefully when a beta is unavailable."""

    def __init__(self, cfg: DictConfig):
        import anthropic

        self.cfg = cfg
        self.client = anthropic.Anthropic()
        self.model = str(cfg.llm_judge.model)
        self.effort = str(cfg.llm_judge.effort)
        self.max_tokens = int(cfg.llm_judge.max_tokens)
        self.send_image = bool(cfg.llm_judge.send_image)
        self.max_edge = int(cfg.llm_judge.max_image_edge)
        self.use_fallbacks = bool(cfg.llm_judge.enable_fallbacks)
        self.images_root = Path(cfg.paths.raw_dir)

    def _content(self, row: dict) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if self.send_image:
            media_type, data = encode_image(self.images_root / row["image"], self.max_edge)
            content.append(
                {"type": "image", "source": {"type": "base64",
                                             "media_type": media_type, "data": data}}
            )
        content.append(
            {
                "type": "text",
                "text": (
                    f"Reference caption:\n{row['reference']}\n\n"
                    f"Candidate caption:\n{row['prediction'] or '(empty output)'}"
                ),
            }
        )
        return content

    def _request(self, row: dict, *, with_fallbacks: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": self._content(row)}],
            # adaptive thinking: judging a caption against an image is exactly the
            # kind of task where a little reasoning changes the verdict
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
        }
        if with_fallbacks:
            return self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        return self.client.messages.create(**kwargs)

    def judge(self, row: dict) -> Judgement:
        import anthropic

        try:
            try:
                response = self._request(row, with_fallbacks=self.use_fallbacks)
            except anthropic.BadRequestError as error:
                if not self.use_fallbacks:
                    raise
                # the beta is not enabled for this account/endpoint - stop asking
                print(f"[judge] server-side fallbacks unavailable ({error}); continuing without")
                self.use_fallbacks = False
                response = self._request(row, with_fallbacks=False)
        except Exception as error:  # network, auth, rate limit after retries
            return Judgement(row["image_id"], None, f"{type(error).__name__}: {error}")

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            return Judgement(row["image_id"], None, f"refusal ({category})")

        try:
            text = next(b.text for b in response.content if b.type == "text")
            return Judgement(row["image_id"], json.loads(text))
        except (StopIteration, json.JSONDecodeError) as error:
            return Judgement(row["image_id"], None, f"unparseable response: {error}")


def judge_rows(cfg: DictConfig, rows: list[dict]) -> tuple[list[Judgement], dict[str, Any]]:
    """Judge every row (up to max_samples) concurrently."""
    limit = cfg.llm_judge.get("max_samples")
    subset = rows[: int(limit)] if limit else rows
    estimate = estimate_cost(cfg, len(subset))

    print(f"[judge] {estimate['n_calls']} calls to {estimate['model']}, "
          f"~${estimate['estimated_usd']} estimated")
    if bool(cfg.llm_judge.cost_estimate_only):
        print("[judge] cost_estimate_only=true - not calling the API")
        return [], {**estimate, "skipped": True}

    judge = AnthropicJudge(cfg)
    workers = max(1, int(cfg.llm_judge.concurrency))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        judgements = list(pool.map(judge.judge, subset))

    failed = [j for j in judgements if not j.ok]
    if failed:
        print(f"[judge] {len(failed)}/{len(judgements)} calls failed:")
        for judgement in failed[:5]:
            print(f"  {judgement.image_id}: {judgement.error}")

    return judgements, {**estimate, "skipped": False}


def summarise(judgements: list[Judgement], meta: dict[str, Any]) -> dict[str, Any]:
    """Means per axis over the judgements that succeeded."""
    from statistics import mean

    if meta.get("skipped"):
        # cost_estimate_only - nothing was called, and that is not a failure
        return {
            "rubric_version": RUBRIC_VERSION,
            "n_judged": 0,
            "n_failed": 0,
            "skipped_reason": "cost_estimate_only=true - set it to false to actually judge",
            **meta,
        }

    ok = [j for j in judgements if j.ok]
    if not ok:
        return {
            "rubric_version": RUBRIC_VERSION,
            "n_judged": 0,
            "n_failed": len(judgements),
            "error": "every judge call failed - see the log above",
            **meta,
        }

    per_axis = {
        axis: round(mean(float(j.scores[axis]) for j in ok), 3) for axis in NUMERIC_AXES
    }
    clean = sum(1 for j in ok if int(j.scores["hallucination_count"]) == 0)

    return {
        "rubric_version": RUBRIC_VERSION,
        "n_judged": len(ok),
        "n_failed": len(judgements) - len(ok),
        **per_axis,
        "hallucination_free_rate": round(clean / len(ok), 3),
        "note": "a model's opinion, not ground truth. The judge never saw which "
                "system produced a caption.",
        **meta,
    }


def per_sample_rows(judgements: list[Judgement]) -> list[dict[str, Any]]:
    """Flat rows for the predictions file, so individual verdicts stay readable."""
    return [
        {"image_id": j.image_id, **(j.scores or {}), "judge_error": j.error}
        for j in judgements
    ]
