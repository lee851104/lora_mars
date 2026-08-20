"""Gradio front end.

Deliberate choices:

* the model loads lazily on the first request, so the UI comes up even with no
  GPU attached and tells you what is wrong instead of crashing at import;
* the header always states which weights are live - base-only output must
  never be mistaken for fine-tuned output;
* decoding goes through src.models.infer.build_gen_kwargs, the same helper the
  evaluator uses, so the demo and the reported metrics cannot drift apart.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.config import config_from_args
from src.data.split import split_path

try:  # Hugging Face Spaces GPU decorator; a no-op everywhere else
    import spaces
except Exception:  # pragma: no cover - depends on deployment target

    class _SpacesFallback:
        @staticmethod
        def GPU(duration: int = 180):
            def decorator(function):
                return function

            return decorator

    spaces = _SpacesFallback()


_STATE: dict[str, Any] = {"loaded": None, "cfg": None, "error": None}


def _cfg() -> DictConfig:
    if _STATE["cfg"] is None:
        raise RuntimeError("config was never set - launch through main()")
    return _STATE["cfg"]


def ensure_loaded() -> Any:
    """Load the model once, caching both success and failure."""
    if _STATE["loaded"] is not None:
        return _STATE["loaded"]
    if _STATE["error"] is not None:
        raise RuntimeError(_STATE["error"])

    from src.models.loader import load_for_inference

    try:
        _STATE["loaded"] = load_for_inference(_cfg())
    except Exception as error:  # cache so we do not retry a doomed load per click
        _STATE["error"] = str(error)
        raise
    return _STATE["loaded"]


def weights_status() -> str:
    """Markdown describing which weights would be used, without loading them."""
    from src.models.download import describe_local
    from src.models.loader import resolve_adapter_source

    cfg = _cfg()
    source, is_local = resolve_adapter_source(cfg)
    lines = [f"**Base model** `{cfg.model.base_id}`", ""]

    if source is None:
        lines += [
            "**LoRA weights** none found - the demo will serve the **stock base model**.",
            "",
            "Fetch an adapter first:",
            "",
            "```bash",
            'make weights OVERRIDE="lora.repo_id=<user>/<adapter-repo>"',
            "```",
            "",
            f"or train one with `make train` (it writes to `{cfg.lora.local_dir}`).",
        ]
        return "\n".join(lines)

    if is_local:
        info = describe_local(source)
        lines += [
            f"**LoRA weights** local `{source}` ({info.get('size_mb')} MB)",
            f"- adapter base: `{info.get('base_model')}`",
            f"- r={info.get('r')}, alpha={info.get('lora_alpha')}",
        ]
        meta_path = Path(source) / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            lines.append(
                f"- trained on {meta.get('n_train')} records, "
                f"split_hash `{meta.get('split_hash')}`"
            )
    else:
        lines.append(f"**LoRA weights** hub `{source}` (downloads on first request)")

    return "\n".join(lines)


@spaces.GPU(duration=180)
def describe_image(
    image: Any,
    instruction: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> tuple[str, str]:
    """Returns (caption, run-details markdown)."""
    if image is None:
        return "", "Upload an image first."

    cfg = _cfg()
    prompt = (instruction or "").strip() or str(cfg.serving.default_instruction)

    try:
        loaded = ensure_loaded()
    except Exception as error:
        return "", f"**Model failed to load**\n\n```\n{error}\n```"

    try:
        from src.models.infer import generate_caption

        generation = generate_caption(
            loaded.model,
            loaded.tokenizer,
            image,
            prompt,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            # greedy mode drops these entirely rather than passing them and
            # having transformers ignore them
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
        )
    except Exception as error:  # pragma: no cover - runtime/GPU dependent
        return "", f"**Generation failed**\n\n```\n{traceback.format_exc(limit=3)}{error}\n```"

    mode = "sampling" if do_sample else "greedy (deterministic)"
    details = (
        f"- decoding: **{mode}**"
        + (f", temperature={temperature}, top_p={top_p}" if do_sample else "")
        + f"\n- prompt tokens: {generation.prompt_tokens}"
        + f"\n- generated tokens: {generation.new_tokens}"
        + f"\n- weights: {'LoRA' if loaded.has_adapter else 'BASE MODEL ONLY'}"
    )
    return generation.text, details


def example_images(cfg: DictConfig, limit: int = 4) -> list[str]:
    """Sample thumbnails from the test split, when the data is present."""
    path = split_path(cfg, "test")
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    raw_dir = Path(cfg.paths.raw_dir)
    found = []
    for record in records[:limit]:
        candidate = raw_dir / record["image"]
        if candidate.exists():
            found.append(str(candidate))
    return found


def build_demo(cfg: DictConfig):
    import gradio as gr

    _STATE["cfg"] = cfg
    default_instruction = str(cfg.serving.default_instruction)

    # Gradio 6 moved `theme` from the Blocks constructor to launch()
    with gr.Blocks(title="AstroVision LoRA") as demo:
        gr.Markdown("# AstroVision LoRA\nLlama-3.2-11B-Vision + LoRA, astronomy image captioning.")
        gr.Markdown(weights_status())

        with gr.Tab("Describe"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Astronomy image", type="pil", sources=["upload"])
                    instruction_input = gr.Textbox(
                        label="Instruction",
                        value=default_instruction,
                        lines=3,
                    )
                    with gr.Accordion("Decoding", open=False):
                        max_tokens = gr.Slider(
                            32, 512, value=int(cfg.serving.max_new_tokens), step=32,
                            label="Max new tokens",
                        )
                        do_sample = gr.Checkbox(
                            value=bool(cfg.serving.do_sample),
                            label="Sample (off = greedy, reproducible)",
                        )
                        temperature = gr.Slider(
                            0.1, 2.0, value=float(cfg.serving.temperature), step=0.1,
                            label="Temperature (sampling only)",
                        )
                        top_p = gr.Slider(
                            0.1, 1.0, value=float(cfg.serving.top_p), step=0.05,
                            label="Top-p (sampling only)",
                        )
                    # keep the sampling knobs visibly inert when greedy
                    do_sample.change(
                        lambda flag: (gr.update(interactive=flag), gr.update(interactive=flag)),
                        inputs=do_sample,
                        outputs=[temperature, top_p],
                    )
                    run_btn = gr.Button("Describe", variant="primary")
                    clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    output = gr.Textbox(label="Generated description", lines=10)
                    details = gr.Markdown()

            examples = example_images(cfg)
            if examples:
                gr.Examples(examples=[[p] for p in examples], inputs=[image_input])

            run_btn.click(
                describe_image,
                inputs=[image_input, instruction_input, max_tokens, do_sample, temperature, top_p],
                outputs=[output, details],
            )
            clear_btn.click(
                lambda: (None, default_instruction, "", ""),
                outputs=[image_input, instruction_input, output, details],
            )

        with gr.Tab("Status"):
            status_md = gr.Markdown(weights_status())
            gr.Button("Refresh").click(lambda: weights_status(), outputs=status_md)
            gr.Markdown(
                "Only the generated continuation is shown - the echoed prompt is "
                "stripped before decoding, the same way `make eval` scores it. "
                "See `MODEL_CARD.md` for limits and out-of-scope uses."
            )

    return demo


def main(cfg: DictConfig) -> None:
    demo = build_demo(cfg)
    import gradio as gr

    demo.launch(
        server_name=str(cfg.serving.host),
        server_port=int(cfg.serving.port),
        share=bool(cfg.serving.share),
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main(config_from_args("Gradio front end"))
