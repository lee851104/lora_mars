"""Gradio front end.

Deliberate choices:

* the model loads lazily on the first request, so the UI comes up even with no
  GPU attached and tells you what is wrong instead of crashing at import;
* the header always states which weights are live - base-only output must
  never be mistaken for fine-tuned output;
* the comparison view runs the SAME model twice, once inside
  PeftModel.disable_adapter(), so "before vs after fine-tuning" costs no extra
  VRAM and cannot drift out of sync;
* decoding goes through src.models.infer.build_gen_kwargs, the same helper the
  evaluator uses, so the demo and the reported metrics cannot drift apart.
"""

from __future__ import annotations

import json
import traceback
from contextlib import nullcontext
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

ABOUT_TEMPLATE = """
### 這是什麼

一個視覺語言模型（`{base_id}`），用 LoRA 在 250 張天文照片上微調過。
你上傳一張圖，它用天文資料集的語彙描述你看到的東西。

「原廠模型」和「微調後」是**同一個模型**——微調只是在原本的權重上加了一層很薄的
adapter（LoRA），比較欄位是把那層關掉再跑一次。所以你看到的差異，就是微調的效果。

### 它不能做什麼

- **不能當科學判讀**。它不會測距離、亮度、成分，講出來的任何數字都是文字模仿。
- **不能當天體辨識的依據**。它可能把仙女座說成銀河系。
- **不能餵非天文照片**。訓練資料只有五類天文場景，餵人像或街景會得到「自信但荒謬」的答案。
- **只會英文**。訓練資料全英文。

### 為什麼描述會怪

訓練資料只有 250 筆，而且預設只訓練 30 步——約 240 個樣本次，通常約一個 epoch。模型主要學到的是
**輸出風格**（句子長度、用語、句型），不是新的天文知識。完整的限制與已知偏誤寫在
專案的 `MODEL_CARD.md`。

---

**授權**：{license_notice}
"""


def about_text(cfg: DictConfig) -> str:
    """Rendered from config so the model name and licence notice always match
    the weights actually configured - the Llama presets carry an attribution
    requirement that the Apache-2.0 ones do not."""
    return ABOUT_TEMPLATE.format(
        base_id=cfg.model.base_id,
        license_notice=cfg.model.get("license_notice") or "（未在設定檔中指定）",
    )


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


def has_adapter_configured() -> bool:
    from src.models.loader import resolve_adapter_source

    source, _ = resolve_adapter_source(_cfg())
    return source is not None


def status_badge() -> str:
    """One line the user can read at a glance."""
    from src.models.loader import resolve_adapter_source

    cfg = _cfg()
    source, is_local = resolve_adapter_source(cfg)
    if source is None:
        return (
            "⚠️ **找不到 LoRA 權重 —— 現在跑的是原廠模型**，"
            "看到的輸出不是微調結果。"
        )
    where = "本機" if is_local else "Hugging Face"
    loaded = "（已載入）" if _STATE["loaded"] is not None else "（首次產生時載入，約需數分鐘）"
    return f"✅ LoRA 權重：`{source}`（{where}）{loaded}"


def weights_detail() -> str:
    """The long version, for the 狀態 tab."""
    from src.models.download import describe_local
    from src.models.loader import resolve_adapter_source

    cfg = _cfg()
    source, is_local = resolve_adapter_source(cfg)
    lines = [f"- 基礎模型：`{cfg.model.base_id}`"]

    if source is None:
        lines += [
            "- LoRA 權重：**沒有**",
            "",
            "先取得權重，兩條路任選一條：",
            "",
            "```bash",
            'make weights OVERRIDE="lora.repo_id=<user>/<adapter-repo>"   # 下載現成的',
            "make train                                                   # 自己訓練",
            "```",
            "",
            f"兩者都會寫到 `{cfg.lora.local_dir}`。",
        ]
    elif is_local:
        info = describe_local(source)
        lines += [
            f"- LoRA 權重：本機 `{source}`（{info.get('size_mb')} MB）",
            f"- adapter 的基礎模型：`{info.get('base_model')}`",
            f"- r={info.get('r')}, alpha={info.get('lora_alpha')}",
        ]
        meta_path = Path(source) / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            lines.append(
                f"- 訓練資料：{meta.get('n_train')} 筆，split_hash `{meta.get('split_hash')}`"
            )
    else:
        lines.append(f"- LoRA 權重：Hugging Face `{source}`（第一次產生時才下載）")

    if _STATE["error"]:
        lines += ["", f"- ❌ 上次載入失敗：`{_STATE['error']}`"]
    elif _STATE["loaded"] is not None:
        lines += ["", "- 模型已在記憶體中，可以直接產生。"]

    return "\n".join(lines)


@spaces.GPU(duration=240)
def preload() -> tuple[str, str]:
    """Load the model up front so the first description does not look hung.

    Decorated because on ZeroGPU there is no CUDA device outside a @spaces.GPU
    call, and unsloth needs one at import time.
    """
    try:
        loaded = ensure_loaded()
    except Exception as error:
        return status_badge(), f"❌ 載入失敗\n\n```\n{error}\n```"
    return status_badge(), f"✅ 載入完成\n\n```\n{loaded.describe()}\n```"


def _run_once(loaded: Any, image: Any, prompt: str, params: dict, *, with_adapter: bool) -> str:
    """One generation pass. with_adapter=False temporarily turns the LoRA off."""
    from src.models.infer import generate_caption

    model = loaded.model
    disable = getattr(model, "disable_adapter", None)
    if not with_adapter:
        if disable is None:
            return "（這個模型沒有掛 adapter，無法關閉）"
        context = disable()
    else:
        context = nullcontext()

    with context:
        return generate_caption(model, loaded.tokenizer, image, prompt, **params).text


@spaces.GPU(duration=240)
def describe_image(
    image: Any,
    instruction: str,
    compare: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> tuple[str, str, str]:
    """Returns (fine-tuned output, base output, run details)."""
    if image is None:
        return "", "", "請先上傳一張圖片。"

    cfg = _cfg()
    prompt = (instruction or "").strip() or str(cfg.serving.default_instruction)

    try:
        loaded = ensure_loaded()
    except Exception as error:
        return "", "", f"**模型載入失敗**\n\n```\n{error}\n```"

    params = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        # greedy mode drops these entirely rather than passing them and having
        # transformers ignore them
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
    }

    try:
        tuned = _run_once(loaded, image, prompt, params, with_adapter=True)
        base = ""
        if compare and loaded.has_adapter:
            base = _run_once(loaded, image, prompt, params, with_adapter=False)
    except Exception as error:  # pragma: no cover - runtime/GPU dependent
        detail = f"{traceback.format_exc(limit=3)}{error}"
        return "", "", f"**產生失敗**\n\n```\n{detail}\n```"

    mode = "sampling（每次結果不同）" if do_sample else "greedy（每次結果相同）"
    details = [
        f"- 解碼方式：**{mode}**"
        + (f"，temperature={temperature}、top_p={top_p}" if do_sample else ""),
        f"- 權重：{'LoRA 微調後' if loaded.has_adapter else '⚠️ 原廠模型（沒有 LoRA）'}",
        f"- 上限 {int(max_new_tokens)} 個 token",
    ]
    if compare and not loaded.has_adapter:
        details.append("- 沒有 LoRA 權重，無法比較")
    return tuned, base, "\n".join(details)


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
    can_compare = has_adapter_configured()

    # Gradio 6 moved `theme` from the Blocks constructor to launch()
    with gr.Blocks(title="天文影像描述 · LoRA Demo") as demo:
        gr.Markdown("# 🔭 天文影像描述\n上傳一張天文照片，看模型怎麼描述它。")
        badge = gr.Markdown(status_badge())

        with gr.Tab("試用"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="天文照片", type="pil", sources=["upload"])
                    run_btn = gr.Button("產生描述", variant="primary", size="lg")

                    examples = example_images(cfg)
                    if examples:
                        gr.Examples(
                            examples=[[p] for p in examples],
                            inputs=[image_input],
                            label="沒有圖？點一張試試",
                        )

                    compare = gr.Checkbox(
                        value=can_compare,
                        interactive=can_compare,
                        label="同時顯示原廠模型的描述（比較微調效果，時間會變兩倍）",
                    )

                    with gr.Accordion("進階設定（不用動也可以）", open=False):
                        instruction_input = gr.Textbox(
                            label="指令（英文）",
                            value=default_instruction,
                            lines=2,
                        )
                        max_tokens = gr.Slider(
                            32, 512, value=int(cfg.serving.max_new_tokens), step=32,
                            label="最多產生幾個 token",
                        )
                        do_sample = gr.Checkbox(
                            value=bool(cfg.serving.do_sample),
                            label="隨機取樣（關掉 = 每次結果都一樣）",
                        )
                        temperature = gr.Slider(
                            0.1, 2.0, value=float(cfg.serving.temperature), step=0.1,
                            label="Temperature（只在取樣時有效）",
                        )
                        top_p = gr.Slider(
                            0.1, 1.0, value=float(cfg.serving.top_p), step=0.05,
                            label="Top-p（只在取樣時有效）",
                        )
                        # the sliders go visibly inert when greedy, so nobody
                        # thinks temperature is doing something it is not
                        do_sample.change(
                            lambda flag: (gr.update(interactive=flag), gr.update(interactive=flag)),
                            inputs=do_sample,
                            outputs=[temperature, top_p],
                        )

                with gr.Column(scale=1):
                    tuned_out = gr.Textbox(
                        label="✨ 微調後（LoRA）", lines=7, placeholder="描述會出現在這裡"
                    )
                    base_out = gr.Textbox(
                        label="原廠模型（未微調）",
                        lines=7,
                        placeholder="勾選比較後才會有內容",
                        visible=can_compare,
                    )
                    details = gr.Markdown()

            compare.change(lambda flag: gr.update(visible=flag), inputs=compare, outputs=base_out)

            gr.Markdown(
                "> 第一次產生要先下載並載入約 8 GB 的模型，可能要好幾分鐘，"
                "之後就快了。想先載好可以到「狀態」分頁按載入。"
            )

            run_btn.click(
                describe_image,
                inputs=[
                    image_input, instruction_input, compare,
                    max_tokens, do_sample, temperature, top_p,
                ],
                outputs=[tuned_out, base_out, details],
            )

        with gr.Tab("說明"):
            gr.Markdown(about_text(cfg))

        with gr.Tab("狀態"):
            detail_md = gr.Markdown(weights_detail())
            load_msg = gr.Markdown()
            with gr.Row():
                gr.Button("先載入模型", variant="primary").click(
                    preload, outputs=[badge, load_msg]
                ).then(weights_detail, outputs=detail_md)
                gr.Button("重新檢查").click(weights_detail, outputs=detail_md).then(
                    status_badge, outputs=badge
                )

    return demo


def main(cfg: DictConfig) -> None:
    import gradio as gr

    demo = build_demo(cfg)
    demo.launch(
        server_name=str(cfg.serving.host),
        server_port=int(cfg.serving.port),
        share=bool(cfg.serving.share),
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main(config_from_args("Gradio front end"))
