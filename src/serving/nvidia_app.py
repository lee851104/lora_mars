"""CPU-only Gradio demo for Render backed by NVIDIA's hosted vision API."""

from __future__ import annotations

import os
import time
from typing import Any

from src.serving.nvidia_client import DEFAULT_MODEL, NvidiaAPIError, generate

# 本機可使用 repo 根目錄的 .env；Render 則直接提供同名環境變數。
# load_dotenv 預設不會覆蓋 Render 已設定的值。
try:
    from dotenv import load_dotenv
except ImportError:  # 只有不安裝 Render 依賴的純測試環境會走到這裡
    pass
else:
    load_dotenv()

DEFAULT_PROMPT = (
    "You are an astronomy image assistant. Describe only what is visibly supported "
    "by the image. Do not invent mission names, locations, measurements, or dates. "
    "If uncertain, say so. Answer in concise English."
)


def model_name() -> str:
    return os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL).strip()


def answer(
    image: Any,
    question: str,
    max_tokens: int,
    temperature: float,
    access_code: str = "",
) -> tuple[str, str]:
    """Gradio callback. Errors are deliberately safe for a public page."""
    if image is None:
        return "", "請先上傳圖片。"

    required_code = os.environ.get("DEMO_ACCESS_CODE", "").strip()
    if required_code and access_code != required_code:
        return "", "存取碼不正確。"

    prompt = (question or "").strip() or DEFAULT_PROMPT
    started = time.monotonic()
    try:
        result = generate(
            image,
            prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            max_image_edge=int(os.environ.get("MAX_IMAGE_EDGE", "1280")),
        )
    except (NvidiaAPIError, ValueError) as error:
        return "", str(error)
    except Exception:
        return "", "伺服器發生未預期錯誤，請稍後再試。"

    seconds = time.monotonic() - started
    return result.text, f"模型：`{result.model}` · API 回應時間：{seconds:.1f} 秒"


def build_demo():
    import gradio as gr

    code_required = bool(os.environ.get("DEMO_ACCESS_CODE", "").strip())
    with gr.Blocks(title="天文影像互動展示 · NVIDIA API") as demo:
        gr.Markdown(
            f"""
# 🔭 天文影像互動展示

上傳圖片並輸入問題，由 NVIDIA 託管的視覺模型產生回答。

> **目前不是微調後的 Gemma 3 LoRA。** 這個臨時版本使用 NVIDIA API 的
> `{model_name()}` 基礎模型。微調權重仍保存在
> [Hugging Face](https://huggingface.co/lee851104/gemma3-4b-astronomy-lora)，
> 後續改用 Hugging Face GPU 時才會接回 LoRA。
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="圖片",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                )
                question = gr.Textbox(
                    label="指令或問題（建議使用英文）",
                    value=DEFAULT_PROMPT,
                    lines=5,
                )
                access_code = gr.Textbox(
                    label="展示存取碼",
                    type="password",
                    visible=code_required,
                    placeholder="由網站擁有者提供",
                )
                with gr.Accordion("進階設定", open=False):
                    max_tokens = gr.Slider(
                        32, 512, value=256, step=32, label="最多產生 token 數"
                    )
                    temperature = gr.Slider(
                        0.0, 1.0, value=0.2, step=0.1, label="Temperature"
                    )
                submit = gr.Button("送出", variant="primary", size="lg")

            with gr.Column(scale=1):
                output = gr.Textbox(label="模型回答", lines=16)
                status = gr.Markdown()

        submit.click(
            answer,
            inputs=[image, question, max_tokens, temperature, access_code],
            outputs=[output, status],
        )
        question.submit(
            answer,
            inputs=[image, question, max_tokens, temperature, access_code],
            outputs=[output, status],
        )

        gr.Markdown(
            f"""
### 使用限制

- 模型可能產生錯誤或虛構內容，不能作為科學判讀依據。
- NVIDIA 官方對此模型的影像加文字能力只正式支援英文。
- 目前模型：`{model_name()}`。本頁不會在瀏覽器保存 NVIDIA API 金鑰。
- Built with Llama. 使用受 Llama 3.2 Community License 約束。
"""
        )

    return demo


def main() -> None:
    import gradio as gr

    port = int(os.environ.get("PORT", "7860"))
    demo = build_demo()
    demo.queue(default_concurrency_limit=2, max_size=10).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
