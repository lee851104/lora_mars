"""Hugging Face Spaces entry point.

Spaces runs `python app.py` from the repo root, so this file must live at the
top level and be the thing that calls launch(). Everything real lives in
src/serving/app.py - this is only the adapter between the two.

The Space needs to know where the LoRA weights are. Set a Space variable
(Settings -> Variables and secrets):

    LORA_REPO_ID = <user>/<adapter-repo>

If it is unset, the config value is used; if that is empty too, the UI says so
plainly rather than serving the base model as if it were fine-tuned.
"""

from __future__ import annotations

import os

from src.config import load_config
from src.serving.app import build_demo


def build():
    cfg = load_config()

    repo_id = os.environ.get("LORA_REPO_ID", "").strip()
    if repo_id:
        cfg.lora.repo_id = repo_id

    # Spaces exposes the app on 7860 itself; sampling on by default reads better
    # in a demo than identical greedy output every time.
    return build_demo(cfg)


demo = build()

if __name__ == "__main__":
    import gradio as gr

    demo.launch(theme=gr.themes.Soft())
