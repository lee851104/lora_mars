"""Inference. The generation-parameter and prompt-stripping helpers are pure
and torch-free so they can be unit tested without a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.features.conversation import inference_messages


@dataclass
class Generation:
    """One decoded sample plus the token accounting behind it."""

    text: str
    prompt_tokens: int
    new_tokens: int
    full_text: str


def build_gen_kwargs(
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> dict[str, Any]:
    """Assemble generate() kwargs, dropping sampling knobs when greedy.

    transformers silently ignores `temperature` when `do_sample=False` (and
    warns about it). The original notebook passed `temperature=1.0` with the
    default `do_sample=False`, which reads like sampling but is pure greedy
    decoding. Making the coupling explicit here removes that trap: a greedy
    call never carries sampling parameters at all.
    """
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "use_cache": True,
    }
    if not do_sample:
        return kwargs
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    if min_p is not None:
        kwargs["min_p"] = float(min_p)
    return kwargs


def strip_prompt(sequence: Sequence[int], prompt_tokens: int) -> Sequence[int]:
    """Drop the echoed prompt from a generate() output sequence.

    `model.generate` returns prompt + continuation. Decoding the whole thing
    (as the original notebook did) glues the instruction onto every prediction
    and then feeds that to BLEU/ROUGE, so the scores describe the prompt as
    much as the model. Works for lists and for torch tensors alike.
    """
    if prompt_tokens < 0:
        raise ValueError(f"prompt_tokens must be >= 0, got {prompt_tokens}")
    return sequence[prompt_tokens:]


def prompt_length(inputs: Any) -> int:
    """Number of prompt tokens in a processor batch (batch size 1)."""
    return int(inputs["input_ids"].shape[1])


def generate_caption(
    model: Any,
    tokenizer: Any,
    image: Any,
    instruction: str,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    device: str = "cuda",
) -> Generation:
    """Describe one image. Returns only the newly generated text."""
    import torch

    messages = inference_messages(instruction)
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        prompt_text,
        add_special_tokens=False,  # the chat template already added them
        return_tensors="pt",
    ).to(device)

    n_prompt = prompt_length(inputs)
    gen_kwargs = build_gen_kwargs(max_new_tokens, do_sample, temperature, top_p, min_p)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # .detach().cpu() so a caller collecting many results does not pin VRAM
    sequence = outputs[0].detach().cpu()
    continuation = strip_prompt(sequence, n_prompt)

    return Generation(
        text=tokenizer.decode(continuation, skip_special_tokens=True).strip(),
        prompt_tokens=n_prompt,
        new_tokens=int(len(continuation)),
        full_text=tokenizer.decode(sequence, skip_special_tokens=True).strip(),
    )
