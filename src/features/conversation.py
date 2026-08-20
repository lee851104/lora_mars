"""Multimodal conversation format. Pure functions - no file I/O, no GPU."""

from __future__ import annotations

import os
from typing import Any

Record = dict[str, Any]
Conversation = dict[str, Any]


def user_turn(instruction: str, image_path: str | None = None) -> dict[str, Any]:
    """The user turn.

    At inference time only an image placeholder is needed - the actual pixels go
    to the processor as a positional argument - so image_path is None. At
    training time the collator reads the file, so the path is required.
    """
    image_part: dict[str, Any] = {"type": "image"}
    if image_path is not None:
        image_part["image"] = image_path
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            image_part,
        ],
    }


def assistant_turn(caption: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": caption}]}


def to_conversation(record: Record, images_root: str, instruction: str) -> Conversation:
    """Turn one {image_id, image, text} record into a training conversation,
    carrying the id along so results stay traceable.
    """
    image_path = os.path.join(images_root, record["image"])
    return {
        "image_id": record["image_id"],
        "messages": [
            user_turn(instruction, image_path),
            assistant_turn(record["text"]),
        ],
    }


def inference_messages(instruction: str) -> list[dict[str, Any]]:
    """Inference messages: a single user turn with no image path."""
    return [user_turn(instruction)]


def conversation_image_path(conversation: Conversation) -> str:
    """Read the image path back out, so callers never hand-index content[1]."""
    for part in conversation["messages"][0]["content"]:
        if part.get("type") == "image" and part.get("image"):
            return part["image"]
    raise KeyError("this conversation's user turn carries no image path")


def conversation_reference(conversation: Conversation) -> str:
    """Read the ground-truth caption back out."""
    for part in conversation["messages"][1]["content"]:
        if part.get("type") == "text":
            return part["text"]
    raise KeyError("this conversation's assistant turn carries no text")
