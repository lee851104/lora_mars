"""Small NVIDIA NIM client used by the CPU-only Render demo.

The browser never receives the API key.  Render accepts the image, resizes it,
and sends one request to NVIDIA's hosted multimodal endpoint.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MAX_IMAGE_EDGE = 1280


class NvidiaAPIError(RuntimeError):
    """A message that is safe to display in the public demo."""


@dataclass(frozen=True)
class NvidiaResult:
    text: str
    model: str


def image_data_uri(image: Image.Image, max_edge: int = DEFAULT_MAX_IMAGE_EDGE) -> str:
    """Convert a PIL image into a bounded JPEG data URI."""
    if image is None:
        raise ValueError("image is required")
    if max_edge < 64:
        raise ValueError("max_edge must be at least 64")

    converted = image.convert("RGB")
    if max(converted.size) > max_edge:
        scale = max_edge / max(converted.size)
        size = (
            max(1, round(converted.width * scale)),
            max(1, round(converted.height * scale)),
        )
        converted = converted.resize(size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    converted.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def request_payload(
    image: Image.Image,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.2,
    max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
) -> dict[str, Any]:
    """Build the OpenAI-compatible multimodal request accepted by NVIDIA NIM."""
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("prompt is required")

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cleaned_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri(image, max_image_edge)},
                    },
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 0.9,
        "stream": False,
    }


def _safe_http_error(status: int) -> str:
    if status in {401, 403}:
        return "NVIDIA API 金鑰無效或沒有這個模型的權限。"
    if status == 413:
        return "圖片仍然太大，請換一張較小的圖片。"
    if status == 429:
        return "NVIDIA API 暫時達到速率或試用額度限制，請稍後再試。"
    if status >= 500:
        return "NVIDIA API 暫時無法使用，請稍後再試。"
    return f"NVIDIA API 請求失敗（HTTP {status}）。"


def generate(
    image: Image.Image,
    prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    api_url: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
    max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
    timeout: float = 120,
    session: Any | None = None,
) -> NvidiaResult:
    """Call NVIDIA and return only the assistant text.

    ``session`` is injectable so unit tests never touch the network.
    """
    import requests

    token = (api_key or os.environ.get("NVIDIA_API_KEY", "")).strip()
    if not token:
        raise NvidiaAPIError("伺服器尚未設定 NVIDIA_API_KEY。")

    selected_model = (model or os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL)).strip()
    selected_url = (api_url or os.environ.get("NVIDIA_API_URL", DEFAULT_API_URL)).strip()
    payload = request_payload(
        image,
        prompt,
        model=selected_model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_image_edge=max_image_edge,
    )
    client = session or requests

    try:
        response = client.post(
            selected_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as error:
        raise NvidiaAPIError("NVIDIA API 回應逾時，請稍後再試。") from error
    except requests.RequestException as error:
        raise NvidiaAPIError("無法連線到 NVIDIA API，請稍後再試。") from error

    if not 200 <= response.status_code < 300:
        raise NvidiaAPIError(_safe_http_error(response.status_code))

    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        raise NvidiaAPIError("NVIDIA API 回傳了無法解析的內容。") from error
    if not text:
        raise NvidiaAPIError("NVIDIA API 沒有產生文字。")

    return NvidiaResult(text=text, model=selected_model)
