from __future__ import annotations

import base64
from pathlib import Path

import pytest
import requests
from omegaconf import OmegaConf
from PIL import Image

from src.serving import nvidia_app, nvidia_client

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {"choices": [{"message": {"content": "A galaxy."}}]}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.call = None

    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        if self.error:
            raise self.error
        return self.response


def test_image_data_uri_downscales_and_encodes_jpeg():
    uri = nvidia_client.image_data_uri(Image.new("RGB", (2000, 1000)), max_edge=400)
    prefix, data = uri.split(",", 1)

    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(data).startswith(b"\xff\xd8")


def test_request_payload_is_multimodal_and_non_streaming():
    payload = nvidia_client.request_payload(
        Image.new("RGB", (8, 8)), "Describe this.", max_tokens=64, temperature=0.1
    )

    assert payload["model"] == nvidia_client.DEFAULT_MODEL
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0] == {
        "type": "text",
        "text": "Describe this.",
    }
    assert payload["messages"][0]["content"][1]["type"] == "image_url"


def test_generate_keeps_key_in_authorization_header_only():
    session = FakeSession()
    result = nvidia_client.generate(
        Image.new("RGB", (8, 8)), "Describe.", api_key="nvapi-secret", session=session
    )

    _url, kwargs = session.call
    assert result.text == "A galaxy."
    assert kwargs["headers"]["Authorization"] == "Bearer nvapi-secret"
    assert "nvapi-secret" not in str(kwargs["json"])


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "金鑰無效"), (429, "速率或試用額度"), (503, "暫時無法使用")],
)
def test_generate_returns_safe_http_errors(status, expected):
    session = FakeSession(FakeResponse(status_code=status))

    with pytest.raises(nvidia_client.NvidiaAPIError, match=expected):
        nvidia_client.generate(
            Image.new("RGB", (8, 8)), "Describe.", api_key="secret", session=session
        )


def test_generate_maps_network_timeout():
    session = FakeSession(error=requests.Timeout("private network details"))

    with pytest.raises(nvidia_client.NvidiaAPIError, match="回應逾時"):
        nvidia_client.generate(
            Image.new("RGB", (8, 8)), "Describe.", api_key="secret", session=session
        )


def test_answer_requires_optional_access_code(monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "let-me-in")

    output, status = nvidia_app.answer(Image.new("RGB", (8, 8)), "Describe.", 64, 0.2)

    assert output == ""
    assert status == "存取碼不正確。"


def test_answer_does_not_expose_unexpected_exception(monkeypatch):
    monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
    monkeypatch.setattr(nvidia_app, "generate", lambda *_args, **_kwargs: 1 / 0)

    output, status = nvidia_app.answer(Image.new("RGB", (8, 8)), "Describe.", 64, 0.2)

    assert output == ""
    assert "未預期錯誤" in status


def test_render_blueprint_uses_cpu_proxy_and_secret_key():
    blueprint = OmegaConf.load(REPO_ROOT / "render.yaml")
    service = blueprint.services[0]

    assert service.type == "web"
    assert service.runtime == "python"
    assert service.plan == "free"
    assert service.startCommand == "python -m src.serving.nvidia_app"
    assert service.healthCheckPath == "/"
    secrets = {item.key: item for item in service.envVars}
    assert secrets["NVIDIA_API_KEY"].sync is False

    requirements = (REPO_ROOT / "requirements-render.txt").read_text(encoding="utf-8")
    packages = "\n".join(
        line for line in requirements.splitlines() if line and not line.startswith("#")
    ).lower()
    assert "gradio" in packages
    assert "requests" in packages
    assert "python-dotenv" in packages
    assert "torch" not in packages
    assert "unsloth" not in packages


def test_real_dotenv_is_ignored_but_example_is_tracked():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".env" in gitignore.splitlines()
    assert (REPO_ROOT / ".env.example").is_file()
    assert "NVIDIA_API_KEY=" in (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
