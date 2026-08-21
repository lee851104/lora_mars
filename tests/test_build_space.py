from pathlib import Path

from src.serving.build_space import build_space


def test_static_space_contains_only_static_assets(cfg, tmp_path: Path):
    staged = build_space(cfg, tmp_path / "static", static=True)

    assert (staged / "index.html").is_file()
    assert (staged / "style.css").is_file()
    assert (staged / "app.js").is_file()
    assert "sdk: static" in (staged / "README.md").read_text(encoding="utf-8")
    html = (staged / "index.html").read_text(encoding="utf-8")
    assert "UI PREVIEW" in html
    assert 'src="app.js"' in html
    assert not (staged / "app.py").exists()
    assert not (staged / "src").exists()


def test_gradio_space_keeps_the_python_app(cfg, tmp_path: Path):
    staged = build_space(cfg, tmp_path / "gradio")

    assert (staged / "app.py").is_file()
    assert (staged / "src" / "serving" / "app.py").is_file()
    assert "sdk: gradio" in (staged / "README.md").read_text(encoding="utf-8")
