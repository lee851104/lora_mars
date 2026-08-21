"""Build the small animated workflow used by the GitHub README."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 960, 430
BG = "#070911"
SURFACE = "#111727"
LINE = "#29334e"
TEXT = "#f4f7ff"
MUTED = "#98a3be"
ACCENT = "#9d8cff"
CYAN = "#56d9ff"
GREEN = "#62e6a7"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ["C:/Windows/Fonts/seguisb.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE = font(28, bold=True)
LABEL = font(13, bold=True)
BODY = font(15)
SMALL = font(11, bold=True)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)


def build_frame(frame_index: int, total_frames: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    progress = frame_index / total_frames

    draw.ellipse((700, -260, 1120, 160), fill="#17163d")
    draw.ellipse((-200, 250, 250, 700), fill="#0b2534")
    draw.text((54, 37), "ASTROVISION LORA", font=SMALL, fill=CYAN)
    draw.text((54, 62), "Image to caption, with a traceable pipeline", font=TITLE, fill=TEXT)
    draw.text((54, 102), "UPLOAD  →  GEMMA 3 4B + LORA  →  CAPTION", font=BODY, fill=MUTED)

    cards = [
        ((54, 155, 286, 350), "01", "UPLOAD IMAGE", "Astronomy photo\nheld in the browser"),
        ((364, 155, 596, 350), "02", "LORA INFERENCE", "Gemma 3 4B\n+ 119 MB adapter"),
        ((674, 155, 906, 350), "03", "CAPTION", "Grounded English\ndescription"),
    ]
    active = min(2, int(progress * 3.25))
    for index, (box, number, heading, body) in enumerate(cards):
        outline = [CYAN, ACCENT, GREEN][index] if index == active else LINE
        rounded(draw, box, SURFACE, outline)
        draw.rounded_rectangle(
            (box[0] + 18, box[1] + 18, box[0] + 52, box[1] + 52),
            radius=9,
            fill="#1b2235",
            outline=outline,
        )
        draw.text((box[0] + 27, box[1] + 27), number, font=SMALL, fill=outline)
        draw.text((box[0] + 18, box[1] + 76), heading, font=LABEL, fill=TEXT)
        draw.multiline_text(
            (box[0] + 18, box[1] + 112), body, font=BODY, fill=MUTED, spacing=8
        )

    line_y = 252
    draw.line((286, line_y, 364, line_y), fill=LINE, width=3)
    draw.line((596, line_y, 674, line_y), fill=LINE, width=3)
    x = 286 + (674 - 286) * progress
    glow = 5 + round(2 * math.sin(progress * math.tau * 3))
    draw.ellipse((x - glow, line_y - glow, x + glow, line_y + glow), fill=CYAN)

    caption = "A rust-toned planetary surface extends toward a hazy horizon."
    if active == 2:
        shown = caption[: max(1, int(len(caption) * ((progress * 3) - 2)))]
        draw.rounded_rectangle((690, 300, 890, 333), radius=8, fill="#0a0e18")
        draw.text((700, 309), shown[:29], font=font(11), fill=TEXT)

    draw.text((54, 384), "UI FLOW PREVIEW · NO METRIC VALUES ARE SIMULATED", font=SMALL, fill="#6f7b99")
    return image


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo-flow.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = [build_frame(index, 30) for index in range(30)]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    print(f"wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
