#!/usr/bin/env python3
"""
brand_snail.py

Erzeugt eine einfache "Schnecken"-Animation (Ticker), bei der ein Markenlogo
rechts→links über eine LED-Bandengröße (Standard 7080×108 px) wandert.

Nur lokaler Modus: Logo wird über `--graphic` angegeben, Farben/FPS lassen sich
frei konfigurieren.

Beispiel:
    python brand_snail.py --graphic nike_logo.png --output nike_snail.mp4 \
      --bg-color "#000000" --speed 240 --duration 12 --fill

Abhängigkeiten:
    pip install pillow moviepy cairosvg
    (ffmpeg empfohlen: brew install ffmpeg)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.VideoClip import VideoClip

try:
    import cairosvg  # type: ignore
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(ch * 2 for ch in h)
    if len(h) != 6:
        return (0, 0, 0)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def f(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ensure_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def build_text_image(text: str, font_size: int, color: Tuple[int, int, int]) -> Image.Image:
    font = ensure_font(font_size)
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    w, h = d.textbbox((0, 0), text, font=font)[2:4]
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(img)
    d2.text((0, 0), text, font=font, fill=color)
    return img


def load_logo_local(path: str, max_h: int) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    if h:
        scale = max_h / h
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    return img


def convert_svg_to_png(path: str) -> Image.Image:
    if not HAS_CAIROSVG:
        raise SystemExit("SVG Logo gefunden, bitte cairosvg installieren oder PNG nutzen.")
    with open(path, "rb") as fh:
        svg_data = fh.read()
    png_bytes = cairosvg.svg2png(bytestring=svg_data)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def load_graphic(path: str, logo_max_height: int) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        img = convert_svg_to_png(path)
    else:
        img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    if h:
        scale = logo_max_height / h
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    return img


@dataclass
class RenderConfig:
    speed: float
    gap: int
    logo_max_height: int
    duration: float
    fps: int
    output: str
    fill: bool
    copies: int


def render_frame(t: float, cfg: RenderConfig, canvas_w: int, canvas_h: int,
                 bg_rgb: Tuple[int, int, int], logo_img: Image.Image,
                 domain_text_img: Optional[Image.Image],
                 claim_text_img: Optional[Image.Image]) -> Image.Image:
    frame = Image.new("RGB", (canvas_w, canvas_h), bg_rgb)
    if domain_text_img is not None:
        frame.paste(domain_text_img, (24, 8), domain_text_img)
    if claim_text_img is not None:
        frame.paste(claim_text_img, (24, 56), claim_text_img)

    lw, lh = logo_img.size
    period = lw + cfg.gap
    base_x = int(canvas_w - ((t * cfg.speed) % period))
    y = (canvas_h - lh) // 2

    if cfg.fill:
        start_x = base_x
        while start_x > 0:
            start_x -= period
        x = start_x
        while x < canvas_w + period:
            if x + lw > 0:
                frame.paste(logo_img, (x, y), logo_img)
            x += period
    else:
        for i in range(max(1, cfg.copies)):
            xi = base_x - i * period
            if xi < canvas_w and xi + lw > 0:
                frame.paste(logo_img, (xi, y), logo_img)
        extra_x = base_x + max(1, cfg.copies) * period
        if extra_x < canvas_w and extra_x + lw > 0:
            frame.paste(logo_img, (extra_x, y), logo_img)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphic", required=True, help="Pfad zum Logo (PNG, SVG, ...)")
    parser.add_argument("--bg-color", default="#000000", help="Hintergrundfarbe (Hex)")
    parser.add_argument("--width", type=int, default=7080)
    parser.add_argument("--height", type=int, default=108)
    parser.add_argument("--speed", type=float, default=220.0)
    parser.add_argument("--gap", type=int, default=48)
    parser.add_argument("--logo-max-height", type=int, default=88)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", default="output_snail.mp4")
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--fill", action="store_true", help="durchgehende Kette ohne Lücken")
    parser.add_argument("--domain-text", type=str, default=None)
    parser.add_argument("--claim", type=str, default=None)
    parser.add_argument("--text-color", default=None, help="Textfarbe (Hex, optional)")
    args = parser.parse_args()

    if not os.path.exists(args.graphic):
        raise SystemExit(f"Logo nicht gefunden: {args.graphic}")

    bg_rgb = hex_to_rgb(args.bg_color)
    text_rgb = hex_to_rgb(args.text_color) if args.text_color else ((255, 255, 255) if relative_luminance(bg_rgb) < 0.5 else (0, 0, 0))

    logo_img = load_graphic(args.graphic, args.logo_max_height)

    domain_img = build_text_image(args.domain_text, 28, text_rgb) if args.domain_text else None
    claim_img = build_text_image(args.claim, 26, text_rgb) if args.claim else None

    cfg = RenderConfig(speed=args.speed,
                       gap=args.gap,
                       logo_max_height=args.logo_max_height,
                       duration=args.duration,
                       fps=args.fps,
                       output=args.output,
                       fill=args.fill,
                       copies=max(1, args.copies))

    def make_frame(tt: float) -> np.ndarray:
        return np.array(render_frame(tt, cfg, args.width, args.height, bg_rgb,
                                     logo_img, domain_img, claim_img))

    clip = VideoClip(lambda tt: make_frame(tt), duration=cfg.duration).with_fps(cfg.fps)
    ext = os.path.splitext(cfg.output)[1].lower()
    if ext == ".gif":
        clip.write_gif(cfg.output, fps=cfg.fps, program="ffmpeg")
    else:
        clip.write_videofile(cfg.output, fps=cfg.fps, codec="libx264", audio=False,
                             preset="medium")
    print(f"✅ Fertig: {cfg.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
