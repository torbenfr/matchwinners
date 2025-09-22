#!/usr/bin/env python3
"""
brand_snail.py

Erzeugt eine einfache "Schnecken"-Animation (Ticker), bei der ein Markenlogo
rechts→links über eine LED-Bandengröße (Standard 3456×108 px) wandert.

Betriebsarten:
  1) Brandfetch-Modus  → --identifier <domain> (z. B. pluto.tv) + --api-key <KEY>
  2) Lokaler Modus     → --identifier local + --graphic <Pfad-zum-Logo> (+ --bg-override)

Beispiele:
  Brandfetch:
    python brand_snail.py --identifier pluto.tv --api-key $BRANDFETCH_API_KEY \
      --output pluto_snail.mp4 --speed 240 --duration 12

  Lokal (ohne Brandfetch):
    python brand_snail.py --identifier local --graphic Pluto_TV_logo.png \
      --bg-override "#000000" --output pluto_snail.mp4 --speed 240 --duration 12

Abhängigkeiten:
    pip install requests pillow moviepy cairosvg
    (ffmpeg empfohlen: brew install ffmpeg)

Hinweise:
  • MoviePy ≥ 2.x: Import aus moviepy.video.VideoClip und .with_fps()
  • Für SVG-Logos wird cairosvg empfohlen (optional).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass
from math import ceil
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.VideoClip import VideoClip

try:
    import cairosvg  # type: ignore
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False

BRANDFETCH_URL = "https://api.brandfetch.io/v2/brands/{identifier}"

@dataclass
class LogoFormat:
    src: str
    fmt: str
    width: int
    height: int
    background: str

@dataclass
class LogoEntry:
    type: str
    theme: Optional[str]
    formats: List[LogoFormat]

@dataclass
class BrandColor:
    hex: str
    type: Optional[str]
    brightness: Optional[float]

@dataclass
class BrandData:
    name: Optional[str]
    domain: Optional[str]
    logos: List[LogoEntry]
    colors: List[BrandColor]

@dataclass
class RenderConfig:
    speed: float = 220.0
    gap: int = 48
    logo_max_height: int = 88
    duration: float = 10.0
    fps: int = 30
    output: str = "output.mp4"
    show_domain: bool = False
    claim: Optional[str] = None
    graphic_path: Optional[str] = None
    identifier: str = ""
    copies: int = 2
    fill: bool = False  # durchgehende Kette ohne Lücken

def fetch_brand(identifier: str, api_key: str) -> BrandData:
    url = BRANDFETCH_URL.format(identifier=identifier)
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=20)

    if r.status_code in (401, 403):
        raise SystemExit(f"Brandfetch verweigert den Zugriff ({r.status_code}).")
    if r.status_code == 404:
        raise SystemExit(f"Brand nicht gefunden (404): {identifier}")

    r.raise_for_status()
    data = r.json()

    logos: List[LogoEntry] = []
    for L in data.get("logos", []) or []:
        formats = []
        for F in L.get("formats", []) or []:
            formats.append(LogoFormat(
                src=F.get("src"),
                fmt=(F.get("format") or "").lower(),
                width=int(F.get("width") or 0),
                height=int(F.get("height") or 0),
                background=str(F.get("background") or "transparent")
            ))
        logos.append(LogoEntry(
            type=L.get("type", ""),
            theme=L.get("theme"),
            formats=formats,
        ))

    colors: List[BrandColor] = []
    for C in data.get("colors", []) or []:
        colors.append(BrandColor(
            hex=C.get("hex", "#000000"),
            type=C.get("type"),
            brightness=C.get("brightness"),
        ))

    return BrandData(
        name=data.get("name"),
        domain=data.get("domain"),
        logos=logos,
        colors=colors,
    )

def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(ch*2 for ch in h)
    if len(h) != 6:
        return (0, 0, 0)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def f(c: float) -> float:
        c = c / 255.0
        return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

def pick_background_color(colors: List[BrandColor]) -> Tuple[int, int, int]:
    if not colors:
        return (0, 0, 0)
    cand_rgb = [(c, hex_to_rgb(c.hex)) for c in colors]
    chosen = min(cand_rgb, key=lambda x: relative_luminance(x[1]))
    return chosen[1]

def pick_logo_format(entries: List[LogoEntry], prefer_theme: Optional[str]) -> Optional[LogoFormat]:
    if not entries:
        return None
    fmt_order = {"svg": 0, "png": 1, "webp": 2, "jpg": 3, "jpeg": 3}
    for e in entries:
        formats_sorted = sorted(e.formats, key=lambda f: (fmt_order.get(f.fmt, 99), -(f.width * f.height)))
        if formats_sorted:
            return formats_sorted[0]
    return None

def load_logo_local(path: str, max_h: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    if h:
        scale = max_h / h
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    return img

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

def render_frame(t: float, cfg: RenderConfig, canvas_w: int, canvas_h: int,
                 bg_rgb: Tuple[int,int,int], logo_img: Image.Image,
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
        # durchgehende Kette
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--api-key", required=False)
    parser.add_argument("--width", type=int, default=3456)
    parser.add_argument("--height", type=int, default=108)
    parser.add_argument("--speed", type=float, default=220.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--fill", action="store_true", help="durchgehende Kette ohne Lücken")
    parser.add_argument("--gap", type=int, default=48)
    parser.add_argument("--logo-max-height", type=int, default=88)
    parser.add_argument("--show-domain", action="store_true")
    parser.add_argument("--claim", type=str, default=None)
    parser.add_argument("--graphic", type=str, default=None)
    parser.add_argument("--bg-override", type=str, default=None)
    args = parser.parse_args()

    cfg = RenderConfig(speed=args.speed,
        duration=args.duration,
        fps=args.fps,
        output=args.output,
        gap=args.gap,
        logo_max_height=args.logo_max_height,
        show_domain=args.show_domain,
        claim=args.claim,
        graphic_path=args.graphic,
        identifier=args.identifier,
        copies=max(1, args.copies),
        fill=args.fill)

    W, H = args.width, args.height

    if args.identifier.lower() == "local":
        if not args.graphic:
            raise SystemExit("Lokaler Modus: Bitte --graphic <Pfad_zum_Logo> angeben.")
        bg_rgb = hex_to_rgb(args.bg_override) if args.bg_override else (0,0,0)
        logo_img = load_logo_local(args.graphic, max_h=cfg.logo_max_height)
        claim_img = None
        if cfg.claim:
            text_color = (255,255,255) if relative_luminance(bg_rgb) < 0.5 else (0,0,0)
            claim_img = build_text_image(cfg.claim, font_size=26, color=text_color)
        def make_frame(t: float):
            return np.array(render_frame(t, cfg, W, H, bg_rgb, logo_img, None, claim_img))
        clip = VideoClip(lambda t: make_frame(t), duration=cfg.duration).with_fps(cfg.fps)
        ext = os.path.splitext(cfg.output)[1].lower()
        if ext == ".gif":
            clip.write_gif(cfg.output, fps=cfg.fps, program="ffmpeg")
        else:
            clip.write_videofile(cfg.output, fps=cfg.fps, codec="libx264", audio=False, preset="medium")
        print(f"✅ Fertig (lokal): {cfg.output}")
        return

    if not args.api_key:
        raise SystemExit("Brandfetch-Modus: Bitte --api-key angeben.")
    brand = fetch_brand(args.identifier, args.api_key)
    bg_rgb = hex_to_rgb(args.bg_override) if args.bg_override else pick_background_color(brand.colors)
    logo_fmt = pick_logo_format(brand.logos, None)
    if not logo_fmt:
        raise SystemExit("Kein Logo gefunden.")
    # Logo laden
    resp_logo = requests.get(logo_fmt.src, timeout=20)
    resp_logo.raise_for_status()
    logo_img = Image.open(io.BytesIO(resp_logo.content)).convert("RGBA")
    w, h = logo_img.size
    if h:
        scale = cfg.logo_max_height / h
        logo_img = logo_img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)

    domain_img = None
    if cfg.show_domain and (brand.domain or brand.name):
        text = (brand.domain or brand.name or "").replace("https://", "").replace("http://", "")
        text_color = (255,255,255) if relative_luminance(bg_rgb) < 0.5 else (0,0,0)
        domain_img = build_text_image(text, font_size=28, color=text_color)

    claim_img = None
    if cfg.claim:
        text_color = (255,255,255) if relative_luminance(bg_rgb) < 0.5 else (0,0,0)
        claim_img = build_text_image(cfg.claim, font_size=26, color=text_color)

    def make_frame(t: float):
        return np.array(render_frame(t, cfg, W, H, bg_rgb, logo_img, domain_img, claim_img))

    clip = VideoClip(lambda t: make_frame(t), duration=cfg.duration).with_fps(cfg.fps)
    ext = os.path.splitext(cfg.output)[1].lower()
    if ext == ".gif":
        clip.write_gif(cfg.output, fps=cfg.fps, program="ffmpeg")
    else:
        clip.write_videofile(cfg.output, fps=cfg.fps, codec="libx264", audio=False, preset="medium")
    print(f"✅ Fertig: {cfg.output}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
