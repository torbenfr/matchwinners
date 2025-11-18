#!/usr/bin/env python3
"""
brand_domain.py

Animates a domain name sliding vertically across an ultra-wide canvas (default
7080×108 px). Local-only configuration with optional side logos and fade
controls.

Example:
    python brand_domain.py --domain "NIKE.COM" --font-path "Futura.ttf" \
      --bg-color "#000000" --font-color "#FFFFFF" --side-logo nike_logo.png

Dependencies:
    pip install pillow moviepy cairosvg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import io
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


def load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    tried = []

    def try_path(p: str) -> Optional[ImageFont.FreeTypeFont]:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            if p.lower().endswith(".ttc"):
                for idx in range(8):
                    try:
                        return ImageFont.truetype(p, size=size, index=idx)
                    except Exception:
                        continue
        except Exception:
            pass
        tried.append(p)
        return None

    if font_path and os.path.exists(font_path):
        font = try_path(font_path)
        if font:
            return font

    system_candidates = [
        "/System/Library/Fonts/Supplemental/Inter.ttc",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for p in system_candidates:
        if os.path.exists(p):
            font = try_path(p)
            if font:
                return font

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        raise SystemExit(
            "Konnte keine skalierbare Schrift laden. Bitte --font-path auf eine TrueType/OpenType-Datei setzen."
        )


def text_bbox(text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    return d.textbbox((0, 0), text, font=font)


def render_text_autosize(text: str, font_path: Optional[str], color: Tuple[int, int, int], target_h: int) -> Image.Image:
    lo, hi = 1, 2000
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(font_path, size=mid)
        _, t, _, b = text_bbox(text, font)
        h = b - t
        if h <= target_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    font = load_font(font_path, size=best)
    l, t, r, b = text_bbox(text, font)
    w, h = r - l, b - t
    w = max(1, w)
    h = max(1, h)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((-l, -t), text, font=font, fill=color)
    return img


def fade_factor(t: float, total: float, fade_in: float, fade_out: float) -> float:
    if total <= 0:
        return 1.0
    factor = 1.0
    if fade_in > 0:
        factor = min(factor, max(0.0, min(1.0, t / fade_in)))
    if fade_out > 0:
        factor = min(factor, max(0.0, min(1.0, (total - t) / fade_out)))
    return max(0.0, min(1.0, factor))


def paste_with_alpha(frame: Image.Image, img: Image.Image, pos: Tuple[int, int], alpha: float) -> None:
    if alpha <= 0.0:
        return
    if alpha >= 0.999:
        frame.paste(img, pos, img)
        return
    temp = img.convert("RGBA") if img.mode != "RGBA" else img.copy()
    r, g, b, a = temp.split()
    a = a.point(lambda p: int(p * alpha))
    temp.putalpha(a)
    frame.paste(temp, pos, temp)


def load_logo(path: str, max_height: int) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        if not HAS_CAIROSVG:
            raise SystemExit("SVG-Logo gefunden, aber cairosvg ist nicht installiert.")
        with open(path, "rb") as fh:
            svg_bytes = fh.read()
        png = cairosvg.svg2png(bytestring=svg_bytes)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
    else:
        img = Image.open(path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    w, h = img.size
    if h:
        scale = max_height / h
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    return img


def ease_out_cubic(x: float) -> float:
    return 1 - (1 - x) ** 3


def ease_in_cubic(x: float) -> float:
    return x ** 3


@dataclass
class RenderConfig:
    width: int
    height: int
    fps: int
    bg_rgb: Tuple[int, int, int]
    text_img: Image.Image
    slide_from: str
    exit_to: str
    in_duration: float
    hold_duration: float
    out_duration: float
    total_duration: float
    text_fade_in: float
    text_fade_out: float
    logo_img: Optional[Image.Image]
    logo_positions_left: List[int]
    logo_positions_right: List[int]
    logo_padding: int
    logo_total_duration: float
    logo_fade_in: float
    logo_fade_out: float
    logo_in_duration: float
    logo_hold_duration: float
    logo_out_duration: float
    freeze_text: bool
    freeze_logos: bool


def position_for_progress(progress: float, cfg: RenderConfig) -> Tuple[int, int]:
    tw, th = cfg.text_img.size
    total = cfg.in_duration + cfg.hold_duration + cfg.out_duration
    t = max(0.0, min(progress, total))
    cx = (cfg.width - tw) // 2
    cy = (cfg.height - th) // 2
    if cfg.freeze_text:
        return (cx, cy)

    if t <= cfg.in_duration:
        p = 0.0 if cfg.in_duration == 0 else t / cfg.in_duration
        e = ease_out_cubic(p)
        if cfg.slide_from == "top":
            start_y = -th
            return (cx, int(round(start_y + e * (cy - start_y))))
        start_y = cfg.height
        return (cx, int(round(start_y - e * (start_y - cy))))

    if t <= cfg.in_duration + cfg.hold_duration:
        return (cx, cy)

    t_out = t - (cfg.in_duration + cfg.hold_duration)
    p = 0.0 if cfg.out_duration == 0 else t_out / cfg.out_duration
    e = ease_in_cubic(min(1.0, p))
    exit_dir = cfg.exit_to if cfg.exit_to != "auto" else cfg.slide_from
    target_y = -th if exit_dir == "top" else cfg.height
    return (cx, int(round(cy + e * (target_y - cy))))


def animate_logo(t: float, final_x: int, side: str, cfg: RenderConfig, logo_width: int) -> Tuple[int, float]:
    total = cfg.logo_total_duration
    tt = max(0.0, min(t, total)) if total > 0 else t
    fade = fade_factor(tt, total, cfg.logo_fade_in, cfg.logo_fade_out) if total > 0 else 1.0
    if cfg.freeze_logos or total == 0:
        return final_x, fade
    in_d = cfg.logo_in_duration
    hold_d = cfg.logo_hold_duration
    out_d = cfg.logo_out_duration
    start = -logo_width - cfg.logo_padding if side == "left" else cfg.width + cfg.logo_padding
    exit_pos = start
    if tt <= in_d:
        p = 0.0 if in_d == 0 else tt / in_d
        e = ease_out_cubic(p)
        pos = int(round(start + e * (final_x - start)))
    elif tt <= in_d + hold_d:
        pos = final_x
    elif out_d == 0:
        pos = exit_pos
    else:
        t_out = tt - (in_d + hold_d)
        p = 0.0 if out_d == 0 else min(1.0, t_out / out_d)
        e = ease_in_cubic(p)
        pos = int(round(final_x + e * (exit_pos - final_x)))
    return pos, fade


def render_frame(t: float, cfg: RenderConfig) -> Image.Image:
    frame = Image.new("RGB", (cfg.width, cfg.height), cfg.bg_rgb)
    x, y = position_for_progress(t, cfg)
    text_alpha = fade_factor(t, cfg.total_duration, cfg.text_fade_in, cfg.text_fade_out)
    paste_with_alpha(frame, cfg.text_img, (x, y), text_alpha)

    if cfg.logo_img:
        lw, lh = cfg.logo_img.size
        y_logo = (cfg.height - lh) // 2
        for lx in cfg.logo_positions_left:
            pos_x, alpha = animate_logo(t, lx, "left", cfg, lw)
            if pos_x + lw > 0 and pos_x < cfg.width:
                paste_with_alpha(frame, cfg.logo_img, (pos_x, y_logo), alpha)
        for rx in cfg.logo_positions_right:
            pos_x, alpha = animate_logo(t, rx, "right", cfg, lw)
            if pos_x < cfg.width and pos_x + lw > 0:
                paste_with_alpha(frame, cfg.logo_img, (pos_x, y_logo), alpha)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Pfad zu einer JSON-Konfiguration")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument("--bg-color", type=str, default="#000000")
    parser.add_argument("--font-color", type=str, default=None)
    parser.add_argument("--width", type=int, default=7080)
    parser.add_argument("--height", type=int, default=108)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", default="output_domain.mp4")
    parser.add_argument("--slide-from", choices=["top", "bottom"], default="top")
    parser.add_argument("--exit-to", choices=["auto", "top", "bottom"], default="bottom")
    parser.add_argument("--in-duration", type=float, default=0.8)
    parser.add_argument("--hold-duration", type=float, default=2.5)
    parser.add_argument("--out-duration", type=float, default=0.8)
    parser.add_argument("--height-factor", type=float, default=0.98)
    parser.add_argument("--height-padding", type=int, default=0)
    parser.add_argument("--text-fade-in", type=float, default=0.0)
    parser.add_argument("--text-fade-out", type=float, default=0.0)
    parser.add_argument("--side-logo", type=str, default=None)
    parser.add_argument("--side-logo-height-factor", type=float, default=0.98)
    parser.add_argument("--side-logo-padding", type=int, default=120)
    parser.add_argument("--logo-fill-edges", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--freeze-logos", action="store_true")
    parser.add_argument("--logo-in-duration", type=float, default=0.8)
    parser.add_argument("--logo-hold-duration", type=float, default=2.5)
    parser.add_argument("--logo-out-duration", type=float, default=0.8)
    parser.add_argument("--logo-fade-in", type=float, default=0.0)
    parser.add_argument("--logo-fade-out", type=float, default=0.0)
    parser.add_argument("--pix-fmt", type=str, default="yuv420p")
    parser.add_argument("--crf", type=int, default=18)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cli_args = vars(args)
        for k, v in data.items():
            if k in cli_args and cli_args[k] in (None, False, 0, ""):
                setattr(args, k, v)

    if not args.domain:
        raise SystemExit("Bitte --domain angeben oder im JSON setzen.")

    bg_rgb = hex_to_rgb(args.bg_color)
    text_rgb = hex_to_rgb(args.font_color) if args.font_color else (
        (255, 255, 255) if relative_luminance(bg_rgb) < 0.5 else (0, 0, 0)
    )

    target_h = max(1, int(round(args.height * args.height_factor)) - max(0, args.height_padding))
    text_img = render_text_autosize(args.domain, args.font_path, text_rgb, target_h)

    logo_img = None
    logo_positions_left: List[int] = []
    logo_positions_right: List[int] = []
    if args.side_logo:
        if not os.path.exists(args.side_logo):
            raise SystemExit(f"Seitenlogo nicht gefunden: {args.side_logo}")
        max_logo_h = max(1, int(round(args.height * float(args.side_logo_height_factor))))
        logo_img = load_logo(args.side_logo, max_logo_h)
        lw, _ = logo_img.size
        tw, _ = text_img.size
        cx = (args.width - tw) // 2
        left_x = cx - args.side_logo_padding - lw
        right_x = cx + tw + args.side_logo_padding
        if left_x < 0 or right_x + lw > args.width:
            print("⚠️ Hinweis: Seitenlogo übersprungen (zu wenig Platz).", file=sys.stderr)
            logo_img = None
        else:
            logo_positions_left.append(left_x)
            logo_positions_right.append(right_x)
            if args.logo_fill_edges:
                distance = lw + args.side_logo_padding
                pos = left_x - distance
                while pos + lw > 0:
                    logo_positions_left.append(pos)
                    pos -= distance
                pos = right_x + distance
                while pos < args.width:
                    logo_positions_right.append(pos)
                    pos += distance
            logo_positions_left.sort()
            logo_positions_right.sort()

    text_total = max(0.0, args.in_duration) + max(0.0, args.hold_duration) + max(0.0, args.out_duration)
    logo_total = 0.0
    if logo_img is not None:
        logo_total = max(0.0, args.logo_in_duration) + max(0.0, args.logo_hold_duration) + max(0.0, args.logo_out_duration)
    total_duration = max(0.01, max(text_total, logo_total))
    exit_to = args.exit_to if args.exit_to != "auto" else "bottom"

    cfg = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        bg_rgb=bg_rgb,
        text_img=text_img,
        slide_from=args.slide_from,
        exit_to=exit_to,
        in_duration=max(0.0, args.in_duration),
        hold_duration=max(0.0, args.hold_duration),
        out_duration=max(0.0, args.out_duration),
        total_duration=total_duration,
        text_fade_in=max(0.0, args.text_fade_in),
        text_fade_out=max(0.0, args.text_fade_out),
        logo_img=logo_img,
        logo_positions_left=logo_positions_left,
        logo_positions_right=logo_positions_right,
        logo_padding=args.side_logo_padding,
        logo_total_duration=logo_total,
        logo_fade_in=max(0.0, args.logo_fade_in),
        logo_fade_out=max(0.0, args.logo_fade_out),
        logo_in_duration=max(0.0, args.logo_in_duration),
        logo_hold_duration=max(0.0, args.logo_hold_duration),
        logo_out_duration=max(0.0, args.logo_out_duration),
        freeze_text=bool(args.freeze_text),
        freeze_logos=bool(args.freeze_logos),
    )

    clip = VideoClip(lambda tt: np.array(render_frame(tt, cfg)), duration=total_duration).with_fps(cfg.fps)
    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".gif":
        clip.write_gif(args.output, fps=args.fps, program="ffmpeg")
    else:
        ffmpeg_params = ["-pix_fmt", args.pix_fmt, "-crf", str(args.crf), "-tune", "animation"]
        clip.write_videofile(args.output, fps=args.fps, codec="libx264", audio=False,
                             preset="medium", ffmpeg_params=ffmpeg_params)
    print(f"✅ Fertig: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
