#!/usr/bin/env python3
"""
brand_item.py

Slides one or multiple foreground images (e.g. product shots) vertically across
an ultra-wide canvas. Items are centered within evenly spaced sections. Optional
side logos fade in/out and fill available gaps without overlapping items.

Example:
    python brand_item.py --image nike_shoe.png --item-count 3 --item-spacing 180 \
      --side-logo nike_logo.png --logo-fill-edges --logo-fade-in 0.5 --logo-fade-out 0.5

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
from PIL import Image
from moviepy.video.VideoClip import VideoClip

try:
    import cairosvg  # type: ignore
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False


def load_image(path: str, max_height: Optional[int] = None, max_width: Optional[int] = None) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        if not HAS_CAIROSVG:
            raise SystemExit("SVG-Datei benötigt cairosvg. Bitte installieren oder PNG nutzen.")
        with open(path, "rb") as fh:
            svg_bytes = fh.read()
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    else:
        img = Image.open(path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    w, h = img.size
    scale = 1.0
    if max_height and h > max_height:
        scale = min(scale, max_height / float(h))
    if max_width and w > max_width:
        scale = min(scale, max_width / float(w))
    if scale < 1.0:
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    return img


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
    item_img: Image.Image
    item_offsets: List[Tuple[int, int]]
    group_width: int
    group_height: int
    slide_from: str
    exit_to: str
    in_duration: float
    hold_duration: float
    out_duration: float
    total_duration: float
    freeze_item: bool
    logo_img: Optional[Image.Image]
    logo_positions_left: List[List[int]]
    logo_positions_right: List[List[int]]
    logo_padding: int
    logo_total_duration: float
    logo_fade_in: float
    logo_fade_out: float
    logo_in_duration: float
    logo_hold_duration: float
    logo_out_duration: float
    freeze_logos: bool
    item_fade_in: float
    item_fade_out: float


def position_for_progress(progress: float, cfg: RenderConfig) -> Tuple[int, int]:
    total = cfg.in_duration + cfg.hold_duration + cfg.out_duration
    t = max(0.0, min(progress, total))
    cx = (cfg.width - cfg.group_width) // 2
    cy = (cfg.height - cfg.group_height) // 2
    if cfg.freeze_item:
        return (cx, cy)

    if t <= cfg.in_duration:
        p = 0.0 if cfg.in_duration == 0 else t / cfg.in_duration
        e = ease_out_cubic(p)
        if cfg.slide_from == "top":
            start_y = -cfg.group_height
            return (cx, int(round(start_y + e * (cy - start_y))))
        start_y = cfg.height
        return (cx, int(round(start_y - e * (start_y - cy))))

    if t <= cfg.in_duration + cfg.hold_duration:
        return (cx, cy)

    t_out = t - (cfg.in_duration + cfg.hold_duration)
    p = 0.0 if cfg.out_duration == 0 else t_out / cfg.out_duration
    e = ease_in_cubic(min(1.0, p))
    exit_dir = cfg.exit_to if cfg.exit_to != "auto" else cfg.slide_from
    target_y = -cfg.group_height if exit_dir == "top" else cfg.height
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
    base_x, base_y = position_for_progress(t, cfg)
    item_alpha = fade_factor(t, cfg.total_duration, cfg.item_fade_in, cfg.item_fade_out)
    for dx, dy in cfg.item_offsets:
        paste_with_alpha(frame, cfg.item_img, (base_x + dx, base_y + dy), item_alpha)

    if cfg.logo_img:
        lw, lh = cfg.logo_img.size
        y_logo = (cfg.height - lh) // 2
        for row in cfg.logo_positions_left:
            for lx in row:
                pos_x, alpha = animate_logo(t, lx, "left", cfg, lw)
                if pos_x + lw > 0 and pos_x < cfg.width:
                    paste_with_alpha(frame, cfg.logo_img, (pos_x, y_logo), alpha)
        for row in cfg.logo_positions_right:
            for rx in row:
                pos_x, alpha = animate_logo(t, rx, "right", cfg, lw)
                if pos_x < cfg.width and pos_x + lw > 0:
                    paste_with_alpha(frame, cfg.logo_img, (pos_x, y_logo), alpha)

    return frame


def fill_logo_positions(section_start: int, section_end: int, logo_width: int, padding: int,
                        fill: bool, width: int) -> List[int]:
    available = section_end - section_start
    if available < logo_width:
        return []
    step = logo_width + padding
    if fill and step > 0:
        count = max(1, int((available + padding) // step))
    else:
        count = 1
    total = count * logo_width + (count - 1) * padding
    base = section_start + max(0, (available - total) // 2)
    return [int(base + i * (logo_width + padding)) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Pfad zu einer JSON-Konfiguration")
    parser.add_argument("--image", required=False)
    parser.add_argument("--width", type=int, default=7080)
    parser.add_argument("--height", type=int, default=108)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", default="output_item.mp4")
    parser.add_argument("--bg-color", default="#000000")
    parser.add_argument("--slide-from", choices=["top", "bottom"], default="top")
    parser.add_argument("--exit-to", choices=["auto", "top", "bottom"], default="bottom")
    parser.add_argument("--in-duration", type=float, default=0.8)
    parser.add_argument("--hold-duration", type=float, default=2.5)
    parser.add_argument("--out-duration", type=float, default=0.8)
    parser.add_argument("--item-fade-in", type=float, default=0.0)
    parser.add_argument("--item-fade-out", type=float, default=0.0)
    parser.add_argument("--side-logo", type=str, default=None)
    parser.add_argument("--side-logo-height-factor", type=float, default=0.98)
    parser.add_argument("--side-logo-padding", type=int, default=120)
    parser.add_argument("--logo-fill-edges", action="store_true")
    parser.add_argument("--freeze-item", action="store_true")
    parser.add_argument("--freeze-logos", action="store_true")
    parser.add_argument("--logo-in-duration", type=float, default=0.8)
    parser.add_argument("--logo-hold-duration", type=float, default=2.5)
    parser.add_argument("--logo-out-duration", type=float, default=0.8)
    parser.add_argument("--logo-fade-in", type=float, default=0.0)
    parser.add_argument("--logo-fade-out", type=float, default=0.0)
    parser.add_argument("--item-count", type=int, default=1)
    parser.add_argument("--item-spacing", type=int, default=120)
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

    if not args.image:
        raise SystemExit("Bitte --image angeben oder im JSON setzen.")
    if not os.path.exists(args.image):
        raise SystemExit(f"Bild nicht gefunden: {args.image}")

    bg_rgb = hex_to_rgb(args.bg_color)

    target_max_height = int(round(args.height * 0.9))
    item_img = load_image(args.image, target_max_height)

    item_count = max(1, args.item_count)
    spacing = max(0, args.item_spacing)

    section_width = max(1, args.width // (item_count * 2))
    iw, ih = item_img.size
    if iw > section_width:
        scale = section_width / float(iw)
        item_img = item_img.resize((max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))), Image.LANCZOS)
        iw, ih = item_img.size

    item_offsets: List[Tuple[int, int]] = []
    for idx in range(item_count):
        section_start = idx * 2 * section_width
        item_section_start = section_start + section_width
        offset_x = item_section_start + (section_width - iw) // 2
        item_offsets.append((offset_x, 0))

    min_offset = min((x for x, _ in item_offsets), default=0)
    item_offsets = [(x - min_offset, y) for x, y in item_offsets]
    group_width = max((x + iw) for x, _ in item_offsets) if item_offsets else iw
    group_height = ih

    logo_img = None
    logo_positions_left: List[List[int]] = []
    logo_positions_right: List[List[int]] = []
    if args.side_logo:
        if not os.path.exists(args.side_logo):
            raise SystemExit(f"Logo nicht gefunden: {args.side_logo}")
        max_logo_h = max(1, int(round(args.height * float(args.side_logo_height_factor))))
        logo_img = load_logo(args.side_logo, max_logo_h)
        lw, _ = logo_img.size
        for idx, (offset_x, _) in enumerate(item_offsets):
            section_start = idx * 2 * section_width
            item_section_start = section_start + section_width
            item_actual_x = offset_x
            group_left = max(0, (args.width - group_width) // 2)
            item_left = group_left + item_actual_x
            item_right = item_left + iw
            left_zone_start = section_start
            left_zone_end = min(item_left, item_section_start)
            right_zone_start = item_right
            right_zone_end = min(right_zone_start + section_width, args.width)
            logo_positions_left.append(
                fill_logo_positions(left_zone_start, left_zone_end, lw, args.side_logo_padding,
                                     args.logo_fill_edges, args.width)
            )
            logo_positions_right.append(
                fill_logo_positions(right_zone_start, right_zone_end, lw, args.side_logo_padding,
                                     args.logo_fill_edges, args.width)
            )

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
        item_img=item_img,
        item_offsets=item_offsets,
        group_width=group_width,
        group_height=group_height,
        slide_from=args.slide_from,
        exit_to=exit_to,
        in_duration=max(0.0, args.in_duration),
        hold_duration=max(0.0, args.hold_duration),
        out_duration=max(0.0, args.out_duration),
        total_duration=total_duration,
        freeze_item=bool(args.freeze_item),
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
        freeze_logos=bool(args.freeze_logos),
        item_fade_in=max(0.0, args.item_fade_in),
        item_fade_out=max(0.0, args.item_fade_out),
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
