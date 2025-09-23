#!/usr/bin/env python3
"""
brand_item.py

Animates a foreground image (e.g. product shot) sliding across an LED banner canvas.
Supports JSON-driven configuration (like brand_slide), optional side logos,
and simple x/y-axis tilt during the float-in.

Dependencies:
    pip install pillow moviepy requests
"""
from __future__ import annotations
import argparse, json, math, os, sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import requests
from moviepy.video.VideoClip import VideoClip

BRANDFETCH_URL = "https://api.brandfetch.io/v2/brands/{identifier}"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit("Konfigurationsdatei muss ein JSON-Objekt enthalten.")
    return data


@dataclass
class BrandColor:
    hex: str
    type: Optional[str]
    brightness: Optional[float]


@dataclass
class BrandData:
    name: Optional[str]
    domain: Optional[str]
    colors: List[BrandColor]


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(ch * 2 for ch in h)
    if len(h) != 6:
        return (0, 0, 0)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def f(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def pick_background_color(colors: List[BrandColor]) -> Tuple[int, int, int]:
    if not colors:
        return (0, 0, 0)
    rgb = [(c, hex_to_rgb(c.hex)) for c in colors]
    return min(rgb, key=lambda x: relative_luminance(x[1]))[1]


def ease_out_cubic(x: float) -> float:
    return 1 - (1 - x) ** 3


def ease_in_cubic(x: float) -> float:
    return x ** 3


def fetch_brand(identifier: str, api_key: str) -> BrandData:
    url = BRANDFETCH_URL.format(identifier=identifier)
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    colors = [BrandColor(hex=c.get("hex", "#000000"), type=c.get("type"), brightness=c.get("brightness"))
              for c in (data.get("colors") or [])]
    return BrandData(name=data.get("name"), domain=data.get("domain"), colors=colors)


def load_image(path: str, max_height: Optional[int], max_width: Optional[int]) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    scale = 1.0
    if max_height and h > max_height:
        scale = min(scale, max_height / float(h))
    if max_width and w > max_width:
        scale = min(scale, max_width / float(w))
    if scale < 1.0:
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
    return img


def apply_axis_rotation(img: Image.Image, axis: str, angle_deg: float) -> Image.Image:
    if axis not in ("x", "y") or angle_deg <= 1e-6:
        return img
    w, h = img.size
    rad = math.radians(min(89.0, max(0.0, angle_deg)))
    cos_v = abs(math.cos(rad))
    if axis == "x":
        new_h = max(1, int(round(h * cos_v)))
        scaled = img.resize((w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        y_off = (h - new_h) // 2
        canvas.paste(scaled, (0, y_off), scaled)
        return canvas
    new_w = max(1, int(round(w * cos_v)))
    scaled = img.resize((new_w, h), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x_off = (w - new_w) // 2
    canvas.paste(scaled, (x_off, 0), scaled)
    return canvas


@dataclass
class RenderConfig:
    width: int; height: int; fps: int
    bg_rgb: Tuple[int, int, int]
    item_img: Image.Image
    slide_from: str
    exit_to: Optional[str]
    in_duration: float; hold_duration: float; out_duration: float
    rotate_axis: str; rotate_angle: float; rotate_speed: float
    logo_img: Optional[Image.Image] = None
    logo_padding: int = 120
    logo_positions_left: Optional[List[int]] = None
    logo_positions_right: Optional[List[int]] = None
    logo_static_positions: Optional[List[int]] = None
    logo_in_duration: float = 0.8
    logo_hold_duration: float = 2.5
    logo_out_duration: float = 0.8
    logo_slide_from_left: str = "left"
    logo_slide_from_right: str = "right"
    freeze_item: bool = False
    freeze_logos: bool = False
    logo_fill_edges: bool = False
    item_start_left: Optional[int] = None
    item_start_right: Optional[int] = None
    item_exit_left: Optional[int] = None
    item_exit_right: Optional[int] = None
    item_count: int = 1
    item_spacing: int = 0
    item_offsets: List[Tuple[int, int]] = field(default_factory=list)
    group_width: int = 0
    group_height: int = 0


def position_for_progress(progress: float, cfg: RenderConfig, item_size: Tuple[int, int]) -> Tuple[int, int]:
    W, H = cfg.width, cfg.height
    iw, ih = item_size
    total = cfg.in_duration + cfg.hold_duration + cfg.out_duration
    t = max(0.0, min(progress, total))
    cx = (W - iw) // 2; cy = (H - ih) // 2
    if cfg.freeze_item:
        return (cx, cy)
    if t <= cfg.in_duration:
        p = 0.0 if cfg.in_duration == 0 else t / cfg.in_duration; e = ease_out_cubic(p)
        if cfg.slide_from == "left":
            start = cfg.item_start_left if cfg.item_start_left is not None else -iw
            return (int(round(start + e * (cx - start))), cy)
        if cfg.slide_from == "right":
            start = cfg.item_start_right if cfg.item_start_right is not None else W
            return (int(round(start - e * (start - cx))), cy)
        if cfg.slide_from == "top":
            return (cx, int(round(-ih + e * (cy + ih))))
        return (cx, int(round(H - e * (H - cy))))
    if t <= cfg.in_duration + cfg.hold_duration:
        return (cx, cy)
    t_out = t - (cfg.in_duration + cfg.hold_duration)
    p = 0.0 if cfg.out_duration == 0 else t_out / cfg.out_duration; e = ease_in_cubic(p)
    exit_dir = cfg.exit_to or cfg.slide_from
    target_x, target_y = cx, cy
    if exit_dir == "left":
        target_x = cfg.item_exit_left if cfg.item_exit_left is not None else -iw
    elif exit_dir == "right":
        target_x = cfg.item_exit_right if cfg.item_exit_right is not None else W
    elif exit_dir == "top":
        target_y = -ih
    elif exit_dir == "bottom":
        target_y = H

    new_x = int(round(cx + e * (target_x - cx)))
    new_y = int(round(cy + e * (target_y - cy)))
    return (new_x, new_y)


def render_frame(t: float, cfg: RenderConfig) -> Image.Image:
    base_item = cfg.item_img
    angle = 0.0
    if cfg.rotate_axis in ("x", "y") and cfg.rotate_angle > 0 and cfg.rotate_speed > 0:
        angle = min(cfg.rotate_angle, cfg.rotate_speed * t)
    item = apply_axis_rotation(base_item, cfg.rotate_axis, angle) if angle else base_item
    frame = Image.new("RGB", (cfg.width, cfg.height), cfg.bg_rgb)

    group_size = (cfg.group_width, cfg.group_height)
    anchor_x, anchor_y = position_for_progress(t, cfg, group_size)
    for dx, dy in cfg.item_offsets:
        frame.paste(item, (anchor_x + dx, anchor_y + dy), item)

    if cfg.logo_img is not None and (cfg.logo_positions_left or cfg.logo_positions_right or cfg.logo_static_positions):
        lw, lh = cfg.logo_img.size
        y_logo = (cfg.height - lh) // 2

        def anim_pos(final_x: int, side: str) -> int:
            if cfg.freeze_logos:
                return final_x
            total = cfg.logo_in_duration + cfg.logo_hold_duration + cfg.logo_out_duration
            tt = max(0.0, min(t, total))
            in_d = cfg.logo_in_duration
            hold_d = cfg.logo_hold_duration
            out_d = cfg.logo_out_duration
            slide_from = cfg.logo_slide_from_left if side == "left" else cfg.logo_slide_from_right
            start_left = -lw - cfg.logo_padding
            start_right = cfg.width + cfg.logo_padding
            start = start_left if slide_from == "left" else start_right
            exit_pos = start
            if tt <= in_d:
                p = 0.0 if in_d == 0 else tt / in_d
                e = ease_out_cubic(p)
                return int(round(start + e * (final_x - start)))
            if tt <= in_d + hold_d:
                return final_x
            if out_d == 0:
                return exit_pos
            t_out = tt - (in_d + hold_d)
            p = 0.0 if out_d == 0 else min(1.0, t_out / out_d)
            e = ease_in_cubic(p)
            return int(round(final_x + e * (exit_pos - final_x)))

        for lp in (cfg.logo_positions_left or []):
            lx = anim_pos(lp, "left")
            if lx + lw > 0 and lx < cfg.width:
                frame.paste(cfg.logo_img, (lx, y_logo), cfg.logo_img)
        for rp in (cfg.logo_positions_right or []):
            rx = anim_pos(rp, "right")
            if rx < cfg.width and rx + lw > 0:
                frame.paste(cfg.logo_img, (rx, y_logo), cfg.logo_img)
        for sx in (cfg.logo_static_positions or []):
            if sx + lw > 0 and sx < cfg.width:
                frame.paste(cfg.logo_img, (sx, y_logo), cfg.logo_img)
    return frame


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, help="Pfad zu einer JSON-Konfiguration")

    config_args, remaining = config_parser.parse_known_args()
    config_data: Dict[str, Any] = {}
    config_base_dir = os.getcwd()
    if getattr(config_args, "config", None):
        config_path = os.path.abspath(config_args.config)
        if not os.path.exists(config_path):
            raise SystemExit(f"Konfigurationsdatei nicht gefunden: {config_path}")
        config_data = load_config(config_path)
        config_base_dir = os.path.dirname(config_path)

    defaults: Dict[str, Any] = {
        "identifier": None,
        "api_key": None,
        "image": None,
        "width": 7080,
        "height": 108,
        "fps": 60,
        "output": "output_item.mp4",
        "bg_override": None,
        "slide_from": "left",
        "exit_to": "auto",
        "in_duration": 0.8,
        "hold_duration": 2.5,
        "out_duration": 0.8,
        "pix_fmt": "yuv420p",
        "crf": 18,
        "image_height_factor": 0.8,
        "image_max_height": None,
        "image_max_width": None,
        "rotate_axis": "none",
        "rotate_angle": 45.0,
        "rotate_speed": 90.0,
        "side_logo": None,
        "side_logo_max_height": None,
        "side_logo_height_factor": 0.98,
        "side_logo_padding": 120,
        "logo_in_duration": 0.8,
        "logo_hold_duration": 2.5,
        "logo_out_duration": 0.8,
        "logo_slide_from_left": "left",
        "logo_slide_from_right": "right",
        "freeze_item": False,
        "freeze_logos": False,
        "logo_fill_edges": False,
        "item_count": 1,
        "item_spacing": 120
    }
    defaults.update({k: v for k, v in config_data.items() if k in defaults})

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--identifier", default=defaults.get("identifier"))
    parser.add_argument("--api-key", default=defaults.get("api_key"))
    parser.add_argument("--image", default=defaults.get("image"), help="Pfad zu einem PNG/JPG mit Transparenz")
    parser.add_argument("--width", type=int, default=int(defaults.get("width", 7080)))
    parser.add_argument("--height", type=int, default=int(defaults.get("height", 108)))
    parser.add_argument("--fps", type=int, default=int(defaults.get("fps", 60)))
    parser.add_argument("--output", default=defaults.get("output", "output_item.mp4"))

    parser.add_argument("--bg-override", default=defaults.get("bg_override"))

    parser.add_argument("--slide-from", choices=["left", "right", "top", "bottom"],
                        default=defaults.get("slide_from", "left"))
    parser.add_argument("--exit-to", choices=["auto", "left", "right", "top", "bottom"],
                        default=defaults.get("exit_to", "auto"))
    parser.add_argument("--in-duration", type=float, default=float(defaults.get("in_duration", 0.8)))
    parser.add_argument("--hold-duration", type=float, default=float(defaults.get("hold_duration", 2.5)))
    parser.add_argument("--out-duration", type=float, default=float(defaults.get("out_duration", 0.8)))

    parser.add_argument("--image-height-factor", type=float,
                        default=float(defaults.get("image_height_factor", 0.8)))
    parser.add_argument("--image-max-height", type=int, default=defaults.get("image_max_height"))
    parser.add_argument("--image-max-width", type=int, default=defaults.get("image_max_width"))

    parser.add_argument("--rotate-axis", choices=["none", "x", "y"],
                        default=defaults.get("rotate_axis", "none"))
    parser.add_argument("--rotate-angle", type=float, default=float(defaults.get("rotate_angle", 45.0)))
    parser.add_argument("--rotate-speed", type=float, default=float(defaults.get("rotate_speed", 90.0)))

    parser.add_argument("--pix-fmt", default=defaults.get("pix_fmt", "yuv420p"))
    parser.add_argument("--crf", type=int, default=int(defaults.get("crf", 18)))

    parser.add_argument("--side-logo", default=defaults.get("side_logo"))
    parser.add_argument("--side-logo-max-height", type=int, default=defaults.get("side_logo_max_height"))
    parser.add_argument("--side-logo-height-factor", type=float,
                        default=float(defaults.get("side_logo_height_factor", 0.98)))
    parser.add_argument("--side-logo-padding", type=int, default=int(defaults.get("side_logo_padding", 120)))
    parser.add_argument("--logo-in-duration", type=float, default=float(defaults.get("logo_in_duration", 0.8)))
    parser.add_argument("--logo-hold-duration", type=float, default=float(defaults.get("logo_hold_duration", 2.5)))
    parser.add_argument("--logo-out-duration", type=float, default=float(defaults.get("logo_out_duration", 0.8)))
    parser.add_argument("--logo-slide-from-left", choices=["left", "right"],
                        default=defaults.get("logo_slide_from_left", "left"))
    parser.add_argument("--logo-slide-from-right", choices=["left", "right"],
                        default=defaults.get("logo_slide_from_right", "right"))
    parser.add_argument("--freeze-item", action="store_true", dest="freeze_item",
                        default=bool(defaults.get("freeze_item", False)))
    parser.add_argument("--no-freeze-item", action="store_false", dest="freeze_item", help=argparse.SUPPRESS)
    parser.add_argument("--freeze-logos", action="store_true", dest="freeze_logos",
                        default=bool(defaults.get("freeze_logos", False)))
    parser.add_argument("--no-freeze-logos", action="store_false", dest="freeze_logos", help=argparse.SUPPRESS)
    parser.add_argument("--logo-fill-edges", action="store_true", dest="logo_fill_edges",
                        default=bool(defaults.get("logo_fill_edges", False)))
    parser.add_argument("--no-logo-fill-edges", action="store_false", dest="logo_fill_edges", help=argparse.SUPPRESS)
    parser.add_argument("--item-count", type=int, default=int(defaults.get("item_count", 1)))
    parser.add_argument("--item-spacing", type=int, default=int(defaults.get("item_spacing", 120)))

    args = parser.parse_args(remaining)

    if not args.image:
        parser.error("--image muss gesetzt werden (oder in der JSON-Konfiguration).")

    if not args.identifier and (args.api_key or args.side_logo):
        # allow local mode without identifier
        args.identifier = "local"

    if args.identifier and args.identifier.lower() != "local" and not args.api_key:
        parser.error("Brandfetch-Modus benötigt --api-key.")

    if args.identifier and args.identifier.lower() != "local":
        brand = fetch_brand(args.identifier, args.api_key)
        bg_rgb = hex_to_rgb(args.bg_override) if args.bg_override else pick_background_color(brand.colors)
    else:
        bg_rgb = hex_to_rgb(args.bg_override) if args.bg_override else (0, 0, 0)

    img_path = args.image
    candidate_paths = [img_path]
    if not os.path.isabs(img_path):
        candidate_paths.insert(0, os.path.join(config_base_dir, img_path))
        candidate_paths.append(os.path.abspath(img_path))
    resolved_image = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not resolved_image:
        raise SystemExit(f"Bild nicht gefunden: {args.image}")

    target_max_height = args.image_max_height
    if not target_max_height and args.image_height_factor:
        target_max_height = max(1, int(round(args.height * float(args.image_height_factor))))
    item_count = max(1, int(args.item_count))
    spacing = max(0, int(round(args.item_spacing)))

    item_img = load_image(resolved_image, target_max_height, args.image_max_width)
    iw, ih = item_img.size

    total_width = item_count * iw + spacing * (item_count - 1)
    if total_width > args.width and total_width > 0:
        scale = args.width / float(total_width)
        iw = max(1, int(round(iw * scale)))
        ih = max(1, int(round(ih * scale)))
        item_img = item_img.resize((iw, ih), Image.LANCZOS)
        total_width = item_count * iw + spacing * (item_count - 1)

    item_offsets: List[Tuple[int, int]] = [(idx * (iw + spacing), 0) for idx in range(item_count)]
    group_width = total_width
    group_height = ih

    logo_img: Optional[Image.Image] = None
    logo_positions_left: List[int] = []
    logo_positions_right: List[int] = []
    logo_static_positions: List[int] = []

    logo_padding = max(0, int(args.side_logo_padding))
    if args.side_logo:
        candidate = args.side_logo
        candidate_paths = [candidate]
        if not os.path.isabs(candidate):
            candidate_paths.insert(0, os.path.join(config_base_dir, candidate))
            candidate_paths.append(os.path.abspath(candidate))
        logo_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not logo_path:
            raise SystemExit(f"Seitenlogo nicht gefunden: {args.side_logo}")
        if args.side_logo_max_height:
            max_logo_h = args.side_logo_max_height
        else:
            factor = args.side_logo_height_factor if args.side_logo_height_factor else 0.98
            max_logo_h = max(1, int(round(args.height * factor)))
        logo_img_candidate = load_image(logo_path, max_logo_h, None)
        lw, _ = logo_img_candidate.size

        group_x = max(0, (args.width - group_width) // 2)
        item_intervals = [(group_x + offset_x, group_x + offset_x + iw) for offset_x, _ in item_offsets]

        pos = 0
        while pos <= args.width - lw:
            overlaps = False
            for start, end in item_intervals:
                if pos < end and (pos + lw) > start:
                    pos = end + logo_padding
                    overlaps = True
                    break
            if overlaps:
                continue
            logo_static_positions.append(pos)
            pos += lw + logo_padding

        logo_img = logo_img_candidate

    item_start_left = -group_width
    item_start_right = args.width
    item_exit_left = -group_width
    item_exit_right = args.width

    text_total = max(0.0, args.in_duration) + max(0.0, args.hold_duration) + max(0.0, args.out_duration)
    logo_total = 0.0
    if logo_img is not None:
        logo_total = max(0.0, args.logo_in_duration) + max(0.0, args.logo_hold_duration) + max(0.0, args.logo_out_duration)
    total_duration = max(0.01, max(text_total, logo_total))

    exit_to = None if args.exit_to == "auto" else args.exit_to

    cfg = RenderConfig(width=args.width, height=args.height, fps=args.fps,
                       bg_rgb=bg_rgb, item_img=item_img,
                       slide_from=args.slide_from,
                       exit_to=exit_to,
                       in_duration=max(0.0, args.in_duration),
                       hold_duration=max(0.0, args.hold_duration),
                       out_duration=max(0.0, args.out_duration),
                       rotate_axis=args.rotate_axis,
                       rotate_angle=max(0.0, args.rotate_angle),
                       rotate_speed=max(0.0, args.rotate_speed),
                       logo_img=logo_img,
                       logo_padding=logo_padding,
                       logo_positions_left=logo_positions_left or None,
                       logo_positions_right=logo_positions_right or None,
                       logo_static_positions=logo_static_positions or None,
                       logo_in_duration=max(0.0, args.logo_in_duration),
                       logo_hold_duration=max(0.0, args.logo_hold_duration),
                       logo_out_duration=max(0.0, args.logo_out_duration),
                       logo_slide_from_left=args.logo_slide_from_left,
                       logo_slide_from_right=args.logo_slide_from_right,
                       freeze_item=bool(args.freeze_item),
                       freeze_logos=bool(args.freeze_logos),
                       logo_fill_edges=bool(args.logo_fill_edges),
                       item_start_left=item_start_left,
                       item_start_right=item_start_right,
                       item_exit_left=item_exit_left,
                       item_exit_right=item_exit_right,
                       item_count=item_count,
                       item_spacing=spacing,
                       item_offsets=item_offsets,
                       group_width=group_width,
                       group_height=group_height)

    def make_frame(tt: float) -> np.ndarray:
        return np.array(render_frame(tt, cfg))

    clip = VideoClip(lambda tt: make_frame(tt), duration=total_duration).with_fps(cfg.fps)
    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".gif":
        clip.write_gif(args.output, fps=cfg.fps, program="ffmpeg")
    else:
        ffmpeg_params = ["-pix_fmt", args.pix_fmt, "-crf", str(args.crf), "-tune", "animation"]
        clip.write_videofile(args.output, fps=cfg.fps, codec="libx264", audio=False,
                             preset="medium", ffmpeg_params=ffmpeg_params)
    print(f"✅ Fertig: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
