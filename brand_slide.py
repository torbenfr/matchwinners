#!/usr/bin/env python3
"""
brand_slide.py (autosize to height with correct baseline; crisp draw; optional pix_fmt)

- Schriftgröße wird per binärer Suche so bestimmt, dass die tatsächliche Text-Bounding-Box
  <= (height * height_factor - height_padding) ist.
- Zeichnet mit korrektem Baseline-Offset (-bbox.left, -bbox.top), damit nichts abgeschnitten wird.
- Kein nachträgliches Resampling → scharf auf Pixel-Ebene.
- Optional: --pix-fmt (z. B. yuv444p) und --crf zur Kontrolle der H.264-Qualität.

pip install pillow moviepy requests
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from moviepy.video.VideoClip import VideoClip

BRANDFETCH_URL = "https://api.brandfetch.io/v2/brands/{identifier}"

@dataclass
class BrandColor:
    hex: str
    type: Optional[str]
    brightness: Optional[float]

@dataclass
class BrandFontFile:
    url: str
    fmt: str

@dataclass
class BrandFont:
    name: str
    files: List[BrandFontFile]

@dataclass
class BrandData:
    name: Optional[str]
    domain: Optional[str]
    colors: List[BrandColor]
    fonts: List[BrandFont]

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(ch*2 for ch in h)
    if len(h) != 6:
        return (0,0,0)
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def relative_luminance(rgb: Tuple[int,int,int]) -> float:
    def f(c: float) -> float:
        c = c/255.0
        return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055) ** 2.4
    r,g,b = rgb
    return 0.2126*f(r) + 0.7152*f(g) + 0.0722*f(b)

def pick_background_color(colors: List[BrandColor]) -> Tuple[int,int,int]:
    if not colors:
        return (0,0,0)
    rgb = [(c, hex_to_rgb(c.hex)) for c in colors]
    return min(rgb, key=lambda x: relative_luminance(x[1]))[1]

def pick_text_color_for_bg(bg: Tuple[int,int,int]) -> Tuple[int,int,int]:
    return (255,255,255) if relative_luminance(bg) < 0.5 else (0,0,0)

def ease_out_cubic(x: float) -> float: return 1 - (1-x)**3
def ease_in_cubic(x: float) -> float: return x**3


def load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """
    Robustes Laden von TTF/OTF/TTC. Bei .ttc werden mehrere Indizes probiert.
    Fällt auf systemweite TrueType-Fonts zurück. Bricht ab, wenn nichts Skalierbares gefunden wird,
    statt auf ein pixeliges Bitmap-Default zu fallen.
    """
    tried = []

    def try_path(p: str) -> Optional[ImageFont.FreeTypeFont]:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            # TTC-Unterstützung: mehrere Indizes probieren
            if p.lower().endswith(".ttc"):
                for idx in range(0, 8):
                    try:
                        return ImageFont.truetype(p, size=size, index=idx)
                    except Exception:
                        continue
        except Exception:
            pass
        tried.append(p)
        return None

    # 1) expliziter Pfad
    if font_path and os.path.exists(font_path):
        fnt = try_path(font_path)
        if fnt: return fnt

    # 2) übliche System-Schriften
    system_candidates = [
        "/System/Library/Fonts/Supplemental/Inter.ttc",  # macOS Inter (TTC)
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for p in system_candidates:
        if os.path.exists(p):
            fnt = try_path(p)
            if fnt: return fnt

    # 3) letzter Versuch: DejaVuSans via font name lookup (manche Systeme)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        pass

    # 4) Abbruch – kein skalierbarer Font gefunden
    raise SystemExit("Konnte keine skalierbare Schrift laden. "
                     f"Versuche --font-path auf eine .ttf/.otf/.ttc zu setzen. Getestet: {', '.join(tried) or '—'}")

def fetch_brand(identifier: str, api_key: str) -> BrandData:
    url = BRANDFETCH_URL.format(identifier=identifier)
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=20); r.raise_for_status()
    data = r.json()
    colors = [BrandColor(hex=c.get("hex","#000000"), type=c.get("type"), brightness=c.get("brightness"))
              for c in (data.get("colors") or [])]
    fonts: List[BrandFont] = []
    for F in (data.get("fonts") or []):
        name = F.get("name") or F.get("family") or "BrandFont"
        files: List[BrandFontFile] = []
        files_dict = F.get("files") or {}
        if isinstance(files_dict, dict):
            for k,v in files_dict.items():
                if isinstance(v,str): files.append(BrandFontFile(url=v, fmt=k.lower()))
        for src in (F.get("sources") or []):
            url = src.get("url"); fmt = (src.get("format") or "").lower()
            if url: files.append(BrandFontFile(url=url, fmt=fmt or "ttf"))
        fonts.append(BrandFont(name=name, files=files))
    return BrandData(name=data.get("name"), domain=data.get("domain"), colors=colors, fonts=fonts)

def download_brand_font_tmp(fonts: List[BrandFont]) -> Optional[str]:
    if not fonts: return None
    prefer = ["ttf","otf","woff2","woff"]
    for bf in fonts:
        candidates = sorted(bf.files, key=lambda f: prefer.index(f.fmt) if f.fmt in prefer else 99)
        for f in candidates:
            try:
                resp = requests.get(f.url, timeout=20); resp.raise_for_status()
                ext = "." + (f.fmt if f.fmt in ("ttf","otf") else "ttf")
                tmp = os.path.join(os.getcwd(), f"_brandfont{ext}")
                with open(tmp,"wb") as fh: fh.write(resp.content)
                if ext in (".ttf",".otf"): return tmp
                os.remove(tmp)
            except Exception:
                continue
    return None

def text_bbox(text: str, font: ImageFont.ImageFont) -> Tuple[int,int,int,int]:
    dummy = Image.new("RGBA",(10,10),(0,0,0,0))
    d = ImageDraw.Draw(dummy)
    return d.textbbox((0,0), text, font=font)

def render_text_autosize(text: str, font_path: Optional[str], color: Tuple[int,int,int],
                         target_h: int) -> Image.Image:
    """
    Binäre Suche nach größter Font-Size, deren tatsächliche Bounding-Box-Höhe <= target_h.
    Zeichnet mit Baseline-Offset (-bbox.left, -bbox.top), damit nichts abgeschnitten wird.
    Kein Resampling → 1:1 Pixel.
    """
    lo, hi = 1, 2000
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(font_path, size=mid)
        l, t, r, b = text_bbox(text, font)
        h = b - t
        if h <= target_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    font = load_font(font_path, size=best)
    l, t, r, b = text_bbox(text, font)
    w, h = r - l, b - t
    w = max(1, w); h = max(1, h)
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # Wichtig: negativen Top/Left-Offset berücksichtigen
    draw.text((-l, -t), text, font=font, fill=color)
    return img

@dataclass
class RenderConfig:
    width: int; height: int; fps: int
    bg_rgb: Tuple[int,int,int]; text_rgb: Tuple[int,int,int]
    text_img: Image.Image; slide_from: str
    in_duration: float; hold_duration: float; out_duration: float

def position_for_progress(progress: float, cfg: RenderConfig) -> Tuple[int,int]:
    W,H = cfg.width, cfg.height
    tw,th = cfg.text_img.size
    total = cfg.in_duration + cfg.hold_duration + cfg.out_duration
    t = max(0.0, min(progress, total))
    cx = (W - tw)//2; cy = (H - th)//2
    if t <= cfg.in_duration:
        p = 0.0 if cfg.in_duration==0 else t/cfg.in_duration; e = ease_out_cubic(p)
        if cfg.slide_from=="left":  return (int(-tw + e*(cx+tw)), cy)
        if cfg.slide_from=="right": return (int(W - e*(W-cx)), cy)
        if cfg.slide_from=="top":   return (cx, int(-th + e*(cy+th)))
        return (cx, int(H - e*(H-cy)))  # bottom
    if t <= cfg.in_duration + cfg.hold_duration:
        return (cx, cy)
    t_out = t - (cfg.in_duration + cfg.hold_duration)
    p = 0.0 if cfg.out_duration==0 else t_out/cfg.out_duration; e = ease_in_cubic(p)
    if cfg.slide_from=="left":  return (int(cx - e*(cx+tw)), cy)
    if cfg.slide_from=="right": return (int(cx + e*(W-cx)), cy)
    if cfg.slide_from=="top":   return (cx, int(cy - e*(cy+th)))
    return (cx, int(cy + e*(H-cy)))  # bottom

def render_frame(t: float, cfg: RenderConfig) -> Image.Image:
    frame = Image.new("RGB", (cfg.width, cfg.height), cfg.bg_rgb)
    x,y = position_for_progress(t, cfg)
    frame.paste(cfg.text_img, (x,y), cfg.text_img)
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True, help="'local' oder Brand-Identifier")
    parser.add_argument("--api-key", required=False)
    parser.add_argument("--width", type=int, default=3456)
    parser.add_argument("--height", type=int, default=108)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="output_slide.mp4")

    parser.add_argument("--text", required=True)
    parser.add_argument("--font-path", type=str, default=None, help="Pfad zu .ttf/.otf (lokal/Fallback)")
    parser.add_argument("--font-color", type=str, default=None)
    parser.add_argument("--bg-override", type=str, default=None)

    parser.add_argument("--slide-from", choices=["left","right","top","bottom"], default="left")
    parser.add_argument("--in-duration", type=float, default=0.8)
    parser.add_argument("--hold-duration", type=float, default=2.5)
    parser.add_argument("--out-duration", type=float, default=0.8)

    parser.add_argument("--height-factor", type=float, default=0.98,
                        help="Anteil der Canvas-Höhe (Standard 0.98)")
    parser.add_argument("--height-padding", type=int, default=0,
                        help="Zusätzliche Luft in Pixeln, die von der Zielhöhe abgezogen wird")

    # Encoding-Qualität
    parser.add_argument("--pix-fmt", type=str, default="yuv420p",
                        help="FFmpeg pix_fmt (z.B. yuv420p, yuv444p)")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF (niedriger = bessere Qualität)")

    args = parser.parse_args()

    # Farben/Font bestimmen
    if args.identifier.lower()=="local":
        bg_rgb = hex_to_rgb(args.bg_override) if args.bg_override else (0,0,0)
        text_rgb = hex_to_rgb(args.font_color) if args.font_color else pick_text_color_for_bg(bg_rgb)
        font_path = args.font_path
    else:
        if not args.api_key: raise SystemExit("Brandfetch-Modus benötigt --api-key.")
        brand = fetch_brand(args.identifier, args.api_key)
        bg_rgb = hex_to_rgb(args.bg_override) if args.bg_override else pick_background_color(brand.colors)
        text_rgb = hex_to_rgb(args.font_color) if args.font_color else pick_text_color_for_bg(bg_rgb)
        font_path = download_brand_font_tmp(brand.fonts)

    # Zielhöhe berechnen
    target_h = max(1, int(round(args.height * args.height_factor)) - max(0, args.height_padding))
    text_img = render_text_autosize(args.text, font_path, text_rgb, target_h)

    total_duration = max(0.01, args.in_duration + args.hold_duration + args.out_duration)
    cfg = RenderConfig(width=args.width, height=args.height, fps=args.fps,
                       bg_rgb=bg_rgb, text_rgb=text_rgb, text_img=text_img,
                       slide_from=args.slide_from,
                       in_duration=max(0.0,args.in_duration),
                       hold_duration=max(0.0,args.hold_duration),
                       out_duration=max(0.0,args.out_duration))

    def make_frame(tt): return np.array(render_frame(tt, cfg))
    clip = VideoClip(lambda tt: make_frame(tt), duration=total_duration).with_fps(cfg.fps)
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
