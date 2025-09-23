# Animator Scripts

## Overview
This repository contains three CLI utilities for generating brand-driven motion assets sized for ultra-wide LED banners (default 7080×108 px). Each script renders text or logos with crisp typography by autosizing the artwork and provides slide or ticker style animation presets. All tools can pull brand colors, fonts, and logos from Brandfetch or operate fully offline with local assets.

- `brand_domain.py` – animates a domain name that slides in, holds, then slides out.
- `brand_slide.py` – animates arbitrary text with the same slide/hold/slide timing.
- `brand_item.py` – slides in a foreground PNG/JPG image with optional 3D-style tilt and the same logo tooling.
- `brand_snail.py` – creates a side-scrolling “snail”/ticker of a logo, optionally chaining copies to fill the track.

## Installation
Ensure Python 3.9+ is available, then install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pillow moviepy requests cairosvg
```

`cairosvg` is only required when Brandfetch delivers SVG logos, but installing it up front avoids surprises. The scripts rely on FFmpeg for efficient MP4/GIF encoding; install it via `brew install ffmpeg`, `choco install ffmpeg`, or the package manager of your platform.

## Brandfetch Access
For remote brand lookups, request an API key at [brandfetch.com](https://brandfetch.com). Export it once per shell session:

```bash
export BRANDFETCH_API_KEY="sk_live_..."
```

Pass the key with `--api-key $BRANDFETCH_API_KEY` (or inline). If you prefer to stay offline, select `--identifier local` and provide your color, font, and artwork inputs manually.

## Shared Behavior
- Canvas defaults to 7080×108 px with 60 FPS output; adjust via `--width`, `--height`, and `--fps`.
- Autosizing keeps typography within a configurable fraction of the canvas height using `--height-factor` and `--height-padding` (where applicable).
- Encoding defaults to H.264 MP4. Supplying an `--output` ending in `.gif` switches to GIF encoding automatically.
- Use `--pix-fmt` and `--crf` (where available) to fine-tune FFmpeg quality settings.

## Scripts
### brand_domain.py
Produces a slide-in/hold/slide-out animation for a domain name. The script can fetch colors and fonts from Brandfetch, falling back to local overrides when desired.

Key options:
- `--identifier` – `local` for manual input or a Brandfetch identifier/domain.
- `--domain` – explicit domain string; defaults to the Brandfetch record or identifier.
- `--font-path` – path to a local `.ttf/.otf/.ttc`; useful in local mode or when you prefer a custom font.
- `--font-color` / `--bg-override` – override auto-selected colors with hex values.
- `--slide-from`, `--in-duration`, `--hold-duration`, `--out-duration` – control entry timing/direction; pair with `--exit-to` when you need the domain to leave via a different edge.
- `--side-logo` mirrors the slide script: scale via `--side-logo-height-factor` (0.98× canvas height by default) or `--side-logo-max-height`, adjust spacing with `--side-logo-padding`, and tile extra copies with `--logo-fill-edges` when space allows.
- Separate logo timing (`--logo-in-duration` etc.), per-side directions, and the `--freeze-text` / `--freeze-logos` switches let you lock elements in place or run staggered animations.
- Provide `--config config.json` to ingest every setting (domain, fonts, colors, logo layout) from JSON; CLI flags still override last-minute tweaks.

Examples:

```bash
# Remote brand
python brand_domain.py --identifier pluto.tv --api-key $BRANDFETCH_API_KEY --output pluto_domain.mp4

# Local-only
python brand_domain.py --identifier local --domain "www.example.com" \
  --bg-override "#101820" --font-color "#FEE715" --font-path ./fonts/Example.ttf
```

Example `domain_config.json`:

```json
{
  "identifier": "local",
  "domain": "www.vodafone.de",
  "output": "vodafone_domain.mp4",
  "bg_override": "#E00201",
  "font_color": "#FFFFFF",
  "side_logo": "vodafone.png",
  "side_logo_height_factor": 0.98,
  "logo_fill_edges": true
}
```

Render with:

```bash
python brand_domain.py --config domain_config.json
```

### brand_slide.py
Same animation engine as `brand_domain.py`, but you supply arbitrary text with `--text`. Ideal for slogans or CTAs that accompany the brand domain animation.

Notable options:
- `--text` (required) – string to render.
- Shares the color, font, slide direction, autosize, and encoding flags with `brand_domain.py`.
- Optional `--side-logo` adds flanking marks; scale them via `--side-logo-height-factor` (default 0.98× canvas height) or an explicit `--side-logo-max-height`, and tweak spacing with `--side-logo-padding`.
- Provide `--config config.json` to load every parameter (text, colors, timings, logos) from JSON; CLI flags still override for ad hoc tweaks.
- `--exit-to` mirrors the domain script and lets you send the slide off via another edge (default `auto` = same as `--slide-from`).
- Logos can slide independently of the text via `--logo-in-duration`, `--logo-hold-duration`, `--logo-out-duration`, plus per-side `--logo-slide-from-left/right` directions.
- Use `--freeze-text` and/or `--freeze-logos` when you want static placement instead of animated slides.
- `--logo-fill-edges` duplicates the side logos outward (respecting padding) to blanket the remaining space when there’s room.

Example:

```bash
python brand_slide.py --identifier pluto.tv --api-key $BRANDFETCH_API_KEY \
  --text "Stream something new" --slide-from top --output pluto_slide.mp4
```

Example `slide_config.json` for a Vodafone bumper:

```json
{
  "identifier": "local",
  "text": "Vodafone GigaNetz",
  "output": "vodafone_slide.mp4",
  "bg_override": "#E00201",
  "font_color": "#FFFFFF",
  "side_logo": "vodafone.png",
  "side_logo_height_factor": 0.98,
  "logo_in_duration": 0.5,
  "logo_hold_duration": 2.8,
  "logo_out_duration": 0.6,
  "logo_slide_from_left": "left",
  "logo_slide_from_right": "right",
  "logo_fill_edges": true
}
```

Run with:

```bash
python brand_slide.py --config slide_config.json
```

### brand_item.py
Slides a PNG/JPG (with transparency if available) instead of text. JSON/CLI options mirror `brand_slide.py` but replace text settings with image sizing and rotation controls.

Highlights:
- Provide `--image` (or `"image"` in JSON) to point at the foreground asset; scale it via `--image-height-factor`, `--image-max-height`, or `--image-max-width`.
- `--rotate-axis {none,x,y}` + `--rotate-angle`/`--rotate-speed` apply a simple tilt while the item floats in.
- Shares side-logo, freeze, fill, slide/exit (`--exit-to`) and other timing options with the text/domain scripts; `--config` works the same way for end-to-end presets.
- `--item-count` (default 1) renders the asset multiple times across the canvas; control spacing with `--item-spacing` (pixels between items).
- Logos are only placed in free intervals (left/right and between items) so they never overlap the repeated imagery; enable `--logo-fill-edges` to tile swooshes throughout the available gaps.

Example `item_config.json`:

```json
{
  "identifier": "local",
  "image": "vodafone.png",
  "output": "vodafone_item.mp4",
  "bg_override": "#000000",
  "slide_from": "left",
  "image_height_factor": 0.6,
  "rotate_axis": "y",
  "rotate_angle": 35,
  "rotate_speed": 120
}
```

Generate with:

```bash
python brand_item.py --config item_config.json
```

### brand_snail.py
Builds a continuous ticker where the logo marches from right to left. Works with Brandfetch logos or local artwork.

Highlights:
- `--identifier` / `--api-key` mirror the other scripts; use `--identifier local --graphic path.png` for offline runs.
- `--speed`, `--gap`, `--copies`, and `--fill` adjust motion spacing (use `--fill` for an uninterrupted belt of logos).
- `--logo-max-height` scales the logo to fit within the band while preserving aspect ratio.
- `--show-domain` and `--claim` overlay additional text blocks in the upper portion of the frame.
- Supply `--config config.json` to inject every option via JSON; flags still override values if you need last-second tweaks.

Examples:

```bash
# Brandfetch-powered ticker
python brand_snail.py --identifier pluto.tv --api-key $BRANDFETCH_API_KEY \
  --output pluto_snail.mp4 --speed 240 --duration 12 --show-domain

# Local logo, solid background, custom slogan
python brand_snail.py --identifier local --graphic ./assets/logo.png \
  --bg-override "#000000" --claim "Watch anywhere" --fill --duration 12
```

Example `snail_config.json` for Vodafone:

```json
{
  "identifier": "local",
  "graphic": "vodafone.png",
  "output": "vodafone_snail.mp4",
  "duration": 12,
  "speed": 240,
  "fill": true,
  "gap": 60,
  "copies": 3,
  "bg_override": "#000000"
}
```

Generate the animation with:

```bash
python brand_snail.py --config snail_config.json
```

## Tips
- Delete any temporary `_brandfont.*` artifacts if you re-run scripts with different fonts; the tools overwrite them automatically, but manual cleanup keeps the workspace tidy.
- When experimenting with FFmpeg settings, append `--pix-fmt yuv444p --crf 14` for higher chroma fidelity on LED walls.
- MoviePy writes progress logs to stderr; if you need quiet output, add `--logger none` when you invoke the scripts programmatically.

Happy animating!
