# Brand Config Pipeline

`brand_config.json` captures the full Nike example sequence so you can re-render
all component clips (snail → headline → domain → item → snail) and combine them
into a single timeline.

## Prerequisites
- Python 3.9+
- Dependencies already required by the individual scripts (`pillow`, `moviepy`,
  `requests`, `cairosvg` when using SVG logos)
- Local assets referenced in the config (`nike_logo.png`, `nike_shoe.png`,
  `Futura Std Extra Bold Condensed Oblique.ttf`)

Ensure the scripts (`brand_snail.py`, `brand_slide.py`, `brand_domain.py`,
`brand_item.py`) sit in the same directory as the config.

## Usage
Render and assemble the full sequence with:

```bash
python render_brand_sequence.py --config brand_config.json
```

Flags:
- `--skip-existing` skips segments whose output file already exists.
- `--config` lets you point at alternate brand configs.

The helper script renders each segment with the inline JSON config, then
concatenates them into `nike_full2.mp4` using the canvas defaults (7080×108, 60
fps).

## Config Anatomy
- `canvas`: shared frame size and fps hint; kept for documentation.
- `segments`: ordered definitions of each module run.
  - `script`: which CLI to execute.
  - `output`: per-segment filename.
  - `config`: inline JSON passed to that script (any value acceptable on the
    corresponding CLI).
  - `reuse`: reuse the output from another segment (avoids rerendering).
- `concat`: final ordering and render parameters for the stitched output.

## Customising
Duplicate `brand_config.json`, swap out assets/text, or tweak durations and
colors. Keep widths/heights/fps consistent across segments; the concat step
expects identical dimensions.

If you need additional segments, add them to `segments` (giving each a unique
`name`) and extend `concat.order` accordingly.
