#!/usr/bin/env python3
"""
render_brand_sequence.py

Reads a brand_config.json file, renders each segment with the appropriate
script, and concatenates the resulting clips into a single composite video.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


@dataclass
class Segment:
    name: str
    script: str
    output: Path
    config: Optional[Dict]
    reuse: Optional[str]


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_segment(segment: Segment, base_dir: Path, skip_existing: bool) -> None:
    if segment.reuse:
        return
    if skip_existing and segment.output.exists():
        print(f"⏭️  Skipping existing segment {segment.name} ({segment.output})")
        return
    if segment.config is None:
        raise SystemExit(f"Segment '{segment.name}' is missing a config block.")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        json.dump(segment.config, tmp, indent=2)
        tmp_path = Path(tmp.name)
    cmd = [sys.executable, str(base_dir / segment.script), "--config", str(tmp_path)]
    print(f"▶️  Rendering {segment.name} via {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def concatenate_clips(order: List[Path], output: Path, codec: str, preset: str,
                      pixel_format: str, ffmpeg_params: List[str]) -> None:
    print(f"🎞️  Concatenating {len(order)} clips into {output}")
    clips = [VideoFileClip(str(p)) for p in order]
    try:
        size = clips[0].size
        fps = clips[0].fps
        writer = FFMPEG_VideoWriter(
            str(output),
            size,
            fps,
            codec=codec,
            preset=preset,
            ffmpeg_params=ffmpeg_params,
            pixel_format=pixel_format,
        )
        try:
            for clip in clips:
                if clip.size != size:
                    raise SystemExit("All clips must share the same resolution for concat.")
                for frame in clip.iter_frames(fps=fps, dtype="uint8"):
                    writer.write_frame(frame)
        finally:
            writer.close()
    finally:
        for clip in clips:
            clip.close()
    print(f"✅  Wrote {output}")


def build_segment_map(raw_segments: List[Dict], base_dir: Path) -> Dict[str, Segment]:
    mapping: Dict[str, Segment] = {}
    for raw in raw_segments:
        name = raw["name"]
        if name in mapping:
            raise SystemExit(f"Duplicate segment name '{name}' in config.")
        script = raw["script"]
        output = base_dir / raw.get("output", f"{name}.mp4")
        config = raw.get("config")
        reuse = raw.get("reuse")
        mapping[name] = Segment(name=name, script=script, output=output, config=config, reuse=reuse)
    return mapping


def resolve_order(order_names: List[str], mapping: Dict[str, Segment]) -> List[Path]:
    paths: List[Path] = []
    for name in order_names:
        if name in mapping:
            paths.append(mapping[name].output)
        else:
            paths.append(Path(name))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="brand_config.json", help="Path to brand_config JSON")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip rendering segments whose output file already exists")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent

    segments_cfg = cfg.get("segments") or []
    mapping = build_segment_map(segments_cfg, base_dir)

    for segment in mapping.values():
        if segment.reuse:
            if segment.reuse not in mapping:
                raise SystemExit(f"Segment '{segment.name}' reuses unknown segment '{segment.reuse}'.")

    for segment in mapping.values():
        run_segment(segment, base_dir, skip_existing=args.skip_existing)

    concat_cfg = cfg.get("concat") or {}
    order_names = concat_cfg.get("order")
    if not order_names:
        raise SystemExit("Config is missing concat.order list.")

    order_paths = resolve_order(order_names, mapping)
    for p in order_paths:
        if not p.exists():
            raise SystemExit(f"Cannot concatenate; clip missing: {p}")

    codec = concat_cfg.get("codec", "libx264")
    preset = concat_cfg.get("preset", "medium")
    pixel_format = concat_cfg.get("pixel_format", "yuv420p")
    ffmpeg_params = concat_cfg.get("ffmpeg_params", [])
    final_path = Path(cfg.get("final_output", "brand_sequence.mp4"))

    concatenate_clips(order_paths, final_path, codec, preset, pixel_format, ffmpeg_params)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
