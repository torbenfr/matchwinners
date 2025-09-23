#!/usr/bin/env python3
"""Flask backend for the Matchwinners Animator wizard."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from segment_runner import run_segment, SEGMENT_ALIAS

BASE_DIR = Path(__file__).resolve().parent
FONTS_DIR = BASE_DIR / "fonts"
PREVIEW_DIR = BASE_DIR / "previews"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_FILES = {
    "logo_snail": BASE_DIR / "snail_config.json",
    "text": BASE_DIR / "slide_config.json",
    "domain": BASE_DIR / "domain_config.json",
    "item": BASE_DIR / "item_config.json",
}

for folder in (FONTS_DIR, PREVIEW_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------
@app.route("/")
def index() -> Response:
    return send_from_directory(BASE_DIR, "matchwinners_animator.html")


@app.route("/matchwinners_animator.js")
def serve_js() -> Response:
    return send_from_directory(BASE_DIR, "matchwinners_animator.js")


@app.route("/matchwinners_animator.css")
def serve_css() -> Response:
    return send_from_directory(BASE_DIR, "matchwinners_animator.css")


@app.route("/mw-emblem_g.png")
def serve_logo() -> Response:
    return send_from_directory(BASE_DIR, "mw-emblem_g.png")


@app.route("/fonts/<path:filename>")
def serve_font(filename: str) -> Response:
    return send_from_directory(FONTS_DIR, filename)


@app.route("/previews/<path:filename>")
def serve_preview(filename: str) -> Response:
    return send_from_directory(PREVIEW_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename: str) -> Response:
    return send_from_directory(OUTPUT_DIR, filename)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _load_template(segment_type: str) -> Dict:
    path = TEMPLATE_FILES.get(segment_type)
    if path and path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    return {}


def _merge_globals(segment_type: str, config: Dict, global_settings: Dict) -> Dict:
    cfg = dict(config)
    if not isinstance(global_settings, dict):
        return cfg

    width = global_settings.get("width")
    height = global_settings.get("height")
    fps = global_settings.get("fps")
    font_path = global_settings.get("font_path")

    if width is not None:
        cfg["width"] = width
    if height is not None:
        cfg["height"] = height
    if fps is not None:
        cfg["fps"] = fps

    if segment_type in {"text", "domain"} and font_path:
        cfg["font_path"] = font_path

    return cfg


def _normalise_output_path(filename: str) -> Path:
    safe_name = secure_filename(filename) or f"animation_{uuid.uuid4().hex[:8]}.mp4"
    if not safe_name.lower().endswith(".mp4"):
        safe_name += ".mp4"
    return OUTPUT_DIR / safe_name


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/api/fonts", methods=["GET"])
def list_fonts() -> Response:
    fonts = []
    for font in sorted(FONTS_DIR.glob("*")):
        if font.suffix.lower() not in {".ttf", ".otf", ".ttc", ".otc"}:
            continue
        fonts.append({
            "label": font.stem.replace("_", " "),
            "file": font.name,
            "path": str(font.relative_to(BASE_DIR)),
        })
    return jsonify({"fonts": fonts})


@app.route("/api/fonts", methods=["POST"])
def upload_font() -> Response:
    if "font" not in request.files:
        return jsonify({"error": "No font file uploaded."}), 400
    file = request.files["font"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400
    dest = FONTS_DIR / filename
    file.save(dest)
    return jsonify({
        "message": "Font uploaded.",
        "font": {
            "label": dest.stem.replace("_", " "),
            "file": dest.name,
            "path": str(dest.relative_to(BASE_DIR)),
        }
    })


@app.route("/api/templates", methods=["GET"])
def templates() -> Response:
    payload = {}
    for key in TEMPLATE_FILES:
        payload[key] = {
            "defaults": _load_template(key),
            "script": SEGMENT_ALIAS[key],
        }
    return jsonify(payload)


@app.route("/api/preview", methods=["POST"])
def create_preview() -> Response:
    data = request.get_json(force=True) or {}
    segment_type = data.get("segmentType")
    config = data.get("config") or {}
    global_settings = data.get("globalSettings") or {}
    preview_id = data.get("previewId") or uuid.uuid4().hex

    if segment_type not in SEGMENT_ALIAS:
        return jsonify({"error": f"Unknown segment type: {segment_type}"}), 400

    cfg = _merge_globals(segment_type, config, global_settings)
    preview_name = f"preview_{preview_id}.mp4"
    cfg["output"] = str((PREVIEW_DIR / preview_name).relative_to(BASE_DIR))

    try:
        output_path = run_segment(segment_type, cfg, BASE_DIR)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    url = f"/previews/{output_path.name}?cb={uuid.uuid4().hex[:6]}"
    return jsonify({"previewUrl": url, "output": output_path.name})


@app.route("/api/generate", methods=["POST"])
def generate_animation() -> Response:
    data = request.get_json(force=True) or {}
    segments = data.get("segments") or []
    global_settings = data.get("globalSettings") or {}
    filename = data.get("filename") or "animation.mp4"

    if not isinstance(segments, list) or not segments:
        return jsonify({"error": "At least one segment is required."}), 400

    job_id = uuid.uuid4().hex[:8]
    job_dir = PREVIEW_DIR / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []
    try:
        for idx, segment in enumerate(segments):
            seg_type = segment.get("type")
            config = segment.get("config") or {}
            if seg_type not in SEGMENT_ALIAS:
                raise ValueError(f"Unsupported segment type: {seg_type}")
            cfg = _merge_globals(seg_type, config, global_settings)
            seg_name = segment.get("name") or f"segment_{idx}"
            output_name = f"{job_id}_{idx}_{seg_name}.mp4"
            cfg["output"] = str((job_dir / output_name).relative_to(BASE_DIR))
            output_path = run_segment(seg_type, cfg, BASE_DIR)
            generated_paths.append(output_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    final_path = _normalise_output_path(filename)

    try:
        from render_brand_sequence import concatenate_clips  # local import to avoid cycle

        codec = data.get("codec", "libx264")
        preset = data.get("preset", "medium")
        pixel_format = data.get("pixel_format", "yuv420p")
        ffmpeg_params = data.get("ffmpeg_params", ["-tune", "animation"])
        concatenate_clips(generated_paths, final_path, codec, preset, pixel_format, ffmpeg_params)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to concat clips: {exc}"}), 500

    url = f"/outputs/{final_path.name}?cb={uuid.uuid4().hex[:6]}"
    return jsonify({
        "message": "Animation generated.",
        "outputUrl": url,
        "filename": final_path.name,
    })


if __name__ == "__main__":
    app.run(debug=True)
