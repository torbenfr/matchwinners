#!/usr/bin/env python3
"""Utility helpers to invoke the brand segment scripts from JSON configs."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class CommandSpec:
    script: str
    arg_map: Dict[str, str]
    flag_keys: Iterable[str] = ()
    path_keys: Iterable[str] = ()
    passthrough_keys: Iterable[str] = ()
    use_config_file: bool = False

    def build_command(self, config: Dict, working_dir: Path) -> Tuple[List[str], Path]:
        script_path = (working_dir / self.script).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        cfg = dict(config)
        if "output" not in cfg or not cfg["output"]:
            cfg["output"] = f"{Path(self.script).stem}_output.mp4"
        output_path = _resolve_path(cfg["output"], working_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_config_file:
            cfg = _normalise_paths(cfg, working_dir, set(self.path_keys))
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
                json.dump(cfg, tmp, indent=2)
                tmp_path = Path(tmp.name)
            cmd = [sys.executable, str(script_path), "--config", str(tmp_path)]
            return cmd, output_path

        args: List[str] = [sys.executable, str(script_path)]
        path_key_set = set(self.path_keys)
        flag_key_set = set(self.flag_keys)

        for key, flag in self.arg_map.items():
            if key not in cfg:
                continue
            value = cfg[key]
            if value is None:
                continue
            if key in flag_key_set:
                if bool(value):
                    args.append(flag)
                continue
            if key in path_key_set:
                resolved = _resolve_path(str(value), working_dir)
                value = str(resolved)
            args.extend([flag, str(value)])

        for key in self.passthrough_keys:
            if key in cfg and cfg[key] is not None:
                args.extend([key, str(cfg[key])])

        return args, output_path


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _normalise_paths(config: Dict, base_dir: Path, path_keys: Iterable[str]) -> Dict:
    result = dict(config)
    path_set = set(path_keys)
    for key in path_set:
        if key in result and result[key]:
            result[key] = str(_resolve_path(str(result[key]), base_dir))
    return result


SCRIPT_SPECS: Dict[str, CommandSpec] = {
    "brand_snail.py": CommandSpec(
        script="brand_snail.py",
        arg_map={
            "identifier": "--identifier",
            "api_key": "--api-key",
            "width": "--width",
            "height": "--height",
            "speed": "--speed",
            "duration": "--duration",
            "fps": "--fps",
            "output": "--output",
            "copies": "--copies",
            "gap": "--gap",
            "logo_max_height": "--logo-max-height",
            "claim": "--claim",
            "graphic": "--graphic",
            "bg_override": "--bg-override",
        },
        flag_keys={"fill", "show_domain"},
        path_keys={"graphic"},
    ),
    "brand_slide.py": CommandSpec(
        script="brand_slide.py",
        arg_map={
            "identifier": "--identifier",
            "api_key": "--api-key",
            "width": "--width",
            "height": "--height",
            "fps": "--fps",
            "output": "--output",
            "text": "--text",
            "font_path": "--font-path",
            "font_color": "--font-color",
            "bg_override": "--bg-override",
            "slide_from": "--slide-from",
            "in_duration": "--in-duration",
            "hold_duration": "--hold-duration",
            "out_duration": "--out-duration",
            "height_factor": "--height-factor",
            "height_padding": "--height-padding",
            "pix_fmt": "--pix-fmt",
            "crf": "--crf",
        },
        path_keys={"font_path"},
    ),
    "brand_domain.py": CommandSpec(
        script="brand_domain.py",
        arg_map={
            "identifier": "--identifier",
            "api_key": "--api-key",
            "width": "--width",
            "height": "--height",
            "fps": "--fps",
            "output": "--output",
            "domain": "--domain",
            "font_path": "--font-path",
            "font_color": "--font-color",
            "bg_override": "--bg-override",
            "slide_from": "--slide-from",
            "in_duration": "--in-duration",
            "hold_duration": "--hold-duration",
            "out_duration": "--out-duration",
            "height_factor": "--height-factor",
            "height_padding": "--height-padding",
            "pix_fmt": "--pix-fmt",
            "crf": "--crf",
        },
        path_keys={"font_path"},
    ),
    "brand_item.py": CommandSpec(
        script="brand_item.py",
        arg_map={"output": "--output"},
        path_keys={"image", "side_logo"},
        use_config_file=True,
    ),
}


SEGMENT_ALIAS = {
    "logo_snail": "brand_snail.py",
    "text": "brand_slide.py",
    "domain": "brand_domain.py",
    "item": "brand_item.py",
}


def resolve_script(segment_type_or_script: str) -> CommandSpec:
    script_name = SEGMENT_ALIAS.get(segment_type_or_script, segment_type_or_script)
    if script_name not in SCRIPT_SPECS:
        raise KeyError(f"Unsupported segment type/script: {segment_type_or_script}")
    return SCRIPT_SPECS[script_name]


def run_segment(segment_type_or_script: str, config: Dict, working_dir: Path) -> Path:
    spec = resolve_script(segment_type_or_script)
    cmd, output_path = spec.build_command(config, working_dir)
    try:
        subprocess.run(cmd, cwd=str(working_dir), check=True)
    finally:
        if spec.use_config_file:
            # remove temp config file if present at the end of command list
            for token in reversed(cmd):
                if token.endswith(".json") and Path(token).exists():
                    try:
                        Path(token).unlink()
                    except OSError:
                        pass
                    break
    return output_path


__all__ = ["run_segment", "resolve_script", "SEGMENT_ALIAS", "SCRIPT_SPECS"]
