#!/usr/bin/env python3
"""Reconcile repository-owned ComfyUI settings without touching UI secrets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


MANAGED_KEYS = {
    "ENABLE_IMAGE_GENERATION",
    "IMAGE_GENERATION_ENGINE",
    "IMAGE_GENERATION_MODEL",
    "IMAGE_SIZE",
    "IMAGE_STEPS",
    "ENABLE_IMAGE_PROMPT_GENERATION",
    "COMFYUI_BASE_URL",
    "COMFYUI_API_KEY",
    "COMFYUI_WORKFLOW",
    "COMFYUI_WORKFLOW_NODES",
}


def compact_json(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return json.dumps(json.load(handle), separators=(",", ":"), ensure_ascii=True)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: configure-open-webui.py <repository-root> <target-env>")

    repository_root = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()
    if not target.is_file():
        raise SystemExit(f"Open WebUI environment file is missing: {target}")

    existing = target.read_text(encoding="utf-8").splitlines()
    preserved = [
        line
        for line in existing
        if not (line and not line.startswith("#") and line.partition("=")[0] in MANAGED_KEYS)
    ]
    settings = {
        "ENABLE_IMAGE_GENERATION": "true",
        "IMAGE_GENERATION_ENGINE": "comfyui",
        "IMAGE_GENERATION_MODEL": "pascal-llm/realvisxl.safetensors",
        "IMAGE_SIZE": "1024x1024",
        "IMAGE_STEPS": "50",
        "ENABLE_IMAGE_PROMPT_GENERATION": "true",
        "COMFYUI_BASE_URL": "http://host.docker.internal:8188",
        "COMFYUI_API_KEY": "",
        "COMFYUI_WORKFLOW": compact_json(
            repository_root / "config/comfyui/workflow-sdxl-api.json"
        ),
        "COMFYUI_WORKFLOW_NODES": compact_json(
            repository_root / "config/comfyui/workflow-nodes.json"
        ),
    }
    output = preserved + [f"{key}={value}" for key, value in settings.items()]
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
