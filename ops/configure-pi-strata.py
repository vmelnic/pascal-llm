#!/usr/bin/env python3
"""Register the X99E Strata model and enable it in Pi."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


MODEL = {
    "id": "swift-flash-next-iq3-s",
    "name": "Swift 1.5 Flash-Next IQ3_S · X99E Strata",
    "reasoning": True,
    "input": ["text"],
    "contextWindow": 262144,
    "maxTokens": 32768,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "thinkingLevelMap": {
        "off": "none",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "xhigh",
        "xhigh": "xhigh",
        "max": "xhigh",
    },
    "samplingParams": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
}
MODEL_ID = MODEL["id"]


def atomic_write(path: Path, data: dict) -> None:
    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.chmod(mode)
    os.replace(temporary, path)


def backup_once(path: Path) -> Path:
    backup = path.with_name(path.name + ".before-strata")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path.home() / ".pi/agent/models.json")
    parser.add_argument("--settings", type=Path, default=Path.home() / ".pi/agent/settings.json")
    parser.add_argument("--provider", default="pascal-llm")
    args = parser.parse_args()

    data = json.loads(args.config.read_text(encoding="utf-8"))
    provider = (data.get("providers") or {}).get(args.provider)
    if not isinstance(provider, dict):
        raise SystemExit(f"Pi provider is missing: {args.provider}")
    models = provider.get("models")
    if not isinstance(models, list):
        raise SystemExit(f"Pi provider has no model catalog: {args.provider}")

    provider["models"] = [MODEL] + [model for model in models if model.get("id") != MODEL_ID]

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    enabled_models = settings.get("enabledModels")
    if not isinstance(enabled_models, list) or not all(isinstance(item, str) for item in enabled_models):
        raise SystemExit(f"Pi settings have no enabledModels string list: {args.settings}")
    model_ref = f"{args.provider}/{MODEL_ID}"
    settings["enabledModels"] = [f"{model_ref}:xhigh"] + [
        item for item in enabled_models if item.split(":", maxsplit=1)[0] != model_ref
    ]
    settings["defaultProvider"] = args.provider
    settings["defaultModel"] = MODEL_ID

    config_backup = backup_once(args.config)
    settings_backup = backup_once(args.settings)
    atomic_write(args.config, data)
    atomic_write(args.settings, settings)
    print(f"Pi model configured: {model_ref}")
    print(f"Pi default model enabled first: {model_ref}:xhigh")
    print(f"Original model configuration backup: {config_backup}")
    print(f"Original settings backup: {settings_backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
