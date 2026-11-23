#!/usr/bin/env python3
"""Register the X99E Strata model and enable it in Pi."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit, urlunsplit


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
# Inkling 2 Mini SQ-20 shares the same P100 service and port; `ops/strata.sh start inkling` selects it.
INKLING_MODEL = {
    **copy.deepcopy(MODEL),
    "id": "inkling-2-mini-sq20",
    "name": "Inkling 2 Mini SQ-20 · X99E Strata TP3",
    "samplingParams": {"temperature": 1.0, "top_p": 1.0},
}
MODELS = [MODEL, INKLING_MODEL]


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
    parser.add_argument("--provider", default="pascal-strata")
    parser.add_argument("--base-url", help="Strata's OpenAI base URL, including /v1")
    args = parser.parse_args()

    data = json.loads(args.config.read_text(encoding="utf-8"))
    if args.provider == "pascal-llm":
        raise SystemExit("Strata requires its own provider; pascal-llm preserves the port-8080 client contract")
    providers = data.get("providers") or {}
    source = providers.get("pascal-llm")
    if not isinstance(source, dict):
        raise SystemExit("Pi provider is missing: pascal-llm")
    provider = providers.get(args.provider)
    if not isinstance(provider, dict):
        provider = copy.deepcopy(source)
    base_url = args.base_url
    if not base_url:
        old = urlsplit(source["baseUrl"])
        base_url = urlunsplit((old.scheme, f"{old.hostname}:8081", "/v1", "", ""))
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path != "/v1":
        raise SystemExit("--base-url must be an HTTP(S) OpenAI URL ending in /v1")
    provider["baseUrl"] = base_url
    provider["models"] = copy.deepcopy(MODELS)
    providers[args.provider] = provider
    source["models"] = [model for model in source.get("models", []) if model.get("id") != MODEL_ID]
    data["providers"] = providers

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    enabled_models = settings.get("enabledModels")
    if not isinstance(enabled_models, list) or not all(isinstance(item, str) for item in enabled_models):
        raise SystemExit(f"Pi settings have no enabledModels string list: {args.settings}")
    model_ref = f"{args.provider}/{MODEL_ID}"
    inkling_ref = f"{args.provider}/{INKLING_MODEL['id']}"
    settings["enabledModels"] = [f"{model_ref}:xhigh", f"{inkling_ref}:xhigh"] + [
        item for item in enabled_models
        if item.split(":", maxsplit=1)[0] not in {model_ref, inkling_ref, f"pascal-llm/{MODEL_ID}",
                                                    f"pascal-inkling/{INKLING_MODEL['id']}"}
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
