#!/usr/bin/env python3
"""Validate and normalize the upstream Strata run configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


MODEL_ID = "swift-flash-next-iq3-s"
EXPECTED_GPUS = [0, 1, 2]
EXPECTED_CONTEXT = "32768"


def option_value(args: list[str], option: str) -> str | None:
    try:
        index = args.index(option)
    except ValueError:
        return None
    if index + 1 >= len(args):
        raise ValueError(f"{option} has no value")
    return args[index + 1]


def set_option(args: list[str], option: str, value: str) -> None:
    try:
        index = args.index(option)
    except ValueError:
        args.extend((option, value))
        return
    if index + 1 >= len(args):
        raise ValueError(f"{option} has no value")
    args[index + 1] = value


def normalize(config: dict) -> dict:
    args = config.get("args")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("config args must be a string list")

    native = option_value(args, "--native")
    if native is None or "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S" not in native:
        raise ValueError("config does not select the Swift 1.5 IQ3_S artifact")
    if "--resident-experts" not in args or "--mmap-experts" in args:
        raise ValueError("config must use the upstream resident low-RAM mode")
    if option_value(args, "--mtp") is None:
        raise ValueError("config has no MTP draft layer")

    set_option(args, "--max-context", EXPECTED_CONTEXT)
    set_option(args, "--kv", "fp16")
    config["args"] = args
    config["gpu"] = EXPECTED_GPUS
    config["gpus_asked"] = True
    config["parallel"] = 2
    config["host"] = "0.0.0.0"
    config["port"] = 8080
    config["api_key"] = "local"
    config["open_browser"] = False
    config["model_name"] = MODEL_ID
    config["cuda"] = 12
    return config


def validate(config: dict) -> None:
    args = config["args"]
    checks = {
        "model_name": config.get("model_name") == MODEL_ID,
        "gpu": config.get("gpu") == EXPECTED_GPUS,
        "parallel": config.get("parallel") == 2,
        "host": config.get("host") == "0.0.0.0",
        "port": config.get("port") == 8080,
        "cuda": config.get("cuda") == 12,
        "context": option_value(args, "--max-context") == EXPECTED_CONTEXT,
        "kv": option_value(args, "--kv") == "fp16",
        "resident experts": "--resident-experts" in args and "--mmap-experts" not in args,
        "P40 exclusion": 3 not in config.get("gpu", []),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("invalid Strata contract: " + ", ".join(failed))

    executable = Path(str(config.get("exe", "")))
    tokenizer = Path(str(config.get("tokenizer", "")))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Strata executable is missing or not executable: {executable}")
    if not (tokenizer / "vocab.json").is_file():
        raise ValueError(f"Strata tokenizer is incomplete: {tokenizer}")


def atomic_write(path: Path, config: dict) -> None:
    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(config, stream, indent=1)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.chmod(mode)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if not args.check:
        config = normalize(config)
        atomic_write(args.config, config)
    validate(config)
    print(f"Strata config valid: {args.config} ({MODEL_ID}, GPUs 0,1,2, FP16 KV, parallel=2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
