#!/usr/bin/env python3
"""Reject GPU or TCP-port collisions with other managed inference services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path.home() / ".config/pascal-llm"
SERVICES = ("pascal-llm", "pascal-strata", "pascal-image", "pascal-comfyui")


def env_values(path: Path, *names: str) -> list[str]:
    # Match the launchers' shell configuration semantics without printing any
    # unrelated environment values, notably credentials.
    result = subprocess.run(
        ["bash", "-c", 'source "$1" || exit; shift; for name; do printf "%s\\n" "${!name}"; done',
         "placement", str(path), *names],
        check=True, capture_output=True, text=True,
    )
    values = result.stdout.splitlines()
    if len(values) != len(names):
        raise ValueError(f"invalid placement declaration: {path}")
    return values


def devices(value: str) -> set[int]:
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", value):
        raise ValueError(f"invalid CUDA device set: {value!r}")
    return {int(item) for item in value.split(",")}


def port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise ValueError(f"invalid TCP port: {value!r}")
    return number


def profile(kind: str) -> Path:
    selection = "active-model" if kind == "models" else "active-image-model"
    name = (CONFIG / selection).read_text().strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        raise ValueError(f"invalid active profile: {selection}")
    return ROOT / "config" / kind / f"{name}.env"


def placement(service: str, pid: int) -> tuple[set[int], int]:
    if service != "pascal-strata" and pid:
        # Running-process values take precedence over files synchronized before
        # a restart, so an old resident model cannot be mistaken for a new one.
        environment = dict(
            item.split(b"=", 1)
            for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            if b"=" in item
        )
        prefix = {"pascal-llm": "PASCAL", "pascal-image": "PASCAL_IMAGE", "pascal-comfyui": "PASCAL_COMFYUI"}[service]
        gpu = environment.get(f"{prefix}_CUDA_DEVICES".encode())
        tcp = environment.get(f"{prefix}_PORT".encode())
        if gpu is not None and tcp is not None:
            return devices(gpu.decode()), port(tcp.decode())
    if service == "pascal-strata":
        data = json.loads((ROOT / "work/strata-next/strata-swift-iq3_s.json").read_text(encoding="utf-8-sig"))
        gpu = data["gpu"]
        if not isinstance(gpu, list) or not gpu or any(type(item) is not int or item < 0 for item in gpu):
            raise ValueError("invalid Strata GPU declaration")
        return set(gpu), port(str(data["port"]))
    if service == "pascal-llm":
        gpu = env_values(profile("models"), "PASCAL_CUDA_DEVICES")[0]
        tcp = env_values(CONFIG / "server.env", "PASCAL_PORT")[0]
    elif service == "pascal-image":
        gpu = env_values(profile("images"), "PASCAL_IMAGE_CUDA_DEVICES")[0]
        tcp = env_values(CONFIG / "image-server.env", "PASCAL_IMAGE_PORT")[0]
    else:
        gpu, tcp = env_values(CONFIG / "comfyui.env", "PASCAL_COMFYUI_CUDA_DEVICES", "PASCAL_COMFYUI_PORT")
    return devices(gpu), port(tcp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True, choices=SERVICES)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--port", required=True)
    args = parser.parse_args()
    requested_devices, requested_port = devices(args.devices), port(args.port)
    for service in SERVICES:
        if service == args.service:
            continue
        state = subprocess.run(
            ["systemctl", "--user", "show", f"{service}.service", "--property=ActiveState,MainPID"],
            capture_output=True, text=True,
        )
        if state.returncode:
            raise ValueError(f"cannot inspect {service}.service")
        properties = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
        if properties.get("ActiveState") not in {"active", "activating", "reloading", "deactivating"}:
            continue
        peer_devices, peer_port = placement(service, int(properties.get("MainPID", "0")))
        overlap = requested_devices & peer_devices
        if overlap:
            raise ValueError(f"{args.service} overlaps {service} on physical CUDA devices {sorted(overlap)}")
        if requested_port == peer_port:
            raise ValueError(f"{args.service} overlaps {service} on TCP port {requested_port}")
    print(f"Placement admitted: {args.service}, physical CUDA {sorted(requested_devices)}, port {requested_port}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Placement refused: {error}") from error
