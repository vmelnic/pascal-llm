# Web interface

Status: Open WebUI is deployed through Docker Compose and is connected to a
native ComfyUI service on x99e.

| Interface | Version | URL on LAN | URL on WireGuard | Purpose |
|---|---|---|---|---|
| Open WebUI | 0.11.3 slim | `http://<configured-host>:3000` | direct multi-user chat over the OpenAI-compatible endpoint |
| ComfyUI | 0.3.72 | `http://<configured-host>:8188` | native SDXL workflow and API on the isolated P40 |

The image is version-pinned. Its data lives in the `open-webui-data` Docker
volume. Secrets and mutable provider settings live in the ignored
`config/open-webui.env` file on x99e. The container uses
`http://host.docker.internal:8080/v1` and API key `local`.

The active `config/models/<profile>.env` is the model source of truth. Open
WebUI obtains the only resident model from the server's `/v1/models` response.
Its one-second model cache bounds stale selections during a model switch. No
model ID is pinned in the web configuration.

This includes the optional `qwen-abliterated` profile: after
`./ops/model.sh start qwen-abliterated`, Open WebUI discovers
`qwen3.8-27b-abliterated-ud-q4-k-xl` through the same endpoint. Starting
`qwen` restores the official model; neither choice requires rebuilding or
reconfiguring the UI container.

The deployment configuration is environment-owned so an old database provider
setting cannot pin a non-resident model. Accounts and chats remain persistent.

`ops/ui.sh` reconciles the image settings without replacing the generated
Open WebUI secret. The container reaches ComfyUI at
`http://host.docker.internal:8188`; its model selector is populated from
ComfyUI and exposes `pascal-llm/realvisxl.safetensors`,
`pascal-llm/juggernaut.safetensors` and
`pascal-llm/animagine.safetensors`. RealVisXL is the configured default;
Juggernaut and Animagine remain explicit manual choices. Qwen only prepares
the text prompt. Open WebUI passes the selected checkpoint to ComfyUI, so no
LLM-side model guess or family router is involved. The repository workflow
uses tiled VAE decode because untiled FP32 SDXL decode already exhausted the
P40.

Docker is enabled at boot and the container uses `restart: unless-stopped`.
The Qwen systemd user service remains disabled at boot and is controlled
separately with `ops/model.sh`; the web interface remains reachable but cannot
generate while the model endpoint is stopped.

```bash
./ops/ui.sh start
./ops/ui.sh status
./ops/ui.sh logs
./ops/ui.sh stop
```

ComfyUI itself is not containerized and is deliberately disabled at boot:

```bash
./ops/comfyui.sh install
./ops/comfyui.sh start
./ops/comfyui.sh status
./ops/comfyui.sh models
./ops/comfyui.sh logs
./ops/comfyui.sh stop
```

The installer pins ComfyUI, CPython and the PyTorch/cu126 build in one managed
environment under `work/`. Existing model payloads are linked from
`PASCAL_MODEL_ROOT`, not copied. Custom nodes are disabled by default. ComfyUI
runs with its single backend user; Open WebUI owns account isolation. Enabling
ComfyUI's independent `--multi-user` registry makes Open WebUI's `default`
backend identity invalid. The API has no authentication layer, so port 8188
must remain on the trusted LAN/VPN.
The legacy OpenAI-compatible `pascal-image` service remains available on port
8081, but its launcher and the ComfyUI launcher reject concurrent ownership of
the P40.

No reverse proxy is used. Port 3000 is bound directly on all host addresses,
which covers the LAN and WireGuard addresses. Do not expose this plain-HTTP
port directly to the public Internet.

The first registered Open WebUI account becomes its administrator. User
accounts and chats persist across container recreation.
