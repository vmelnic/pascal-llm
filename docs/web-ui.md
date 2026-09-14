# Web interface

Status: Open WebUI is deployed through Docker Compose on x99e.

| Interface | Version | URL on LAN | URL on WireGuard | Purpose |
|---|---|---|---|---|
| Open WebUI | 0.11.3 slim | `http://<configured-host>:3000` | direct multi-user chat over the OpenAI-compatible endpoint |

The image is version-pinned. Its data lives in the `open-webui-data` Docker
volume. Secrets and mutable provider settings live in the ignored
`config/open-webui.env` file on x99e. The container uses
`http://host.docker.internal:8080/v1` and API key `local`.

The active `config/models/<profile>.env` is the model source of truth. Open
WebUI obtains the only resident model from the server's `/v1/models` response.
Its one-second model cache bounds stale selections during a model switch. No
model ID is pinned in the web configuration.

The deployment configuration is environment-owned so an old database provider
setting cannot pin a non-resident model. Accounts and chats remain persistent.

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

No reverse proxy is used. Port 3000 is bound directly on all host addresses,
which covers the LAN and WireGuard addresses. Do not expose this plain-HTTP
port directly to the public Internet.

The first registered Open WebUI account becomes its administrator. User
accounts and chats persist across container recreation.
