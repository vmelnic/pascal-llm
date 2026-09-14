# Web interfaces

Status: deployed as two independent Docker Compose services on x99e.

| Interface | Version | URL on LAN | URL on WireGuard | Purpose |
|---|---|---|---|---|
| Open WebUI | 0.11.3 slim | `http://<configured-host>:3000` | direct multi-user chat over the OpenAI-compatible endpoint |
| AnythingLLM | 1.16.1 | `http://<configured-host>:3001` | workspaces, documents, RAG and agent features |

Both images are version-pinned. Their data lives in separate named Docker
volumes. Secrets and mutable provider settings live in ignored files under
`config/` on x99e. Both containers use
`http://host.docker.internal:8080/v1`, API key `local`, and model
`qwen3.8-27b-ud-q4-k-m`.

Docker is enabled at boot and each container uses `restart: unless-stopped`.
The Qwen systemd user service remains disabled at boot and is controlled
separately with `ops/model.sh`; the web interfaces remain reachable but cannot
generate while the model endpoint is stopped.

```bash
./ops/ui.sh start
./ops/ui.sh status
./ops/ui.sh logs
./ops/ui.sh stop
```

No reverse proxy is used. Ports 3000 and 3001 are bound directly on all host
addresses, which covers the LAN and WireGuard addresses. Do not expose these
plain-HTTP ports directly to the public Internet.

The first registered Open WebUI account becomes its administrator. Complete
AnythingLLM's initial account/workspace setup in its browser interface. User
accounts, chats and workspaces persist across container recreation.
