# Pascal LLM

Copy the public deployment template before using the operational scripts:

```bash
cp .env.example .env
```

Set `PASCAL_REMOTE_HOST`, `PASCAL_REMOTE_ROOT`, `PASCAL_SERVER_URL` and the two
image-service URLs in the ignored `.env`; never commit that file. No host
identity or installation path is compiled into the scripts.

Local multi-user LLM serving on three Tesla P100 GPUs, with isolated one-P40
placement available per model profile. Qwen and Ornith have real Pi tool-flow
evidence; North Mini Code, GPT-OSS and Granite H-Small have 64K direct-chat and
minimal-Pi transport evidence on both placements. The production path is
pinned `llama.cpp` v0.4.0 plus a pinned, zero-fuzz SM60 patch set; this
repository owns its reproducible build, artifact-driven profiles, deployment,
Pi integration and evidence ledger, not another inference engine.

## Verified status

Measured through the real Pi harness with project instructions and a tool
round trip. The Qwen ordinary-context row was refreshed on 2026-09-17 after
deploying the SM60 patch set; older rows retain their recorded runtime:

| Model | Artifact | Compute | Context/KV | Prompt | Decode | Result |
|---|---|---|---|---:|---:|---|
| Qwen3.8-27B | UD-Q4_K_M, 15.33 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | 266.2 tok/s | 38.7-44.1 tok/s | patched Pi tool flow passed |
| Qwen3.8-27B Abliterated | UD-Q4_K_XL, 16.19 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | 17.53 tok/s on uncached `hi` | 36.55 tok/s | direct and minimal Pi transport passed; coding quality unqualified |
| Qwen3.8-27B, two Pi sessions | same | same | one shared unified pool | concurrent | 14.0-18.8 tok/s/session while overlapping | both sessions passed and remained isolated |
| Qwen3.8-27B, populated maximum | same | same | 262,016-token cold prompt, F16 K/V | 63.81 tok/s; TTFT 4,106.2 s | 6.18 tok/s | capacity passed; performance failed |
| Ornith-1.5-35B-A3B | official Q4_K_M, 20.22 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | 592-612 tok/s in Pi | 38.5-40.2 tok/s | direct chat and Pi tool flow passed |
| North Mini Code 1.0 | Q4_K_M, 17.46 GiB | 3 x P100 tensor / 1 x P40 | 65,536 shared, F16 K/V | 200.63 / 47.67 tok/s | 45.11 / 30.42 tok/s | direct and minimal Pi transport passed |
| GPT-OSS-20B | native MXFP4, 11.28 GiB | 3 x P100 tensor / 1 x P40 | 65,536 shared, F16 K/V | 165.07 / 96.08 tok/s | 70.71 / 47.94 tok/s | direct and minimal Pi transport passed |
| Granite 4.0 H-Small | Q4_K_M, 18.14 GiB | 3 x P100 layer / 1 x P40 | 65,536 shared, F16 K/V | 42.09 / 25.20 tok/s | 28.55 / 16.05 tok/s | direct and minimal Pi transport passed |

The 262,144 capacity is now proven by an actually populated 262,016-token
prompt plus 128 generated tokens. Its cold prefill took about 68 minutes
26 seconds and decode fell to 6.18 tok/s, so maximum-context performance does
not meet the target. Four slots share one 262K KV pool dynamically; four
simultaneous 262K conversations do not fit. The weight artifact is quantized;
only the runtime KV contract is exact IEEE F16.

The P40 is intentionally excluded from the text-model collective. It has no
CUDA P2P path to the P100 domain and made the measured hot path slower when
host-staged. It now owns the independent image service instead. Qwen uses
about 12.5-12.7 GiB per P100 with its MTP context; Ornith uses 8.7-9.2 GiB per
P100 without MTP.

Qwen keeps up to 24 GiB of exact target-plus-MTP session state in host RAM.
This lets Pi resume an append-only session while the server process remains
alive: a forced slot eviction followed by a resume restored a 1,990-token
state into an empty slot and evaluated only the 28-token suffix. The cache is
bounded and removes obsolete or oldest entries automatically. It is not a
disk cache and does not survive a model-service restart.

## Operation

The service is manual, not enabled at boot:

```bash
./ops/model.sh profiles
./ops/model.sh download qwen-abliterated
./ops/model.sh start qwen
./ops/model.sh start qwen-abliterated
./ops/model.sh restart ornith
./ops/model.sh start nemotron
./ops/model.sh start north
./ops/model.sh start gpt
./ops/model.sh start granite
./ops/model.sh status
./ops/model.sh logs
./ops/model.sh stop
```

Only one profile is resident at a time. `start <profile>` and
`restart <profile>` select the artifact transactionally; subsequent `start`
without a profile reuses that selection.

The optional web interface starts automatically with Docker after it has been
installed once:

```bash
./ops/ui.sh start
```

- Open WebUI: `http://<configured-host>:3000`
- ComfyUI: `http://<configured-host>:8188`

See [web interface](docs/web-ui.md) for persistence and lifecycle details.

ComfyUI runs natively on the isolated P40; only Open WebUI is containerized.
It reuses the existing RealVisXL, Juggernaut and Animagine checkpoint files
through symbolic links and owns one repository-managed Python/PyTorch
environment:

```bash
./ops/comfyui.sh install
./ops/comfyui.sh start
./ops/comfyui.sh status
./ops/comfyui.sh models
./ops/comfyui.sh logs
./ops/comfyui.sh stop
```

`pascal-comfyui.service` and `pascal-image.service` cannot run together because
both own CUDA device 3. Qwen may remain active on CUDA devices 0-2.

RealVisXL V5.0, Juggernaut XL v9 and Animagine XL 4.0 use the pinned upstream
`stable-diffusion.cpp` server and its OpenAI-compatible Images API on the
isolated P40:

```bash
./ops/image.sh install
./ops/image.sh download realvisxl
./ops/image.sh download juggernaut
./ops/image.sh download animagine
./ops/image.sh start realvisxl
./ops/image.sh start juggernaut
./ops/image.sh start animagine
./ops/image.sh status
```

Example request:

```bash
curl http://10.10.88.7:8081/v1/images/generations \
  -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{"model":"sd-cpp-local","prompt":"a cinematic sunset over the sea","n":1,"size":"1216x832","output_format":"png"}'
```

Only one image profile is resident per image-service process. Image profiles
use physical CUDA device `3`; text profiles use `0,1,2`. The launchers permit
both services to run together only when their profile-declared CUDA sets are
disjoint and fail closed on missing, malformed or overlapping declarations.
RealVisXL is the default photorealistic checkpoint in Open WebUI. Juggernaut
remains an alternative for general cinematic work, and Animagine targets
anime/illustration. Selection is explicit in Open WebUI or ComfyUI; Qwen does
not guess or route the image checkpoint.

| Image profile | P100 FP16 baseline | P40 FP32 + tiled VAE | Result |
|---|---:|---:|---|
| RealVisXL V5.0, 1216x832, 50 steps | not run | 274.01 s | photorealistic ComfyUI sample passed |
| Juggernaut, 1216x832, 35 steps | 120.864 s | 189.574 s | photorealistic sample passed |
| Animagine, 832x1216, 28 steps | 170.187 s | 211.642 s | anime sample passed |

The P40 profile uses FP32 because GP102 has no P100-class fast FP16 path. VAE
tiling is required: an untiled Juggernaut decode reached 24,187 MiB and failed
allocation after sampling completed. The successful paths peaked at 13,537
MiB and left all P100s free. These are single first-request measurements, not
sustained throughput.
The native endpoint is `POST http://10.10.88.7:8081/v1/images/generations`;
OpenAI SDK clients may use `base_url=http://10.10.88.7:8081/v1` and any dummy
local API key. The upstream server does not authenticate requests, so keep it
on the trusted VPN/LAN and do not publish port 8081 to the Internet.
See [image-generation research](docs/image-generation-research.md).

From this repository or any real project, use Pi with its normal `AGENTS.md`,
skills, tools and sessions intact:

```bash
./ops/pi.sh qwen
./ops/pi.sh qwen-abliterated
./ops/pi.sh ornith
./ops/pi.sh nemotron
./ops/pi.sh north
./ops/pi.sh gpt
./ops/pi.sh granite
```

Each `config/models/<profile>.env` owns its context and placement. The new
profiles default to 64K on the three P100s:

```text
PASCAL_CUDA_DEVICES=0,1,2
PASCAL_TENSOR_SPLIT=1,1,1
PASCAL_SPLIT_MODE=tensor
PASCAL_CUDA_P2P=1
PASCAL_CONTEXT_SIZE=65536
```

Granite uses `PASCAL_SPLIT_MODE=layer` because upstream `llama.cpp` does not
implement tensor split for `granitehybrid`. To isolate any profile on the P40,
set the following values in that profile and restart it:

```text
PASCAL_CUDA_DEVICES=3
PASCAL_TENSOR_SPLIT=1
PASCAL_SPLIT_MODE=none
PASCAL_CUDA_P2P=0
```

OpenAI-compatible clients use:

```text
base URL       ${PASCAL_SERVER_URL}/v1
API key        local
Qwen model     qwen3.8-27b-ud-q4-k-m
Qwen abliterated model qwen3.8-27b-abliterated-ud-q4-k-xl
Ornith model   ornith-1.5-35b-a3b-q4-k-m
Nemotron model nvidia-nemotron-3.5-lightning-30b-a3b-q4-0 (prepared, unqualified)
North model     north-mini-code-1.0-q4-k-m
GPT-OSS model   gpt-oss-20b-mxfp4
Granite model   granite-4.0-h-small-q4-k-m
```

Open WebUI discovers the currently resident model from the OpenAI-compatible
endpoint; no UI configuration is tied to Qwen. See
[web interface](docs/web-ui.md).

The host uses `llama.cpp` v0.4.0 and the P100 patch set pinned by
`ops/install-llama.sh`, CUDA SM60, NCCL 2.27.7, tensor split `1,1,1`,
FlashAttention, continuous batching and a shared F16 KV pool. Decode graph
slots are disabled because their tensor-split embedded-MTP path failed the
real Pi gate; two optional graph fusions with reported non-deterministic MoE
output are also disabled. The stable `quantum-llm` repository remains
separate.

See [hardware](docs/hardware.md), [feasibility](docs/feasibility.md),
[Qwen contract](docs/qwen3.8-contract.md),
[Ornith contract](docs/ornith-contract.md),
[Nemotron contract](docs/nemotron-contract.md), [decisions](docs/decisions.md), and
[roadmap](docs/roadmap.md). Measurements are in
[benchmarks](docs/benchmarks.md); the populated-262K scaling analysis and
prefill decision are in [long-context prefill](docs/long-context-prefill.md).
