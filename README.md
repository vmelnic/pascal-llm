# Pascal LLM

Copy the public deployment template before using the operational scripts:

```bash
cp .env.example .env
```

Set `PASCAL_REMOTE_HOST`, `PASCAL_REMOTE_ROOT`, `PASCAL_SERVER_URL`,
`PASCAL_STRATA_SERVER_URL` and the two image-service URLs in the ignored
`.env`; never commit that file. No host
identity or installation path is compiled into the scripts.

Local multi-user LLM serving on three Tesla P100 GPUs, with isolated one-P40
placement available per model profile. Qwen, Ornith, Swift GSQ and Swift
Heretic have real Pi tool-flow evidence; North Mini Code, GPT-OSS and Granite
H-Small have 64K direct-chat and minimal-Pi transport evidence on both
placements. The repository operates two text runtimes with separately checked GPU
placements and TCP ports: pinned `llama.cpp` v0.4.0 with a pinned, zero-fuzz SM60
patch set, and pinned Strata `0.2.20` (commit `a34979fc`, built for SM60 from
source) for full-expert Swift 1.5 IQ3_S and Inkling 2 Mini SQ-20. It owns
their reproducible builds, artifact-driven deployment, Pi integration and
evidence ledger.

## Verified status

Measured through the real Pi harness with project instructions and a tool
round trip. The Qwen ordinary-context row was refreshed on 2026-09-17 after
deploying the SM60 patch set; older rows retain their recorded runtime.
The Gemma P40 row is the separately requested single Pi `hi` transport check,
with normal project instructions and skill discovery, and no tool round trip:

| Model | Artifact | Compute | Context/KV | Prompt | Decode | Result |
|---|---|---|---|---:|---:|---|
| Qwen3.8-27B | UD-Q4_K_M, 15.33 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | 266.2 tok/s | 38.7-44.1 tok/s | patched Pi tool flow passed |
| Qwen3.8-27B Abliterated | UD-Q4_K_XL, 16.19 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | 17.53 tok/s on uncached `hi` | 36.55 tok/s | direct and minimal Pi transport passed; coding quality unqualified |
| Qwen3.8-27B, two Pi sessions | same | same | one shared unified pool | concurrent | 14.0-18.8 tok/s/session while overlapping | both sessions passed and remained isolated |
| Qwen3.8-27B, populated maximum | same | same | 262,016-token cold prompt, F16 K/V | 63.81 tok/s; TTFT 4,106.2 s | 6.18 tok/s | capacity passed; performance failed |
| Ornith-1.5-35B-A3B | official Q4_K_M, 20.22 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | 592-612 tok/s in Pi | 38.5-40.2 tok/s | direct chat and Pi tool flow passed |
| Swift 1.5 Qwen3.8 GSQ-RCO | IQ2_S with embedded MTP, 8.95 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | not isolated in Pi gate | 46.37 tok/s weighted | short-context Pi coding and tools passed; populated 262K unqualified |
| Swift 1.5 Qwen3.8 Heretic | i1 Q4_K_M with embedded MTP, 15.66 GiB | 3 x P100 tensor split | 262,144 shared, F16 K/V | not isolated in Pi gate | 40.55 tok/s weighted | short-context Pi coding and tools passed; populated 262K unqualified |
| Swift 1.5 Flash-Next via Strata 0.1.46.3 | full-expert IQ3_S, 77.99 GiB source | 3 x P100 layer split | 262,144 maximum, F16 K/V, 32K GPU-resident window | 56.5 tok/s short prompt | 39.3-42.9 tok/s | OpenAI, real Pi and two concurrent Pi transports passed; populated 262K unqualified |
| Swift 1.5 Flash-Next via Strata 0.2.20 (selectable) | same prepared IQ3_S pack | 3 x P100 layer split, MTP | same | 246.5 tok/s, 5,666-token Pi `hi` | 38.1 tok/s, 164 tokens | one real Pi `hi` passed in 27.9 s; MTP accepted 91/165 |
| Inkling 2 Mini via Strata 0.2.20 + pascal-llm TP3 (active) | SQ-20 native SF4+MX8, 19.35 GB | 3 x P100 tensor parallel (`--tp 3`, patches/strata-next) | 1 slot x 262,144, F16 K/V in VRAM, global KV split 3/3/2 | ~440 tok/s, 7,001-token Pi `hi` | 58.9 tok/s, 75 tokens (57.5 tok/s after the multi-turn fixes); two agents at once were measured no faster than one after the other | real Pi `hi` through the `pascal-strata` service on 8081 passed in 18.2 s; two concurrent Pi sessions passed (`benchmarks/2026-11-23-two-sessions/`); real multi-turn agents reuse 96% of prompt tokens after the prefill, prefix and TP3-deadlock fixes (`benchmarks/2026-11-23-two-agents-real/`), Inkling itself loops on some tool tasks; KLD 0.0013 vs unmodified engine (its own noise 0.0012) |
| Inkling 2 Mini via unmodified Strata 0.2.20 | same | 3 x P100 layer split; DSpark disabled by the engine in layer split | same | ~229 tok/s | 38.6 tok/s | superseded |
| Gemma 4 26B A4B | official QAT Q4_0, 13.45 GiB | 1 x P40, native SM61 build | 262,144 configured, F16 K/V | 756.13 tok/s, 5,358 tokens | 55.09 tok/s, 105 tokens | one real Pi `hi` passed in 10.67 s; tools and populated maximum unqualified |
| North Mini Code 1.0 | Q4_K_M, 17.46 GiB | 3 x P100 tensor / 1 x P40 | 65,536 shared, F16 K/V | 200.63 / 47.67 tok/s | 45.11 / 30.42 tok/s | direct and minimal Pi transport passed |
| GPT-OSS-20B | native MXFP4, 11.28 GiB | 3 x P100 tensor / 1 x P40 | 65,536 shared, F16 K/V | 165.07 / 96.08 tok/s | 70.71 / 47.94 tok/s | direct and minimal Pi transport passed |
| Granite 4.0 H-Small | Q4_K_M, 18.14 GiB | 3 x P100 layer / 1 x P40 | 65,536 shared, F16 K/V | 42.09 / 25.20 tok/s | 28.55 / 16.05 tok/s | direct and minimal Pi transport passed |

For the `llama.cpp` Qwen profile, 262,144 capacity is proven by an actually
populated 262,016-token prompt plus 128 generated tokens. Its cold prefill took
about 68 minutes 26 seconds and decode fell to 6.18 tok/s, so maximum-context
performance does not meet the target. Four slots share one 262K KV pool
dynamically; four simultaneous 262K conversations do not fit. Strata is
configured and allocation-tested at 262,144 with two slots, but has not run an
actually populated 262K prompt. The weight artifacts are quantized; only each
runtime's KV contract is exact IEEE F16.

The P40 is intentionally excluded from the text-model collective. It has no
CUDA P2P path to the P100 domain and made the measured hot path slower when
host-staged. It can host Gemma independently, or the image service when Gemma is stopped. Qwen uses
about 12.5-12.7 GiB per P100 with its MTP context; Ornith uses 8.7-9.2 GiB per
P100 without MTP.

Qwen keeps up to 24 GiB of exact target-plus-MTP session state in host RAM.
This lets Pi resume an append-only session while the server process remains
alive: a forced slot eviction followed by a resume restored a 1,990-token
state into an empty slot and evaluated only the 28-token suffix. The cache is
bounded and removes obsolete or oldest entries automatically. It is not a
disk cache and does not survive a model-service restart.

## Operation

### llama.cpp profiles (port 8080)

The service is manual, not enabled at boot. All repository model profiles are
listed below; `download <profile>` fetches that profile's declared artifact,
and `start <profile>` selects and starts it:

```bash
./ops/model.sh profiles
./ops/model.sh download gemma4
./ops/model.sh download qwen
./ops/model.sh download qwen-abliterated
./ops/model.sh download swift-gsq
./ops/model.sh download swift-heretic
./ops/model.sh download ornith
./ops/model.sh download nemotron
./ops/model.sh download north
./ops/model.sh download gpt
./ops/model.sh download granite
./ops/model.sh start gemma4
./ops/model.sh start qwen
./ops/model.sh start qwen-abliterated
./ops/model.sh start swift-gsq
./ops/model.sh start swift-heretic
./ops/model.sh start ornith
./ops/model.sh start nemotron
./ops/model.sh start north
./ops/model.sh start gpt
./ops/model.sh start granite
./ops/model.sh restart ornith
./ops/model.sh status
./ops/model.sh logs
./ops/model.sh stop
```

Only one llama.cpp profile is resident at a time; a disjoint Strata service
may run alongside it. The start commands above are alternatives, not a batch
that loads every model together. `start <profile>` and
`restart <profile>` select the artifact transactionally; subsequent `start`
without a profile reuses that selection.

### Strata model (port 8081)

Strata uses a separate lifecycle wrapper and its own port, `8081`. It can run
alongside Gemma on P40 at the unchanged port `8080`. Other llama.cpp profiles
that select P100s must be stopped before Strata starts.

The service runs Strata `0.2.20` from `work/strata-next`, installed by
`ops/install-strata-next.sh`. It reuses the Swift runtime pack prepared by
the pinned `0.1.46.3` setup; `work/strata` and `ops/install-strata.sh` remain
only as the rollback runtime and as the Swift preparation path used by
`./ops/strata.sh download`.

| Model | OpenAI model ID | Physical GPUs | Pi provider | Launch |
|---|---|---|---|---|
| Swift 1.5 Flash-Next, full-expert IQ3_S | `swift-flash-next-iq3-s` | P100 `0,1,2` | `pascal-strata` | `pascal-strata` service, port 8081 |
| Inkling 2 Mini SQ-20 (TP3, active) | `inkling-2-mini-sq20` | P100 `0,1,2` | `pascal-strata` | `pascal-strata` service, port 8081 |

Both models share the one `pascal-strata` service, its P100s and port 8081;
exactly one is loaded. `./ops/strata.sh start inkling` or `start swift` selects
it (stored on X99E in `~/.config/pascal-llm/active-strata-model`; default
`swift`); `start`/`restart` without a name keep the current selection, and
`./ops/strata.sh pi` opens Pi on the active model. `pi-config` registers both
models under the `pascal-strata` provider.

Inkling runs with the pascal-llm tensor-parallel decode
(`patches/strata-next/01-inkling-tp-decode.patch`, `--tp 3`), one request slot of
262,144 tokens with all F16 K/V in VRAM (two slots gave no speedup, see
`benchmarks/2026-11-23-two-agents-real/`); `ops/run-strata-server.sh` writes its run configuration. Its SQ-20 pack
is downloaded with
`PASCAL_MODEL_PROFILE_DIR=config/strata ./ops/download-model.sh inkling-mini-sq20`
on X99E. See `docs/inkling-tp3-plan.md` and `docs/feasibility.md` for the
measurements and why it does not reach 100 tok/s on this host.

The complete command list is:

```bash
./ops/strata.sh download
./ops/strata.sh start [swift|inkling]
./ops/strata.sh restart [swift|inkling]
./ops/strata.sh status
./ops/strata.sh logs
./ops/strata.sh models
./ops/strata.sh pi-config
./ops/strata.sh pi
./ops/strata.sh stop
```

`download` runs setup for the configured Swift IQ3_S model: it downloads and
checks the source shards, prepares the runtime packs and builds the engine.
The prepared runtime model already exists on X99E. `start` and `restart` take
an optional model name, `swift` or `inkling`; without one they keep the
current selection (default `swift`).
`models` queries the running server's `/v1/models` endpoint; it does not list
an upstream download catalog. The llama.cpp profiles above are managed only
by `model.sh` and are not selectable through `strata.sh`.

`start` and `restart` synchronize the repository, install the user service and
wait for OpenAI health at `${PASCAL_STRATA_SERVER_URL}` (port `8081`).
`pi-config` registers both Strata models under the separate `pascal-strata`
provider and selects Swift as Pi's default while preserving the `pascal-llm`
endpoint on port `8080`.
`pi` launches the normal project-aware Pi session on the active Strata model;
additional Pi arguments may follow `pi`. The service is manual and remains
disabled at boot.

### Web and image services

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

`pascal-comfyui.service`, `pascal-image.service` and the Gemma profile all use
CUDA device 3 and cannot run together. The image API also shares port 8081
with Strata. Launch checks reject overlapping GPU selections or TCP ports.
Qwen may remain active on CUDA devices 0-2 when Strata is stopped.

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
use physical CUDA device `3`; Gemma also uses `3`, while the other default text
profiles use `0,1,2`. The launchers permit
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
./ops/pi.sh gemma4
./ops/pi.sh qwen
./ops/pi.sh qwen-abliterated
./ops/pi.sh swift-gsq
./ops/pi.sh swift-heretic
./ops/pi.sh ornith
./ops/pi.sh nemotron
./ops/pi.sh north
./ops/pi.sh gpt
./ops/pi.sh granite
./ops/strata.sh pi
```

Each `config/models/<profile>.env` owns its context and placement. North,
GPT-OSS and Granite default to 64K on the three P100s; Qwen, Ornith, Nemotron
Gemma 4 and both Swift profiles declare a 262K shared context pool. Gemma 4 is
prepared with Google's official QAT Q4_0 text artifact and uses a separate
SM61 build on P40. Initialization and one real Pi `hi` passed; tool use,
concurrent requests and populated maximum context remain unqualified. The 64K defaults are:

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

The concurrent deployment uses these independent OpenAI-compatible endpoints:

| Runtime/model | Physical GPUs | Base URL |
|---|---|---|
| llama.cpp / Gemma 4 | P40 `3` | `${PASCAL_SERVER_URL}/v1` (port `8080`, unchanged) |
| Strata / Inkling 2 Mini SQ-20 or Swift Flash-Next IQ3_S (one at a time) | P100 `0,1,2` | `${PASCAL_STRATA_SERVER_URL}/v1` (port `8081`) |

Pi keeps `pascal-llm` pointed at Gemma and registers both Strata models under
the separate `pascal-strata` provider with `./ops/strata.sh pi-config`. Both wrappers retain
normal project instruction and skill discovery.

Gemma selects the separate `work/llama.cpp/build-sm61` runtime for P40.
The original P100 runtime remains in `work/llama.cpp/build`; other profiles
keep that default. On X99E, `./ops/build-llama-p40.sh` builds SM61 from the
already pinned and patched source without resetting the source or overwriting
the SM60 build. It uses all CPU threads by default; stop Strata first to free
its host RAM, or set `PASCAL_BUILD_JOBS` to limit compiler parallelism. It records the P100 artifact hashes
before and after compilation. To move Gemma back to P100s, change both its
placement and its `PASCAL_LLAMA_BUILD_DIR` selection to the original build,
and stop Strata before starting it on the same P100 domain.

OpenAI-compatible clients use:

```text
base URL       ${PASCAL_SERVER_URL}/v1
API key        local
Qwen model     qwen3.8-27b-ud-q4-k-m
Qwen abliterated model qwen3.8-27b-abliterated-ud-q4-k-xl
Swift GSQ model swift-1.5-qwen3.8-27b-gsq-rco-iq2-s-mtp
Swift Heretic model swift-1.5-qwen3.8-27b-heretic-i1-q4-k-m
Strata Swift model swift-flash-next-iq3-s
Strata Inkling model inkling-2-mini-sq20
Gemma 4 model  gemma-4-26b-a4b-it-qat-q4-0 (Pi hi passed; tools unqualified)
Ornith model   ornith-1.5-35b-a3b-q4-k-m
Nemotron model nvidia-nemotron-3.5-lightning-30b-a3b-q4-0 (prepared, unqualified)
North model     north-mini-code-1.0-q4-k-m
GPT-OSS model   gpt-oss-20b-mxfp4
Granite model   granite-4.0-h-small-q4-k-m
```

Open WebUI discovers the currently resident model from the OpenAI-compatible
endpoint; no UI configuration is tied to Qwen. See
[web interface](docs/web-ui.md).

The profile runtime uses `llama.cpp` v0.4.0 and the P100 patch set pinned by
`ops/install-llama.sh`, CUDA SM60, NCCL 2.27.7, tensor split `1,1,1`,
FlashAttention, continuous batching and a shared F16 KV pool. Decode graph
slots are disabled because their tensor-split embedded-MTP path failed the
real Pi gate; two optional graph fusions with reported non-deterministic MoE
output are also disabled. The alternative Strata runtime is pinned by
`ops/install-strata-next.sh` (v0.2.20); it uses layer split on CUDA devices 0/1/2, exact FP16
KV, host KV streaming, two request slots and a 262,144-token maximum context.
P40 device 3 is excluded from both text collectives. The stable `quantum-llm`
repository remains separate.

See [hardware](docs/hardware.md), [feasibility](docs/feasibility.md),
[Qwen contract](docs/qwen3.8-contract.md),
[Ornith contract](docs/ornith-contract.md),
[Nemotron contract](docs/nemotron-contract.md), [decisions](docs/decisions.md), and
[roadmap](docs/roadmap.md). Measurements are in
[benchmarks](docs/benchmarks.md); the populated-262K scaling analysis and
prefill decision are in [long-context prefill](docs/long-context-prefill.md).
