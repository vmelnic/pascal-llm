# Hardware baseline

Status: measured through 2026-10-21.

## Host

| Component | Installed |
|---|---|
| Motherboard | ASUS X99-E-10G WS |
| CPU | Intel Xeon E5-2696 v4, 22 cores / 44 threads |
| RAM | 64 GiB DDR4-2133, four channels |
| Boot/model NVMe | Samsung 980 PRO 1 TB |
| Secondary storage | WD Blue SSD 1 TB |
| Network | Intel X550 10 GbE |
| OS | Ubuntu Server 26.04.1, kernel 7.0.0-31-generic |

## Accelerators

| CUDA index | PCI address | GPU | Architecture | VRAM | Link |
|---:|---|---|---|---:|---|
| 0 | 05:00.0 | Tesla P100 PCIe | SM60 | 16 GiB HBM2 | PCIe 3.0 x16 |
| 1 | 06:00.0 | Tesla P100 PCIe | SM60 | 16 GiB HBM2 | PCIe 3.0 x16 |
| 2 | 09:00.0 | Tesla P100 PCIe | SM60 | 16 GiB HBM2 | PCIe 3.0 x16 |
| 3 | 0a:00.0 | Tesla P40 | SM61 | 24 GiB GDDR5X | PCIe 3.0 x16 |

Aggregate device memory is 72 GiB. It is four separate address/performance
domains, not one transparent 72 GiB VRAM allocation.

The measured topology reports `PIX` inside pairs GPU0/GPU1 and GPU2/GPU3, and
`PHB` across pairs. There is no NVLink.

The CUDA 12.4 runtime gate measured the actual directed connectivity:

| Source/destination domain | CUDA P2P | 4 KiB latency | 128 MiB bandwidth |
|---|---|---:|---:|
| P100 0 <-> P100 1 (`PIX`) | yes | 5.89-5.93 us | 12.25-12.26 GiB/s |
| P100 0/1 <-> P100 2 (`PHB`) | yes | 5.80-5.89 us | 9.50-9.54 GiB/s |
| any P100 <-> P40 | no | 3.66-3.79 us staged legs combined | 11.31-12.27 GiB/s per leg |

The P40 figures are two independently timed pinned-host legs. Their small-copy
sum does not include CPU orchestration between legs, and their bulk bandwidth
cannot be interpreted as direct GPU-to-GPU bandwidth.

## Driver and firmware state

- NVIDIA server driver: 580.178.04.
- CUDA compiler/runtime: Ubuntu CUDA 12.4.131, with native `sm_60` and `sm_61`
  targets. CUDA 13 is intentionally excluded because it removed Pascal offline
  compilation and library support.
- NCCL: 2.27.7 from NVIDIA's CUDA repository. Ubuntu's 2.22.3 package failed
  the first tensor all-reduce against this driver with a CUDA stub-library
  error; the official package passed the same unmodified upstream path.
- NVIDIA modules loaded: `nvidia`, `nvidia_uvm`, `nvidia_modeset`, `nvidia_drm`.
- `nouveau` is not loaded.
- Secure Boot remains enabled; the Ubuntu-signed prebuilt NVIDIA module is used.
- Above 4G Decoding is enabled and firmware exposes a 256 GiB high-MMIO window.
- All GPU BARs are assigned. Remaining boot BAR warnings concern unused X550
  SR-IOV VF resources; the physical NIC is operational.

## Connectivity

- Management uses the WireGuard address configured in the local `.env`.
- The repository does not contain the LAN address or SSH account.
- VPN route is limited to `10.10.88.0/24`.
- Public and Hugging Face traffic uses physical Ethernet, not WireGuard.
- `wg-quick@x99e` is enabled at boot and passed a reboot gate.

Secrets and the WireGuard profile are intentionally outside this repository.

## Active serving placement

The original Qwen profile exposes only CUDA devices 0-2. When selected, its
three P100s form the tensor-parallel compute domain and each holds approximately
11 GiB after initialization. That profile is currently inactive while the
Strata service below owns the same P100 domain; Gemma can instead run on the isolated P40.

Placement is declared directly by each model profile. North Mini Code,
GPT-OSS-20B and Granite H-Small were also initialized at a configured 65,536
context exclusively on CUDA device 3. `nvidia-smi` reported only the P40 UUID
for each process. This is an isolated alternative placement, not a combined
P100/P40 execution domain.

The image services also use CUDA device 3 independently. Juggernaut and
Animagine passed as complete FP32 graphs with tiled VAE at 13,537 MiB peak;
RealVisXL passed the same placement and isolation contract through ComfyUI.
Qwen remained healthy on CUDA devices 0-2. The launcher admits concurrent text
and image services only when their profile-declared device sets are disjoint.

Native ComfyUI is the Open WebUI image backend and uses the same P40 and model
payloads. Its PyTorch cu126 install passed direct FP32 CUDA execution on SM61.
ComfyUI and the OpenAI-compatible `stable-diffusion.cpp` service are mutually
exclusive; neither may claim device 3 while the other is active.

## Strata deployment

### Active runtime: Strata 0.2.20 (2026-11-23)

`ops/install-strata-next.sh` checks out release `v0.2.20` (commit
`a34979fc94f3d126b9bb8210f4ad311345688b52`) into `work/strata-next` and builds
`strata` (engine 0.1.66) and `strata-inkling` (0.1.66) with CUDA 12.4.131,
GCC 13, `STRATA_EXPERIMENTAL_SM60=ON` and only SM60 images. CUDA 12.4's front
end rejects GCC 13's AMX intrinsic headers included through one `.cu` file;
the script pre-defines only those header guards for CUDA compilation. Binary
SHA-256: `strata`
`094a3e314c17a757211cad674d3fa892a18bfa3b77b5fafe769069fe4ea9827a`,
`strata-inkling`
`1ab016a5074ea2a2b150748ab10eb033ce4888c3cdb41d2e773347e5f56d3f37`.
The front end moved to `terminal/serve/server.py` with its venv in
`terminal/.venv`, created through `work/bin/uv` because the host has no
`python3-venv`.

`pascal-strata.service` runs the active model (`swift` or `inkling`, chosen by
`./ops/strata.sh start <model>`). Swift runs from
`work/strata-next/strata-swift-iq3_s.json`. It is the previous run
configuration with only `exe`, `cwd` and `log` changed; it reuses the
unchanged prepared pack, MTP runtime and expert profile under
`work/strata-data` and `work/strata/data`. After start the P100s held
15,441/14,931/14,931 MiB. The 0.1.46.3 description below remains the rollback
runtime and the Swift preparation path.

Inkling 2 Mini SQ-20 is the other model of the same `pascal-strata` service
(port 8081, `./ops/strata.sh start inkling`). It runs `strata-inkling` with the
pascal-llm tensor-parallel decode (`--tp 3`): every P100 holds the 3.67 GB
trunk, a 1/3 feature shard of all 3,072 experts and 3 or 2 of the 8 global-visit
KV caches. At 262,144 context the P100s held 12,927/11,903/12,927 MiB after start.

### Rollback runtime: Strata 0.1.46.3

The pinned Strata `0.1.46.3` source at commit
`9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4` is cloned under
`work/strata`. `ops/install-strata.sh` configures its engine with the installed
CUDA 12.4.131 compiler, the installed GCC/G++ 13.4.0 host toolchain,
`STRATA_EXPERIMENTAL_SM60=ON` and `CMAKE_CUDA_ARCHITECTURES=60`. GCC 13 is
explicit because the host's default GCC 15 objects cannot link against the GCC
13 C++ runtime path selected by CUDA 12.4.

The setup-produced service binary is `work/strata/engine-cuda12/strata`. It is
78,041,128 B, has SHA-256
`a3f2673ab6a9974130189085cf2e1e4ceed8400cbd90877b89e56ade34b8acea`,
resolved dynamic dependencies and 67 embedded SM60 CUDA images. It has no SM61
image. The earlier `build-sm60-gcc13` binary remains a build-gate artifact; it
is not the executable used by the service.

`pascal-strata.service` runs Swift 1.5 IQ3_S with layer split across CUDA
devices 0/1/2, resident experts, exact FP16 KV, a 262,144-token maximum context
and two request slots. KV outside a 32,768-cell-per-QSA-layer GPU window is
kept in pinned host RAM; no device boundary is described as unified memory.
The configuration reserves 1,115 MiB for late CUDA graph capture instead of
giving that space to expert caches. `pascal-llm.service`
was inactive in that historical measurement. As of 2026-11-18, Gemma instead
runs concurrently on P40 device 3 at the unchanged port 8080, and Strata uses
port 8081. Device/port admission checks replace the blanket service conflict.
After graph capture and concurrent real Pi requests in the original deployment, a
snapshot reported 16,003 MiB, 15,405 MiB and 15,629 MiB allocated on P100
devices 0, 1 and 2; P40 device 3 remained at 0 MiB. The service cgroup used
58,680,033,280 B and the host reported 2,402,856,960 B available RAM. No
driver, CUDA toolkit or system package was installed, removed or changed.

Operate this deployment only through `ops/strata.sh`: `start`, `stop`,
`restart`, `status`, `logs` and `models` manage or inspect the remote service;
`pi-config` registers `swift-flash-next-iq3-s` and `inkling-2-mini-sq20` under
a separate `pascal-strata` provider at port 8081 in `~/.pi/agent/models.json`, preserving `pascal-llm`
at port 8080. It places
`pascal-strata/swift-flash-next-iq3-s:xhigh` first in the `enabledModels` list in
`~/.pi/agent/settings.json`, declares `contextWindow: 262144`, and selects it as
Pi's default model. The separate maximum response allowance remains 32,768
tokens. The configurator creates one-time `*.before-strata` backups. `pi`
launches the normal repository-aware Pi harness against Strata. The service is
deliberately not enabled at boot.

## Separate Gemma P40 runtime (2026-11-18)

Gemma selects `work/llama.cpp/build-sm61`, built from the same pinned patched
source with GCC 13.4.0 and CUDA 12.4.131. Its CUDA library contains SM61 images;
`work/llama.cpp/build` remains the unchanged P100 SM60 build. The profile's
`PASCAL_LLAMA_BUILD_DIR` selects the separate runtime. Port 8080, the model ID,
exact F16 KV and the 262144 configured context remain unchanged. The requested
single real Pi `hi` passed; see the benchmark and feasibility records.
