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
mutually exclusive Strata service below owns the same P100 domain.

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
devices 0/1/2, resident experts, exact FP16 KV, 32,768 context and two request
slots. `pascal-llm.service` is inactive and conflicts with it. A post-concurrency
snapshot reported 15,995 MiB, 15,813 MiB and 16,041 MiB allocated on P100
devices 0, 1 and 2 respectively; P40 device 3 remained at 0 MiB. The Strata
process RSS was 34,163,208,192 B and the host still reported 27,238,043,648 B
available RAM. No driver, CUDA toolkit or system package was installed, removed
or changed.

Operate this deployment only through `ops/strata.sh`: `start`, `stop`,
`restart`, `status`, `logs` and `models` manage or inspect the remote service;
`pi-config` registers `swift-flash-next-iq3-s` under the existing `pascal-llm`
provider in `~/.pi/agent/models.json` and adds
`pascal-llm/swift-flash-next-iq3-s:xhigh` to the `enabledModels` list in
`~/.pi/agent/settings.json`. It preserves Pi's current default provider/model
and creates one-time `*.before-strata` backups. `pi` launches the normal
repository-aware Pi harness against Strata. The service is deliberately not
enabled at boot.
