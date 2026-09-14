# Hardware baseline

Status: measured through 2026-09-14.

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

The deployed Qwen process exposes only CUDA devices 0-2. The three P100s form
the tensor-parallel compute domain and each holds approximately 11 GiB after
initialization. The P40 is visible to the host but idle for this service.

Placement is declared directly by each model profile. North Mini Code,
GPT-OSS-20B and Granite H-Small were also initialized at a configured 65,536
context exclusively on CUDA device 3. `nvidia-smi` reported only the P40 UUID
for each process. This is an isolated alternative placement, not a combined
P100/P40 execution domain.
