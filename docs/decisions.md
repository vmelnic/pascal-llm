# Decision ledger

Status: 2026-09-14.

## Accepted

- Keep this project separate from `quantum-llm`; the stable RTX 3090 serving
  path is not the experimental branch for Pascal-only execution.
- Treat aggregate VRAM as explicit placement capacity, never transparent
  unified memory.
- Start with Qwen3.8 feasibility because compact weights plus exact-F16 262K KV
  fit aggregate device capacity.
- Use the three P100s as the first coherent compute domain: measured CUDA P2P
  is available between every pair.
- Keep P40 outside the hot Qwen tensor-parallel path unless a complete-local
  placement beats the full three-P100 latency equation. It has no P2P with the
  P100s and is not needed for baseline capacity.
- Keep model declarations artifact-driven. SM60 and SM61 may require different
  providers, but common scheduling must not branch on the model family.
- Use a dedicated batch-one SM60 FP16 MMV with FP32 accumulation for resident
  layers. It measured 28.400 ms per full-MLP equivalent and makes the
  44-expanded/20-compact MLP layout pass at a projected 34.740 ms.
- Stop developing a custom full-model runtime after the exact 262K component
  inequality failed. Use pinned, unmodified upstream `llama.cpp` for serving.
- Use the approved `Qwen3.8-27B-UD-Q4_K_M.gguf` artifact. This changes weight
  fidelity relative to full precision; F16 refers only to the KV cache.
- Keep Huihui Qwen3.8-27B Abliterated UD-Q4_K_XL as an explicitly selected
  alternative profile. Its refusal-direction edit is a semantic fidelity
  change and must never replace the official Qwen default implicitly.
- Use tensor split across the three P100s. With NCCL 2.27.7 this path passed
  model initialization, Pi tool use and concurrent-session gates. Keep the P40
  outside the serving process because it lacks P2P with the P100 domain.
- Keep one unified 262K F16 KV pool shared dynamically by four server slots.
  This enables concurrent users but does not promise 262K to every slot at the
  same time.
- Start the model service manually. It is deliberately disabled at boot.
- Use upstream `stable-diffusion.cpp` instead of implementing an image API or
  diffusion runtime. RealVisXL, Juggernaut and Animagine share one
  artifact-driven server path and the native OpenAI Images compatibility
  endpoint.
- Run the complete image graph independently on P40 device `3`, while text
  remains on P100 devices `0,1,2`. FP32 parameters plus tiled VAE passed at
  13,537 MiB peak for both qualified profiles. The launchers compare declared
  CUDA sets and allow concurrent services only when they are disjoint.
- Use pinned native ComfyUI as the Open WebUI image backend. It reuses the
  qualified checkpoints by symbolic link and owns exactly one managed
  CPython/PyTorch-cu126 environment. Keep the OpenAI-compatible
  `stable-diffusion.cpp` service as an alternative, never a concurrent second
  owner of the P40. Custom ComfyUI nodes remain disabled by default.
- Use RealVisXL V5.0 standard as the Open WebUI default for photorealism while
  keeping Juggernaut and Animagine explicitly selectable. Do not make Qwen
  infer the checkpoint. Frontend selection is deterministic and is passed to
  ComfyUI as workflow input.

## Not transferable as positive evidence

Earlier P100 experiments used an RTX 3090 primary plus two P100 auxiliaries and
pinned-host boundaries. Their failures warn against dense sharding and remote
attention, but they do not directly measure the new P100/P40-only topology.
They may be reopened only through the explicit complete-latency gate in
`feasibility.md`, not by repeating the old implementation.

## Rejected until a new prerequisite changes the result

- NVMe or host-RAM streaming of hot dense weights per decoded token.
- Calling 72 GiB of separate device memory one unified allocation.
- Full-model FP16 expansion without an exact per-device capacity proof.
- Assuming topology labels mean CUDA P2P works without a runtime gate. The
  current three-P100 matrix has now passed that gate; P40 has not.
- Porting SM86 FP4 or FlashAttention kernels by changing compiler architecture
  flags only.
- Using P40 merely as overflow storage if every token must copy its hot shard
  over PCIe.
- Quality-changing KV quantization, sparsification or reduced model semantics
  without explicit user approval and a quality gate.
- Three-P100 scalar packed-FP4 execution for the 15 tok/s Qwen target: the real
  MLP shapes alone project to 48.689 ms/token. An exact-product FP16x2 rewrite
  regressed to 71.623 ms and was rolled back.
- Heterogeneous per-layer MLP sharding across three P100s and one P40. The P40
  has no P2P path; after charging host staging and reduction, the integrated
  MLP measured 55.561 ms/token, slower than the three-P100 baseline.
- Mixed 44-layer resident-FP16 plus 20-layer compact-FP4 MLP using cuBLAS on
  three P100s. Its measured projection is 44.662 ms for MLP alone, versus the
  35 ms prerequisite. Capacity passes, but the execution primitive does not.
- Exact-F16 attention at populated 262K on three P100s. The fused implementation
  measured 95.182 ms; the single BLAS GQA correction measured 52.558 ms, still
  above the <=11 ms component budget. MLP plus attention already totals
  87.298 ms/token before the rest of the model.
- A separate upstream `layer`-mode engine for cold prefill followed by state
  conversion into the production `tensor`-mode engine. On the same exact
  32,768-token cold prompt, tensor mode completed in 126.979 s. Layer mode on
  three P100s had processed only 26,624 tokens after 134.53 s; adding P40 had
  processed only 20,480 after 244.15 s. Both were stopped fail-fast. Conversion
  and restart work can only worsen these results.
- P40 as a layer-parallel cold-prefill stage. Its extra capacity does not repair
  the heterogeneous pipeline utilization, and the measured arm was already
  slower than the complete three-P100 tensor baseline at 62.5% progress.
- Flash Attention as an SDXL speed default on P100 SM60. For an identical
  Juggernaut request, `--diffusion-fa` had not completed sampling after more
  than 208 seconds versus 111.84 seconds without it. It saves workspace but is
  a latency regression on this measured path.
- Untiled FP32 SDXL decode on P40. Juggernaut sampling completed, but VAE
  allocation reached 24,187 MiB and aborted. Upstream VAE tiling is the
  accepted correction; it completed with 13,537 MiB peak.
- DreamShaper XL Lightning. Its measured four-step path was fast, but the
  inspected image had material color and edge artifacts. The profile and its
  checkpoint were removed rather than presenting a failed quality result.

## Open decision

The original `Qwen3.8-27B + populated 262K + exact-F16 KV + >=15 tok/s` target
is closed for the custom runtime measured here. The upstream GGUF service is
useful at ordinary Pi context sizes and measured 25.4-25.5 tok/s decode. Its
populated-262K run passed capacity but failed performance: 4,105.894 s cold
prefill and 6.18 tok/s decode. Do not infer maximum-context throughput from the
short-context result or reopen the rejected custom kernel path without a new
quantitative prerequisite.

The active exact-prefill direction is append-only hybrid-state reuse, not a
new inference engine. Qualify the existing in-memory cache with both attention
KV and Gated DeltaNet state before adding persistence. P40 and exact Pascal
kernel work remain rejected for this goal because their best quantitative
bounds do not approach the required 6.84x cold-prefill speedup. See
`long-context-prefill.md`.
