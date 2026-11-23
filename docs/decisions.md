# Decision ledger

Status: 2026-11-23.

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
  inequality failed. Serve with pinned `llama.cpp` v0.4.0 plus the pinned
  29-patch SM60 set. The installer accepts patches only at zero fuzz and zero
  offset, then performs a clean CUDA build.
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
- Deploy the pinned Strata checkout under `work/strata` with Swift 1.5 IQ3_S as
  the first service artifact. IQ3_S keeps all 512 experts and is the highest
  fidelity Strata tier offered for this release; it is still mixed-bit
  quantized GSQ/RCO and is not BF16-equivalent. Run it in upstream `resident`
  mode on P100 devices 0/1/2 only, with exact FP16 KV, 262,144 maximum context,
  a 32,768-cell GPU-resident KV window and two request slots. Keep P40 device 3
  outside this gate. Reserve 1,115 MiB for late graph capture as directed by
  the engine's three-P100 startup probe.
- Permit the Q2_0 MTP draft pack because every speculative token is verified by
  the target model; the draft changes proposal efficiency, not accepted target
  tokens. Keep it reported separately from the IQ3_S target weights.
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
- Keep the patch set's decode scheduler slots disabled. With tensor split and
  embedded MTP, the first real Pi request aborted in `ggml-backend-meta.cpp`
  at `GGML_ASSERT(bcj.nodes[i])`. `LLAMA_DEC_SLOTS=0` passed the corrected Pi
  gate; the two graph fusions documented upstream as non-deterministic on MoE
  also remain disabled.
- Use a 24 GiB process-local RAM prompt cache for official Qwen. It stores and
  restores both target and MTP state and is consumed/re-published as slots are
  reused. Admission removes an obsolete contained prefix first, then the
  oldest entries until the byte budget fits. It survives Pi exit/resume but
  not a server restart; disk persistence remains unqualified.
- Use at most four embedded-MTP proposals and a SHA-256-pinned 65,801-token
  draft vocabulary for official Qwen. Its independent Romanian-plus-coding
  holdout coverage is 97.8452%. The full target head still verifies and samples
  every accepted output; only proposal coverage changes. Tensor-split startup
  gathers the reduced head from one complete host-staged read because partial
  row reads are invalid on the distributed meta buffer.

- Serve Swift 1.5 IQ3_S through Strata 0.2.20 (2026-11-23). It reuses the
  prepared 0.1.46.3 pack unchanged and passed one real Pi `hi` at 38.1 tok/s;
  0.1.46.3 stays as rollback.
- Run Inkling 2 Mini only through Strata's `strata-inkling` with the SQ-20
  pack. SQ-20 is upstream's native SF4+MX8 encoding of the official weights
  (reported KLD 0.00000); SQ-15/SQ-10 are fidelity changes needing explicit
  approval.

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
- Treating the functional 262K-configured IQ3_S deployment as the project's
  populated-262K or BF16-quality gate. Exact per-device
  activation/intermediate telemetry and local quality comparison remain open
  even though API and concurrent Pi transport passed at the new configured
  limit and the earlier tool/cancellation gates passed at 32K.
- Adding the P40 to this Strata service. The accepted engine contains only
  SM60 cubins, the three-P100 service already reserves 15.0-15.6 GiB per P100,
  and a four-device split would add another PCIe boundary without establishing
  a fidelity or concurrency benefit.
- CUDA backend sampling for the current Qwen tensor-split path. A real Pi turn
  bounded all CPU sampler work at 0.165 ms per generated token, already below
  the approved 1 ms/token prerequisite. Removing the tensor-split safety guard
  cannot produce a material decode gain on this evidence.
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

- Inkling 2 Mini on llama.cpp (2026-11-23). Upstream `inkling2_loop` has no
  tensor split; layer split assigns the 24 virtual loop slots and their KV by
  slot index, producing 448 graph splits per token. Measured 15 tok/s decode
  and 61 tok/s prefill on 3 x P100. The shared llama.cpp tree was not changed.
  The `inkling-mini` llama.cpp profile was removed; Inkling is served only by
  Strata (`./ops/strata.sh start inkling`).
- maya-inkling-mini (`maya-ik2`) and UnbiasedStrata for Inkling 2 Mini. Their
  >100 tok/s results are RTX 3090/A6000; the kernels use `mma.sync`,
  `ldmatrix` and `cp.async`, and the build skips `maya-ik2` below sm_80
  (`CMakeLists.txt` line 899). NInfer accepts only sm_70+.
- One persistent TP3 kernel per P100 for Inkling 2 Mini decode (2026-11-23).
  Go/no-go gate: one fused SWA visit <= 200 us (engine TP3 path ~320 us).
  The PoW (`tools/poc/inkling-tp3-megakernel-visit.cu`) measured 230.8 us,
  then 230.3 us after one corrective patch; rejected after the second failure.
  Even before global visits, loop controllers and the head it implies
  ~11.1 ms/token (~90 tok/s), so it cannot reach 100 tok/s.
- Inkling 2 Mini as the agent model (2026-11-24). Fast (56-60 tok/s in Pi with
  TP3 and suffix drafts, 10/10 on a read-only project task), but unreliable in
  long tool sessions: at ~106K context it emitted ~600 near-identical `bash`
  calls in 20 minutes while its thinking said to use `edit`; earlier it paged
  the same `grep` output for over an hour. The server forwards `edit` calls
  intact (a 4,052-character `newText` parsed exactly) and a clean Pi turn uses
  `read` -> `edit` -> `read` correctly, so the cause is the model. Swift stays
  the agent model; a harness guard should stop a turn after repeated calls
  without progress.
- Swift 1.5 Flash-Next faster decode on 3 x P100 (2026-11-23). The layer split
  runs one GPU at a time (one busy 78%, none 19%). Tensor parallel through
  Strata's `strata-split` builds for sm_60 but needs every expert in VRAM
  (51.8 > 51.5 GB) or a pinned host copy of all experts plus the embedding
  table (~80 > 60 GiB RAM). In the layer split, overlap/P2P hand-off (+1.6%),
  PLE I/O and host sampling (no effect) were ruled out; the remaining
  between-window gaps bound the gain at ~10%. Rejected: `docs/swift-tp3-plan.md`,
  `benchmarks/2026-11-23-swift-decode-profile/`.
- Inkling 2 Mini SQ-15 pack for speed (2026-11-23). Lossy (96.68% top-1
  upstream), and its expert codecs are not uniform SF4/MX8, so the TP3 row
  split cannot load it; on the layer split it decodes at 20.5 tok/s against
  44.6 for SQ-20 on the same build. SQ-20 stays.
- Inkling DSpark/MTP drafts on TP3 (2026-11-23). Embedded 3-layer drafter
  (`mtp_config`), replicated on every rank: 51.0 tok/s against 62.4 plain.

## Open decision

The original `Qwen3.8-27B + populated 262K + exact-F16 KV + >=15 tok/s` target
is closed for the custom runtime measured here. The GGUF service is useful at
ordinary Pi context sizes. The previous unpatched runtime measured 25.4-25.5
tok/s; the active four-token/reduced-head MTP configuration measured
38.68-44.14 tok/s across two bounded real Pi tool-flow runs. Its
populated-262K run passed capacity but failed performance: 4,105.894 s cold
prefill and 6.18 tok/s decode. Do not infer maximum-context throughput from the
short-context result or reopen the rejected custom kernel path without a new
quantitative prerequisite.

The active exact-prefill direction is append-only hybrid-state reuse, not a
new inference engine. Process-local RAM restore now passes a Pi resume gate
with target and MTP state and suffix-only evaluation; populated-262K restore
and disk persistence remain separate, unqualified gates. P40 and exact Pascal
kernel work remain rejected for this goal because their best quantitative
bounds do not approach the required 6.84x cold-prefill speedup. See
`long-context-prefill.md`.
