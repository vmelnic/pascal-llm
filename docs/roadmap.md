# Roadmap

Status: dependency order, 2026-09-14.

1. Completed: the custom runtime feasibility program was stopped after MLP
   plus populated-262K attention reached 87.298 ms/token before the rest of the
   model. Preserve the measurements; do not integrate those kernels.
2. Completed: deploy pinned, unmodified upstream `llama.cpp` with the approved
   Qwen3.8 UD-Q4_K_M artifact on the coherent three-P100 domain.
3. Completed: restore tensor mode with official NCCL 2.27.7; initialize the
   model and verify equal residency on all three P100s. P40 remains idle.
4. Completed: qualify a real Pi instruction/tool flow. Measured decode is
   25.4-25.5 tok/s at ordinary context.
5. Completed: qualify two concurrent independent Pi processes. Both completed
   correctly; observed overlapping decode was 14.0-18.8 tok/s per session.
6. Completed: a controlled cold request populated 262,016 prompt tokens and
   generated 128 tokens. Capacity passed; cold prefill was 63.81 tok/s with a
   4,106.2-second TTFT, and full-context decode was 6.18 tok/s. Reused-prefix
   and representative Pi coding-context behavior remain separate gates.
7. Completed and rejected: upstream layer-mode cold prefill. Three-P100 layer
   mode and a three-P100-plus-P40 layer pipeline were both already slower than
   the complete three-P100 tensor baseline before reaching 32,768 tokens. Do
   not build a layer-prefill-to-tensor-decode state converter.
8. Open: qualify exact in-memory hybrid-prefix reuse on a real append-only
   Pi-shaped request. Require reported cached tokens, suffix-only prefill,
   coherent Gated DeltaNet state and no duplicated full-prefix KV across two
   sessions. Size `cache-ram` from reported state bytes; do not enable disk
   persistence before restore is proven on this hybrid model.
9. Active but not yet qualified: the current service loads the embedded MTP
   head with F16 draft KV and at most three proposals. Verify that the server
   exposes speculative slots, then record acceptance, cold/reused prefill and
   useful Pi decode separately. Do not infer the community RTX 5060 Ti result
   on the three-P100 topology.
10. Open after real use exposes a failure: cancellation/recovery and sustained
    four-slot pressure. Do not add schedulers, caches or custom kernels in
    anticipation of an unobserved problem.
11. Completed for base text: the official Ornith-1.5-35B-A3B Q4_K_M GGUF is
    the second artifact-driven profile. Its revision and hashes were validated,
    direct streaming chat passed, Qwen passed the common-launcher regression,
    and Ornith completed a real Pi read-tool loop. Keep MTP and vision disabled
    until each is qualified independently; populated 262K and concurrency also
    remain separate gates.
12. Prepared, not qualified: NVIDIA Nemotron-3.5-Lightning-30B-A3B Q4_0 with
    its separate MTP artifact. The generic launcher and Pi catalog are ready;
    the running Qwen process was not changed. Its next gate is an explicit
    model switch followed by initialization, direct chat, Qwen regression and
    a real Pi tool loop, in that order.
13. Completed as transport qualification only: North Mini Code 1.0 Q4_K_M,
    GPT-OSS-20B native MXFP4 and Granite 4.0 H-Small Q4_K_M were added as
    artifact-driven 64K profiles. Each passed direct `hi` and minimal Pi `hi`
    on both three-P100 and isolated one-P40 placement. Qwen passed the common
    launcher regression after placement became profile-owned.
14. Open only when one candidate is selected for real use: qualify its normal
    Pi instructions, tools, coding task, concurrency and populated-context
    behavior. The minimal Pi smoke is not a production qualification.
15. Completed: Juggernaut XL v9 and Animagine XL 4.0 Opt passed the pinned
    upstream `stable-diffusion.cpp` OpenAI Images path on one P100, then on the
    isolated P40. P40 uses FP32 parameters and tiled VAE; measured walls were
    189.574 and 211.642 seconds with 13,537 MiB peak and zero P100 residency.
16. Completed: text/image launcher admission is profile-driven. Qwen started
    and remained healthy on P100 devices `0,1,2` while Animagine remained
    resident on P40 device `3`. Missing, invalid or overlapping declarations
    fail closed. DreamShaper was removed after failing visual-quality review.
17. Completed: native ComfyUI v0.3.72 is installed on the P40 with one managed
    CPython 3.12/PyTorch-cu126 environment. Juggernaut completed the repository
    SDXL workflow, Animagine is discovered through the same link-only model
    path, SQLite is persistent, and Open WebUI reaches the backend through the
    Docker host gateway. The legacy image service and ComfyUI fail closed on
    concurrent P40 ownership.
18. Completed for functionality: standard RealVisXL V5.0 is linked into
    ComfyUI and configured as Open WebUI's default. Its 1216x832, 50-step gate
    produced a coherent photorealistic PNG in 274.01 seconds while Qwen stayed
    on P100 devices 0-2.
19. Completed for transport only: Huihui Qwen3.8-27B Abliterated UD-Q4_K_XL
    is a separate artifact-driven profile with pinned revision, byte size and
    SHA-256. It passed direct `hi`, embedded-MTP telemetry and minimal Pi
    selection. Open WebUI discovers it whenever that profile is resident.
    Real coding/tool quality remains open and the official Qwen profile stays
    the default.
