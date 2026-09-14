# Qwen3.8 feasibility gate

Status: the custom exact-component design fails the populated-262K 15 tok/s
prerequisite. The upstream GGUF service is useful at ordinary context sizes;
its populated-maximum gate passed capacity but failed performance at 63.81
prompt tok/s, 4,106.2 s TTFT and 6.18 decode tok/s.

## Nemotron-3.5-Lightning-30B-A3B admission

The third prepared serving target is the official GGUF conversion at revision
`8a08a1c81dadcc75d35dbb96016cfd344b632e67d`. Its main Q4_0 artifact and
separate Q4_0 MTP artifact are already present under `PASCAL_MODEL_ROOT`.
This is a weight-fidelity change relative to the BF16 source model; F16 below
describes the target and draft caches, not the weights.

The artifact declares 52 base blocks: 26 routed-MoE blocks, 20 recurrent SSM
blocks and six full-attention blocks. It has 128 experts, activates six experts
per routed block, uses two 128-wide KV heads and declares a native 1,048,576
token context. Pascal initially exposes 262,144 positions so all qualified
models share one explicit deployment boundary.

Capacity at that deployed context is:

```text
main Q4_0 GGUF                       18,898,091,584 B = 17.6002 GiB
separate Q4_0 MTP GGUF               1,155,907,520 B =  1.0765 GiB
six-layer target F16 KV              1,610,612,736 B =  1.5000 GiB
one-layer MTP F16 KV                   268,435,456 B =  0.2500 GiB
four-slot recurrent-state bound        173,015,040 B =  0.1611 GiB
admission subtotal                  22,106,062,336 B = 20.5879 GiB
three-P100 physical capacity                           48.0000 GiB
remaining before graphs/workspaces                     27.4121 GiB
```

The recurrent bound charges, for every one of four sequences, all 20 SSM
blocks at `4096 * (128 + 4) * sizeof(float)`. Runtime allocation telemetry is
authoritative, but capacity is not the admission blocker.

The tensor payload inventory gives an approximate single-token hot-weight
read of `2,767,055,616` bytes for the target: all non-embedding dense tensors,
the vocabulary head and `6/128` of routed-expert storage. One MTP invocation
adds approximately `265,123,328` bytes before acceptance is known. At the
P100's nominal 732 GB/s, the impossible-to-beat sequential layer-placement
floors are about 3.78 ms per target call and 0.36 ms per MTP call. A fully
populated target decode must additionally read 1.5 GiB of F16 KV, whose nominal
floor is about 2.20 ms. These omit dequantization, SSM/attention arithmetic,
pipeline bubbles, launches, output sampling and speculative rejection; they
are not throughput claims.

Upstream `llama.cpp` explicitly rejects tensor split for
`nemotron_h_moe`, so the only valid prepared multi-P100 profile is layer split.
P40 remains excluded: the artifacts and exact caches fit the P100 domain, and
the P40 has no peer path to it. The MTP artifact is passed through the generic
external-draft capability rather than a model-name branch.

Qualification remains ordered and unrun: clean initialization, direct
streaming chat, Qwen regression after the common-launcher change, then a real
Pi tool loop with project instructions. No readiness or performance claim is
made by this admission calculation.

## Ornith-1.5-35B-A3B admission

The second serving target is the official
`ornith-ai/Ornith-1.5-35B-A3B-GGUF` `Q4_K_M` artifact. This is a weight-fidelity
change from the source FP8/BF16 checkpoint and must be reported as Q4_K_M; it
is not FP8 execution. The source architecture has 40 layers, 10 full-attention
layers, 30 Gated DeltaNet layers, two KV heads of width 256, 256 routed experts
and top-8 routing. Native context is 262,144 positions.

Capacity at maximum aggregate context is:

```text
official Q4_K_M artifact             21,713,463,040 B = 20.2222 GiB
exact target F16 KV                   5,368,709,120 B =  5.0000 GiB
derived recurrent F32 state              65,863,680 B =  0.0613 GiB
text subtotal, MTP disabled          27,148,035,840 B = 25.2836 GiB
three-P100 physical capacity                            48.0000 GiB
remaining before runtime workspace                      22.7164 GiB
```

The recurrent estimate includes one `32 x 128 x 128` F32 matrix and the
three-position Q/K/V convolution history per linear-attention layer. Runtime
allocation telemetry remains authoritative. If the embedded MTP layer is later
qualified, its maximum F16 KV adds 0.5 GiB plus workspace.

The official BF16 GGUF is 71.067 GB and cannot fit with KV and runtime reserve.
The Transformers FP8 checkpoint is not a native Pascal execution format. The
official GGUF therefore avoids a local conversion and is the first admissible
artifact. P40 is excluded from the initial hot path because capacity already
fits the coherent three-P100 domain and P40 has no P2P link to it.

Admission gates are dependency ordered: exact HF revision and file-size
validation, GGUF metadata inspection, clean model initialization, direct
streaming chat, Qwen regression, then a real Pi tool round trip. A configured
262K limit is not a populated-context result, and no throughput claim is made
before those gates run.

## Target

```text
model              Qwen3.8-27B
model state         existing compact QPack/FP4 representation
context             262,144 actually populated positions
KV reference        exact IEEE binary16 values
host                3 x P100 16 GiB + 1 x P40 24 GiB
desired decode      >=15 useful output tokens/s
per-token budget    <=66.667 ms, including every GPU and transfer
```

## Capacity

Known values from the validated source project:

```text
QPack artifact                         14,775,390,208 bytes = 13.7607 GiB
hot target weights                     13,625,700,352 bytes = 12.6899 GiB
exact target F16 KV                     17,179,869,184 bytes = 16.0000 GiB
exact MTP F16 KV                         1,073,741,824 bytes =  1.0000 GiB
target recurrent state                     158,859,264 bytes =  0.1480 GiB
hot target weights + target/MTP KV      31,879,311,360 bytes = 29.6899 GiB
aggregate physical VRAM                77,309,411,328 bytes = 72.0000 GiB
```

Therefore aggregate capacity is not the blocker for Qwen3.8. This statement
does not prove a feasible per-device layout: executable weights, recurrent
state, embeddings, logits, MTP state, communication buffers, attention
workspaces and a failure reserve must all fit on their owning devices.

## Hot traffic lower bound

Dense weights are touched for every generated token. Using nominal peak memory
bandwidth only as an impossible-to-beat lower bound:

```text
3 x P100 nominal HBM2                 3 x 732 GB/s
1 x P40 nominal GDDR5X                    346 GB/s
aggregate nominal bandwidth              2,542 GB/s
weight-read floor, all four          13.626 GB / 2,542 GB/s = 5.36 ms/token
weight-read floor, P100 only         13.626 GB / 2,196 GB/s = 6.20 ms/token
```

These floors ignore dequantization, arithmetic, attention, recurrent state,
collectives, kernel launches and imbalance. They show only that the 66.667 ms
target is not disproved by raw aggregate bandwidth.

Expanding the compact model permanently to roughly two bytes per parameter is
not automatically feasible: approximately 54 GiB of dense weights plus 16 GiB
of KV would consume almost the entire 72 GiB before runtime state and reserve.
The exact expanded byte count must be taken from the source tensors before that
layout is considered.

## Measured communication boundary

All three P100s support directed CUDA peer access. Direct-copy measurements
were 5.8-5.9 us for 4 KiB and 9.50-12.26 GiB/s for 128 MiB, depending on the
PCIe path. This is sufficient to keep a three-P100 tensor-parallel layout open,
but the exact collective count and byte width must still be charged to every
token.

P40 has no CUDA peer access with the P100s. Every boundary is host-staged.
Because the complete 13.7607 GiB QPack plus 17 GiB target/MTP KV and 151.5 MiB
recurrent state fit the 48 GiB P100 domain before workspaces, P40 is not
required for base capacity and is excluded from the hot tensor-parallel design
unless a complete organ can execute locally on it and reduces the complete
latency equation.

## Compatibility blocker

The existing primary provider is not portable as-is:

- P100 is SM60 and does not provide the DP4A path used by current packed FP4
  kernels.
- P40 is SM61 and supports DP4A, but has materially lower memory bandwidth and
  different FP16 behavior from P100.
- The existing FlashAttention path targets newer CUDA architectures.
- A P100-oriented provider needs a numerically verified packed-FP4-to-FP16
  execution path, or a proven resident representation that fits with all other
  state. It cannot silently decode the full model per token.

The first direct SM60 packed implementation achieved 53.99-61.27 GiB/s per
card and projected the 64-layer MLP alone at 48.689 ms/token. An FP16x2
exact-product rewrite regressed to 71.623 ms and was rolled back. Thus the
three-P100-only compact path fails the complete 66.667 ms target before the
remaining program is counted.

The SM61 DP4A gate achieved 55.735 GiB/s for gate/up and 42.338 GiB/s for down.
An ideal communication-free four-card MLP split projected 36.879 ms/token, but
the integrated 64-layer gate measured 55.561 ms/token after charging the real
P100 peer broadcasts, host-staged P40 boundary and four-way reduction. That is
slower than the 48.689 ms three-P100 scalar result and leaves only 11.106 ms of
the complete 66.667 ms token budget for all non-MLP weights, exact 16 GiB KV
attention, recurrent operations, collectives and launch overhead.

Even an impossible-to-beat nominal-bandwidth floor consumes approximately
1.784 ms for the remaining 4,534,546,432 target-weight bytes and 7.823 ms for
reading the 16 GiB target KV once across three P100s. The resulting 65.168 ms
subtotal omits the actual attention arithmetic, recurrent state, normalization,
residuals, MTP, imbalance and runtime overhead. The current heterogeneous
executor therefore does not provide a credible 15 tok/s path.

## Next prerequisite: mixed resident representation

The next candidate does not stream weights and does not use P40 in every base
layer. It stores 44 MLP layers expanded to FP16 on the three P100s and keeps 20
MLP layers in the verified compact FP4/UE8M0 representation. The input embedding,
source QPack and optional MTP organ can reside on P40 because they need not cross
the P100/P40 boundary once per base layer.

```text
full MLP, compact FP4/UE8M0                    8.4668 GiB
full MLP, expanded FP16                       31.8750 GiB
44 expanded + 20 compact MLP layers           24.5599 GiB
remaining hot target weights                   4.2231 GiB
exact target KV                               16.0000 GiB
target recurrent state                         0.1480 GiB
P100-domain subtotal                          44.9310 GiB
P100 physical capacity                        48.0000 GiB
physical reserve                               3.0690 GiB
reserve after current driver use              about 2.735 GiB
```

This is capacity-feasible without MTP state in the P100 domain. It is not yet a
throughput result. The 20 compact layers consume a projected 15.215 ms using the
measured scalar SM60 path. To keep the mixed 64-layer MLP at or below 35 ms, the
44 expanded layers may consume at most 19.785 ms. Expressed as a full 64-layer
equivalent, resident FP16 execution must complete within 28.778 ms, or sustain
at least 1,107.6 GiB/s aggregate (369.2 GiB/s per P100). A real integrated
three-P100 FP16 MLP gate must pass that threshold before this layout proceeds.

The CUDA 12.4 cuBLAS gate measured 42.832 ms for the full 64-layer equivalent,
or 744.189 GiB/s aggregate. Substitution into the mixed equation yields
44.662 ms for MLP alone, so the cuBLAS-backed mixed representation is rejected.
No thermal or hardware slowdown was active; sampled SM clocks reached the
1,328 MHz maximum and memory remained at 715 MHz.

A dedicated batch-one SM60 MMV then measured 28.400 ms for the full 64-layer
equivalent, or 1,122.368 GiB/s aggregate, with an independent CPU oracle error
of zero. This passes the 28.778 ms prerequisite. Substitution into the mixed
layout gives 34.740 ms for its 64 MLP layers and leaves 31.927 ms of the target
budget. The next gate must charge the remaining hot matrices, recurrent layers,
16 full-attention layers over the populated 16 GiB KV, collectives, norms,
residuals and launch overhead. Passing the MLP gate alone is not a 15 tok/s
claim.

The exact populated-context attention gate fails the remaining budget. A fused
GQA-reuse kernel measured 95.182 ms for 16 layers; its single allowed correction
using strided-batched QK/PV and F32 softmax measured 52.558 ms. The latter reads
the complete 16 GiB target KV at 304.426 GiB/s aggregate. MLP plus attention is
therefore 87.298 ms/token before the remaining program, already above the
66.667 ms target. Aggregate capacity passes, but 15 tok/s at populated 262K
does not.

## Required measurements before model runtime code

Each result is a design prerequisite, not a generic benchmark:

1. Inventory the exact Qwen program by operation and bytes: dense matrices,
   embeddings, recurrent state, attention layers, KV, vocabulary head, MTP and
   maximum workspaces.
2. The compact scalar SM60, exact-product FP16x2 SM60 and DP4A SM61 gates are
   complete. None establishes a complete 15 tok/s path.
3. Completed: the mixed resident representation fits and its measured MLP
   projection is 34.740 ms.
4. Completed and failed: exact-F16 populated-262K attention alone is 52.558 ms
   versus its <=11 ms budget. Do not integrate a full token or service under the
   15 tok/s acceptance criterion.

Failure of the representative packed operation or the complete latency
inequality stops the Qwen path before a full runtime is built.

## Deployed upstream path

The production service does not integrate the rejected experimental kernels.
It uses the approved 16,464,440,224-byte `UD-Q4_K_M` GGUF in pinned upstream
`llama.cpp`, equally tensor-sharded over three P100s:

```text
quantized GGUF artifact              16,464,440,224 bytes = 15.3337 GiB
262,144-token F16 target KV          17,179,869,184 bytes = 16.0000 GiB
artifact + full target KV subtotal   33,644,309,408 bytes = 31.3337 GiB
P100 physical capacity               51,539,607,552 bytes = 48.0000 GiB
```

The remaining capacity covers CUDA/NCCL contexts, graph buffers, workspaces
and allocator reserve. Actual initialized use was 33,855 MiB across the three
P100s. `--ctx-size 262144 --parallel 4 --kv-unified` creates one shared pool:
aggregate live KV population must remain within the pool even though any one
slot may grow toward the configured maximum when the others are small.

Real Pi tool-flow decode measured 25.4-25.5 tok/s at a roughly 2.4K-token
prompt. Two overlapping Pi sessions measured 14.0-18.8 tok/s per active slot.
At 262,016 populated prompt tokens, cold prefill measured 4,105.894 s and
full-context decode measured 6.18 tok/s. Capacity is proven; maximum-context
performance is not. See `long-context-prefill.md` for the measured scaling and
the exact-prefix reuse decision.
