# Measurement ledger

Status: through 2026-09-14. Component gates and service results are kept
separate.

## New 64K profile transport gates

North Mini Code 1.0 Q4_K_M (revision
`6ff6563002170723a6f7a672bf4c99775be6c0dd`, 18,744,024,640 bytes), GPT-OSS-20B
MXFP4 (revision `ef9b12f2ff56c69cf32153a02784e7a3c88bf524`, 12,109,566,624 bytes) and
Granite 4.0 H-Small Q4_K_M (revision
`6522095069a86fc3186b8632850e4373dd850cac`, 19,476,621,984 bytes) were loaded
with F16 K/V and a configured shared 65,536-token context. Each direct request
was an uncached `hi`; each Pi request disabled context files, tools and session
state and therefore proves transport only.

| Model | Placement | VRAM | Prompt | Output | Prompt rate | Decode rate | Minimal Pi |
|---|---|---:|---:|---:|---:|---:|---|
| North Mini Code | 3 x P100, tensor | not recorded | 117 | 47 | 200.63 tok/s | 45.11 tok/s | passed |
| North Mini Code | 1 x P40 | 21,088 MiB | 117 | 62 | 47.67 tok/s | 30.42 tok/s | passed |
| GPT-OSS-20B | 3 x P100, tensor | not recorded | 68 | 22 | 165.07 tok/s | 70.71 tok/s | passed |
| GPT-OSS-20B | 1 x P40 | 12,826 MiB | 68 | 22 | 96.08 tok/s | 47.94 tok/s | passed |
| Granite H-Small | 3 x P100, layer | not recorded | 31 | 10 | 42.09 tok/s | 28.55 tok/s | passed |
| Granite H-Small | 1 x P40 | 20,638 MiB | 31 | 10 | 25.20 tok/s | 16.05 tok/s | passed |

Granite first failed closed because `granitehybrid` does not implement tensor
split in the pinned runtime. Its profile was corrected once to layer split and
then passed. The P40 rows were verified against its UUID with no P100 process
residency. These short results do not qualify real coding, tool calling,
concurrency, populated 64K, or sustained throughput.

## Ornith artifact and text qualification

Runtime: the same pinned upstream `llama.cpp` commit and three-P100 tensor
placement as Qwen. Artifact: official
`Ornith-1.5-35B-Q4_K_M.gguf`, revision
`12393612fd4f730ff5aadc23e9b8f9648aa49ceb`, SHA-256
`42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
The 21,713,463,040-byte file and the separately downloaded projector both
matched their official hashes before promotion. KV was F16, the shared pool
was configured for 262,144 positions, MTP was disabled, and P40 was idle.

The model became ready 4.97 seconds after process start. Pre-request VRAM was
8,869, 9,365 and 9,377 MiB on P100 devices 0-2. A direct streaming `hi` with
the official general-use sampling (`temperature=0.6`, `top_p=0.95`,
`top_k=20`) returned a coherent answer:

| Prompt | Generated | Prompt rate | Decode rate | Client TTFT | Client total |
|---:|---:|---:|---:|---:|---:|
| 11 tokens | 46 tokens | 31.39 tok/s | 39.44 tok/s | 0.441 s | 1.520 s |

A real Pi request then loaded the repository instructions, used its read tool
on `README.md`, and returned the exact requested heading. No context files or
tools were disabled. The two server turns were:

| Turn | Prompt | Prompt rate | Generated | Decode rate | Server total |
|---|---:|---:|---:|---:|---:|
| instruction/tool request | 2,406 tokens | 592.25 tok/s | 49 tokens | 40.15 tok/s | 5.258 s |
| tool result/final answer | 998 tokens | 612.00 tok/s | 18 tokens | 38.48 tok/s | 2.073 s |

Pi wall time was 7.80 seconds and the exact result was `# Pascal LLM`. This
qualifies the text transport and one real tool round trip. It does not qualify
vision, MTP, populated 262K, coding quality, or sustained concurrency.

After the common launcher change, Qwen was restarted through its own profile
and passed a direct streaming `hi`. Its process retained F16 target/draft KV,
MTP `n_max=3`, three-P100 tensor split and an idle P40. The request measured
48.06 prompt tok/s, 35.28 decode tok/s and MTP acceptance `35/60 = 0.5833`.
This is a launcher regression gate, not a replacement for the representative
Qwen measurements below.

## Upstream service and Pi qualification

Runtime: upstream `llama.cpp` commit
`4a89937354190cef5a97baf8eeb17336105eb72d`, compiled for SM60 with NCCL
2.27.7. Model: `Qwen3.8-27B-UD-Q4_K_M.gguf`, tensor split equally over the
three P100s. The P40 was excluded. KV was F16 and the unified pool was
configured for 262,144 tokens shared by four slots.

The model became ready 5.49 seconds after process start. Initialized VRAM was
11,187, 11,207 and 11,461 MiB on P100 devices 0-2 respectively.

One real Pi request loaded the repository instructions, read a file through a
tool call and returned the requested project summary. Server timings for its
two API turns were:

| Turn | Prompt | Prompt rate | Generated | Decode rate |
|---|---:|---:|---:|---:|
| initial/tool request | 2,446 tokens | 292.36 tok/s | 89 tokens | 25.41 tok/s |
| tool result/final answer | 556 tokens | 242.52 tok/s | 149 tokens | 25.51 tok/s |

Two independent Pi processes were then started concurrently with different
file-reading tasks and response markers. The server assigned separate slots;
both completed with the correct marker and no cross-session content. During
the overlapping turns, observed per-slot decode rates were 14.05-14.36 tok/s
and 16.61-18.77 tok/s. Once contention ended, subsequent turns returned to
25.01-25.43 tok/s.

These results qualify ordinary Pi transport, tool use and two-session
isolation. They do not qualify a populated 262K prompt, four-way sustained
load, cancellation/recovery under pressure or useful coding quality.

## Populated 262K cold-context baseline

The same running service processed one controlled request whose tokenized
prompt contained 262,016 tokens and whose output limit was 128 tokens. Prompt
cache reuse and speculative decoding were disabled, so this is a cold F16-KV
baseline rather than a reused-prefix or MTP result. The generated counting
continuation was structurally correct and ended at the requested output limit.

| Populated prompt | Cached prompt | Output | Cold prefill | TTFT | Decode | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 262,016 | 0 | 128 | 4,105.894 s, 63.81 tok/s | 4,106.161 s | 20.534 s, 6.18 tok/s | 4,126.429 s |

The prompt plus output reached the configured 262,144-token ceiling. This
proves capacity and exact F16 KV allocation on the three-P100 service, but
fails both practical maximum-context gates: cold TTFT is about 68 minutes
26 seconds and decode is below 15 tok/s. The earlier ordinary-context decode
rate must not be extrapolated to a fully populated KV pool.

## Cold-prefill split-mode gate

Date: 2026-09-13. This gate tested whether a separate layer-parallel prefill
engine could materially beat the deployed tensor-parallel engine before any
state-conversion or dual-engine code was written. All arms used the same
`UD-Q4_K_M` artifact, F16 K/V, a cold raw prompt of exactly 32,768 repeated
tokens, prompt-cache reuse disabled and one requested output token.

| Placement | Measured progress | Wall time | Verdict |
|---|---:|---:|---|
| tensor, 3 x P100 | 32,768 / 32,768 | 126.979 s, 258.06 prompt tok/s | baseline |
| layer, 3 x P100 | 26,624 / 32,768 | 134.53 s | stopped: already slower than the complete baseline |
| layer, 3 x P100 + P40 | 20,480 / 32,768 | 244.15 s | stopped: already 1.92x the complete baseline |

Both layer arms were deliberately censored once their elapsed time exceeded
the complete tensor baseline; no final throughput is inferred from partial
progress. The P40 arm used an equal `1,1,1,1` layer split. During its run the
P40 and one P100 were active while other stages were idle at the sampled
instant, consistent with the heterogeneous pipeline failing to keep all four
devices productively overlapped.

The result rejects a dual-placement design in which cold prefill runs in
upstream layer mode and its hybrid attention/recurrent state is then converted
for tensor-mode decode. The prefill producer is slower before the conversion,
restart and state-transfer costs are charged.

## Host and P2P

CUDA 12.4.131, NVIDIA driver 580.178.04. All GPUs negotiated PCIe 3.0 x16.

| Path | P2P | 4 KiB | 128 MiB |
|---|---|---:|---:|
| P100 0 <-> P100 1 | yes | 5.89-5.93 us | 12.25-12.26 GiB/s |
| P100 0/1 <-> P100 2 | yes | 5.80-5.89 us | 9.50-9.54 GiB/s |
| P100 <-> P40 | no | 3.66-3.79 us staged-leg sum | 11.31-12.27 GiB/s per leg |

## Qwen-shape compact FP4 MLP on three P100s

The gate used the real Qwen hidden/intermediate geometry, block-32 FP4 E2M1
weights, UE8M0 scales and Q8 activation semantics. The independent CPU oracle
had zero maximum and relative-L2 error.

Aligned intermediate split: `5,824 + 5,792 + 5,792`.

| SM60 implementation | Effective per-card weight bandwidth | Projected 64-layer MLP |
|---|---:|---:|
| scalar integer baseline | 53.99-61.27 GiB/s | 48.689 ms |
| FP16x2 exact-product correction | 36.95-41.19 GiB/s | 71.623 ms |

The correction was rejected and rolled back. The scalar MLP alone consumes
73.0% of the complete 66.667 ms/token target. Remaining dense/recurrent
weights, saturated KV attention, collectives and runtime overhead cannot fit
the remaining 17.978 ms. A three-P100-only scalar packed provider therefore
fails the 15 tok/s prerequisite.

## Qwen-shape compact FP4 MLP on P40

The SM61 provider used DP4A with the same FP4/UE8M0/Q8 arithmetic and the same
independent CPU oracle. Maximum and relative-L2 error were zero.

| GPU | Gate/up | Down | Representative local MLP |
|---|---:|---:|---:|
| P40 | 0.529 ms, 55.735 GiB/s | 0.348 ms, 42.338 GiB/s | 0.878 ms |

This is comparable to, not several times faster than, the P100 scalar path.
Allocating the 17,408 intermediate channels proportionally to the four measured
local times gives a communication-free 64-layer MLP projection of about
36.879 ms.

## Integrated heterogeneous MLP

The complete 64-layer MLP gate used aligned channel shards
`4,416 + 4,640 + 4,544 + 3,808`. Devices 0-2 used the scalar SM60 kernel;
device 3 used DP4A. Each layer included hidden-state broadcast, host-staged P40
input/output, local gate/up + SwiGLU + down, and reduction on P100 0.

```text
integrated 64-layer MLP     55.561 ms/token
per-layer                    0.868 ms
MLP-only rate               17.998 token/s
```

The first run exposed an invalid 3,824-wide P40 shard: its down-projection rows
were not 16-byte aligned for vector DP4A loads. The corrected 3,808-wide shard
passed a clean build and execution. The integrated result is 6.872 ms slower
than the three-P100 scalar baseline, so P40 host staging does not improve
single-token latency in this design.

## Resident FP16 MLP on three P100s

The gate expanded one real-shape MLP layer to resident FP16, sharded its
intermediate width as `5,824 + 5,792 + 5,792`, and executed 64 layer-equivalents
with FP32 accumulation through CUDA 12.4 cuBLAS. The timing includes P100 P2P
input broadcast, gate/up, SwiGLU, down projection and output reduction. An
independent CPU matrix oracle had zero maximum and relative-L2 error.

```text
integrated 64-layer FP16 MLP       42.832 ms/token
effective aggregate weight rate  744.189 GiB/s
effective rate per P100           248.063 GiB/s
required mixed-layout equivalent   28.778 ms/token
```

For the capacity-feasible 44-expanded/20-compact layout, the measured
projection is `44/64 * 42.832 + 20/64 * 48.689 = 44.662 ms/token` for MLP alone.
It therefore misses the 35 ms prerequisite. The installed CUDA 12.4 headers do
not expose an FP16 `cublasGemvEx`; replacing `GemmEx(n=1)` with a nonexistent
API is not a correction.

## Dedicated resident FP16 MMV on three P100s

The dedicated batch-one kernel uses one 256-thread block per output row,
aligned `half2` loads, FP32 accumulation and block-wide reduction. This follows
the mature CUDA MMV execution shape used by llama.cpp rather than treating a
single vector as a generic GEMM. The same three-device broadcast, SwiGLU,
down-projection and reduction boundary remained in the timed path.

```text
independent CPU oracle maximum error     0
independent CPU oracle relative L2       0
integrated 64-layer FP16 MMV        28.400 ms/token
effective aggregate weight rate  1,122.368 GiB/s
effective rate per P100             374.123 GiB/s
required gate                         <=28.778 ms/token
```

The capacity-feasible mixed projection is now
`44/64 * 28.400 + 20/64 * 48.689 = 34.740 ms/token`. This passes the 35 ms MLP
prerequisite by 0.260 ms. It is not end-to-end throughput evidence; all
non-MLP operations must fit the remaining 31.927 ms/token.

Algorithm reference: [llama.cpp CUDA floating-point MMV](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvf.cu).

## Exact-F16 262K GQA attention on three P100s

Both gates populated the complete 16 GiB target KV and executed all 16
full-attention layers. Context tokens were sharded `87,382 + 87,381 + 87,381`;
only online-softmax statistics and 24 x 256 partial outputs crossed the measured
P100 P2P links.

| Implementation | 16-layer attention | Effective KV rate | Decision |
|---|---:|---:|---|
| fused 32-token shared-memory tiles | 95.182 ms | 168.099 GiB/s | reject |
| strided-batched GQA QK/PV + F32 softmax | 52.558 ms | 304.426 GiB/s | reject |

The BLAS correction reuses each KV head across its six query heads and improved
attention by 1.81x. Its independent CPU oracle measured maximum error
`2.91e-10` and relative L2 `1.73e-7`; compared with an unrounded F32-softmax
reference, FP16 probability staging measured maximum error `1.14e-6` and
relative L2 `1.323e-3` on the bounded oracle input.

The required attention budget was <=11 ms. The best measured subtotal is
`34.740 + 52.558 = 87.298 ms/token` for MLP plus attention, or at most
11.455 tok/s before recurrent projections, recurrent-state updates, LM head,
normalization, residuals and runtime overhead.
