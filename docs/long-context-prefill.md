# Long-context prefill: measured limits and viable directions

Status: research decision, 2026-09-13. This document is about the deployed
Qwen3.8-27B service on the X99E host. It does not turn a configured context
limit into a performance claim.

## Executive result

The populated-262K run established two independent facts:

1. The model, exact F16 KV and runtime state fit on the three P100s.
2. The cold path is not interactive: 262,016 prompt tokens took 4,105.894 s
   to prefill and full-context decode fell to 6.18 tok/s.

The dominant cold-prefill term grows quadratically and is consistent with the
16 full-attention layers, not with model-file I/O, NVMe, or the P40 being idle.
Using the measured 32K and 262K points, an empirical two-term fit attributes
about 85.1% of the 262K wall time to the quadratic term. This is a scaling
decomposition, not an operation profiler.

There is one strong exact direction for real coding sessions: compute a stable
project/conversation prefix once, retain the complete hybrid model state, and
resume only the appended suffix. At the measured scaling, 99.2% exact-prefix
reuse reduces the estimated remaining prefill from 68.4 minutes to about one
minute. It does not make the first arbitrary 262K prompt cheap, and it does not
fix the measured 6.18 tok/s full-context decode.

No cold-prefill mechanism found in current systems or research closes the
required 6.84x total speedup on this hardware while also preserving the exact
model and exact full attention. P40 participation, chunk scheduling, layer
placement, and published Pascal kernel patches all fail that prerequisite by
large margins. Dynamic sparse prefill can plausibly cross it only by changing
attention semantics and therefore remains an approval-gated research branch.

## What 262K represents in practice

`262,144` is a token capacity, not a file-size or line-count promise. Tokenizer,
language, comments, generated files and minification change every conversion.
Useful orders of magnitude are:

- roughly 180,000-200,000 English prose words, or about 600-800 ordinary book
  pages at 250-300 words per page;
- often 50,000-100,000 lines of source code, with a very wide range between
  verbose Java, compact Python and minified JavaScript;
- one large agent transcript containing its initial instructions and tool
  schemas, many source files read through tools, terminal output, patches,
  diagnostics, reasoning and prior answers;
- several complete small repositories, but only a fraction of a large
  monorepo.

A tokenizer census already taken for these repositories gives a concrete local
example. The selected source/text files in `pascal-llm` were 58,043 Qwen
tokens, so four such snapshots fit in 262K. The corresponding selected corpus
in `quantum-llm` was 3,667,363 tokens, so 262K covered about 7.1%. These counts
exclude binary/model artifacts and should not be generalized to other projects.
Pi also does not insert a repository automatically: context grows from its
instructions, conversation and the files/tool results it actually reads.

Qwen documents 262,144 tokens as the native context of the 27B model. The
language model contains 64 layers arranged as 16 repetitions of three Gated
DeltaNet layers followed by one full-attention layer.[^1]

## Exact measured boundary

### Capacity

The measured request used MTP off and prompt-cache reuse off:

```text
Qwen GGUF weights                    16,464,440,224 B = 15.3337 GiB
target F16 KV                        17,179,869,184 B = 16.0000 GiB
target recurrent state                 158,859,264 B =  0.1480 GiB
MTP state during this measurement                 0 =  0.0000 GiB
known state subtotal                                    31.4817 GiB
initialized CUDA residency                 33,855 MiB = 33.0615 GiB
three-P100 physical capacity                            48.0000 GiB
remaining physical headroom                            14.9385 GiB
```

If embedded MTP is enabled, its exact F16 draft KV adds 1 GiB plus its own
workspace. It can improve decode only when accepted proposals amortize target
verification; it cannot reduce the initial target-model prefill.

The exact target KV equation is:

```text
16 attention layers * 262,144 tokens * 2 (K,V)
* 4 KV heads * 256 values/head * 2 bytes = 17,179,869,184 bytes
```

Thus capacity is proven. The 48 GiB is three separate HBM domains, not one
transparent allocation.

### Measured scaling

The two comparable cold points are:

```text
T(32,768)  =   131.09272 s
T(262,016) = 4,105.89448 s
```

Fit the minimum useful empirical model:

```text
T(N) = aN + bN^2
a = 0.0023325924 s/token
b = 5.0904539e-8 s/token^2
```

At `N = 262,016`:

```text
linear-like contribution       611.18 s = 14.89%
quadratic contribution       3,494.72 s = 85.11%
```

The model architecture explains the shape. Gated DeltaNet layers carry fixed
recurrent state and scale linearly in sequence length; the 16 full-attention
layers compute causal attention over pairs of prompt positions.[^2]

For 24 query heads of width 256 in 16 layers, QK plus PV at 262,016 positions
contains approximately:

```text
2 * N * (N + 1) * 24 * 256 * 16 = 13.50 PFLOP
```

That number excludes projections, FFNs, Gated DeltaNet, softmax, dequantizing
weights, collectives and launch overhead. Even the attention arithmetic alone
would take about 241 s at the impossible-to-sustain aggregate 56.1 TFLOP/s
FP16 peak of three PCIe P100s, or about 484 s at their aggregate FP32 peak.
NVIDIA specifies 18.7 TFLOP/s FP16 and 732 GB/s HBM2 bandwidth for each PCIe
P100.[^3] A 60-second exact cold prefill is therefore outside the hardware
ceiling before the rest of the model is counted.

### Required speedup

The current 4,105.9 s cold result requires:

| Cold-prefill target | Required end-to-end speedup |
|---:|---:|
| 30 minutes | 2.28x |
| 20 minutes | 3.42x |
| 10 minutes | 6.84x |
| 5 minutes | 13.69x |
| 1 minute | 68.43x |

For a ten-minute result, even a 4x improvement of every linear operation would
still require the quadratic attention term to improve by 7.81x. Deleting all
linear work entirely would still require 5.82x attention. This is the
fail-fast threshold for any cold-prefill proposal.

### Decode traffic remains separate

At full context, one target decode token must at least touch approximately:

```text
hot target weights       13.626 GB
exact target KV          17.180 GB
recurrent state           0.159 GB
subtotal                 30.965 GB/token
```

At the three P100s' nominal aggregate 2,196 GB/s, the impossible bandwidth
floor is about 14.1 ms/token before arithmetic, collectives and runtime costs.
The measured result was 161.8 ms/token. Prefix reuse skips past prefill work;
it does not reduce these full-context decode bytes. A useful 15 tok/s result
still needs a separate decode improvement.

## Direction A: exact hybrid-prefix state reuse

This is the only direction that changes real repeated-work latency by orders of
magnitude without changing the model's result.

For an existing exact prefix of length `P`, only the suffix needs prefill:

```text
T_delta(N, P) = a(N - P) + b(N^2 - P^2)
```

Applying the measured fit at 262,016 total tokens:

| Exact prefix reused | New suffix | Estimated remaining prefill |
|---:|---:|---:|
| 90.0% | 26,202 tokens | 725.1 s (12.1 min) |
| 95.0% | 13,101 tokens | 371.3 s (6.2 min) |
| 99.0% | 2,620 tokens | 75.7 s |
| 99.2% | 2,096 tokens | 60.6 s |
| 99.5% | 1,310 tokens | 37.9 s |
| 99.9% | 262 tokens | 7.6 s |

This matches the shape of a real agent conversation: each API turn normally
resends an identical history and appends one user message or tool result.
Automatic prefix caching in vLLM explicitly targets long-document queries and
multi-round conversations and skips computation only for their shared
prefix.[^4] SGLang's RadixAttention organizes the same opportunity as a radix
tree and schedules requests around cached prefixes.[^5]

Qwen3.8 is harder than an attention-only transformer. A valid checkpoint must
contain both:

- the exact F16 K/V of all 16 full-attention layers up to the boundary;
- the exact recurrent and convolution states of all 48 Gated DeltaNet layers
  at that same boundary.

The prefix must also be token-identical at the same positions. Reordered files,
changed whitespace, a modified system prompt or inserting a tool result in the
middle invalidates everything after the first difference. Marconi documents
why in-place recurrent state makes partial rollback and hybrid-model cache
admission materially harder than ordinary KV prefix caching.[^6]

The deployed server already enables `--cache-prompt` and a 16 GiB host prompt
cache. Current upstream llama.cpp exposes context checkpoints, a RAM cache and
slot save/restore APIs.[^7] That is evidence for a mechanism, not evidence that
our pinned hybrid path works correctly at 262K. Current llama.cpp bug reports
show successful-looking disk restores that subsequently re-prefill hybrid
models, so disk persistence must be treated as unqualified until a real
restore reports reuse and avoids computation.[^8]

The immediate design is therefore deliberately small:

1. Preserve Pi's natural append-only token prefix; do not insert changing
   metadata before stable history.
2. Qualify the existing in-memory hybrid checkpoint path before adding a new
   cache implementation.
3. Account the full state. One maximum prefix is already 16.148 GiB before
   checkpoint metadata; the configured 16 GiB RAM cache cannot be assumed to
   retain it. Select a host budget only after the server reports actual cache
   bytes.
4. Key reusable state by model artifact, tokenizer/template, runtime/state ABI,
   KV dtype, complete prefix tokens and tenant/session salt.
5. For multi-user service, prove that a shared base prefix is physically shared
   rather than copied into every GPU slot. Four separately materialized 262K
   states cannot fit the one 16 GiB GPU KV pool.
6. Treat saved state transactionally: publish only after the complete attention
   and recurrent checkpoint exists; invalidate on mismatch; never silently fall
   back while reporting a cache hit.

A RAM or NVMe checkpoint is not used during every token. It is loaded once to
replace thousands of seconds of recomputation. A 16.15 GiB state has an ideal
sequential-transfer floor of a few seconds from a fast local NVMe and below a
second from DDR4 before software and PCIe overhead. This is the correct use of
the memory hierarchy: cold state movement once, not hot K/V movement on every
decode token. LMCache implements this general tiered-cache pattern for modern
vLLM deployments, including CPU RAM and local disk.[^9]

This direction does not claim modular reuse of arbitrary source files. Exact
KV for a file depends on the tokens before it. Schemes such as CacheBlend
selectively recompute cached chunks and report comparable task quality, but
they do not reproduce full recomputation token-for-token and remain a separate
fidelity tradeoff.[^10]

## Direction B: exact cold-prefill kernels

### What current industry kernels imply

FlashAttention reduces HBM traffic by tiled online softmax while preserving
full attention, but the official CUDA FlashAttention-2 implementation targets
Ampere, Ada and Hopper; FlashAttention-3 targets Hopper.[^11] llama.cpp carries
a separate CUDA implementation that runs on Pascal, so the service works, but
it cannot use the Tensor-Core paths that produce modern long-context numbers.

There is current, directly relevant external work on P100:

- an exact FP32-class Pascal retile reports only 4-9% improvement over the
  corrected FP32 tile path;
- a broader P100 patch set reports substantial decode wins but its long-context
  F16-KV change is chiefly a workspace/capacity optimization;
- a separate Pascal engine reports 24-61% prefill improvements at roughly
  3K-20K contexts, not a 6.84x populated-262K improvement.[^12]

Those results are useful code-reading material, but they fail the ten-minute
prerequisite before we edit the pinned production engine. Rebuilding a large
fork for a likely 1.1-1.6x result would improve a number, not solve the active
problem.

The only credible exact kernel project would need to demonstrate, first, a
path to at least 7.8x on the measured quadratic term together with about 4x on
linear work. Existing Pascal measurements do not support that premise.

### Batch and chunk size

Increasing micro-batch size can amortize graph, launch and quantized-GEMM
overhead. It primarily improves the linear term and may increase attention
utilization. Even an impossible 8x linear speedup still requires 6.67x on
attention for a ten-minute total. It is a secondary lever only.

Chunked prefill is valuable for production fairness: it interleaves a long
prefill with live decodes and reduces pipeline bubbles. Sarathi's reported
benefit is serving capacity and bounded time-between-tokens; it does not erase
the arithmetic of one cold prompt.[^13] It should prevent a 68-minute request
from freezing other users, not be advertised as a 68-minute TTFT solution.

## Direction C: exact multi-GPU attention

Ring/context parallel attention partitions the sequence and exchanges exact K/V
blocks. Published systems show near-linear scaling on large H100 clusters, and
NVIDIA documents that context-parallel attention communicates K/V while model
weights are duplicated across the context-parallel group.[^14][^15]

That capacity rule blocks the simple version here:

```text
per P100 physical VRAM                         16.0 GiB
replicated Qwen GGUF weights                   15.33 GiB
one third of target KV                          5.33 GiB
minimum before recurrent state/workspace       20.66 GiB
```

Pure context parallelism across the P100s does not fit. The present tensor
split already consumes all three GPUs; combining tensor parallel size 3 with
context parallel size 3 conventionally requires nine GPUs, not three. A custom
two-dimensional weight-and-sequence sharding runtime would violate the current
upstream-runtime boundary and still be limited by Pascal arithmetic.

Layer-pipeline placement avoids weight replication, but it does not remove
work. It adds pipeline scheduling and stage imbalance, and P40 boundaries are
host-staged. It is not a hidden 3x speedup over tensor parallelism because the
three P100s already participate in every layer.

This was subsequently tested as a fail-fast production-artifact gate rather
than left as an assumption. The deployed three-P100 tensor path completed an
exactly 32,768-token cold prompt in 126.979 s. Three-P100 layer mode had reached
only 26,624 tokens after 134.53 s; the equal four-device layer split had reached
only 20,480 after 244.15 s. Both layer requests were cancelled once they were
already slower than the complete baseline. Therefore a prefill/decode
disaggregation design cannot start with upstream layer mode on this host:
state reshaping, process turnover and restore would be additional costs after
an already slower prefill.

## Direction D: P40 as a fourth compute device

NVIDIA specifies 12 TFLOP/s FP32 and 346 GB/s memory bandwidth for P40, versus
9.3 TFLOP/s FP32, 18.7 TFLOP/s FP16 and 732 GB/s for one P100.[^3][^16] The P40
has no CUDA P2P path to the installed P100 domain.

Even granting perfect FP32 attention scaling, adding 12 TFLOP/s to the three
P100s' 27.9 TFLOP/s improves the quadratic term by at most `39.9/27.9 = 1.43x`.
Amdahl's law with the measured 85.1% quadratic share caps total improvement at:

```text
1 / (0.149 + 0.851 / 1.43) = 1.34x
```

That optimistic bound moves 68.4 minutes only to about 50.9 minutes before
host staging, imbalance or weaker FP16 paths. The previously measured
four-device MLP also regressed from 48.689 ms to 55.561 ms after its real P40
boundary was charged. P40 is therefore rejected for the cold-prefill hot path.

P40 is also inferior to available host RAM as a retained-state tier: both
require PCIe movement before the P100s can consume state, while RAM is larger,
does not occupy a compute slot and does not burn a 250 W accelerator merely to
hold one checkpoint. P40 remains useful for an isolated task that consumes its
result infrequently, not for every Qwen layer or token.

## Direction E: cutting or approximating the model

### Layer cutting

Removing FFN/Gated-DeltaNet work does not address the dominant term: deleting
the fitted linear component entirely still leaves 58.2 minutes of attention.
Conversely, keeping only one of the 16 full-attention layers gives an estimated
`611 + 3,495/16 = 829 s`, still 13.8 minutes before accounting for the quality
collapse. A ten-minute result cannot be reached by attention-layer deletion
unless linear work also becomes faster. This is a different distilled model,
not a serving optimization for Qwen3.8-27B.

### Sparse prefill

MInference reports up to 10x prefill reduction on A100 by dynamically selecting
sparse attention patterns, without retraining, across several long-context
models.[^17] Applying that claim only as arithmetic, not as a prediction:

```text
current linear + attention/10       = 611 + 349 = 960 s
linear/4 + attention/10             = 153 + 349 = 502 s
```

This is the first published class that can cross a sub-ten-minute prerequisite
in combination with a strong linear-kernel improvement. It changes which
attention scores are evaluated, has no evidence on this Qwen3.8 checkpoint or
Pascal kernel path, and conflicts with the current exact-attention contract.
It may be opened only after explicit approval with long-context coding,
retrieval and reasoning quality gates. The prior Qwen Flash/QSA harness failure
is negative evidence against substituting another sparse model as a shortcut.

FlashPrefill V2 strengthens the magnitude evidence for this class: it reports
17.54x BF16 speedup over a dense FA3/4-aligned baseline at 128K by selecting
blocks and compensating omitted blocks with pooled K/V statistics. It is not an
exact method, and its released operator requires Hopper SM90. The algorithm is
therefore relevant only as an approval-gated quality tradeoff; its CUDA code is
not portable evidence for P100.[^20]

Tail-Replay is also not a cold-prefill solution. It accelerates cache hits in
hybrid Gated-DeltaNet models by retaining exact full-attention KV but
approximately reconstructing recurrent state from a recent tail. Its reported
9.1-14.3x speedup is for matched 32K prefixes, with 92.8-99.9% retained benchmark
quality, not an arbitrary first 262K ingestion and not exact recurrent
state.[^21]

### KV quantization

KIVI and KVQuant show that asymmetric, per-channel/per-token and outlier-aware
KV representations can retain quality better than uniform low-bit caches on
their evaluated models.[^18][^19] They mainly reduce capacity and decode
bandwidth. They do not remove the cold QK/PV attention pairs unless a sparse or
compressed-domain attention kernel also changes execution. This project has
already rejected uniform FP8/Q4 and K1 as an exact-F16 solution after quality
failures. No KV quantization is proposed under the current contract.

## Decision and dependency order

1. **Qualify existing exact in-memory reuse.** Use one real append-only Pi-shaped
   request, repeat it with a small suffix, and require server telemetry to show
   cached tokens plus suffix-only prompt evaluation. This is an acceptance
   gate, not a throughput sweep.
2. **Verify hybrid correctness.** The cached arm must match a cold arm at the
   state/logit boundary and produce coherent output; attention KV and Gated
   DeltaNet state must resume together. One failure permits one local correction;
   a second stops implementation and triggers an end-to-end reassessment.
3. **Size the RAM cache from reported bytes.** The current 16 GiB cannot be
   assumed to hold a 16.148 GiB maximum state plus checkpoints. Keep one useful
   maximum project anchor first; do not promise four.
4. **Make reuse survive real sessions.** Preserve deterministic prompt ordering,
   bind state to model/template/runtime identity, isolate tenants and expose hit
   tokens, restored bytes, suffix-prefill time and fallbacks.
5. **Prove multi-user physical sharing.** Two Pi agents extending the same base
   prefix must not duplicate 16 GiB of device KV. If upstream cannot share it,
   long-context concurrency remains capacity-limited and must be stated.
6. **Do not modify the cold kernel yet.** Published exact Pascal work misses the
   6.84x prerequisite. Revisit only if a specific upstream/Pascal change shows
   a new full-context bound, not a small-context microbenchmark.
7. **Keep approximate sparse prefill parked.** It is the only cold-compute idea
   with sufficient published magnitude, but it needs explicit fidelity approval
   and cannot inherit evidence from other models or A100.

The practical product resulting from steps 1-5 is: an expensive first project
ingestion, followed by exact, append-only agent turns whose prefill cost is
proportional to the small new suffix. It is not a claim of fast arbitrary cold
262K, and the separate full-context decode target remains unsolved.

## Sources

[^1]: [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
[^2]: [Marconi paper: background on hybrid recurrent/attention models](https://openreview.net/pdf?id=RUaMUu7vMX)
[^3]: [NVIDIA Tesla P100 specifications](https://www.nvidia.com/en-au/data-center/tesla-p100/)
[^4]: [vLLM Automatic Prefix Caching documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
[^5]: [SGLang paper: RadixAttention](https://openreview.net/attachment?id=VqkAKQibpq&name=pdf)
[^6]: [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379)
[^7]: [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
[^8]: [llama.cpp hybrid slot-restore defect](https://github.com/ggml-org/llama.cpp/issues/25913)
[^9]: [LMCache tiered KV-cache storage](https://docs.lmcache.ai/)
[^10]: [CacheBlend](https://arxiv.org/abs/2405.16444)
[^11]: [Official FlashAttention implementation and hardware requirements](https://github.com/Dao-AILab/flash-attention)
[^12]: [Measured P100 llama.cpp patches](https://github.com/shinbunbun/llama-cpp-p100-patches) and [PXA engine measurements](https://github.com/poisonxa16/pxa/blob/main/docs/ENGINE.md)
[^13]: [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
[^14]: [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783)
[^15]: [NVIDIA Megatron Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
[^16]: [NVIDIA Tesla P40 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf)
[^17]: [MInference 1.0](https://arxiv.org/abs/2407.02490)
[^18]: [KIVI](https://arxiv.org/abs/2402.02750)
[^19]: [KVQuant](https://arxiv.org/abs/2401.18079)
[^20]: [FlashPrefill V2](https://arxiv.org/abs/2608.19758) and [released SM90 implementation](https://github.com/qhfan/FlashPrefillv2)
[^21]: [Tail-Replay](https://arxiv.org/abs/2608.30310)
