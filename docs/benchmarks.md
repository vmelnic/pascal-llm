# Measurement ledger

Status: through 2026-09-14. Component gates and service results are kept
separate.

## Native ComfyUI functional gate

ComfyUI v0.3.72 at commit
`828b1b9953175b6df79459f417d1032869d0b46a` ran natively with CPython
3.12.11 and PyTorch 2.7.1/cu126. The install gate identified the Tesla P40 as
SM61 and executed finite FP32 matrix-multiplication and convolution results.
The three P100s remained owned only by the concurrently running Qwen process.

The repository SDXL API workflow loaded the existing Juggernaut checkpoint by
symbolic link, generated one coherent 1216x832 PNG with 35 DPM++ 2M/Karras
steps and tiled VAE decode, and completed in 110.99 seconds. The 1,141,924-byte
output passed decode and visual inspection. ComfyUI also discovered Animagine
and RealVisXL through the same link-only path. Open WebUI loaded the same
workflow and node mapping and reached ComfyUI from inside its container.

The standard, non-Lightning RealVisXL V5.0 checkpoint then generated a
1,407,143-byte 1216x832 RGB PNG with 50 DPM++ 2M/Karras steps, CFG 5.0 and
tiled VAE decode. Completion wall was 274.01 seconds. The fixed-seed output
passed visual inspection for coherent photorealistic skin, eyes, hair,
lighting and scene composition. During denoising the P40 was at 100% load;
Qwen remained isolated on P100 devices 0-2.

Two deployment faults were caught before qualification: the ComfyUI
`--cuda-device 0` flag overrode an outer physical-device mask and selected a
P100, and the pinned release omitted its direct `requests` dependency. The
launcher now relies only on `CUDA_VISIBLE_DEVICES=3`; the installer closes the
missing dependency, validates real Pascal CUDA execution and runs the official
startup preflight. The SQLite path is explicit and persistent.

## Qwen3.8-27B Abliterated transport gate

The pinned `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` artifact is
17,378,626,464 bytes with SHA-256
`ebbc66b45cf36bf47dc052d560337ff047a8b4eef851c8919d83d623703b6aa4`.
It loaded through the same three-P100 tensor path, 262,144 shared context,
F16 K/V and embedded MTP settings as the official Qwen profile. Steady
residency was 13,363, 13,109 and 13,347 MiB on P100 devices 0-2.

One uncached direct `hi` with thinking disabled through
`chat_template_kwargs.enable_thinking=false` produced 11 tokens at 36.55
tok/s after a 13-token prefill at 17.53 tok/s. MTP proposed and accepted 9/9
tokens. Minimal Pi transport then returned a coherent greeting and exited
successfully. These results qualify artifact loading, API transport, Pi model
selection and MTP execution only; they do not qualify coding quality, refusal
behavior, long populated context or concurrent sessions.

## Juggernaut XL v9 image-generation gate

Runtime: upstream `stable-diffusion.cpp` commit
`42d6c0ab92fe6595776b28e3f7c8925db79b31f5`, compiled for CUDA SM60/SM61.
Artifact: `RunDiffusion/Juggernaut-XL-v9` revision
`cf419233522daa0b9ea36c3aff98fa2cab1fb0fb`, single-file SDXL checkpoint,
7,105,348,188 bytes, SHA-256
`c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca`.

The first request used the native OpenAI Images route, DPM++ 2M with Karras,
35 steps, CFG 5.0 and the model-card landscape resolution. The endpoint
returned HTTP 200 and a decodable, visually coherent photorealistic PNG.

| Placement | Output | Sampling | VAE decode | HTTP wall | Peak VRAM | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| P100 device 0 only | 1216x832 RGB PNG | 111.84 s | 7.19 s | 120.864 s | 14,735 MiB | 100% |

The runtime reported 6,624.11 MB of parameters on VRAM and 0 MB on RAM:
1,564.36 MB text encoders, 4,900.07 MB diffusion model and 159.68 MB VAE.
Its loader reported FP16 conditioner/diffusion weights and an FP32 VAE; the
artifact is therefore not described as uniformly FP16.
P100 devices 1-2 and the P40 retained zero process residency. This qualifies
one Juggernaut profile and the API transport; it does not yet qualify warm
latency, concurrent replicas, animation/cartoon quality or a public service.

The upstream-recommended CUDA `--diffusion-fa` optimization was then tested
with the identical model, prompt, seed, resolution, sampler and 35 steps. It
reduced denoising residency from about 7,609 to 7,275 MiB, but sampling had not
completed after more than 208 seconds, versus 111.84 seconds without Flash
Attention. The run was stopped fail-fast and the P100 default was reverted.

## Animagine image-generation gate

The artifact was pinned, size/SHA-256 verified and executed through the same
`sd-server` binary and OpenAI Images route. It used no CPU parameter offload or
other device.

| Profile | Preset | Output | Sampling | VAE decode | HTTP wall | Peak VRAM | Visual verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Animagine XL 4.0 Opt | Euler A, 28 steps, CFG 5 | 832x1216 | 157.73 s | 10.69 s | 170.187 s | 14,735 MiB | passed anime sample |

Animagine used the official tag-format prompt, quality suffix and negative
prompt. Its output was coherent and visually strong.

## Isolated P40 image-generation gates

Both remaining profiles were then run as complete FP32 graphs on physical CUDA
device `3`, with upstream VAE tiling and no parameter offload. Physical CUDA
devices `0,1,2` retained zero process residency during both requests.

| Profile | Output | Sampling | VAE decode | HTTP wall | Peak P40 | Visual verdict |
|---|---:|---:|---:|---:|---:|---|
| Juggernaut XL v9 | 1216x832 | 162.33 s | 25.29 s | 189.574 s | 13,537 MiB | passed photorealistic sample; no tile seams |
| Animagine XL 4.0 Opt | 832x1216 | 183.10 s | 26.42 s | 211.642 s | 13,537 MiB | passed anime sample; no tile seams |

The first untiled Juggernaut P40 attempt is a negative capacity result, not a
latency sample: sampling completed in 112.58 seconds, then VAE decode reached
24,187 MiB and aborted with CUDA OOM. Tiling was the single corrective change.
After both image gates, Qwen started concurrently on P100 devices `0,1,2` and
reported healthy while Animagine remained resident on P40. `nvidia-smi`
reported approximately 12.6-12.8 GiB for the Qwen process on each P100 and
12.9 GiB for the image process on the P40. Both services and Open WebUI were
then stopped; all four GPUs reported zero process residency.

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

## Patched production Qwen Pi gate

Date: 2026-09-17. Runtime: llama.cpp v0.4.0 commit
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e` with the 29 patches from
`llama-cpp-p100-patches` commit
`52469952952ee6446207acc574c72055a0685ac4`, applied at zero fuzz and zero
offset. Decode scheduler slots were disabled after the first tensor-split MTP
Pi request hit `GGML_ASSERT(bcj.nodes[i])`; the corrected production build ran
with `LLAMA_DEC_SLOTS=0`. The two optional fusions reported upstream as
non-deterministic on MoE graphs were also disabled.

A bounded real Pi request loaded project instructions, used the read tool once
on `README.md`, returned one correct sentence and stopped. No context files,
skills or normal Pi instructions were disabled; only the tool allowlist was
restricted to `read` to keep the gate bounded.

| Turn | Prompt | Prompt rate | Generated | Decode rate | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| initial/tool | 2,231 tokens | 260.67 tok/s | 168 tokens | 40.07 tok/s | 119/150 = 79.33% |
| tool/final | 59 tokens | 69.54 tok/s | 358 tokens | 30.47 tok/s | 245/336 = 72.92% |

The previous ordinary-context Pi qualification below measured 25.4-25.5
tok/s on the old unpatched commit. The new result is a real improvement on the
active Pi path, but the prompts and output lengths differ, so it is not a
controlled percentage speedup. The populated-262K gate was not rerun.

The same production service was then configured with a 24 GiB RAM prompt
cache. A Pi session containing the codeword `cedar` was evicted from its GPU
slot by a forced task on another slot; `/slots` reported the original slot at
zero prompt tokens. Resuming the Pi session selected a different empty slot,
restored a 1,990-token state, evaluated only 28 suffix tokens in 856.23 ms and
returned `cedar`. Source inspection confirms that the cache serializes and
restores both target and embedded-MTP state. This qualifies process-local
append-only resume at ordinary context. It does not qualify a maximum-size
checkpoint, disk persistence or shared active GPU state across users.

### Four-token MTP and reduced draft vocabulary

The same bounded Pi gate was repeated after raising the embedded-MTP proposal
limit from three to four. With the full draft head, the two turns measured
40.85 and 34.88 decode tok/s. This passed functionally but did not establish a
controlled speedup because the stochastic reasoning lengths differed from the
three-token run.

The draft vocabulary was then reduced to 65,801 token IDs. It contains IDs
0-65,535, all distinct output IDs from two historical Qwen coding sessions and
the tokenizer special IDs. A separate Romanian-plus-coding Pi session held out
from construction contained 79,405 output-token occurrences; 77,694 were in
the candidate set, for 97.8452% coverage. The file is SHA-256 pinned as
`91114dcc3c08638d1316d6442a2bff6e88752bedaecd31526bbf09a50880d455`.
The target model still verifies every proposal; this optimization can reduce
MTP acceptance but cannot bypass target sampling.

The original row-at-a-time draft-head gather was incompatible with the
tensor-split meta buffer and aborted during model load. The deployed local
patch reads the complete distributed output tensor once at startup and gathers
the selected rows in host staging memory. A clean build then passed the same
Pi read-tool request twice:

| Run | Turn | Prompt rate | Generated | Decode rate | MTP acceptance |
|---|---|---:|---:|---:|---:|
| qualification | initial/tool | 266.28 tok/s | 94 tokens | 43.54 tok/s | 72/92 = 78.26% |
| qualification | tool/final | 69.19 tok/s | 356 tokens | 40.11 tok/s | 252/416 = 60.58% |
| telemetry repeat | initial/tool | 266.15 tok/s | 99 tokens | 44.14 tok/s | 76/96 = 79.17% |
| telemetry repeat | tool/final | 69.15 tok/s | 608 tokens | 38.68 tok/s | 423/740 = 57.16% |

The telemetry repeat also bounded CPU sampling. Its final turn spent 100.21 ms
in all sampler work for 608 generated tokens, or at most 0.165 ms per generated
token. That total includes prompt-token acceptance, so the decode-only sampler
cost is lower. CUDA backend sampling therefore failed the precondition of more
than 1 ms/token and was not enabled for tensor split.

## Historical upstream service and Pi qualification

Runtime: historical upstream `llama.cpp` commit
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
