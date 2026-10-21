## Strata Swift 1.5 IQ3_S deployment gate (2026-10-21)

The selected artifact is the full-expert Swift 1.5 IQ3_S release, not IQ2_XS
and not the expert-pruned Coder release. IQ2_XS is a materially more aggressive
2-bit quality trade-off. IQ3_S is still a quantized artifact and is not
bit-identical to BF16, but Strata records it as the tier that matches the full
model on the published aggregate benchmarks. Swift 1.5 is itself a fine-tune
whose advertised advantage is shorter reasoning. These are bounded upstream
claims, not a local quality result; real Romanian coding and tool use through
Pi remain the quality gate.

The source is pinned to Hugging Face revision
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`. Its two required GGUF shards are:

| File | Exact bytes | SHA-256 |
|---|---:|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 44,900,816,736 | `46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 38,842,920,384 | `70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608` |

The exact base download is therefore `83,743,737,120 B = 77.9924328625 GiB`.
Preparation produced the following separate artifacts; none is hidden inside
the base-download total:

| Artifact | Exact bytes | Meaning |
|---|---:|---|
| native dense pack | 1,538,035,200 | non-routed target tensors converted for Strata |
| native expert pack | 50,292,326,400 | canonical target expert arena |
| 31 fetched BF16 MTP tensors | 5,214,301,696 | original speculative draft layer |
| Q2_0 MTP GGUF | 889,017,216 | derived draft pack |
| MTP runtime dense binary | 116,099,072 | draft non-expert runtime payload |
| MTP runtime expert binary | 707,788,800 | draft expert runtime payload |
| MTP runtime vocabulary | 425,196 | draft vocabulary map |

The complete prepared target directory, including tokenizer and manifests, is
51,840,341,389 B. The model filesystem had 679,207,936,000 B free before the
download and 535,900,266,496 B free after preparation and startup.

The model has 48 trunk layers, 36 GDN and 12 QSA, width 2,560, 512 routed
experts, top-10 routing and one width-640 shared expert per layer. One target
pass therefore invokes 48 * (10 + 1) expert MLPs. Counting each gate, up and
down matrix multiply-add as two operations gives the exact expert-matrix work

`F_expert = 48 * 11 * 3 * 640 * 2,560 * 2 = 5,190,451,200 FLOP/token`.

This deliberately excludes the dense GDN/QSA, gated-residual, PLE, embedding,
head and MTP-draft work instead of disguising an expert-only count as whole
model FLOPs. MTP adds one draft-layer pass for each proposal, while a verifier
pass checks a window of rows; accepted proposals must be measured before an
amortized FLOP/token number is reported.

The production candidate uses context `T = 32,768`, two isolated request slots,
and exact IEEE binary16 KV. Strata's exact allocator charge is 2,112 bytes per
token per QSA/draft layer: 2,048 bytes for two 256-wide K/V heads and 64 bytes
of index state. Across the 12 target QSA layers plus one MTP draft layer:

`KV(T,S) = 13 * 2,112 * T * S`.

That is 899,678,208 B per populated 32K session and 1,799,356,416 B for two
fully populated sessions. At 262,144 it is 7,197,425,664 B per session. The
32K service gate does not qualify the project's eventual populated-262K gate.
The 36 persistent GDN recurrent plus convolution states are separately

`R_gdn(S) = 36 * (128*48*128 + 10,240*(4-1)) * 4 * S`
`= 117,669,888 * S B`.

PLE history is another `9 * (4*2,560) * 4 = 368,640 B` per allocated stage
session. RoPE tables, page tables, QSA scratch, prompt buffers, verify windows,
expert cache, CUDA workspace and allocator/startup reserve remain separate
charges and must come from the engine's startup plan rather than an estimate.

The upstream full-RAM estimate for IQ3_S is approximately 62 GiB, while the
host exposes exactly 65,162,358,784 B = 60.6871757507 GiB. A whole expert arena
in RAM is therefore rejected for the two-slot service. The selected upstream
`--low-ram resident` layer-split mode writes the canonical expert pack, keeps
the hottest non-overlapping layer experts in the three P100 caches and copies
only the remainder into RAM once. If `E` is the exact expert-pack payload and
`C_i` the distinct expert payload resident on P100 `i`, its host charge is

`E_host = E - (C_0 + C_1 + C_2)`.

The realized expert arena is `E = 50,292,326,400 B`. Startup logged a cache fill
of 5,850 experts and 10.31 GiB of VRAM, but did not expose distinct `C_i`
payload counters. After the concurrent Pi gate, the process RSS was
33,362,508 KiB = 34,163,208,192 B. This is process residency, not an expert-only
counter, so it cannot be substituted for `E_host` without mislabeling dense,
MTP, allocator and server charges.

For P100 `i`, admission is the independently checked inequality

`D_i + C_i + KV_i(T,S) + R_i(S) + A_i(B,S) + I_i(B,S) + P_i + M_i + U_i`
`<= 17,179,869,184 B`,

where `D_i` is that stage's dense weights, `A_i` live activations, `I_i`
intermediates, `P_i` prompt/verify buffers, `M_i` the MTP/head charge on its
owner and `U_i` CUDA/workspace/startup reserve. No term may be paid from the
P40 or from another device's free memory. Device selection is exactly CUDA
indices `0,1,2`; index `3` is excluded.

For a layer-split verifier window of `R` rows, one boundary transfers
`(4*2,560 + 2,560 + 4) * 4 * R = 51,216*R B`: four residual streams, the block
output and four inject scalars, all FP32. Three stages have two boundaries, so
the useful forward hand-off traffic is exactly `102,432*R B` per full target
window, before protocol flags and any implementation staging. Expert traffic
is recorded without averaging: for routed expert `(l,e)` with packed byte size
`B_l,e` and residency indicator `h_l,e`, the nonresident payload is
`X_expert = sum B_l,e * (1-h_l,e)` over the actual route. Startup and request
telemetry must distinguish GPU cache hits, CPU expert work, pinned-host PCIe
reads and SSD reads; none is called unified memory.

Both shards passed the upstream hash check and the independent wrapper check.
The production executable is 78,041,128 B, reports Strata `0.1.46.3`, has
SHA-256 `a3f2673ab6a9974130189085cf2e1e4ceed8400cbd90877b89e56ade34b8acea`,
has resolved dynamic libraries and contains 67 `sm_60` CUDA images with no
SM61 image. No installed driver, CUDA toolkit or system package changed.

The loaded device-memory snapshot after concurrent Pi turns was, separately,
15,995 MiB on P100 0, 15,813 MiB on P100 1, 16,041 MiB on P100 2 and 0 MiB on
P40 3. These are whole-process CUDA allocations: Strata does not break them
into dense weights, expert cache, FP16 KV, activations, intermediates and
workspace. The exact FP16 KV allocator charge remains the equation above;
activations and intermediates therefore remain bounded by successful admission
but are not yet independently measured. This missing decomposition blocks a
full capacity proof, not the active functional deployment gate.

The service passed these active-path checks:

- `/health` reported loaded model `swift-flash-next-iq3-s`, service `strata` and
  maximum context 32,768; `/v1/models` reported the same model and context.
- A direct OpenAI request completed at 56.8 prompt tok/s and 36.6 generated
  tok/s. Its deliberately tiny 32-token cap ended inside reasoning, so this is
  transport evidence only, not a response-quality result.
- A real Pi turn received the repository `AGENTS.md`, used its `read` tool and
  returned the requested exact host value. Pi kept normal context-file and
  skill discovery enabled.
- Two independent real Pi processes started together, each performed a
  distinct `read` tool loop and returned the correct file evidence in 25.60 s
  and 27.51 s wall time. The server log shows both requests making interleaved
  progress; neither corrupted or indefinitely blocked the other.
- Interrupting a separate Pi tool loop produced server telemetry
  `cancel=True` at 5,445 prompt tokens. A new Pi `read` loop immediately
  returned `RECOVERY_OK` in 5.93 s, establishing slot recovery after client
  cancellation.

This qualifies the requested 32K, two-session Pi service. It does not qualify
the project's 262,144 actually populated-token target, independently measured
activation/intermediate capacity, or BF16-equivalent/local coding quality.

## Strata SM60 build-only gate (2026-10-21)

The active request authorizes source acquisition, configuration and compilation
only. It does not select or download a Strata model, stop the production Qwen
service, start a second inference service, or alter the installed NVIDIA driver
and CUDA toolkit. The build gate is therefore limited to a pinned source
identity, an SM60-only CUDA artifact, resolved runtime dependencies and no
regression in the active service.

Strata commit `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4` was cloned into
`work/strata` and built by `ops/install-strata.sh`. The successful configuration
uses CUDA 12.4.131, GCC/G++ 13.4.0, `STRATA_ENABLE_CUDA=ON`,
`STRATA_EXPERIMENTAL_SM60=ON`, `CMAKE_CUDA_ARCHITECTURES=60` and
`STRATA_BUILD_TESTS=OFF`. The host default GCC 15 configuration compiled all
objects but failed at the final link because CUDA 12.4 added the GCC 13 runtime
search path and GCC 13's `libstdc++` does not export GCC 15's
`__cxa_call_terminate`. Pinning both C and C++ to the already-installed GCC 13
toolchain made CUDA host compilation and the final link use one ABI.

The resulting `build-sm60-gcc13/strata` reports version `0.1.46.3`, is
78,041,128 bytes and has SHA-256
`0f7628e2c33441f08fe3c764806abfd61d596ef63b600cebb68281d1e4c11463`.
`ldd` reports no missing library, and all 67 embedded CUDA images reported by
`cuobjdump --list-elf` are `sm_60`; no SM61 image was built. The executable
reports only the upstream `qwen4exp` and `qwen35moe` families. The production
`llama-server` remained active on devices 0/1/2 at its prior approximately
11 GiB per-device residency, its health endpoint remained `ok`, and device 3
remained at zero allocated MiB. No GPU kernel or model inference was run by this
gate.

This compilation charges no model weights, KV, recurrent state, activations or
intermediates because none was loaded. It is not evidence that a Strata model
fits or that its scheduler meets the production contract. Before any inference
gate, select and hash the exact supported artifact and populate, separately for
each P100 `i`,
`W_i + KV_i(T,S) + recurrent_i(S) + activations_i(B,S) + intermediates_i(B,S)
+ workspace_i + startup_i <= 17,179,869,184 B`, where `T` is the actually
populated context, `S` the independent session count and `B` the active batch.
Every pipeline transfer must then be counted at its actual device boundary.
The current Qwen3.8-27B artifact is outside the two families reported by this
binary, so it cannot supply those measurements. A selected supported model,
exact-F16 KV, three-P100 load, populated context and concurrent real Pi tool
loops remain separate prerequisites.

## Magnitude Swift integration gate (2026-10-10)

The active request authorizes completing Swift execution in Magnitude on CUDA
12.4 and P100 devices 0/1/2. P40 remains excluded. The next dependency is explicit
IEEE binary16 history independent of BF16 activations. The named `dense-f16`
policy must derive F16 key/value descriptors, allocation charges, append/read
operations and tuning inputs from the same codec identity. Existing `dense`
continues to mean the activation dtype. No activation or weight conversion is
authorized by selecting a history policy.

For 16 target attention layers and one enabled attention MTP layer, each having
four KV heads of width 256, key and value storage is
`2 * (16 + 1) * T * 4 * 256 * 2 = 69,632*T B`. At T=262,144 this is
18,253,611,008 B. Without MTP it is 65,536*T B. Each appended token writes
69,632 B; a decode over all T previous tokens reads at least 69,632*T B before
any repeated head/partition reads. Dense BF16 and F16 have equal byte charges
but different payloads and semantic identities. A graph must accept its actual
history descriptors; equal byte counts do not make the types interchangeable.
BF16 projected inputs remain `B*N*2 B`, F32 residuals `B*5120*4 B`, F32 logits
`B*248320*4 B`, and the existing attention scratch/staging charges remain
separate from persistent history. Attention work is `393216*T FLOP/token`
for the target, in addition to 51,247,841,280 target matrix FLOP/token.

For physical device i, a future placement must satisfy
`W_i + 4096*T*n_attention_i + recurrent_i + activations_i + intermediates_i
+ workspace_i + startup_i <= 14,912,061,440 B` using the measured stable fit
capacity. The sum of attention counts includes an MTP layer only on its owning
device. This equation is a prerequisite, not an implemented placement. A
pipeline boundary transfers `B*5120*4 B` residual bytes; tensor-parallel ring
traffic is `4/3*B*N*4 B` per participant and reduced tensor. Actual transfer
counts and timing must come from the selected execution graph.

Completion of the codec dependency requires mixed BF16-activation/F16-history
append and read tests against an independent F32 attention reference, including
history after append, slab boundaries, fresh spans and empty spans. Completion
of the request still requires the missing exact GSQ decoders, per-device
placement and transfers, the original artifacts, and concurrent real Pi tool
loops. No kernel result alone establishes model readiness or tokens per second.

The mixed-history decode gate passes 1/8-row Swift attention cases. Its prefill
fixture exposes the next exact resource mismatch: width 256, two warps, two
stages and 32 keys require `(2*16*256 + 2*32*(256+256))*2 + (3+1)*16
= 81,984 B`, above P100's 49,152 B/block. A 16-key tile with register queries
requires `max(2*16*256, 2*16*(256+256))*2 + (3+1)*16 = 32,832 B`.
Make key tile size a native launch parameter; carry it through source geometry
and the declared shared bound. Preserve history values, softmax state and
matrix operand policy. The correction must pass the same independent reference
and byte-exact history checks before it is accepted.

## Magnitude native SM60 CUDA gate (2026-10-09)

The user rejected the Vulkan direction and requested adapting Magnitude to
existing CUDA libraries. Use the installed CUDA 12.4 NVRTC and matching builtins
through the existing `SEISMIC_NVRTC_DIRECTORY` development setting. P100 devices
0/1/2 are the only execution targets; P40 is excluded. New device support must
implement every reachable native helper used by Swift before lowering a device
floor. A profile must never advertise unavailable physical instructions.

The first acceptance gate is independent numerical execution of the native
CUDA helper ABI on SM60: narrow conversions, cooperative matrix fragment
loads/products, integer arithmetic and copy/barrier semantics. SIMT matrix
products retain the same decoded F16/BF16 operands and F32 accumulation;
there is no weight/KV re-quantization. This gate is a prerequisite to model
loading and cannot establish model throughput or Pi correctness.

Capacity remains per device: `W_i + KV_i + recurrent_i + activations_i +
intermediates_i + workspace_i + reserve_i <= capacity_i`. Measured CUDA
capacity is 17,059,545,088 B per P100; Magnitude's stable fit capacity is
14,912,061,440 B after its 2 GiB reserve. Original stored weights remain
9,596,985,344 B (GSQ) and 16,799,719,424 B (Heretic), with hot target traffic
8,841,271,296 and 15,821,244,416 B/token, respectively. Exact target/head
F16 KV costs `(65536 + 4096)*T B` (18,253,611,008 B at 262,144 positions);
recurrent reference state costs 158,859,264 B before checkpoints. Reference
hidden/MLP/logits F32 storage is `B*5120*4`, `B*17408*4`, `B*248320*4 B`.
Target matrix work is 51,247,841,280 FLOP/token, attention `393216*T` FLOP/token.
A hardware-independent 16x8x16 matrix helper computes 2,048 FMA operations per
warp invocation; its register exchange traffic must be measured separately
from global weight reads. Three-device F32 ring traffic is `4/3*B*N*4 B` per
participating device and reduced tensor. No device boundary is unified memory.

Model admission must still supply missing GSQ weight codecs, exact F16 KV,
individually fitting placements and measured transfers. Real Pi with independent
concurrent sessions remains the production gate; no short helper or direct
request can substitute for it.

### Native projection shared-memory prerequisite

The first real dense-projection gate rejects the fixed small-band declaration:
`3*(64*144 + 2*10240) + 2*2048 = 93,184 B`, while P100 reports
49,152 B/block. This is a verified resource mismatch, not a numerical failure.
The independent prefill-attention gate passes on P100.

New plan: make row tile and pipeline depth generic launch parameters, retaining
old 64/128-row multistage candidates while adding a 32-row single-stage candidate.
A single stage needs a load/completion/block-barrier/multiply sequence; no following
load may overwrite a tile before every consuming warp finishes. Preserve k order,
F32 accumulation and epilogue rounding. The worst per-stream codes/record charge is
`max(P*10240, P*6144 + 8192)` B: Q8 codes dominate with multiple stages,
Q6_K double-buffered coefficient records with one stage. With 8 warps and one
weight tile per warp, shared bound is
`P*R*144 + S*max(P*10240, P*6144 + 8192) + S*2048` B.
For `R=32, P=1`, this is 37,376 B for two streams and 20,992 B for one.
Grid row bands must be `ceil(M/R)`; declared and authored geometry must agree.
Weights, KV and activations stay unchanged. Workspace/scratch are separately charged
from these per-block shared bytes. Completion requires the native projection gate,
not merely recompilation; full model/Pi qualification remains separate.

## Magnitude Swift metadata admission (2026-10-09)

Active gate: load both original Swift GGUF files with Magnitude after discovery,
without aliases or changing stored weights. Both headers declare 65 blocks and
one trailing MTP block; target block count is `65 - 1 = 64`. Recurrent flags
may describe the 64 target blocks or all 65 stored blocks. The target has 48
recurrent and 16 attention blocks; the MTP suffix must declare attention.
Malformed lengths, non-Boolean flags and recurrent MTP are rejected.

Stored weights remain 9,596,985,344 B (GSQ) and 16,799,719,424 B (Heretic).
Hot target weight traffic remains 8,841,271,296 B and 15,821,244,416 B/token,
respectively, before additional activation and attention traffic. Target matrix
work is 51,247,841,280 FLOP/token and attention work is `393216*T` FLOP/token.
Exact F16 KV is `(65536 + 4096)*T B`, or 18,253,611,008 B at 262,144 positions.
Recurrent state is 158,859,264 B before checkpoints. Reference F32 hidden,
MLP and logits storage is `B*5120*4`, `B*17408*4` and `B*248320*4 B`.
Any Magnitude BF16 activation/intermediate policy is a separately reported
numerical configuration, not proof of the reference semantics. Three-device
ring traffic per device is `4/3*B*N*4 B` per reduced F32 tensor; each device
must fit its placed weights, KV, recurrent state, live activations,
intermediates and reserve within 17,179,869,184 B. P40 is excluded.

Header interpretation tests and real load requests qualify metadata admission
only. Generation, exact F16 KV, three-P100 execution and concurrent real Pi
qualification remain separate gates requiring measured execution.

## Swift Heretic throughput follow-up (2026-10-08)

The user deferred admission/cache work and requested generation throughput.
Keep the exact 16,799,719,424 B stored tensors, 15,821,244,416 B hot target
pass, 51,247,841,280 FLOP/token matrix work, F16 KV and three-P100 placement
below. The 262,144-position target/MTP pool still costs 17 GiB; each device
must separately fit placed weights, its KV shard, recurrent state, activation,
intermediates and CUDA/NCCL reserve. No P40, weight re-quantization, KV change,
or smaller production context is permitted.

For speculative depth d, measure accepted proposals A_d and complete verification
steps V_d. Mean emitted tokens per step are approximately 1 + A_d/V_d:
`t_token(d) = (t_target_verify(d+1,T) + t_draft(d,T) + t_catchup +
t_sampler + t_collectives + t_launch) / (1 + A_d/V_d)`.
Full attention remains `393216*T` FLOP per target token; hidden and MLP
activations are `B*5120*4` and `B*17408*4` bytes, logits `B*248320*4` bytes.
For three-device ring collectives, per-device traffic is `4/3*B*N*4` bytes
per reduced FP32 activation, plus measured additional transfers. Internal
AllReduce is an alternative exact FP32 reduction path, not shared memory.

Measured outcome: the original MTP-three profile produced 37.53 tok/s in
one fixed 2,917-prompt-token / 512-output-token diagnostic; MTP two produced
42.54 tok/s. Real Pi with actual instructions, skills and tools produced
989 output tokens in 24,391.78 ms (40.55 tok/s weighted), and two independent
Pi sessions produced 19.34-26.49 tok/s during overlap. All three sessions
passed the unchanged five-test queue gate. These short-context measurements
do not qualify throughput at 262,144 actually populated tokens. The selected
production profile now uses MTP two; target weights, F16 KV, 262,144-position
shared pool, four slots and three-P100 placement remain unchanged.

The opt-in SM60 CUDA-graph experiment measured 39.68 versus 39.58 tok/s
with the same internal-AllReduce configuration and byte-identical diagnostic
output. This 0.24% difference does not establish a gain. The experimental
source patch was removed from production and retained only under ignored
`work/speed-source/`. Thread-count and single-CTA NCCL experiments were not
selected. Native telemetry did not show hardware thermal throttling during
these comparisons. Sustained 50 tok/s remains unqualified.

## Separate exact MTP artifact placement: measured and not selected

An experimental source adapter retained the existing shared embedding/head/
norm and MTP tensors, preserving all 18 selected tensors byte-for-byte and
all metadata. The pinned loader already treats missing trunk tensors as
optional for an MTP-only artifact. The original target artifact was unchanged.

Exact duplicated payload is `715161600 + 1042944000 + 20480 + 263313408
= 2021439488 B`; the derived file is 2,032,385,824 B including metadata and
alignment. The draft context still holds `4096*262144 = 1073741824 B` of
F16 KV. Target weights and target KV stay sharded across three P100s.
Single-device draft placement avoids its tensor collectives; target hidden
rows still cross target -> host -> draft-device boundaries, costing
`B*5120*4 B` plus token/position arrays and any sequence-state transfers.
Observed residency was 15,283 / 12,217 / 12,471 MiB on devices 0/1/2;
P40 remained unallocated. The fixed diagnostic measured 42.85 versus
42.54 tok/s with identical output and acceptance counts. The 0.74%
difference is insufficient to justify the tighter device-0 memory margin.
This placement and artifact are inactive; production uses the original
embedded draft and normal tensor placement.

## Swift Heretic i1-Q4_K_M: same optimization gate

Active request (2026-10-08): stop GSQ, transfer its MTP-three/reduced-head
configuration to `swift-1.5-qwen3.8-27b-heretic-i1-q4-k-m`, start and test
through real Pi. The installed GGUF has SHA-256
`fc3b8413d6b1a707c40cb1df43d247a3de64237be7da15e3f04e345557cb3054`.
Its 866 tensors have the same inspected geometry as GSQ: 64 target layers,
one MTP layer, 5,120 hidden channels, and 248,320 tokens. Token strings,
merges, token types and EOS were directly verified against the pinned draft
subset's source. No weights, target vocabulary or F16 KV change is required.

Exact stored bytes: target MLP 10,361,241,600; target other 4,417,058,816;
embedding 715,161,600; target head 1,042,944,000; MTP 263,313,408.
Their sum is 16,799,719,424 B; the file is 16,810,715,936 B, including
10,996,512 B metadata/alignment. The optimistic hot target pass is
`W_hot = 16,799,719,424 - 263,313,408 - 715,161,600
= 15,821,244,416 B`. Geometry-derived target matrix work remains
51,247,841,280 FLOP/token. The exact F16 KV, recurrent state, activation,
intermediate, attention-work and three-device communication equations below
apply unchanged. Device capacity still includes placed weights, KV, live
intermediates, checkpoints and runtime reserves individually. The gate is
<=33.333 ms/token at ordinary populated Pi context, followed by independent
concurrent Pi tool sessions; a direct native request is diagnostic only.

The transferred configuration passed the single-session real Pi tool gate:
1,489 generated tokens in 37,339.55 ms (39.88 tok/s weighted), with four
turns reporting 33.65-43.97 tok/s. Pi loaded actual AGENTS.md and the
qualification skill, used read/write/bash tools, and passed all five queue
behavior tests. The direct 256-token diagnostic measured 39.20 tok/s.
Observed device residency was 13,383 / 13,143 / 13,377 MiB on P100 devices
0/1/2; P40 remained unallocated. Two concurrent independent real Pi
sessions also passed all five behavior tests, with actual instructions,
skill and tools enabled, without modifying tests or crossing assigned
directories. Overlapping turns reported 18.98-25.76 tok/s each; the last
turn after overlap reported 31.13 tok/s. The active service remains on
`swift-heretic` with all four slots available.

## Swift GSQ-RCO IQ2_S MTP: 30-50 tok/s gate

Active request: optimize the user's selected Swift artifact on CUDA devices
0,1,2, with the production Pi harness. The immediate comparison uses ordinary
Pi context (the observed failing request populated 3,004 prompt tokens).
The existing populated-262K and concurrent-session gates remain distinct;
50 tok/s at short context must not be claimed at maximum population or as
50 tok/s per concurrent session.

Artifact: `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`, revision
`d74895bbe5db4bec1e0024e7cc87d59c02d7631a`,
`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf`, SHA-256
`6ce61b532e45d7113e2b48b8dff9c104ff521230fa119eb0c3e75a44fb9f1cbf`.
The latest request authorizes MTP and runtime tuning for 30-50 tok/s. The
first candidate reduces maximum draft depth to three and reuses the existing
SHA-256-pinned coding/Romanian draft vocabulary after verifying identical
token strings, merges, token types and EOS. Source-specific padding metadata
is retained. The full target head, stored weights and F16 KV are unchanged. The artifact has 866 tensors, 64 target layers, one MTP
layer, 16 target full-attention layers, 48 recurrent layers, 5,120 hidden
channels, 24 query heads, four KV heads of dimension 256, and 248,320 tokens.

Exact stored tensor accounting from its GGUF directory:

```text
file including metadata/alignment             9,607,981,056 B
stored target MLP tensors                      5,613,731,840 B
stored target other tensors                    2,552,109,056 B
stored target embedding                          407,244,800 B
stored target vocabulary head                    675,430,400 B
stored MTP tensors                               348,469,248 B
all stored tensors                             9,596,985,344 B
metadata/alignment                                10,995,712 B
```

Capacity and traffic equations, with T actually populated positions:

```text
K_target(T) = T * 16 * 2(K,V) * 4 * 256 * 2 B = 65,536 T B
K_mtp(T)    = T *  1 * 2(K,V) * 4 * 256 * 2 B =  4,096 T B
K_target(262144) = 17,179,869,184 B; K_mtp = 1,073,741,824 B
R_target = 48 * (48*128*128 + 4*(2*16*128 + 48*128)) * 4 B
         = 158,859,264 B per recurrent state; checkpoints are additional
A_hidden(B) = B * 5120 * 4 B; I_MLP(B) = B * 17408 * 4 B per tensor
L_target(B) = B * 248320 * 4 B; L_draft(B,Vd) = B * Vd * 4 B
M_device(i) = W_placed(i) + K_placed(i) + R_placed(i)
            + A_live(i) + I_live(i) + checkpoints(i) + CUDA/NCCL(i)
            + allocator_reserve(i) <= 17,179,869,184 B
```

One optimistic weight-read pass excludes the embedding (a row lookup):
`W_hot_target = 9,596,985,344 - 348,469,248 - 407,244,800
= 8,841,271,296 B`. Attention adds `65,536 T B` and MTP adds its
weights, KV, vocabulary head and intermediate traffic on every proposal.
These are lower bounds; type conversion and buffer replication must be
charged from the runtime rather than treated as free capacity.

The target matrix multiply count excluding the embedding and MTP is
25,623,920,640 elements: `C_target_matmul = 2 * 25,623,920,640
= 51,247,841,280 FLOP/token`. Full-attention QK/PV adds
`C_attn(T) = 4 * 16 * 24 * 256 * T = 393,216 T FLOP/token`.
A draft matrix pass adds `2 * (424,673,280 + 5120*Vd)` FLOP, because
the draft reuses the target vocabulary head. GatedDeltaNet, norms, rotary
operations, sampling and recurrent checkpoints add their measured execution
cost; matrix FLOP counts are not a complete latency prediction.

For each distributed reduction j with Bj tokens, Nj output elements and
four-byte accumulation, its message is `Sj = Bj * Nj * 4 B`.
A three-device ring all-reduce transfers `2*(3-1)/3 * Sj = 4/3 Sj`
per device; graph collective counts, additional gathers, PCIe copies and
synchronizations must be included in the observed cycle. There is no P40 or
NVLink boundary in this placement.

The minimum acceptance inequality is `t_complete_cycle / mean_verified_tokens
<= 33.333 ms/token`; 50 tok/s requires <=20 ms/token. The observed Pi turn generated 62 tokens in 8,398.10 ms
(7.26 tok/s); MTP accepted 45/180 proposals, with mean verified length 3.50.
With a mean verified length of 3.50, the cycle must be <=116.67 ms for
30 tok/s or <=70 ms for 50 tok/s. Sampling was only 23.75 ms for the turn.

The complete depth-three/reduced-head correction passed the short-context
production gate on 2026-10-08: real Pi, xhigh, the actual repository AGENTS.md,
an enabled qualification skill, read/write/bash tools, and five independent
queue-behavior tests. Five model turns produced 1,500 tokens in 32,348.78 ms
of generation: 46.37 tok/s weighted; individual turns reported 37.56-51.21
tok/s at 3,264-5,267 populated prompt tokens. A direct greedy diagnostic
improved from 10.37 to 46.39 tok/s with identical 256-token output SHA-256.
Two concurrent independent Pi sessions also completed their tool loops and
passed the supplied tests, with approximately 19-31 tok/s during overlap.
This establishes the requested 30-50 tok/s for one active ordinary-context
Pi session, and bounded concurrent progress, not 30 tok/s per simultaneous
user or at populated 262K.

Sustained-load thermal qualification is separate: the 250 W ceiling and
1,328 MHz application clocks reached 79 C on the third P100 in
96.35 seconds. The test guard lowered power before the reported 82 C hardware
slowdown threshold. Further power/clock qualification was deferred at the user's request to
prioritize model speed. The temporary sustained-load test was stopped and
the 250 W / 1,328 MHz policy used by the passing Pi gate restored. Thermal
stability is not qualified by the short Pi runs.

# Feasibility gates

## RealVisXL V5.0 on the isolated P40

Status: artifact and visual gate passed.

The selected artifact is the author's standard-quality SDXL checkpoint, not
the distilled Lightning variant:

```text
repository             SG161222/RealVisXL_V5.0
revision               ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc
checkpoint             RealVisXL_V5.0_fp16.safetensors
checkpoint bytes       6,938,065,488 B = 6.4616 GiB
checkpoint SHA-256     6a35a7855770ae9820a3c931d4964c3817b6d9e3c6f9c4dabb5b3a94e5643b80
physical image device  Tesla P40, CUDA 3, 24 GiB
```

It has the same SDXL graph class and nearly the same stored size as the
qualified Animagine checkpoint. It passed the runtime gate through ComfyUI:
the completed artifact was discovered by link, the standard 50-step DPM++
2M/Karras workflow returned a visually coherent 1216x832 photorealistic PNG in
274.01 seconds, and Qwen remained isolated on the P100 domain.

The author's model card explicitly targets photorealistic SFW and NSFW output
and recommends either DPM++ SDE/Karras at 30 or more steps or DPM++ 2M/Karras
at 50 or more steps. The latter is selected because the existing workflow and
runtime already qualify that sampler family. CFG 5.0 remains a deployment
choice rather than an author-specified invariant.

## Native ComfyUI on the isolated P40

Status: implemented and passed the functional host gate on 2026-09-14.

ComfyUI is an alternative frontend/runtime for the two already qualified SDXL
checkpoints. It does not add a model copy and it does not change the model
capacity calculation: the existing checkpoint files are exposed to ComfyUI by
symbolic links. ComfyUI and `pascal-image.service` are mutually exclusive
because both own physical CUDA device 3. The text service may remain active on
physical devices 0,1,2.

```text
physical image device                 Tesla P40, CUDA 3, SM61, 24 GiB
largest existing checkpoint           7,105,348,188 B = 6.6174 GiB
FP32 UNet + text encoders + VAE       13,708,687,518 B = 12.7654 GiB
measured tiled sd.cpp peak            13,537 MiB
physical reserve at that peak         11,039 MiB
text-device traffic                   0 B/request by construction
checkpoint duplication                0 B; links target PASCAL_MODEL_ROOT
```

The measured `stable-diffusion.cpp` peak is evidence for model admission, not
a ComfyUI peak prediction. The ComfyUI gate therefore requires a complete
request and direct device telemetry rather than assuming equal allocator or
workspace behavior. VAE decoding remains tiled in the supplied SDXL workflow;
the untiled P40 path already failed at 24,187 MiB.

No compatible Python or PyTorch installation exists on the host: system Python
is 3.14 and Torch is absent. Native deployment is therefore limited to one
repository-managed CPython 3.12 environment and one PyTorch CUDA 12.6 install.
The official cu126 wheel reports a Pascal `sm_60` cubin rather than a separate
`sm_61` entry. Direct FP32 allocation, matrix multiplication and convolution
on the P40 SM61 all passed. The installer therefore fails closed unless the
visible device is exactly a Tesla P40 SM61, the wheel reports Pascal code, and
those CUDA operations execute with finite output.

ComfyUI v0.3.72 imports `requests` from its frontend manager but omits it from
that release's `requirements.txt`. The native installer closes this upstream
dependency explicitly and imports the production server module before systemd
may launch it. The unit also bounds repeated startup failures.

The acceptance gate is functional, not a new image-performance benchmark:

1. `pascal-comfyui.service` listens on `0.0.0.0:8188` with only physical CUDA
   device 3 visible and no process memory on devices 0,1,2;
2. ComfyUI discovers the existing Juggernaut and Animagine checkpoints through
   links, without copying their payloads;
3. the repository API-format SDXL workflow returns a valid PNG;
4. the Open WebUI container reaches the ComfyUI API through the host gateway;
5. stopping the service releases P40 process residency.

The real gate passed with ComfyUI v0.3.72, CPython 3.12.11 and PyTorch
2.7.1/cu126. Juggernaut produced a visually coherent 1216x832 PNG through the
repository workflow in 110.99 seconds while Qwen remained resident on P100
devices 0-2. ComfyUI discovered both existing checkpoints by link, its SQLite
state initialized at the explicit persistent path, and Open WebUI reached the
API through the Docker host gateway. This does not transfer the latency result
to Animagine or qualify sustained concurrent image requests.

## Juggernaut XL v9 on one P100

Status: admitted and passed one direct runtime/API qualification on 2026-09-14.

The selected artifact is the single-file SDXL checkpoint at Hugging Face
revision `cf419233522daa0b9ea36c3aff98fa2cab1fb0fb`:

```text
checkpoint                              7,105,348,188 B =   6.6174 GiB
one P100 physical VRAM                                     16.0000 GiB
capacity left for graphs/activations                        9.3826 GiB
UNet FP16 component                     5,135,149,760 B =   4.7825 GiB
two text encoders                       1,635,526,328 B =   1.5232 GiB
VAE FP32 component                        167,335,342 B =   0.1558 GiB
```

The file-size comparison is an admission bound, not peak-VRAM telemetry. A
successful gate still required the server to report a fully resident CUDA
placement, peak device use below the physical 16 GiB limit, and a valid image
without NaN/Inf or CPU parameter offload. The measured run met these gates:
6,624.11 MB of parameters were assigned to VRAM, 0 MB to RAM, peak device use
was 14,735 MiB, and the output was a valid 1216x832 RGB PNG.
Runtime inspection reported FP16 conditioner/diffusion weights and FP32 VAE
weights; those roles are kept distinct instead of calling the whole file FP16.

At the recommended 35 denoising steps, reading every UNet weight only once per
step plus one pass through the text encoders and VAE gives this optimistic hot
traffic floor:

```text
35 * 5,135,149,760 + 1,635,526,328 + 167,335,342
  = 181,533,103,270 B = 169.0659 GiB/request
nominal P100 HBM2 floor at 732 GB/s = 0.2480 s/request
```

This excludes convolution/matmul arithmetic, CFG batch work, intermediate
activations, sampling, VAE decode and image encoding, so it is not a latency
prediction. The checkpoint crosses NVMe, RAM and PCIe once during cold load;
there is no per-step PCIe weight transfer when `--backend cuda0 --auto-fit off`
succeeds. Unlike an LLM request, this pipeline has no retained context, KV
cache, recurrent state or MTP state.

The acceptance gate used one upstream `stable-diffusion.cpp` server on CUDA
device 0 and a `1216x832`, 35-step, CFG 5.0, DPM++ 2M Karras request through
`POST /v1/images/generations`. It returned HTTP 200 in 120.864 seconds;
sampling took 111.84 seconds and VAE decode took 7.19 seconds. This was the
first runtime request after service start, but the checkpoint had just been
downloaded and may have remained in the host page cache, so it is not claimed
as a physical cold-NVMe measurement. The text LLM service remained stopped
because it uses the same P100 domain.

## Animagine XL 4.0 Opt on one P100

Animagine XL 4.0 Opt uses the same complete-model, single-P100 execution
contract as Juggernaut. Its pinned single-file artifact passes the static
capacity prerequisite:

```text
Animagine checkpoint       6,938,350,040 B = 6.4618 GiB
remaining from 16 GiB                         9.5382 GiB
```

Using the same SDXL component sizes as the already inspected Juggernaut graph,
the optimistic parameter-traffic floor is:

```text
Animagine, 28 steps:
28 * 5,135,149,760 + 1,635,526,328 + 167,335,342
  = 145,587,054,950 B = 135.5885 GiB/request
```

This is an admission bound, not a prediction. The runtime gate passed with
zero parameter offload and zero residency on the other devices. Animagine
produced a valid 832x1216 image with its official 28-step, CFG 5, Euler
Ancestral preset in 170.187 seconds and peaked at 14,735 MiB.

## Isolated image generation on the P40

The text profiles use physical CUDA devices `0,1,2`. Image generation can run
concurrently only if its complete checkpoint, parameters and workspaces stay
on physical CUDA device `3`, the P40. The P40 has no peer path to the P100
domain, so this design performs no cross-device model split and transfers no
denoiser tensors between the services.

P40 does not provide the P100's fast FP16 path. The image profiles therefore
request FP32 execution explicitly. This increases precision rather than
reducing it. Using Juggernaut's inspected component inventory gives the static
capacity prerequisite:

```text
FP32 UNet + text encoders, FP32 VAE
  = 2 * (5,135,149,760 + 1,635,526,328) + 167,335,342
  = 13,708,687,518 B = 12.7654 GiB
conservative measured non-parameter peak allowance       8.4000 GiB
estimated resident peak                                 21.1654 GiB
P40 physical capacity                                   24.0000 GiB
estimated reserve                                        2.8346 GiB
```

The 8.4 GiB allowance is derived from the previous 14,735 MiB P100 peak minus
the reported resident parameter allocation and rounded upward. It proved too
low for the untiled VAE: denoising completed in 112.58 seconds, but decode
reached 24,187 MiB and failed allocation. The single corrective gate enables
upstream VAE tiling while keeping parameters and all compute on the same P40.
It kept the same checkpoint, prompt, dimensions, step count, sampler and seed;
required a valid RGB PNG without visible tile seams and zero process residency
on P100 devices `0,1,2`; and rejected parameter offload or an incomplete
request. The corrective gate passed. The runtime reported 12,230.64 MB of
parameters in P40 VRAM and 0 MB in RAM. Juggernaut completed in 189.574 seconds
and Animagine in 211.642 seconds; both peaked at 13,537 MiB in one-second
sampling and passed visual inspection. Qwen then started concurrently on the
three P100s while the image process remained resident on P40.

## Qwen3.8 feasibility gate

Status: the custom exact-component design fails the populated-262K 15 tok/s
prerequisite. The GGUF service is useful at ordinary context sizes. On
2026-09-17, pinned llama.cpp v0.4.0 plus the pinned SM60 patch set passed a
bounded real Pi read-tool flow at 40.07 tok/s on the initial/tool turn and
30.47 tok/s on the tool/final turn. The older populated-maximum gate passed
capacity but failed performance at 63.81 prompt tok/s, 4,106.2 s TTFT and
6.18 decode tok/s; it has not been rerun and the short-context improvement
must not be transferred to it.

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

## Deployed GGUF path

The production service does not integrate the rejected experimental kernels.
It uses the approved 16,464,440,224-byte `UD-Q4_K_M` GGUF in pinned
`llama.cpp` v0.4.0 with the pinned SM60 patch set, equally tensor-sharded over
three P100s:

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

The historical unpatched Pi tool flow measured 25.4-25.5 tok/s at a roughly
2.4K-token prompt. The deployed patched gate measured 40.07 and 30.47 tok/s on
its two bounded Pi turns. Two overlapping Pi sessions on the older runtime
measured 14.0-18.8 tok/s per active slot.
At 262,016 populated prompt tokens, cold prefill measured 4,105.894 s and
full-context decode measured 6.18 tok/s. Capacity is proven; maximum-context
performance is not. See `long-context-prefill.md` for the measured scaling and
the exact-prefix reuse decision.


## Native Magnitude GSQ codecs and physical placement prerequisite (2026-10-10)

The unchanged original GSQ artifact has 866 tensors. Seven missing source
formats account for 315 tensors and 6,733,721,600 stored bytes. Lossless import
uses four-bit palette indices with binary32 scales: IQ1_S group 32, IQ1_M group
16, IQ2_XXS group 32, IQ2_XS/IQ2_S group 16 and IQ3_XXS group 32. Q2_K
zero-extends its two-bit codes into four-bit storage and retains four-bit
(scale, minimum) fields per 16 and the original F16 factors per 256. There is
no weight requantization. The proposed conversion equals the installed GGML
C decoder bit for bit on 41,287,680 sampled values covering every affected
tensor and on 16,252,928 synthetic values per format covering all 63,488 finite
F16 scale payloads, including signed zero and subnormals. This qualifies the
representation arithmetic; it does not qualify the native import or matrix
kernels, nonfinite payloads, or a complete model.

For row length K and 16-byte plane alignment a(x)=16*ceil(x/16), a palette
row occupies a(K/2)+a(4*K/G) bytes, G=16 or 32. The Q2_K resident row occupies
a(K/2)+a(K/16)+a(4*K/256). Native CUDA mma16 additionally pads each matrix's
row count to 16 and code columns to 64; the artifact matrix dimensions already
satisfy those multiples. The seven missing formats occupy 14,616,330,240
resident bytes. Charging every original tensor once, all F32 parameters at
F32, without tied-role deductions or excluding host embeddings, gives the
conservative weight bound 18,858,921,984 bytes. The exact importer plan remains
a required cross-check because activation/role transformations can change
that charge. Per-tensor inventory is retained in
`out/magnitude-swift/gsq-resident-capacity.json`.

A candidate layer pipeline assigns target blocks 0..21, 22..42 and 43..63 to
P100 0, 1 and 2, respectively, and shared embedding/readout/MTP block 64 to
P100 0. It has 5/5/6 target attention layers; MTP adds one on P100 0. KV_i(T)
is therefore (6/5/6)*4096*T bytes, and recurrent state_i is
(17/16/15)*3,309,568 bytes. These are distinct allocations on three physical
cards. Each device must independently satisfy:

```
weights_i + KV_i(T) + recurrent_i + activations_i(B)
  + intermediates_i(B) + transfer_i(B) + import_peak_i <= 14,912,061,440 B
hidden boundary payload = B*5120*4 = 20,480*B B (F32 residual ABI)
three boundaries per target+MTP/readout pass: 0->1, 1->2, 2->0
transfer time = sum_i(latency_i + 20,480*B / measured_bandwidth_i)
```

The candidate per-device capacity, before activation/workspace/import charges:

| P100 | Weights B | F16 KV B at 262,144 | Recurrent B | Remaining B below stable fit |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7,822,084,992 | 6,442,450,944 | 56,262,656 | 591,262,848 |
| 1 | 5,483,724,800 | 5,368,709,120 | 52,953,088 | 4,006,674,432 |
| 2 | 5,553,112,192 | 6,442,450,944 | 49,643,520 | 2,866,854,784 |

Import peak and maximum populated KV need not coexist if import completes
before admission, but that phase distinction must be represented in the plan.
The matrix compute remains 51,247,841,280 FLOP/target token, full attention
393,216*T FLOP/token. Hot resident weight traffic must be derived from the
actual selected roles rather than the source-file byte count. A sequential
layer pipeline uses each device's bandwidth during its own layers; summing
three bandwidths is not a throughput prediction. Device transfer capabilities,
latency and full token time must be measured before reporting tok/s.

For generic projection staging, new group-16 F32 scales need a 64-byte record
per row per 256 values, compared with IQ4's 32 bytes. At PIPE=1 the existing
consumer barriers permit one record buffer; deeper pipelines require two.
Coefficient scratch must hold four groups per 64 rather than two. The declared
conservative shared bound for S projection streams is:

```
PIPE*ROW_TILE*144
  + S*max(PIPE*10240, PIPE*6144+8192,
          PIPE*4096+8192*min(PIPE,2)) + S*4096
ROW_TILE=32, PIPE=1, S=2 => 41,472 B < 49,152 B
```

Declaration, native buffer count/strides and tuning identity must agree. Native
imports must match the independent source decoder bit for bit; old and new
matrix formats must pass the original numerical gate before model execution.

The integrated host run uses an explicit MTP width of 10. The current source
caps chained draft-head transactions at 8; this is a planning/work bound, not
the stored head depth. Raising the generic work bound to 16 permits the
requested width without changing target verification or source weights. For
the selected gated-delta geometry, one additional tape row costs
`10240*2 + ((48+16)*128+48)*4 = 53,440 B` per recurrent layer and bank.
Width 8 to 10 therefore adds `2*48*53,440 = 5,130,240 B` per target bank;
the F32 delta state and per-token F16 attention history charges are unchanged.
Physical startup/growth claims must still charge the selected width, number
of banks and slab rounding rather than the generic maximum.

The actual CUDA load selected the recurrent scan's declared ROWS=128 default
before searching. At W=128 it needs
`3*(96*128+32*128+1680)+(128/16)*16*17*4 = 62,896 B`, beyond
P100's 49,152 B. The already numerically qualified ROWS=32 form needs
`3*(96*128+32*32+1680)+(32/16)*16*17*4 = 47,152 B`.
Making 32 the generic CUDA scan default preserves arithmetic and permits the
real load's initial measurement; larger forms remain search candidates on
devices with sufficient shared memory. This correction is tested through the
original-model load rather than another independent operator matrix.

Native-load capacity preflight uses the same metadata-only state derivation as
service preview, before opening the device or compiling/timing kernels. For
this artifact, exact F16 target plus MTP history is
`(16+1)*4*256*2*2*T = 69,632*T B`; at 262,144 tokens it is
18,253,611,008 B. This exceeds one P100's physical 17,059,545,088 B
even before weights, BF16 activations, F32 recurrent/intermediate storage and
reserve. The production path selects one physical device, so the three-card
capacity cannot satisfy this guard without explicit per-device placement.
Moving the unchanged derivation earlier changes only rejection order and
avoids preparing kernels for an inadmissible load; it does not reduce context,
change history precision or bypass a capacity check.

## Native Magnitude three-device execution gate (2026-10-10)

The active user request now explicitly requires native execution on all three
P100s. Physical placement is derived from semantic decoder-block weight roles
and the existing state descriptors. Every target block executes entirely on
its owner; entry/readout and the embedded head execute on the first selected
device. Boundaries transport exact F32 residual rows, `20,480*B` bytes for the
selected hidden width; no packed weight, F16 history or F32 bank is converted.

For MTP width 10, the actual per-layer recurrent bank is
`(3+10)*10240*2 + 48*128*128*4 + 10*((48+16)*128+48)*4`
`= 3,741,568 B`. A stateful request requires a pristine seed, its accepted
bank, its successor and (when lookahead is enabled) another successor. Four
banks cost `14,966,272 B` per recurrent layer before slab rounding. For the
earlier illustrative 17/16/15 recurrent-layer partition, these charges are
254,426,624 / 239,460,352 / 224,494,080 B. Placement must derive its actual
partition from the importer roles and descriptor sizes; these counts are not
runtime constants.

Each physical device independently satisfies:
`resident_weights_i + populated_F16_KV_i(T) + committed_F32_BF16_banks_i`
`+ graph_activations_i(B) + graph_intermediates_i(B)`
`+ transfer_buffers_i(B) + startup_peak_i <= stable_capacity_i`.
Weights, F16 KV, BF16 activations and F32 intermediates remain separate terms.
Startup and populated-context terms use their actual overlapping lifetimes.
A contiguous partition searches feasible cuts and minimizes the largest
per-device occupancy; a failed physical fit is not repaired by summing VRAM.
Every owner boundary is an explicit ordered transfer with exact payload bytes
and elapsed time recorded. Peer capability has already been observed for all
six directed links between the three P100s, with separate allocations.

Matrix work remains `51,247,841,280 FLOP/token`; full attention contributes
`393,216*T FLOP/token`. The sequential pipeline time is the sum of actual
per-stage execution times and boundary times; three memory bandwidths are not
added to predict its throughput. MTP acceptance and per-request throughput
remain measurements of the original model and real concurrent Pi tool loops.


The Heretic load failure triggered a typed-interface reassessment before
correction. `PreparedTargetBlockGraph::prepare` declares its hidden input as
F32, independently of the BF16 projection activation. Therefore each owned
receiver boundary must allocate `B*H*4` bytes; for H=5120, `20,480*B` bytes,
with three transfers totaling `61,440*B` bytes per full stage traversal. The
earlier BF16 boundary assumption is invalidated by the actual graph ABI.
Graph charge, receiver storage and direct peer copy must use this same F32
representation. No cast or numerical-policy change is admitted.


The next real Heretic gate rejected attention graph argument 14 before weight
import. Its manually declared dense history port still used the activation
representation. Dense kernel bundles and both checked/native graph entries
must carry their declared history element explicitly; generic graph ports use
that element for target, MTP and draft, independent of activation precision.
The per-device F16 capacity and F32 boundary equations above remain unchanged.
## Native Pascal throughput prerequisite (2026-10-10)

The active gate is original Swift generation in Magnitude on three P100s,
with unchanged weights, BF16 projection operands, F16 history and F32 residuals.
Startup residency alone does not pass this gate. Existing native observations
are 5.066839 tok/s for GSQ and 4.075757 tok/s for Heretic at MTP width ten.

The SM60 compatibility MMA computes a 16x8x16 fragment: 2048 F32 FMAs and
4096 lane-level operand shuffles per warp, independent of valid activation
rows. A single valid row needs 16x1x16 = 256 FMAs. Thus single-row GEMV has
an eightfold arithmetic padding factor before decoder and coefficient work.
The selected GEMV uses NB=ceil(M/8); its padded matrix work scales as
16*(8*NB)*K against useful 16*M*K. This is a source-derived work count,
not a measured bottleneck verdict. BF16 remains exact in F32; converting it
unconditionally to F16 would be a fidelity change and is excluded.

For one speculative round, time is
`t_round = t_draft_chain(P) + t_target_verify(P+1) + t_sample + t_host`,
and emitted throughput is `sum(accepted+1)/sum(t_round)`, with EOS truncation
accounted separately. Ten maximum proposals do not imply ten accepted tokens.
Per-device CUDA event attribution on a bounded real generation will identify
the dominant operator before selecting an optimization. Diagnostic per-launch
events alter submission work; their attributed times are not production tok/s.
Physical stage capacity and the exact 20,480*B-byte F32 boundaries retain the
independent-device equations above; transfers and kernel times are separate.


The real 64-token GSQ attribution recorded 11.729617s of GPU work including
prefill: dense_expand 5.031804s, dense_output 1.999856s, gated_delta_project
1.272364s, attention_output 0.762270s, attention_project 0.590198s,
head_logits_rows 0.425325s and readout_head_rows 0.256720s. Those projection
families sum to 10.338531s (88.14%); FFN alone is 7.031660s (59.95%).
The diagnostic response is coherent and all original numerical settings hold.

Selected change: native SIMT GEMV over the existing exact resident packets.
Four lanes cooperate on each output row, evaluating only actual activation
rows M. Each lane computes its four interleaved K elements per 16-code step;
a fixed four-lane reduction combines that group before the existing F32
scale/bias fold. The temporary accumulators remain the existing C-fragment
layout, with the same cross-K-part reduction and epilogue. Work per warp is
16*M*K FMAs instead of 16*(8*ceil(M/8))*K. Decoders, scales, BF16 operands,
F32 accumulation, F16 history and physical transfers retain their policies.
Every M uses the same association for each row, preserving row/batch
invariance. The existing independent numerical bounds must qualify the new
association; there is no lower-precision accumulator or weight conversion.
Production tok/s must be measured with attribution disabled.


The selected MTP optimization keeps the configured maximum at ten and its
physical state capacity unchanged. An opt-in per-session acceptance controller
starts at min(maximum,3), grows to the next efficient verification window after
three fully accepted rounds, and contracts to the smallest power-of-two
verification window covering an observed rejected prefix. Its draft width is
always bounded by the configured maximum and the remaining output budget.
All proposals still undergo the unchanged target acceptance and sampling path;
no vocabulary subset, target sampling shortcut or precision change is used.
The controller is request-local; checkpoints retain numerical method state
and restoration resets only this performance hint. Fixed width remains the
default unless MAGNITUDE_ADAPTIVE_MTP=1 is selected for the actual run.


Prefill retains matrix multiplication. The selected SM60 MMA-helper scheduling
change shares its two A-row operands and two B-column operands among four
outputs at each ascending K. Each output retains exactly the previous sixteen
F32 RN FMAs in the same order. Shuffles fall from 128 to 64 per lane per
fragment, and operand widening is reused, without changing operand bits,
accumulators or precision. This helper is independently checked against the
existing scalar fmaf oracle; its performance effect is qualified only by the
original model run. No parameter association or numerical tolerance changes.

The 01–21 candidate's production GSQ request measured 7.428979 tok/s.
Renewed installed-CUDA attribution includes 10.092834s of GPU work for the
64-token request plus prefill/warm-up: projections contribute 8.887360s
(88.06%), including dense expansion 3.879788s and output 2.060758s.
The next local correction removes the packed 16-bit round trip solely from
SIMT code operands. Packed-weight codes and codebook entries are integers
in [-128,127], exactly representable in BF16, F16 and F32. Therefore direct
F32 decoding yields the identical widened code bits; scale/bias F32 folds,
group and K-part association, weights and all activation/history policies
stay unchanged. Dense operands still round to the declared activation
element before widening. Four-lane SIMT evaluates 16*M*K FMAs per warp as
before; eliminated work is code conversion, rounding, packing and unpacking,
not useful arithmetic or model values. An independent GPU operand-bit gate
and the original real request qualify this correction.

The operand identity gate exposed an existing F16 signed-byte decoder error:
its first permute places the high operand byte at position two, but the next
permute selects position one (`0x4140`). A native GPU reproducer with low=1,
high=0 returns BF16 (1,0), old F16 (1,1), and direct SIMT (1,0). Correcting
the selector to `0x4240` restores the source integer pair exactly. The active
BF16 projection path and F16 KV representation are unaffected. The same
exhaustive integer oracle must pass without modified expectations or bounds.

The second operand gate passes both exhaustive integer references but rejects
a Dense<Bf16> payload: native widening retains `0x7fe10000`, while an extra
BF16 rounding canonicalizes it to `0x7fff0000`. This invalidates the premise
that every dense operand can use round-then-widen. Editing stopped for a
typed-interface reassessment. The renewed plan preserves three separate
source cases: quantized exact integer codes decode directly; dense weights
already in the activation element widen their native bits directly; other
dense source elements retain the old activation rounding. A compile-time
operand/source trait selects these cases, without model-specific branches.
The same exact GPU gate must cover BF16 and F16 output bits across twelve
decoder cases, including raw dense NaN payloads. No NaN exception or numerical
tolerance relaxation is admitted.

The 01–24 real GSQ request reached 8.200016 tok/s; all 192 output tokens,
finish reason and MTP acceptance (156 proposed, 101 accepted) match 01–21.
Actual CUDA reflection reports dense-expand GEMV 206 registers per thread,
256 threads per block, no spills. At least 52736 registers per block prevent
two blocks fitting the 65536-register P100 SM; active warps are eight out of
64. This remains a failed throughput gate, despite exact operand identity.

Selected corrective change: stream one k-block instead of retaining current
and next four-block superblocks for both weight streams. Codes come from each
decoder's existing fetch(tile,kblock,lane); group scale/bias comes from its
existing coefficient(row,kblock,group) interface. Both address the same
resident planes and fields. K-part boundaries remain the old superblock
partition, and k-block/group/column F32 arithmetic order is unchanged. The
existing C-fragment reduction and epilogue remain. Thus traffic in useful
resident codes and coefficients, 16*M*K FMAs, independent device capacities,
F32 boundaries and F16 history equations remain. Retained raw/code/scale
registers and source-level redundant prefetches decrease; occupancy and real
tok/s must be measured. No lower precision, changed codes or decoder widening
is admitted. Weight-plane row geometry is already literal when the packing
axis is static (native/abi.rs); a speculative dynamic-division optimization
is not selected for those planes.

## Renewed end-to-end Pascal throughput plan after 01–25

The 01–25 original GSQ request measured12.178974729tok/s, with exact192-token
choices and156/101 MTP counters matching01–24. The corrective performance
gate still fails, so editing stopped for renewed real-model attribution.
GPU diagnostics total6.726131878s including warm-up and36-token prefill:
projections82.9401%, recurrent step12.7985%. Main GEMV means are expansion
0.950051ms, down projection0.455179ms, recurrent projection0.414320ms and
recurrent step0.597810ms. The selected path must address projection efficiency
and recurrent addressing; eliminating recurrence alone has ideal Amdahl
limit1/(1-0.127985)=1.14677, so it cannot meet the throughput objective alone.
Existing installed nvprof counters on the real pure expansion kernel will
separate instruction efficiency and device-memory throughput before selecting
the next projection change. No new CUDA libraries or driver changes are used.

The first implementation in this renewed plan is the exact single-row slab
address shortcut. A target recurrent bank occupies179595264B, exceeding the
67108864B slab target; banks_per_slab=max(1,floor(target/bank_bytes))=1.
The original address is table[row_index/rows_per_slab]+(row_index %
rows_per_slab)*row_bytes. At rows_per_slab=1 this equals table[row_index]
exactly for every valid index and every row_bytes. The shortcut removes the
runtime64-bit divide/remainder and zero-offset multiply; all other slab
sizes retain the existing path. Actual cached step disassembly contains3169
static instructions, including1185 in four primary XMAD variants and17 CAL
sites; this is evidence of addressing cost, not a dynamic instruction count.

Weights, exact F16 KV, BF16 activation bits, F32 recurrent/intermediate values,
per-device state capacity and F32 peer-edge bytes are unchanged. Address-table
identity is independently checked on installed CUDA, then the original192
request must retain exact choices and MTP counters. Production throughput is
measured with all diagnostics off. This component is not claimed to meet the
complete throughput objective by itself.

## Shared-activation paired SIMT projection plan after 01–26

The exact single-row slab shortcut reached 13.350953387 tok/s on the original
192-token GSQ request, retaining every choice and the 156/101 proposal/accept
counters. Lookahead is already enabled by the production policy and plain
successors require a single decode row; enabling a CLI flag adds no new path.
The next selected implementation targets paired projection instruction work.
For equal representation-group widths G, a tile evaluates two weight streams
and M activation rows with 2 * 16 * M * K F32 FMAs. Existing execution reads
each activation operand twice and separately forms bias sums for the two
streams. A shared group traversal reads each x once, maintains distinct
p_gate and p_up with the identical per-lane FMA order, and computes the same
fixed four-lane XOR reduction (masks 1, 2) for each product. An activation sum
is shared when either stream has bias. Scale/bias folds, ascending k-block
and group order, original K-part boundaries and final part reduction remain
identical. Unequal group widths retain the original independent traversal.

Useful weight traffic, resident layouts, weights_i and exact original GGUFs
are unchanged. Activation reads per matched pair reduce from 2 * 16 * M * K
16-bit reads to 16 * M * K (L1/shared reuse affects physical traffic); bias
sum additions reduce from two copies to one where both streams have bias.
Registers retain both raw packets and group operands, so occupancy can
decrease: measured resource usage and actual whole-model tok/s determine
acceptance. Fixed one/two-row specializations bound loop overhead while all
other real row counts keep the same traversal and independent accumulators.
All physical capacity, F16 history, BF16 activation and F32 intermediate/peer
transfer equations remain unchanged. Exact native fragment comparisons cover
source formats and mixed pairs before the same actual model request. No
performance result is inferred from the component gate.

## Extended native selection budget after the failed 01–27 correction

The warm correction measured 13.918248361 tok/s with unchanged choices and
156/101 MTP counters. It did not meet the throughput gate. Source inspection
established that the 60-second global census/search clock includes NVRTC
formation; a factored search forms every launch variant before its first
sweep. Observed individual formation times of 47–95 seconds therefore exceed
the unit's 0–0.37-second share. The selected change is an explicit offline
`MAGNITUDE_EXTENDED_TUNING=1` preparation policy, independent of model family.

Let C_i be formation time, E_i be measurement/validation/input time, s_i be
the measured step share and U the sum of unsearched shares. Give each census
up to 60 seconds, then start a separate 600-second search budget per device.
The unit allowance is B_i = max(0, 600 - sum E_j) * s_i/U. The factored
window is extended by its initial all-variant formation C_i; subsequent
units subtract formation from the global elapsed clock. Thus an NVRTC cache
miss does not consume the kernel measurement allowance. Validation, launch
domains, numerical limits, references and served shape classes are unchanged.
Default-only incomplete stored searches are reconsidered under this opt-in
policy; measured alternatives remain reusable. Compilation still contributes
to real startup wall time and is reported separately, never to decode rate.

This changes preparation and kernel selection, not weight/KV/activation
formats or capacity. The existing per-device capacity inequalities, exact
GGUFs, F16 KV, BF16 activations, F32 intermediates and explicit peer traffic
remain unchanged. Acceptance requires measured native choices followed by
the same actual request and real concurrent Pi sessions. Merely completing
selection is not acceptance of the throughput objective.

## Single-row normalization staging correction after 01–28

Extended selection reached only 14.006926902 tok/s on the real 192-token
request. This fails the active speed gate. The one selected corrective change
forms a staged BF16 normalization row for M=1 using the same existing
`form_row`, 256-thread reduction, F32 operations and BF16 rounding as the
current per-GEMV-block form. All six CUDA GemvSource consumers and their
declarations must move together, including dependency waits in the first
attention/recurrent projection. Other backends are unaffected.

For output width N and eight warps grouped by configured split S, C =
ceil(ceil(N/16)/(8/S)) GEMV blocks currently each normalize all H inputs.
For H=5120, N=17408 and S=4, C=544. The correction reduces normalization
from C copies to one, adds one staging launch and a 2H=10240-byte scratch
row, and removes that 2H-byte dynamic shared-memory row from each GEMV CTA.
Normalization input traffic has useful-byte bounds C*(4H+2H) before and
4H+2H+2H after staging (F32 residual, BF16 norm, BF16 staged write).
Projections then read the identical 2H-byte BF16 row from global scratch,
with cache reuse; these are useful-byte equations, not measured DRAM traffic.
Normalization arithmetic reduces from approximately 4HC FLOPs plus C block
reductions to 4H plus one block reduction. The projection's K partitions,
per-lane FMA and scale/bias fold order remain unchanged for each configuration.

For the observed fused expansion's 80 registers/thread and 256-thread CTA,
the 65536-register SM budget permits three CTAs. A CTA using 8336 static
bytes plus 10240 dynamic bytes admits only two under the 49152-byte shared
limit; removing the row permits three, subject to the other actual limits.
No occupancy-to-throughput gain is assumed. The additional scratch row is
charged to its physical owner's graph plan; weights, exact F16 KV, recurrent
state and F32 peer transfer equations remain unchanged. Acceptance is the
real original-model request and concurrent Pi with diagnostics disabled.

The extended search changed the actual token sequence and MTP counters
(153/99 versus 156/101), even though candidates passed the existing bounded
numerical policy. No unchanged-quality conclusion is drawn from that result.
The correction therefore offers `MAGNITUDE_NATIVE_REFERENCE=1`: prepare each
authored native default directly, ignoring stored/search alternatives. This
keeps default arithmetic choices (including K-part boundaries and 16-bit
activation paths) while evaluating the exact staging change. It is a generic
preparation policy, independent of model identity. Existing kernel precision
limits and the model artifacts are not relaxed or rewritten.

### Native Pascal projection preparation after the second 28/29 speed failure

The actual 29 trace records 18 verification batches of M=2 and 11 of M=4, with a 36-token prefill. Projection CUDA events occupy approximately 83% of 6.449172430 seconds including warm-up/prefill; no steady-state throughput is inferred from attribution. Selected next implementation: compile separate native GEMV launches for the actual M=2/M=4 row classes, plus lossless hierarchical IQ2 coefficient storage. Other row classes and all original GGUF weights remain supported.

For IQ2, original F16 d and local integer s produce c=(d*(2*s+1))*2^-3. Finite binary16 d has at most 11 significand bits, odd<=31 has at most 5, so each intermediate product is exactly representable in binary32 (<=16 significant bits); exponent range also fits. Store f=d*2^-3 as F32 once per 256 values and q=2*s+1 as u8 per original group, then recover c=f*q exactly. Signed zero and nonfinite payloads require the actual import/reference gate, including canonical NaN rules; no proof from finite values covers them. Code palettes, F32 product order, F16 KV and BF16 activations stay unchanged.

Per 256 resident values, g32 bytes change from 128+8*4=160 to 128+8+4=140; g16 from 128+16*4=192 to 128+16+4=148. Rows align each plane to 16 bytes and pad N to 16; all actual matrix K are checked against alignment, so weights_i=sum(canonical_byte_len(shape_i)) remains the admission authority. For the inspected GSQ IQ2 tensors, old bytes=9052487680, proposed unpadded bytes=7257169920, saving=1795317760. Other weights are charged unchanged. Device_i peak=weights_i+F16_history_i(T)+recurrent_i+graph_i(B)+intermediate_i(B)+startup_overlap_i+transfer_i <=14912061440, independently on each P100. Dense KV reference remains 69632*T bytes across target/head owners, never unified memory.

Useful streamed weight traffic for these IQ2 tensors changes by the same old/new byte terms per full traversal; cache reuse and KSPLIT replication require actual measurement. Decode work adds one exact F32 factor*integer per original coefficient group while reducing live coefficient packet words from 16/32 to 6/10 per lane pair for g32/g16. Matrix FLOPs remain 51247841280 per token, attention 393216*T, and unchanged MTP verification evaluates every candidate through the target. Dedicated M=2/M=4 launches retain the same KSPLIT=4, within-lane FMAs and mask1/mask2 reductions; they expose the actual row bound to compilation rather than changing arithmetic.

The current device schedule remains three contiguous stages, so all three boundaries still transfer 20480*B F32 bytes per target traversal, with existing observed 41–62 microseconds per edge/row. This candidate does not claim concurrent tensor parallel execution. Selected code must pass exact source import and the original-model throughput gate before any Pi/concurrency speed claim.

### Completed CUDA source-import gate after patch 30

The second patch-30 failure is decisive: registry conversion exists, but actual CUDA `repack_weight.cu` rejects IQ2H bindings before importing bytes. Stop edits and reassess source→registry→native import→resident decoder. The missing source adapter has three required paths: accept original IQ2_XXS→IQ2H32 and IQ2_XS/S→IQ2H16; write unsigned odd coefficients and one canonical F32 factor; write both fields in packet, rows16 and mma16 layouts with original zero-padding rules. No runtime model/family branch is needed.

New plan 31 completes that adapter before restarting original-model qualification. Source traffic remains P*66/74/82 bytes (P=ceil(K/256)*N*B packets); resident writes are P*140 for H32 and P*148 for H16 before canonical plane/row/tile alignment. Code decode work and matrix FLOPs remain unchanged; import adds P*(256/group) byte coefficient writes plus P F16→F32 conversions and exact multiplication by 2^-3. Peak_i includes this source upload concurrently with destination_i, the earlier per-device 14912061440-byte limit and all history/state/graph terms. This source adapter does not add inference device transfers; stage boundaries remain three edges with 20480*B bytes each. Test the actual source adapter against registered physical bytes and the original independent GGML fixtures, then load the original model and measure real decoding.

### Native attention preparation: exclude statically empty segments

Patch 31 passes original-byte/import gates but its actual GSQ preparation spends over 20 minutes compiling three attention GEMV launches. A four-second CPU sample of the owned loader locates the hot work in installed NVRTC 12.4.127. Captured production source has (Q, GR, K, V)=(12288, 0, 1024, 1024). An isolated original M2 source does not finish within 20 seconds; excluding the statically empty projection branches compiles the otherwise identical source in 0.545515292 seconds. This is preparation evidence, not a decoding rate.

The corrective plan excludes a segment j at template compilation only when its static N_j=0, and removes the GEMV default's zero-row invocation. Its original tile count ceil(N_j/16)=0 produces no arithmetic, stores or cross-part barrier, so suppression retains the dependency prologue and all nonempty segment operations. Apply the same static guards to GEMV and GEMM segment dispatch; retain ordinal launch selection, weights, KV, activations and F32 association.

Capacity terms remain exactly weights_i + 69632*T apportioned by target/head owner + recurrent_i + graph_i(B) + intermediate_i(B) + startup_overlap_i + transfer_i <=14912061440 per P100. Useful matrix traffic remains sum_j(N_j*K*resident_bytes_per_value_j) and 2*K*B staged BF16 bytes per projection; zero segments contribute zero. Matrix work is 2*B*K*sum_j(N_j), plus unchanged normalization and epilogues. The three physical stage transfers remain 20480*B bytes each, with no new communication. Compilation removes formation work for unreachable matrix bodies; no occupancy or throughput gain is assumed. Rebuild, load the original model on three P100 and execute its actual 192-token request before further speed decisions.

### Native Pascal row-parallel projection plan after 31/32 convergence

The actual corrected model still measures 13.663491064564354 tok/s, with unchanged 192-token choices/usage and 156/101 MTP counters. Resident IQ2 traffic reduction and preparation fixes did not meet the speed gate. Stop the 31/32 change loop. The new selected plan changes the small-batch projection execution, rather than repeating selection/cache work: one independent CUDA grid-y lane per actual activation row for the measured M=2/M=4 verification classes. Each block evaluates one row through the existing four-lane group product, mask1/mask2 reduction, KSPLIT=4 boundary and F32 scale/bias fold; source and epilogue adapters shift the logical row index only.

For output width N, K inputs and B in {2,4}, CTA count changes from ceil(ceil(N/16)/(8/KSPLIT)) to B times that count. Matrix arithmetic remains 2*B*K*N FLOPs; input/output bytes remain 2*B*K and element_bytes*B*N respectively. Original fused product code decoding is performed once per weight k-block and reused by B columns; the row-parallel path can repeat that decoding B times. Useful weight read upper bound becomes B*W_native; actual L2/DRAM reuse must be measured. This deliberately exchanges additional weight/cache traffic and code decoding for independent scheduling and a one-row live accumulator, addressing the observed 56–88 registers/thread and serial column loops. For 256-thread CTAs, resident register capacity is floor(65536/(256*R)) CTAs before other limits; no throughput gain follows from that equation alone.

The same static shared reduction allocation remains per CTA, with no new global scratch, weights, history or recurrent state. Staged normalized BF16 rows are produced once for the full B-row batch. Per-device capacity and all actual 32 weight charges remain unchanged, including exact F16 KV and F32 intermediates. The three contiguous device stages and 20480*B-byte boundaries remain unchanged; this is row scheduling within native projections, not concurrent tensor parallel execution. Session leases retain ownership of all inputs/outputs. Apply the row view, epilogue offset, grid-y geometry, zero-fill offset and all six small-batch projection entries together. Qualify exact independent row mapping, then measure the original GSQ and Heretic through the actual runtime; concurrent real Pi remains the production gate.

### Sole row-parallel correction: interleave logical rows per weight tile

The first actual row-parallel request preserves choices/usage but falls to 5.678221135430781 tok/s. CUDA's x-fastest launch coordinates currently group the entire output width for each activation row; each row can revisit a matrix after its earlier weight tiles have left the 4 MiB P100 L2. The corrective geometry launches (B, output_tile_groups, 1), maps row=blockIdx.x and tile_group=blockIdx.y*GROUPS+group. All six small-batch families, both row classes, shared tile selection and attention zero-fill mapping must change together. Numerical operations and row association remain unchanged.

CTA count is still B*ceil(ceil(N/16)/(8/KSPLIT)); arithmetic remains 2*B*K*N FLOPs, normalized input bytes 2*B*K and output bytes element_bytes*B*N. Requested weight traffic remains bounded above by B*W_native; adjacent row CTAs can reuse the same weight tile through L2 but neither scheduling order nor a reduction in measured DRAM traffic is guaranteed. Weight, scratch, graph, recurrent and F16 history charges are unchanged on every card. Three contiguous stage boundaries still transfer 20480*B bytes each. Test exact physical grid/row coverage and source/epilogue mapping, then the original actual 192-token request. A second speed failure ends this change loop for end-to-end reassessment.

### Reassessment after the second row-parallel speed failure

Patch 34 measures 10.542328842549137 tok/s with exact original choices; the row-parallel loop is closed. An authored native attention projection on P100, using its actual 12288/0/1024/1024-by-5120 bindings and synthetic resident bytes, independently verifies the regression: fused rows take 0.220 ms (B=2) and 0.330 ms (B=4), while interleaved independent rows take 0.325/0.600 ms. Results match bit for bit. This diagnostic is not original-model throughput qualification.

The new selected plan restores fused M2/M4 launches and optimizes existing lossless resident decode: read all IQ2H local coefficients for a k-block in one aligned word per row, and widen small code bytes through exact F32 mantissa construction instead of integer-to-float conversion. Gate the complete native projection first; adopt only an exact variant with an observed device-event gain before rebuilding the full runtime. Weight and coefficient ABI, group arithmetic, per-part reduction, F16 history, BF16 activations and F32 intermediates remain unchanged.

For G in {16,32}, the logical coefficient count per row per 64-value k-block is L=64/G. Old authored scalar metadata requests are L unsigned-byte reads plus a shared F32 factor; new requests are one aligned 4-byte local word plus the same factor. The G=32 word shares two consecutive k-block records; it stays within the padded original 256-value group. Per two-row lane fragment, issued local read instructions change from 2L to 2, with physical read bytes bounded by 8 plus factor bytes; transaction/cache reuse require measurement. Decoded scales remain canonical_F32_factor * original_odd_byte. Small unsigned u uses float_bits(0x4b000000|u)-8388608, and signed byte s uses float_bits(0x4b000000|(byte(s)^128))-8388736; all intermediate integers are exactly representable, including zero. Code arithmetic adds one exact F32 subtraction per code while replacing I2F conversion; no throughput gain is assumed without actual events.

Matrix work remains 2*B*K*N, source/staged/output byte counts and useful weight traversal W_native stay at the earlier fused values. Resident/startup capacities remain separately bounded by 14912061440 bytes on each P100. No buffer, import, KV, activation or device boundary changes are required. Three contiguous physical stages retain 20480*B-byte transfers each. Exact source-adapter gates remain valid; native output and original-model gates must be rerun.

### Projection decode direction rejected; qualify native CTA geometry

The exact diagnostic rejects the proposed mantissa widening: fused B=2 increases from 0.2215 to 0.2729 ms; combined metadata/widening gives 0.2745 ms. Bulk metadata alone is approximately unchanged at B=2 and reduces B=4 by only about 4%; it does not justify a full runtime adoption. No inference source change is applied. Close this direction and reassess occupancy/CTA grouping at fixed KSPLIT=4.

Next selected candidate controls warp count WARPS at fixed TPW=1 and the original K partition. GROUPS=WARPS/KSPLIT and grid_x=ceil(output_tile_groups/GROUPS), threads=32*WARPS. Vary WARPS in {4,8,16}: work remains 2*B*K*N, logical traffic W_native and original K partitions/reduction unchanged. Register-limited active CTAs are floor(65536/(32*WARPS*R)); shared reduction storage scales linearly with GROUPS, and active-warp limits still apply. These are occupancy bounds, not a predicted speed. No new physical buffer/import/history/intermediate allocation or cross-card transfer is introduced. Validate exact native outputs and event timings before selecting a complete kernel/launch parameter change.

### Keep fused rows and expose independent accumulator chains

The focused CTA sweep does not justify a runtime change: four warps regress fused B=2 from 0.2215 to 0.2502 ms; sixteen warps also regress B=2 to 0.2500 ms and are approximately equal at B=4. Stop that direction. The next complete candidate interleaves only independent activation columns within a warp for B=2/4, decoding one 16-value step at a time. Every individual product retains ascending j/half/x-y FMAs, mask1 then mask2, original group factor/bias and K-part fold. Other batch classes keep their old path.

Useful bytes, matrix work 2*B*K*N, physical weights, shared reduction bytes and device transfers are unchanged. Per-thread temporary product storage is 2*B floats for an unpaired projection or 4*B floats for a paired projection, with B bias sums; decode storage changes from 8*(GROUP/16) floats per stream to 8. This exposes independent FMA chains while reducing retained decoded steps. Register occupancy and instruction count must be measured; no output approximation or speed prediction is implied. Full original-model adoption requires exact fragment and native output gates plus an observed projection-event gain.

### Measure one-k-block lookahead instead of larger live superblocks

Interleaved-column diagnostics preserve exact F32 pair outputs but regress paired B=2 from 0.4331 to 0.5238 ms, while B=4 improves only from 0.4415 to 0.4061 ms. This does not justify wholesale adoption. The new selected candidate overlaps the next native 64-value k-block's raw/metadata reads with the current block's exact computation. It retains the existing B-column loops, all fixed four-lane arithmetic and original K-part partition.

Lookahead retains one extra Raw per stream plus 2*GROUPS scale values and applicable biases per stream. For IQ2H, Raw is 16 bytes per lane/stream, compared with the prior full-superblock four-raw staging. No new global or shared allocation is needed; capacity remains unchanged. Useful traffic W_native and per-traversal arithmetic stay exactly the same; each k-block is fetched once, including the first and last. Per-warp outstanding reads can increase, with a potentially lower register occupancy. Measure exact native F32 outputs, resource attributes and device-event time before adoption. Stage ownership and 20480*B-byte device transfers remain unchanged.

### Selected complete native projection schedule (candidate 35)

The selected native probe preserves exact F32 pair outputs: same-group B=2 uses one-k-block lookahead (0.4331 to 0.3728 ms); same-group B=4 uses interleaved columns (0.4415 to 0.4077 ms). Mixed-group B=2 uses interleaved single-stream columns and measures 0.337 ms versus 0.5149 ms; mixed B=4 keeps the prior fused path. These are synthetic resident-byte diagnostics, not model throughput. Restore the original fused grid/row scheduling across all six families and twelve M2/M4 entries, remove the rejected independent-row adapters, and apply the selected arithmetic schedule in the shared native implementation.

Lookahead is admitted only for paired, nondense equal-group formats at B=2; other classes keep one-k-block streaming. Interleaving admits unpaired B=2 and same-group paired B=4. All paths retain original FMAs, group sums, shuffle order, factor/bias fold and K-part reduction; there is no model/family branch. Capacity and communication terms are exactly those above: no global buffers, physical coefficient/codes, history, activations, intermediate formats or stage edges change. Validate exact fragment bits across registered formats, then rebuild and measure the original model before any concurrent Pi speed claim.

### New end-to-end plan 36: concurrently shard projection output rows

Candidate 35 improves original GSQ to 14.385283143632181 tok/s but the requested rate is unmet. Preserve its exact measured arithmetic as the base; stop minor serial-projection changes. The selected next complete direction partitions independent output tiles among three P100 and forks their native graph nodes before joining the downstream stateful operation. This preserves full K dot products and avoids cross-device summation, so each output's arithmetic stays unchanged. Read-only driver queries establish VMM support on all three cards; multi-context host graph behavior and exact row coverage still require actual host verification.

For matrix [N,K], native output groups have S=16*(WARPS/KSPLIT)*TPW rows (currently S=32). Define g=ceil(N/S), a_i=floor(g*i/3), b_i=floor(g*(i+1)/3), and row range [min(N,S*a_i),min(N,S*b_i)). Empty shards do no work. Paired gate/up weights share ranges; segmented projections partition each segment independently. Every original output row occurs once. Device-i matrix work is 2*B*K*(n_i_end-n_i_begin), with original scale/bias and reduction association. The matrix total remains 2*B*K*N; unchanged model projection work is 51247841280 FLOPs per evaluated token and attention remains 393216*T. No throughput follows from total VRAM.

Resident weight pages must remain physically charged to their owner. For each canonical packed matrix j, row stride R_j and 16-row-compatible cuts r_ji, map physical granules [offset_j+r_ji*R_j, offset_j+r_j(i+1)*R_j) to owner i, with a documented boundary-page owner. Actual page granularity G_i must be queried; boundary pages are shared addresses with one physical owner and measured peer-read traffic, never duplicate physical capacity or unified VRAM. A per-device census computes W_i=sum(owned physical page bytes), including all rounding and nonprojection owner-local constants. The physical acceptance equation is W_i+history_i(T)+recurrent_i+graph_i(B)+scratch_i(B)+peer_inputs_i(B)+startup_source_i+boundary_padding_i <=14912061440. The exact earlier target/head weight bytes are 16713367552/348852224; KV remains 69632*T=18253611008 bytes at 262144 populated tokens, apportioned by its existing owners. Full-context admission needs the computed per-device page census, not an averaged charge.

For an owner-local staged BF16 activation [B,K], copy 2*B*K bytes to each of the other two cards once per projection (or share a correctly leased staged input between consumers). Native shards read their owner-local weight pages and publish disjoint output rows, with owner-local or explicitly peer-visible output storage. Output traffic is element_bytes*B*N; the two remote shares must be represented and measured. Existing hidden stage boundaries retain 20480*B bytes each. The expected runtime expression is sum_projections(max_i(kernel_i)+input_copy+output_visibility+join)+unchanged_stateful_work+existing_stage_edges; measure all terms. Even ideal three-way projection speed-up is bounded by 1/((1-p)+p/3), where earlier diagnostic p≈0.83 is not a promise of the requested final rate.

First verify the native multi-context graph on actual three P100: partition the existing 12288-by-5120 paired projection into three equal 4096-row shards, use immutable local diagnostic bytes, join graph completion, compare every F32 output against the original single-device kernel, and record event/wall times. This diagnostic's intentionally duplicated small fixture buffers are separately charged and do not qualify production capacity. Only after this gate should the complete artifact-owned VMM placement, source ABI, per-segment row geometry, peer-staging graph dependencies, retained session leases and resource census be implemented together. No source model conversion, driver change or P40 participation is permitted. Original GSQ/Heretic and concurrent real Pi remain required production gates.

### Reassessed explicit three-root graph and gather-node coverage

The direct-output three-context native graph works, but the first local-gather fixture had an incorrect CUDA_MEMCPY3D ABI (missing its two reserved pointer fields). After the sole ABI correction against installed /usr/include/cuda.h, exact poisoned coverage failed at F32 index 8192, the first GPU1 output row: the fixture emitted only the final GPU2 copy. This second failure ends that gate's correction loop. Reassess the complete native-kernel arguments, context, dependency and coverage path before further edits.

The new explicit DAG records three kernel handles first and emits exactly two separate gather nodes, each depending on its own kernel. Source/destination pitches are 8*N bytes for the diagnostic paired F32 output, X byte offsets 8*n_begin, width 8*(n_end-n_begin), height B, depth 1. GPU0 writes its own disjoint local range. Before parallel execution every output is poisoned; graph completion must replace all 2*B*N F32 values exactly. The production output-byte factor is taken from its artifact epilogue. No capacity, arithmetic, weight layout or model semantics change follows from this gate repair.

The poisoned explicit-DAG retest further disproves the earlier direct-output exact-coverage claim: group_begin was measured in CTAs, while Shape::tile_group returns 16-row tile groups. The first untouched value is index12288, row6144, matching a GPU1 kernel range of [2048,6144) instead of [4096,8192). The prior nonpoisoned direct test retained the original full reference in GPU0 and therefore could mask missing writes; its exact-coverage verdict is invalidated. The new DAG's first failure admits one correction: tile_group_begin=(WARPS/KSPLIT)*CTA_begin=2*128*i. The existing arithmetic and copied row-range equations stay unchanged. Retest poisoned full output coverage before deriving any parallel speed claim.

### Actual P100 page granularity and physical-weight prerequisite

Installed driver queries return minimum=recommended=2097152 bytes on all three P100. A conservative all-source-tensor inspection retains F32 constants, applies the exact H32/H16 resident row equations to original IQ2 matrices, and marks only matrices >=3 granules as shard candidates. It counts 376 matrices; mapped physical charges are [6016729088,5664407552,5586812928] bytes, plus 176115712 owner-local unsharded bytes. Total conservative physical weights are 17444065280 bytes versus 17063604224 logical upper-bound bytes. This is not the actual canonical role census: tied-role deductions, norm dtype and original local owner assignments remain required. The actual existing target/head logical charges are separately 16713367552/348852224.

For a canonical matrix of N rows and physical row stride R, let canonical bytes L=N*R, granularity G=2097152, row cuts r_i=16*floor(ceil(N/16)*i/3) clipped to N, and page cuts p_i=ceil(r_i*R/G) clipped to ceil(L/G). Map [G*p_i,G*p_(i+1)) once on card i, retain a stable full canonical virtual address, and grant explicit access to all three selected cards. Physical charge_i=G*(p_(i+1)-p_i); no weight copy is retained elsewhere. Boundary-page peer reads are part of measured traffic: each internal logical cut can displace at most G bytes. Worst-case boundary traffic per full traversal is <=2*G per matrix; the 376-matrix upper bound is 1577058304 bytes, not an assumed measured cost. Allocated tail padding is <G per matrix and included in the census. Small/dense constants keep their artifact owner and are explicitly peer-accessible if read by a shard.

The next focused actual native gate replaces the paired fixture's three large duplicated weight buffers with these row-owner VMM mappings. Its F32 output ranges, poisoned coverage and three-context gather DAG remain the already verified ones. Compare every output and event time to the same original local single-device fixture. This verifies native decoding across physical page boundaries and peer access without changing weight ABI or arithmetic. Production capacity, source-import writes into distributed mappings, full batch geometries, graph/session leases and original-model speed remain separate integration requirements.

### Sole physical-placement correction: device-local shard views

The first VMM native gate passes every original-local F32 reference value but regresses the useful comparison: same-group parallel B=2 takes 0.669 ms versus original local 0.373 ms, B=4 0.825 ms versus 0.407 ms. The VMM master single-device times of 15.39/19.26 ms include full remote weight reads and MUST NOT be used as a speed-up baseline. Canonical row cuts do not align to P100's 2 MiB pages; peer reads from their boundary pages erase the projection gain. No production VMM path is adopted.

The sole correction retains a full master virtual view for original import/inspection and gives each device a separate virtual view of the same logical matrix. For shard i, its page range is [floor(r_i*R/G),ceil(r_(i+1)*R/G)); every page it reads is physically local. Nonboundary pages have one physical allocation shared with the corresponding master page. An overlapping boundary page is duplicated only for the adjacent shard that needs a local copy. After canonical native import completes, copy each such immutable page from its primary local view to the adjacent clone before publishing residency. No code/scale/weight value changes; all original logical offsets/row strides remain unchanged in every view.

Physical_i(j)=G*(ceil(r_(i+1)*R/G)-floor(r_i*R/G)), plus unchanged owner-local constants. Virtual address reservations add no physical weight copy. The revised conservative 376-matrix census is recorded in projection-row-shard-alias-capacity-inspection.json; exact role/alias accounting and per-device startup equations remain mandatory. Each copied boundary costs G bytes once at startup; its one-time transfer is separate from inference traffic, which must contain no uncached peer weight read inside a selected shard. Full original imports, pointer/view selection, per-batch graph geometry and retained immutable/session leases must be integrated together after the poisoned native gate succeeds. The probe's full single-device reference uses the master view; only row-sharded parallel launches use partial local views. Never launch a full projection through a partial shard view.

### Complete batch-geometry and allocation-lifetime contract before integration

The device-local alias gate must also cover draft M=1, existing GEMV16 and prefill GEMM. Choose a common physical row boundary aligned to 128 rows (also a multiple of existing 16-row packed tiles), rather than deriving different physical views for each runtime batch. For each canonical matrix, define r_i=min(N,128*floor(ceil(N/128)*i/3)). Native GEMV and GEMM map their respective tile units into these same row ranges; no selected launch may read outside its owner's mapped view. Empty ranges create no work. Partial final output tiles keep the original N guard and precision. The updated conservative physical charges are in projection-row-shard-common-geometry-capacity.json. Exact active-plan roles and all per-device full-context/startup charges still determine admission.

One immutable logical weight retains its original canonical master tensor and per-device local views under one allocation access domain. Before native import finishes, only its master is writable. Boundary clones publish only after source completion and their peer-copy fence. Projection view lookup is refused before publication; no full matrix dispatch may use a partial view. Native graph completion on its owning queue includes every peer kernel and gather node, and its retained allocations keep all physical mappings alive. Charge every primary and cloned page to the physical device's memory domain before allocating it; failure must release all grants and mappings. A master virtual reservation is not a physical charge. Freeing the final owner releases mappings before their accounting grants. Session-owned input staging and output scratch remain separate from immutable weights and retain their own scoped permits/fences.

### Plan 37: physical accounting interface completion

After the second integrated build failure of plan 36, retain its exact matrix, KV, arithmetic and transfer equations. Count each unique representation or scale allocation a once: W_i=sum_a physical_bytes(a,i), W=sum_a sum_i physical_bytes(a,i). Use checked addition for every allocation and device; tied allocations contribute once. This changes the accounting implementation, not physical placement, kernel work or communication. Rebuild the entire CLI before replacing the measured patch-35 process and run the unchanged original-model gate.

### Plan 38: residency-cache physical census

The original target imports but warm-up fails before inference because cache and family charges have different units. Keep logical import progress L=sum canonical_bytes; physical cache C=sum over unique allocations a and physical owners i of physical_bytes(a,i). Head extra H=C_target_and_head-W_target is valid only when both operands use the same physical census; tied allocations remain shared. Apply checked addition and shared allocation deduplication recursively across owner caches. Matrix, KV, activation, scratch and graph equations, page placement and communication remain as plan 36. No new kernel or arithmetic change.

### Parallel projection time attribution prerequisite

For each captured projection j, attribute t_j=t_input_j+max_i(t_kernel_j_i)+t_gather_j plus actual scheduling/overlap costs, using root-context completion after every peer branch. Do not reuse serial diagnostic timings: ordinary launch-detail timing bypasses graphs and the parallel projection path. The focused diagnostic preserves each projection fork/join graph and records the existing CUDA events before/after that graph; other entries retain authored kernels. Diagnostic per-entry launch/event overhead excludes it from production throughput. Capacity adds only graph/event metadata and existing sealed-view temporary allocations, with unchanged immutable page charges, matrix work, exact values, F16 KV and communication bytes. Its output selects the next complete performance correction.

### Selected correction after actual parallel graph attribution

The first real parallel model rate is 17.966034 tok/s with unchanged choices. Root-completed graph projection intervals show actual M2/M4 dense expansion medians 0.264/0.317 ms, down 0.207/0.271 ms, recurrent projection 0.179/0.220 ms. Non-graph launch intervals in this diagnostic can include host graph construction gaps and are not used to rank kernel execution. Direct peer output stores regress versus pitched gathers and are rejected. Native three-P100 local-view probes with poisoned inputs/outputs preserve exact bits while four-warp blocks improve the paired M2/M4 medians from 0.199/0.214 to 0.175/0.202 ms. M16 regresses and keeps eight warps. A checked narrow plane quotient yields a further small measured gain and retains the wide fallback. These are synthetic-binding diagnostics, not model-rate claims.

Selected small-batch correction uses W=max(4,KSPLIT) warps for the dedicated M2/M4 entries only, with unchanged KSPLIT partitions, reduction order and epilogues. Block threads=32W; group count=W/KSPLIT; grid_x=ceil(sum_j ceil(N_j/16)/(W/KSPLIT)), versus W=8 before. Arithmetic 2B K sum_j N_j, useful weights/input/output bytes, resident pages and all communication remain unchanged. Each CTA shared reduction becomes (W/KSPLIT)*KSPLIT*2*NB*4*32*4 bytes; each result retains the same warp-cohort operations. Plane address quotient uses u32 only when both byte offset and width are <=2^32-1, otherwise the original u64 expression; final base, tile, stride and row multiply remain u64. Capacity gains only the reduced per-CTA scratch; global capacities are unchanged. Rebuild and run the unchanged original-model gate with diagnostics disabled.

### Plan 41: complete production replay attribution

After the second parallel performance failure, attribute request time R = sum_submission GPU_root_completion + CPU_prepare_and_graph_construction + scheduling_wait + residual overlap/gaps. Record graph-cache miss construction separately from execution. Add two root-context CUDA event-record graph nodes around each authored launch or complete projection fork/join; each end depends on every peer terminal, and the next launch depends on that end. Per-launch t=elapsed(start,end) has no host construction between the graph event nodes. Retain events with their instantiated graph, read only after the root submission completes, and identify the last repetition if a graph is repeated. These timing nodes add driver event metadata and device execution overhead; do not treat instrumented rate as production throughput. Weight pages, F16 KV, BF16 activations, F32 intermediates, matrix work and all input/gather byte counts remain unchanged. This diagnostic selects the next complete performance change instead of repeating microkernel optimizations.

### Plan 42: bounded parallel scratch per source queue and peer

The concurrent Pi gate proves per-replay retained copies violate the previous scratch estimate. Original request inputs/outputs/KV remain session-owned. For one projection invocation j, construct a disjoint 16-byte-aligned frame containing its unique copied inputs and unique complete peer output buffers: F_j=sum_u align16(input_bytes_u)+sum_v align16(rows_v*pitch_v). Do not count each segmented gather twice when segments share one output buffer. For source CUDA queue q and destination i, retain one current arena of C_qi=next_power_of_two(max_j F_j), and reuse its frame offsets between graph projections. Explicit fork/join frontiers serialize all peer branches before the next projection; source stream order serializes graph invocations and sessions on that queue. Distinct source queues have distinct arenas even on the same physical peer. Old immutable replay graphs retain earlier arena generations, whose geometrically increasing capacities sum to less than 2*C_qi. Thus scratch_i < sum_(q!=i) 2*C_qi, replacing a term proportional to number of cached graphs. Each actual allocation still takes destination-device permits before physical allocation; failed growth leaves the previous arena intact. Graph ownership retains every bound generation.

For the observed prefill bound B=512, dense-down input+output is B*(2*17408+4*5120)+selector alignment =28311552+alignment bytes. Recurrent projection concatenated output is B*(10240+6144+96)*4 and staged input B*5120*2, totaling38993920 bytes before selector alignment; a 64 MiB arena covers this invocation. Two peer-source queues per P100 have cumulative geometric arena storage strictly below256 MiB. The largest recorded prior startup estimate 14516144469 plus268435456 =14784579925 bytes stays below14912061440 by127481515 bytes. This is an active-shape bound, not an unrestricted batch/context proof; larger frames must remain under actual per-device permits. Weights, exact F16 KV, BF16 activations, F32 arithmetic, input/gather traffic and matrix work do not change. Qualify original-model output and both real concurrent Pi tool loops.

### Focused recurrent mapping selection after scratch qualification

The real concurrent Pi tool loop now passes with bounded temporaries. Complete graph spans identify the authored recurrent step as a significant hot path. For B activation rows, NV value heads and W key/value width, the step retains NV*W^2 F32 state elements per bank (4*NV*W^2 bytes; 3145728 at NV=48,W=128). Its state update/remember/output work is 8*B*NV*W^2 FLOPs plus 2*B*NV*W residual scalar operations and the unchanged convolution, normalization, gating and tape work. A publication transfers one full bank read/write plus explicit window/tape versions; actual stop/tape controls determine extra reads/writes.

Mapping uses R rows per warp and V warps per CTA: block rows=R*V, threads=32V, grid=(NV,W/(R*V),slots+1). The authored domains R in {2,4}, V in {4,8} change only ownership of independent rows; W/32=4 per-lane FMA order and warp reductions remain identical. Shared preparation is 2*(2W+R*V)*4 bytes per CTA, and the same per-row prologue repeats over NV*W/(R*V) CTAs. State-row registers double with R=4; fewer CTAs need not improve occupancy. Test all four exact mappings at the active 48-head/128-width geometry for M2/M4, comparing gated output, window, state and tape bits, and rotate 16 independent argument sets beyond P100 L2 for timing. This diagnostic selects one default correction; it does not qualify model throughput. No weights, KV, activation formats or device transfers change.

### Selected exact recurrent default

The focused active-shape gate verifies every state/window/tape/gated bit across all four authored mappings. Select ROWS=4,WARPS=8: block rows=32, threads=256, grid=(48,4,slots+1), versus (48,16,slots+1) and128 threads before. CTA count falls by4; independent row arithmetic and publication/tape rules remain unchanged. Shared preparation changes from2*(256+8)*4=2112 to2*(256+32)*4=2304 bytes per CTA; held state changes from8 to16 F32 values per thread. Matrix weights, F16 KV, BF16 activations, F32 state/intermediates, per-device/global scratch and communication bytes remain unchanged. Measured M2/M4 native-fixture medians improve109.236→69.382 and142.688→91.684 microseconds; only the rebuilt original-model request and concurrent real Pi can establish end-to-end gain.

### Complete MTP projection distribution

The head loader consumes the same row-sharded residency authority, but draft_rows and head_logits_rows are not captured by the existing projection descriptor. Complete their native fork/join path before another model-rate decision. For draft_rows, each output block owns32 rows and repeats the original256-thread embedding/conditioning normalization into8D shared bytes, then original32-lane F32 FMA/reduction over2D. Matrix work is4*M*D^2 FLOPs; useful combine reads are its unchanged resident bytes, plus normalization/table traffic per active CTA. Shared memory stays8D (40960 bytes at D5120). Row cuts remain multiples128, so entire32-output CTAs are owned uniformly and can return before any block barrier. Each peer receives exact conditioning and token/status views, uses its local combine row view, reads the embedding table through the full canonical master, and gathers4*M*owned_D output bytes. Norms remain immutable small local copies.

For head_logits_rows, use the existing plain features input path, original GEMV/GEMM row ownership controls and local vocabulary weight aliases. Useful work is2*O*V*D FLOPs, input2*O*D bytes, output4*O*V bytes; existing dependency joins and bounded scratch arenas apply. Identify sharded weight ports explicitly so an embedding table or other non-output-row matrix never receives a partial projection alias or an unintended full peer replication. All arithmetic associations, original weights, F16 KV, BF16 activation values and F32 results remain unchanged. Account actual extra immutable norm copies and bounded frame maxima per destination; qualify the actual original model and real concurrent Pi.

### Sole correction for partial projection views

Patch45's first request fails with illegal CUDA access. Source inspection establishes that the draft vocabulary view is65536 rows of the248320-row resident output allocation. Local aliases own physical cuts of the full allocation: rows [0,82688),[82688,165504),[165504,248320). The invocation's previous cuts [0,21760),[21760,43648),[43648,65536) do not address those local pages on peers1/2. A partial view is eligible for local-alias dispatch only if offset=0, view_rows=allocation_rows and view_bytes=canonical_bytes for every projection weight. Otherwise execute its original full-canonical-view module on its authored source queue, with no shard controls and no peer input/gather copies. The first65536 rows reside on physical owner0 here; target full-matrix projections and the complete draft combine continue their existing three-owner fork/join. This generic ABI check preserves weights, F16 KV, BF16 inputs, F32 association, allocation/page charges and bounded arena equations. Partial invocation arithmetic remains2*O*65536*D; this correction adds no allocation or transfer. Original-model output and real concurrentPi remain the acceptance gate.

### Plan47: separate exact draft preparation and parallel combine

The45/46 loop is closed after unsafe partial aliases were corrected but actual speed regressed. Draft preparation currently repeats ceil(D/32) times per input row; forD5120 this is160 independent embedding/conditioning decodes and identical fixed256-thread RMS normalizations. Retain the same arithmetic and block reduction sequence in one preparation CTA per input row, writing2D F32 values after the original Act::round. Global joined scratch is8*M*D bytes (40960*M here); each peer copies the same8*M*D bytes once. The projection CTA loads these F32 values into its unchanged8D shared array and performs the original32-lane sequential F32 FMA and reduction for its local32 output rows. Matrix compute stays4*M*D^2 FLOPs and useful combine reads are unchanged; normalization/table traffic drops from ceil(D/32) repetitions to one. Peer outputs retain4*M*owned_D gathered bytes. Token/status, embedding table and norms stay on the original source preparation entry and never enter peer projection dispatch. Split launch barriers and graph dependency frontiers order preparation before every copy/kernel and all gathers before consumers. Additional persistent authored scratch8*M*D and existing bounded peer frame maxima are charged through the existing DSL resource authority; joined transport uses F32 exactly, independent of BF16 activations and exact F16 KV. Full canonical partial-vocabulary fallback remains. Qualify original-model output and concurrent real Pi; no fixture speed is acceptance.

### Plan49: actual projection admission and replay diagnosis

After47/48's second failure, measure the production path before further optimization. Record actual launch name, weight view extents/bytes/offset and RowShardPlan rows/canonical bytes, once per distinct ABI shape, on graph construction only. Enable it with a separate diagnostic environment variable; production defaults remain off. Use existing complete replay event spans to distinguish admitted parallel projections from full-canonical fallback and its peer traffic. Diagnostic host strings and CUDA event metadata add no weight, KV, activation or scratch bytes and do not change matrix compute or transfers. Instrumented rate is not a speed qualification. This evidence determines the next complete plan.

### Plan50: preserve empty matrix aliases in projection admission

Actual49 metadata proves attention's empty gate has extents[0,5120], byte_len0, while sharing the query allocation whose plan owns12288 rows/34406400 bytes. The full-view safety check mistakenly rejects the entire projection, causing full canonical peer reads: warm actual M2/M4 attention_project medians11.096896/13.827280ms. A tensor with canonical_bytes0 addresses no weight values and contributes no matrix FLOPs, physical allocation, copy or gather traffic. Exclude such zero-sized weight views from complete-view admission comparisons only. Every nonempty view still requires offset0, matching rows and canonical bytes; the partial65536-row vocabulary retains canonical fallback. Existing static-empty source guards and zero-width gather skips remain. Restored attention work, page charges, F32 association, F16 KV and three-device fork/join equations are those of the qualified42 path. No new scratch or transfers are introduced. Rebuilt original192 output and concurrentPi are the gate;49's instrumented rate is diagnostic only.

### Plan51: remaining production recurrent cost

The50 real Pi gate passes, but warm direct26.383662 tok/s remains below requested speed. Existing warm49 events show gated_delta_step median0.293584ms versus the earlier exact one-slot fixture0.069382/0.091684ms. Record actual sealed dimensions, launch grid/block/shared bytes, declared launch specialization and mapped-input buffer indexes once per active shape for recurrent/attention/draft entries. This distinguishes real slot geometry, tape/state bindings and host-mapped controls before changing computation. Use existing replay event spans on the actual model. All weight/state/KV/activation capacities, computation and communication remain unchanged; opt-in host metadata and event timings do not qualify speed.

### Plan52: stage mutable mapped upload inputs once per replay

Actual51 shows production gated_delta_step atM2/M4,B1,grid[48,4,2],block256,ROWS4/WARPS8; five controls are host-mapped slices at offsets0/256/512/768/1024 of one276480-byte allocation. Qualification inputs are device-local. Production median0.295552ms remains substantially above the exact DRAM-rotated fixture. Stage only actually read dense scalar tensor spans of mutable mapped upload allocations into local CUDA memory before the replay. For each view, span_bytes=(1+sum_d((extent_d-1)*stride_d))*element_bytes, zero if any extent is zero. Merge overlapping or adjacent intervals of the same allocation, never unread gaps or another session region. F=sum_intervals align16(span_bytes); copy F_useful bytes once per replay and rewrite each matching buffer pointer with its identical byte offset into the local frame. Weights, GPU-produced activations, KV/state and immutable mapped artifacts are excluded.

A source queue owns one geometric arena C=next_power_of_two(max F) using the existing allocation permits and identity-scoped scratch authority; retained earlier generations sum to less than2C. Distinct queues retain distinct frames; complete source stream replay order and copy→kernel→peer-join dependencies serialize reuse. Original bound upload regions and request permits retain CPU source bytes throughout each copy. Copy intervals exclude unowned gaps. Original projection peers keep their explicit input copies; the source kernel uses the local pointer override. Root staging is opt-in for qualification. Compute, every IEEE value and reduction order are unchanged. Extra traffic is the measured read upload bytes per source replay; all CUDA graphs include these nodes in completion. Actual source permits reject growth beyond capacity. Original192 output and concurrentPi qualify the change; mapped transport savings are a hypothesis until measured.

### Plan54: retain packed coefficient blocks for exact M2 projections

Native three-P100 metadata56 probes preserve everyF32 output: same-groupM2 improves0.172016→0.144210ms; mixedM2 measures0.159115ms. M4 same-group regresses and retains the original schedule. An independent installed-CUDA gate compares scalar coefficient reads with authored block/coefficients accessors across13 formats, including exceptional F32/F16 factors, passing212992 exact scale/bias comparisons. Select the block cache only forM2,nondense weights andKblocks divisible4.

Original partition boundaries are superblock-aligned: S=ceil(Kblocks/4), begin=floor(S*part/KSPLIT), end=floor(S*(part+1)/KSPLIT). For each superblock in[begin,end), fetch one authored packed coefficient block per weight stream/tile and use q=0..3 withKblock=4*superblock+q; original per-group F32 coefficient decoding, product/reduction and epilogue order remain unchanged. Useful coefficient payload bytes and weight allocations do not change; repeated scalar loads become the existing packed block loads. Register-local storage is TPW*(sizeof(WA.Block)+PAIR*sizeof(WB.Block)), alongside one current raw/decoded-coefficient set; no new global/shared buffers or peer traffic. Dense,M1,M4+,non-four-block widths retain the prior path including selected lookahead. Source import/layout, F16 KV and BF16 values are unchanged. Integrated original-model rate/output and real concurrentPi are the gate.

### Plan55 prerequisite: bound M2 coefficient-loop code size

Active gate: improve native original Swift throughput on three P100, with exact
weights, binary16 KV and unchanged F32 arithmetic. Patch54 qualifies both
original models with concurrent real Pi, but reaches only31.564/30.312 tok/s.
Warm original Heretic64 replay attribution (staging enabled) identifies the
M2/M4 feed-forward projections as the largest decode spans. Recurrent step is
now0.082496ms, so the earlier mapped-control bottleneck is resolved.

The M2 coefficient-cache loop currently unrolls its four K chunks. The complete
scalar decoder/product body is therefore repeated four times in generated code.
A focused original-layout probe will compare the actual54 source with the same
loop left rolled. No new resident codec, precision or accumulation tree is
selected. For each original split p, `S=ceil(Kblocks/4)`, `begin=floor(S*p/P)` and
`end=floor(S*(p+1)/P)`. Execution remains `sb=begin..end`, then `q=0..3`, with
`kb=4*sb+q` in ascending order. Coefficients remain one exact authored Block per
weight/tile/superblock. Accumulator updates and split reduction retain their
order and rounding. Runtime loop-control instructions may increase; generated
product-body copies decrease from four to one. Performance must be measured.

Weight bytes, F16 KV (`69632*T` bytes including head), activation storage,
shared-memory geometry, queue arenas, peer input/gather transfers and all
physical device charges are unchanged. Register allocation and generated
module bytes must be observed rather than inferred. Three-P100 full poisoned
F32-output comparison and real original-model requests are the acceptance
sequence; an isolated kernel gain alone does not qualify production.

Plan55 first probe: rolled loop retains all835584 F32 outputs but regresses
M2 from0.145350 to0.235395ms. Register count falls87→72 while48 bytes of
thread-local memory appear. The dynamic IQ2H coefficient-word array index is
reachable at runtime q. The sole corrective probe explicitly selects its two
or four existing packed words before the original shift/multiply, retaining
all byte offsets, values and loop order. No production change is made. If this
also regresses, stop this direction and reassess the end-to-end projection.

Plan55 correction also fails its performance gate: explicit word selection
removes the48 local bytes, retains835584 exact outputs, but M2 is0.157133ms
versus0.145334ms baseline. Stop this direction. No inference source or active
production arithmetic changed. Current evidence still places the feed-forward
projections above recurrent execution in the decode critical path.

### Plan56 prerequisite: independent output-row halves, original K splits

Plan55's two speed failures are closed without a production edit. Reassessment
focuses on M2 matrix occupancy: native small-batch GEMV uses four warps/CTA,
KSPLIT4, one16-output-row tile/CTA. A5120-row down projection has320 CTAs
before ownership, about107 active CTAs/P100 after three-way sharding. Its
paired16-row fragment and exact four-K-share reduction remain the reference.

The next focused probe separates the two independent eight-row output halves
into two CTAs, preserving each output's four-lane products and all four original
K partitions. For original tile group g, new blocks2g/2g+1 own row r=0/r=1;
logical n=16*g+lane/4+8*r is unchanged. Physical row cuts and aliases remain
identical. Only the selected independent output half performs products and
stores; split sums retain the same order. Active CTA count doubles, while total
selected F32 products remain constant. Code-plane loads can double; coefficient
loads/register allocation may be eliminated when their output half is unused.
No bandwidth or speed gain is presumed.

For a weight stream with packed traffic B, this probe bounds traffic by2B;
input and pitched gather bytes are unchanged. No new resident weights, KV,
activation/intermediate buffers or queue arenas are allocated. Existing shared
reduction shape is retained in the prototype. Matrix FLOPs remain
`2*M*N*K` per stream; loop and load overhead must be measured. Admission cuts
still use the original16-row logical tile, so every partial local alias is read
only by its owning group. The exact three-P100 poisoned output gate compares
all output rows against the unmodified54 reference before original-model speed
and concurrent real Pi qualification. This is representation/shape generic.

Plan56 first probe build fails because the expanded authored source defines
`group` as its groups-buffer macro. The generated fixture's `Shape::group()`
is macro-expanded to a buffer identifier. The sole correction spells the same
cohort quotient as `threadIdx.x/32/Shape::KSPLIT`, preserving geometry and all
arithmetic. Retest before any production integration.

Plan56 fixture retest fails the same compilation gate. End-to-end inspection
corrects the prior macro diagnosis: the collision is uppercase `GROUPS`, not
the lowercase member `group()`. The authored operator defines GROUPS as its
buffer macro after the projection header. Both error locations refer to the
new fixture's `Shape::GROUPS`. No inference kernel ran and no output/performance
evidence exists for the row-half hypothesis. Stop editing that failed fixture.

### Plan57 prerequisite: isolate the row-half fixture from operator macros

The row-half algorithm and Plan56 equations are unchanged. The invalidated
dependency is the generated fixture's identifiers after the full authored
operator expansion. The new plan explicitly ends the operator's GROUPS macro
before standalone fixture entries and restores the original tile-group formula
inside the fixture. Earlier operator bodies are already preprocessed under
their original bindings. All other transport, alias, fork/join and poisoned
F32 comparisons retain the verified existing fixture path. No production code
changes until this complete isolated fixture compiles and proves both fidelity
and an actual device-event improvement.

Plan57 compiled row halves retain all835584 F32 outputs but paired M2 regresses
0.145334→0.182302ms despite registers87→64. The sole corrective gate narrows
to the active down-projection dimensions N5120/K17408 with a single Q4K stream,
where baseline ownership has only104/104/112 CTAs per P100. This is a diagnostic
of the original Heretic resident format, not a model requantization. Exact
row cuts are[0,1664,3328,5120]. Row stride is9792 bytes:8704 code bytes,
816 scale bytes and272 super-factor bytes. Logical payload is50135040 bytes;
physical VMM pages/boundary clones are recorded independently. Existing input
and full-F32 pitched gather bytes remain explicitly included. If this specific
complete gate also regresses, the row-half direction is closed.

Plan57 corrective gate fails before CUDA compilation: the inherited fixture
builder had already replaced its family condition with `if True`, invalidating
a later string lookup. No down-projection speed evidence exists. Stop editing
that reconstruction. Plan58 constructs the exact Q4K/N5120/K17408 payload and
input buffers directly, removing the inherited Python setup-string dependency.
The numerical algorithm, cuts, ABI, VMM alias/copy graph and Plan57 capacity
and traffic equations are unchanged. The only active production source remains54.

Plan58 baseline first run fails only its diagnostic CUDA context. Independent
compute-sanitizer pinpoints a16-byte code-plane read atblock112; modular
address arithmetic identifies the old2800-byte stride rather than the sealed
9792. Generated source duplicates metadata under both formal PARAM_2 and
semantic QUERY_WEIGHT names. Only PARAM_2 was updated, while the decoder used
the old semantic alias (IQ2H32), invalidating the fixture's claimed Q4K binding.
The sole correction binds packet slots directly to the fully updated formal
PARAM_2/3/4 metadata. Original model health remainsREADY and all P100s visible;
there is no driver or production source change. Retest the complete fixture.

Plan58 corrected down-projection gate passes all348160 F32 outputs, but row
halves regress M2 0.175531→0.194440ms, despite registers56→40 and no local
bytes. Together with the paired-shape regression, this rejects row halves.
No production integration follows. The retained54 path remains the baseline.

### Plan59 prerequisite: combine exact cached coefficients with raw lookahead

End-to-end source inspection establishes that54's M2 coefficient-cache branch
fetches the current raw code block immediately before computing it; this branch
bypasses35's one-K-block lookahead. Preserve54's four-K-chunk coefficient Block
and issue the next original raw fetch before the current products. For each
original `sb=begin..end`, load raw(4*sb) once. For q0..3, issue raw(4*sb+q+1)
only when q<3, compute the original q products/scale/bias folds, then advance
raw. No fetch crosses a superblock or split boundary. The sequence and rounding
of all F32 accumulations and split sums are unchanged.

Raw code bytes read remain B per stream (each original K block exactly once),
coefficient fields remain one Block per stream/tile/superblock. Register-local
live raw storage is bounded by two Raw objects per weight/tile, replacing one;
no global/shared/intermediate buffer or peer transfer is added. Original
weights, F16 KV capacity (`69632*T`), BF16 activation values, physical charges
and queue/session ordering are unchanged. Observe registers/local spills and
full fork/join device timings. Existing poisoned three-P100 IQ2H same/mixed
and exact-dimension Q4K down fixtures are the decisive numerical/performance
prerequisite, followed by original-model192 and concurrent realPi gates.

Plan59 raw lookahead does not improve either complete fixture: paired M2
0.145334→0.146822ms (87→88registers); exact-dimension Q4K down0.175531→0.189507ms
(56→48registers), with all previous F32 values retained. Reject integration.

### Plan60 prerequisite: exact parallel group products, original serial folds

The next complete probe increases independent K-group work without changing
F32 accumulation order. For each original output n/input row m/group g,
compute the original four-lane FMA products P[g,n,m] and bias sum U[g,m],
including the same lane-local order and XOR1/XOR2 reduction. Store these
already-F32 values exactly. A second launch uses each original KSPLIT4 interval:
`S=ceil(Kblocks/4)`, `begin=floor(S*p/4)`, `end=floor(S*(p+1)/4)`, ascending
sb/q/group, then the original `a=fma(scale,P,a); a=fma(-bias,U,a)` when biased.
Finish with the same four partials added in part order. Weights, F16 KV and
BF16 activation values are unchanged; additional intermediates are exact F32.

For one stream, M2/N5120/K17408/Q4K/G32:
`Pbytes=4*M*N*(K/G)=22282240`, `Ubytes=4*M*(K/G)=4352`.
Each physical device owns an independent22286592-byte full-stride work frame;
only owned row ranges are read/written. No unified-memory claim is made.
Three local frames charge66859776 bytes total. Original input/gather copies
are included. Dot launches320*272 blocks before ownership, compared with320
original CTAs; fold uses320 CTAs and the same K intervals. Extra global traffic
is2*Pbytes (write/read), plusUbytes write and its cacheable fold reads; original
code/coefficient bytes remain unchanged. FLOPs retain all original products,
scale/bias folds and split sums; launch/loop overhead is measured.

This is an isolated active-shape hypothesis, not yet an integration decision.
Before integration, general stream/group-width capacity must be derived from
artifact metadata. A proposed64MiB frame cap admits only shapes whose actual
sum of stream product/sum bytes fits; larger shapes retain the existing path.
Queue ownership, root upload snapshots, grants and arena namespaces must be
separate and explicitly checked. No production edit follows merely from a
kernel gain. The exact poisoned original Q4K down oracle precedes both original
model and concurrent realPi acceptance.

Plan60 first complete three-P100 gate preserves all348160 F32 outputs but
M2 regresses0.175531→0.258245ms. The first version uses one warp/CTA in the
product stage and scalar coefficient loads in the fold. Its sole corrective
patch groups four independent K-block warps per product CTA (Y272→68,
block32→128), and reuses the already-qualified authored packed Block decoder
in the fold. Each original P/U, scale/bias fold and KSPLIT4 sum remains exact;
bytes are unchanged. No production edit or extra local correction is allowed
before the complete retest verdict.

Plan60's sole correction also regresses the complete Q4K M2 gate to0.302990ms
while retaining348160 exact F32 values. Stop the staged-product direction; no
production integration. Both stages' extra traffic/launch/fold overhead exceed
the baseline. Retain54 unchanged.

### Plan61 diagnostic prerequisite: decoded-weight FP32 row dot

This isolated hypothesis measures a conventional FP32 GEMV over the same exact
decoded weights and unchanged BF16 input values. Four original K intervals are
retained; 32 lanes cooperate per output row, accumulating decoded-weight FMA
products and reducing their partials at the end. It introduces no weight/KV/
activation codec change or global scratch. It DOES change F32 association
relative to Magnitude's grouped scale/bias folds; exact output-bit equality is
not claimed. This experiment is not a production fidelity decision. Record
absolute/relative/ULP differences, independently compare decoded weight bits
when extending beyond Q4K, and require original-model/Pi/quality evidence before
any production selection. An isolated speed improvement cannot replace those
requirements. The current source remains54 and its qualified original responses
are preserved.

For M2/N5120/K17408, matrix products remain2*M*N*K FP32 FLOPs. Grid has5120
blocks before ownership,128threads/block, four original K intervals per block.
No extra intermediate allocation or transfer is added; physical row ownership
and original input/gather copies remain explicit. Decoder loads may replicate
packed-row fragments and metadata. Actual driver spans decide performance.

Plan61 first row-dot probe is1.067282ms versus0.175531ms. It records FP32
association differences separately and preserves its independent full-vector
transport gate; it is not an exact-bit production candidate. Its sole
corrective probe reads only the required Q4K nibble via the verified mma16
virtual bit mapping, rather than fetching/decoding a full vector for each
logical weight. Coefficients, decoded weight formula, K intervals and this
experiment's own F32 order are unchanged. No production selection follows.

Plan61 direct-byte correction remains slower0.936094ms versus0.175531ms.
Stop that direction. Reassessment identifies the representation's mma16 row
pair/interleaving as a poor fit for one logical-row scalar fetch per lane.

### Plan62 diagnostic prerequisite: reuse full mma16 fragments in FP32 GEMV

Keep the exact resident format, vector fetches and two-row/four-lane mapping;
cache original coefficient Blocks. This isolated experiment decodes each weight
value to its original F32 scalar before multiplying unchanged BF16 inputs,
accumulates through each original K interval, and reduces four lanes once at
its end rather than after every coefficient group. It is standard FP32 matrix
arithmetic with unchanged decoded weights, but its F32 association differs
from54. It is not a production fidelity decision and does not claim original
output-bit identity. Record numerical differences and full-vector three-P100
transport separately. No new allocation, global intermediate, device transfer,
model format or session state is introduced. Product FLOPs are unchanged;
weight dequantization adds per-value work while removing per-group reductions
and bias sums. Driver measurements decide whether this is useful. Current
production remains54; original-model and quality gates are still required.
