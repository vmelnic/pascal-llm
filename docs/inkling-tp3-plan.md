# Inkling 2 Mini TP3 decode plan

Goal: >= 100 tok/s single-stream decode for Inkling 2 Mini SQ-20 on
3 x P100 through real Pi, unchanged weights, F16 KV. Budget and measurements:
`feasibility.md`, section "Measured decode decomposition and the >=100 tok/s
plan". Code lives as patches in `patches/strata-next/` on Strata v0.2.20
(`a34979fc`), built by `ops/install-strata-next.sh`.

Rule: finish each phase, measure its gate, then start the next. No side
experiments. On a failed gate: one local correction, re-measure; a second
failure stops the plan for reassessment.

## Phase 1 - PoC: three P100s read concurrently (fast result)

Scope: new `--tp3` decode mode in `strata-inkling` CUDA backend, M=1, one
slot, greedy, `--bench` only, 32K context.

1. Load: trunk (dense, 3.67 GB) replicated on GPUs 0,1,2; experts assigned
   whole by id (`e % 3 == g`, no repack); KV replicated per GPU.
2. Dense projections: each GPU computes its 1/3 of output rows of every
   `linear`/`linear_multi` (pointer offset into the replicated BF16 rows),
   then an in-kernel P2P all-gather.
3. MoE: router replicated; each GPU runs the unchanged expert kernels on its
   local selected experts; partial outputs P2P all-reduced.
4. Small kernels and attention replicated on identical inputs.
5. P2P exchange kernel: write slice into peers' buffers + epoch flag,
   spin-wait; no host round trip.

Gate 1: `--bench --tp3` decode tok/s > 38.6 (expected ~55-70) and the same
greedy tokens as the unmodified engine for 128 generated tokens.

## Phase 1 result (2026-11-23): Gate 1 passed

`--bench --tp 3` (SQ-20, 128-token prompt, 128 generated, `--draft off`):
46.9 tok/s decode vs 40.3 for the unmodified layer split; all 128 greedy
tokens identical; prefill 310.8 vs 204.6 tok/s. Experts per GPU 1032/1032/1008,
none on host. Per-token profile (rank 1): dense gemv 2.4 ms, all-gather
(154/token) 3.2 ms, MoE all-reduce (48/token) 2.3 ms, experts 4.3 ms, small
kernels ~6.3 ms, idle gaps ~2.8 ms. Rank 0 also runs the full 1.6 ms head, so
ranks 1-2 wait for it. A standalone probe of the same exchange kernel on the
three P100s costs 11.8 us at 1 float, 17.1 us at 2,304, 26.8 us at 6,912:
the floor is synchronization (system fence + flag), not payload.

## Phase 2 - reach >= 100 tok/s (ordered by measured gain)

Done, each with all 128 greedy tokens identical to the unmodified engine:

| Step | Decode |
|---|---:|
| Phase 1 TP3 (expert split by id, fenced exchange) | 46.9 tok/s |
| 2.1 exchange as one 8-byte {value, epoch} store, 8 blocks (floor 11.8 -> 5.4 us) | 52.2 tok/s |
| 2.2 unembed split by vocabulary rows on all ranks | 55.5 tok/s |
| 2.3 MoE row split: features 384/GPU, exact load-time repack | 59.8 tok/s |
| 2.4a SM60 E2M1 decode via PRMT byte table (bit-identical on 2.7e8 words) | 63.6 tok/s |
| 2.4b dedicated M=1 SF4 shard kernels (down 25.9 -> 18.8 us; gate/up 38.6 -> 41.5 us) | 63.5 tok/s |
| 2.5 CUDA graph per rank with device epoch base (no measurable gain: the GPU is busy, not gap-bound) | 63.2 tok/s |
| 2.6 M=1 attention-residual mix, one float4 of every state per thread | 63.9 tok/s |

Frozen baseline: 63.9 tok/s, all 128 greedy tokens identical; patch SHA-256 prefix 5d357b2cbd52102f
(copy kept as `out/01-inkling-tp-decode-63.9tps.patch`).

## Reassessment after 2.4b-2.6 (three corrections, +0.4 tok/s together)

The ~15.6 ms step is ~1,270 kernels that are each latency-bound (5-25 us
regardless of payload), so per-kernel tuning cannot reach 10 ms (estimated
ceiling ~92 tok/s). Measured TP3 step cost by tokens per step (prefill in
chunks, steady state): M=1 15.0 ms, M=2 21.2 ms, M=4 27.0 ms (6.8 ms/token),
M=8 55 ms. The fixed per-step latency amortizes over tokens, so the new
direction is speculative decoding with the SQ-20 embedded DSpark drafter on
the TP ranks: every rank runs the drafter on bit-identical replicated state,
verifies the same draft window through the TP exchanges and applies the same
acceptance. >= 100 tok/s needs ~2.7 accepted tokens per M=4 step, or a faster
M>1 path (dense row split for M <= 16).


Profile after 2.3 (rank 1, ms/token): MoE gate/up 2.71 (143 GB/s), MoE down
1.67 (115 GB/s), dense+head gemv 2.96, exchanges 2.02, small kernels ~5.5,
gaps ~1.3. Remaining steps by measured gain:

4. Expert kernels (superseded by the reassessment below). After 2.4a/b gate/up is 41.5 us/call
   (192 GB/s): every CTA reloads and converts the 9.2 KB input vector and
   synchronizes; next correction is one precomputed half input per visit read
   through the read-only cache, 16 features per CTA. Expected -0.8 ms.
5. Multi-block M=1 versions / fusion of the slowest small kernels
   (`attn_res_mix` 17 us on one block, router, cat_rms, mlp_res, ctrl): -2 ms.
6. One CUDA graph per rank for the M=1 step (device-side epoch): -1.3 ms.
7. Fewer exchanges: one all-gather for all attention-input projections of a
   visit, dense gemv grid tuning for 1/3-size row ranges: -1 ms.

Gate 2: `--bench --tp 3` decode >= 100 tok/s, same greedy tokens.

## Phase 3 - serve and qualify

1. `--serve` with `--tp3` (single slot), 262,144 context: attention split by
   KV heads 3/3/2 so KV is not replicated.
2. `--kld-eval` top-1 agreement vs. the unmodified engine.
3. One real Pi `hi` through the `pascal-strata` provider, tok/s reported.
4. Docs, ops scripts, service.

Gate 3: Pi `hi` decode >= 100 tok/s; KLD/top-1 matches; Swift restored.

## Speculative decoding on TP3 (2026-11-23)

DSpark runs on all three ranks from bit-identical state; ranks are checked to
emit identical tokens every step. Greedy speculative output is identical to
plain greedy for all 128 tokens in every row below (`--draft dspark`, same
benchmark), 2.17 accepted tokens per step.

| Change | Decode |
|---|---:|
| DSpark enabled on TP ranks (beam tree up to 12 nodes) | 40.4 tok/s |
| dense row split also for M <= 16 (prefill M=8: 145.5 -> 188.7 tok/s) | 43.2 tok/s |
| SM60 drafter on the FP16 cuBLAS cache (as SM70/75; drafter only proposes) | 47.0 tok/s |
| target-head exact argmax: BF16 vocab slice per rank + {value, index} merge | 51.0 tok/s |

Step budget: ~42.5 ms per speculative step. Verify alone (dense 5.1, MoE 6.9,
exchanges 3.2, small kernels ~8 ms) is ~23 ms; the drafter's int8 vocab screen
(5 ms) is the largest remaining drafter cost. 100 tok/s needs a step <= 21.7 ms
at 2.17 tokens/step (<= 27 ms at 2.7), so speculative decoding alone cannot
reach the gate on this hardware; plain M=1 decode is bounded near ~92 tok/s by
~1,270 latency-bound kernels per token unless the per-visit pipeline is fused
into a few persistent kernels.

## Phase 3 status (2026-11-23)

- Global-visit KV ownership: the 8 global visits are owned 3/3/2; only the owner
  writes their KV and runs their attention, then broadcasts `d_attn_out`
  (`k_tp_bcast`). 262,144 context allocates on every P100; tokens identical;
  plain 62.4 tok/s, DSpark 51.0 tok/s, 58.1 tok/s at 2K depth.
- Served by the `pascal-strata` service (`./ops/strata.sh start inkling`, port
  8081) with `--tp 3`; prompt checkpoints forwarded to every rank.
- Real Pi `hi`: 57.6 tok/s decode, ~440 tok/s prefill, 17.9 s wall.
- KLD (512 tokens): prefill 0.000889 / top-1 99.22%, single-token steps
  0.001348 / 98.63%, unmodified engine's own step noise 0.001249 / 98.43%.
- Service Pi `hi` on 8081: 58.9 tok/s, 18.2 s. Phase 3 complete.


## Phase 4 - fused M=1 visit pipeline (>= 100 tok/s)

Rule as before: one fusion per step, measured with `--bench --tp 3 --draft off`
(128 greedy tokens identical to the unmodified engine), then the next. Targets
from the eager per-kernel profile (us per call, calls per token):

| Step | Fuse | Kernels now | Expected saving |
|---|---|---|---:|
| 4.1 | router gemv + top-8 select + expert binning | 6.7 + 18.0 + 8.6 us, x48 | ~1.2 ms |
| 4.2 | rel-position table + split-K flash + merge (M=1, one kernel per visit) | 10.7 + 11.3 + 10.2 us, x48 | ~1.0 ms |
| 4.3 | MoE reduce8 + all-reduce + MLP short-conv + residual | 3.6 + 10.6 + 4.7 us, x48 | ~0.5 ms |
| 4.4 | loop controllers (3x down + 3x apply + rms) at M=1 | 12.6 + 14.6 + 6.6 us, x24 | ~0.4 ms |
| 4.5 | one all-gather per visit for all attention-input projections | ~3.2 exchanges/visit -> 2 | ~0.5 ms |
| 4.6 | dense gemv for 1/3-size row ranges (more CTAs per row block) | 15.4 us, x153 | ~0.8 ms |

Gate 4: `--bench --tp 3` >= 100 tok/s with identical tokens, then one real Pi `hi`.

Phase 4 results:

| Step | Decode (128 tokens identical) |
|---|---:|
| start (global-visit KV ownership) | 62.4 tok/s |
| 4.1 fused M=1 router (last-CTA select + binning) | 64.8 tok/s |
| 4.2 fused rel-bias + split-K + merge (on-the-fly bias) | 60.9 tok/s, rejected |
| 4.2 correction: per-split bias in shared memory | 61.9 tok/s, rejected (second failure), reverted |


## Phase 5 - go/no-go: one persistent kernel per P100 (2026-11-23)

Gate: one fused M=1 SWA visit per rank <= 200 us (engine TP3 path: 15.4 ms /
48 visits ~ 320 us), else stop. PoW: `tools/poc/inkling-tp3-megakernel-visit.cu`,
one cooperative launch per token and P100 (112 CTAs x 256 threads) running 48
SWA visits with the engine's per-rank shapes and bytes: depth mix + RMSNorm,
q|r|gate|k|v|sinks BF16 rows 1/3 (8,600 rows), post-QKV/ShortConv4/FP16 KV
store/rel table, SWA-1024 split-K flash, merge * sigmoid(gate), `wo` rows 1/3,
FP32 router + top-8, SF4 gate/up 384-feature shards + SITU, SF4 down, rank-order
all-reduce. 5 grid barriers and 3 in-kernel P2P LL exchanges per visit; random
weights (a timing gate), and all ranks end with a bit-identical residual.

| Run | us/visit | ms/token (48 SWA visits) |
|---|---:|---:|
| first version | 230.8 | 11.08 |
| correction: exact bit-built E4M3, 14 attention splits (one round), rel table in halves, two expert rows in flight per warp | 230.3 | 11.05 |

Rank-0 CTA-0 phase times (us, corrected run): qkv gemv 32.8, flash 16.4,
`wo` 11.3, gate/up 39.9, down 24.6, and waits for other ranks / barriers
~60 (barrier after post-QKV 13.3, after gate/up 13.3, all-reduce 29.7). SF4
gate/up reaches ~200 GB/s, limited by software E2M1/E4M3 decode on SM60, and
cross-rank skew is paid at every exchange. Result: gate failed twice; rejected.
The full model would add 8 global visits, 36 loop-controller passes and the
head, so a fused path is bounded near ~80-90 tok/s on 3 x P100.

## Two slots on TP3 (2026-11-23)

`--parallel 2 --tp 3` needs no engine change: batched decode (`mode 1`) uses the
same row-split projections, owner attention + broadcast per slot and expert
shards. Startup auto-places the second slot's global KV: 161,536 tokens in VRAM
and 100,608 mapped host tokens (2,358 MiB) on the two ranks that own 3 global
visits. `--bench --parallel 2` (same prompt per slot, `STRATA_BENCH_SAME_PROMPT`)
emits the 1-slot greedy tokens in both slots; 88.3 tok/s total (44.2 per slot).
Two concurrent Pi sessions: `benchmarks/2026-11-23-two-sessions/`.

## Multi-turn serving fixes (2026-11-23)

Real Pi tool loops (`benchmarks/2026-11-23-two-agents-real/`) exposed three
defects, now fixed in the patch:

1. Prefill starved a decoding slot (1 decode token per 512-token step).
   `--prefill-share F` adds decode-only steps after each prefill step until the
   decoding slots have had (1 - F) of the time. Prefill chunks are unchanged.
2. No prompt reuse on tool turns. The served prompt ends with an empty
   thinking turn that the next turn does not repeat, so the `prompt_len - 1`
   checkpoint never matched. `--tail-role-token ID` checkpoints in front of
   the last role token (`<|message_model|>` = 200001), as the main Strata
   engine does.
3. Chunked broadcasts (M=512: 12 chunks over 4 receive slots) could overwrite
   a slot a slower rank had not read, because the owner waits for nobody.
   `tp_bcast` now inserts a one-float all-reduce barrier after every 3 chunks,
   and `tp_ld_wait` traps after 60 s without data instead of hanging.
   Greedy tokens are unchanged.

Serving decision (2026-11-23): one slot (`--parallel 1 --tail-role-token 200001`).
Two agents at once finished in 112.4 s against 97.6 s one after the other, and
the second slot pushes the tail of every slot's global KV beyond 161,536 tokens
into mapped host RAM. With one slot, all 262,144 tokens of KV stay in VRAM
(12,941 / 11,917 / 12,941 MiB), and Pi `hi` decodes at 57.5 tok/s with the
7,001-token prompt read in ~16 s (18.1 s wall).
