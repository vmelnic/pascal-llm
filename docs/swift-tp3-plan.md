# Swift 1.5 Flash-Next tensor-parallel decode on 3 x P100 (plan, 2026-11-23)

Goal: faster single-stream decode for Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ3_S
through real Pi, by letting the three P100s work on every layer at once instead
of one after another. Same weights, F16 KV.

## Starting point (measured)

`benchmarks/2026-11-23-swift-decode-profile/`: the service's engine (`strata
--serve`, layer split, MTP) decodes 132 tokens in 2,972 ms under nvprof (44.4
tok/s; 46-48 tok/s in Pi). Exactly one GPU is busy 78.3% of the time, none
19.3%; each GPU is idle 65-75%. GPU time: dense projections 38.8%, routed
experts 18.5% + 6.4% fetched from host RAM, hyper-connections 11.9%, attention
and GDN 5.5%, head 4.7%.

Equations (per token, from that profile):

```text
serial step        t = sum_g busy_g + t_host = 2,525 / 132 + 573 / 132 = 19.1 + 4.3 = 23.4 ms (44 tok/s)
ideal 3-way split  t = 19.1 / 3 + 4.3 = 10.7 ms (~93 tok/s), before exchanges
exchanges          2 per layer (attention sum, FFN sum) x 48 = 96 per token; at Inkling's
                   measured ~2 ms per token for 144 exchanges: ~1.3-2 ms -> ~12.5 ms (~79 tok/s)
memory             routed experts 50.3 GB (IQ3_S) > 3 x 16 GiB: a share stays in pinned host RAM
                   (ExpertCache, mapped); embedding table 27.5 GiB in RAM (as today)
```

These are bounds, not results.

## Direction: port Strata's split engine to sm_60

Strata v0.2.20 already ships `strata-split` (engine/src/split, sFNX, MIT): every
layer of Qwen3.8-Flash-Next split across the cards (query/value heads, FF
224/224/192), exchanges through page-locked host memory (no peer access), MTP,
PLE, an expert cache with RAM residency. Upstream builds it for sm_86 / gfx1201
only. On sm_60 the decode kernels need no tensor cores; what fails is:

| Item | sm_60 change |
|---|---|
| `__dp4a` (sm_61+) in dense, hc, Q4_K experts | exact int32 emulation (`sm60_compat.cuh`) |
| `__match_any_sync`, `__nanosleep` (sm_70+) | 32-shuffle mask; clock spin |
| QSA index scorer on WMMA (`k_idx_score_tc`) | the existing scalar `k_idx_score` |
| prompt attention on WMMA (`k_pf_attn`) | the existing decode-attention path (`FNX_PF_ATTN_DEC`) |
| prompt experts on MMA (`k_moe_mma`) | the existing q8_1 decode-kernel path (`FNX_PF_MOE_Q8`) |

## Phases and gates

Rule as for Inkling: one change at a time, measured; one local correction after
a failed gate, a second failure stops for reassessment.

1. **Build and self-test.** `strata-split-3` builds for sm_60;
   `strata-split-selftest` passes on each P100.
2. **Load and answer.** Swift IQ3_S loads on GPUs 0,1,2 with the experts that do
   not fit in pinned RAM; a greedy reply is coherent and matches the layer split's
   on the first tokens (upstream reports 29/61 identical suite answers between the
   two engines: different arithmetic, same model).
   Gate: decode > 48 tok/s (the service today) at a ~2K prompt.
3. **Serve through Pi.** The server's `--split-engine on` path with the service
   config; real Pi `hi` and task A (as in the effort benchmark).
   Gate: Pi decode above the layer split's 46-48 tok/s, answers correct.

## Phase 1 result (2026-11-23)

`strata-split-3` builds for sm_60 with the five changes above (patch in the
scratch tree, not yet in `patches/strata-next/`). `strata-split-selftest` is
HIP only, so there is no CUDA self-test; Phase 2 is the first real check.

Blocker found before Phase 2, from the source (`model.cpp` `expert_buf`,
`model.hpp` ExpertCache): the split engine either places every routed expert
in VRAM (`FNX_EXPERT_RAM_GIB=off`), or, with its expert cache, keeps a
page-locked host copy of **every** expert and VRAM slots for the hot ones.

```text
all in VRAM   experts 50.3 GB + dense 1.5 GB = 51.8 GB  >  3 x 16 GiB = 51.5 GB (before KV, MTP, buffers)
expert cache  pinned host: experts 50.3 GB + embedding table 27.5 GiB (29.5 GB) = 79.8 GB  >  60 GiB host RAM
              (41 GiB available with Inkling running and Gemma on P40)
```

Neither fits. Options: (a) change the split engine to a static placement, hot
experts only in VRAM and only the cold ones page-locked in RAM, read by the
kernels over PCIe (no host copy of resident experts, no swapping); at a 32K
context about 10 GB stays in RAM, so RAM need is ~27.5 GiB + 10 GB + 3 GiB
buffers (~43 GiB), at the limit with Gemma running; (b) keep the layer split
and remove its 19% host idle; (c) stop here.

## Decision (2026-11-23)

Option (a), the split engine with static expert placement, was not started:
it would cap the context near 32K and leave host RAM at the limit beside Gemma.
Option (b) was investigated (`benchmarks/2026-11-23-swift-decode-profile/`):
stage hand-off, PLE reads and host sampling were ruled out by measurement; the
between-window gaps bound the gain at ~10%. Stopped. Swift stays on the
layer split; the sm_60 port of `strata-split` is kept only as an unapplied
patch under `out/benchmarks/2026-11-23-swift-decode-profile/`.
