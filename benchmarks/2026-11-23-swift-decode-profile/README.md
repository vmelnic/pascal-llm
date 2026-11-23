# Swift decode profile on 3 x P100, layer split (2026-11-23)

Question: during Swift 1.5 Flash-Next IQ3_S decode on the service's engine mode,
how long does each P100 sit idle, and what share of the step is expert work?

## Method

- `profile-serve.py`: `nvprof --print-gpu-trace` around `strata --serve` with the
  service's exact arguments (`strata-swift-iq3_s.json` + `--layer-split auto
  --batch 2`, MTP `--spec 4`, resident experts, F16 KV), `pascal-strata`
  stopped. One `GEN 256` request with a real 2,123-token Pi prompt (task A,
  system + user, Swift template, low effort).
- `analyze.py trace.csv decode_ms`: the decode window is the engine's own
  decode time (2,972.2 ms, from `DONE`) ending at the last kernel. Per device:
  union of kernel and copy intervals; time by kernel family; how many GPUs are
  busy at once.
- Data is not versioned: `out/benchmarks/2026-11-23-swift-decode-profile/`
  (`analysis.txt`, `swift-timing.json`, `swift-prompt.txt`,
  `swift-trace.csv.gz`).

The reply stopped at 132 tokens: 132 tokens in 2,972 ms = 44.4 tok/s under the
profiler (the service measured 46-48 tok/s in Pi), prefill 11.0 s.

## Results

| GPU | Busy | Idle |
|---|---:|---:|
| P100 0 | 731 ms (24.6%) | 75.4% |
| P100 1 | 756 ms (25.4%) | 74.6% |
| P100 2 | 1,038 ms (34.9%) | 65.1% |

| GPUs busy at the same time | Share of the decode window |
|---|---:|
| none | 19.3% |
| exactly one | 78.3% |
| two or three | 2.4% |

| Kernel family (all GPUs) | Time | Share |
|---|---:|---:|
| dense projections (quantized MMVQ / BF16 MMVF) | 1,093 ms | 38.8% |
| routed experts gate/up/down | 520 ms | 18.5% |
| hyper-connections (`fused_gr`) | 335 ms | 11.9% |
| other small kernels | 241 ms | 8.6% |
| expert fetch from host (`fetch_blobs` + HtoD copies) | 181 ms | 6.4% |
| attention and Gated DeltaNet | 155 ms | 5.5% |
| output head (Q6_K, on GPU 2) | 132 ms | 4.7% |
| copies to host (DtoH) | 87 ms | 3.1% |
| router | 49 ms | 1.7% |
| sync waits | 25 ms | 0.9% |
| copies between GPUs (peer) | 0 ms | 0% |

## Findings

- The layer split runs the three P100s one after another: for 78% of the decode
  exactly one GPU works, and each GPU is idle 65-75% of the time.
- For 19% of the decode no GPU works at all (~4.3 ms per token): host-side work
  between kernels (the CPU loop, sampling, drafting, staging).
- Expert work is ~25% of GPU time (18.5% compute + 6.4% fetching non-resident
  experts from host RAM); the largest share is the dense projections (39%).
  Families are grouped by kernel name; the MTP drafter and verify rows use the
  same kernels and are not separated.
- Layer boundaries pass through host memory (DtoH + HtoD pinned copies), not
  peer copies.
- Upper bound for a perfect 3-way tensor split, from these numbers only: GPU
  work 2,525 ms / 3 = 842 ms plus the 573 ms with no GPU busy = 1.42 s for 132
  tokens, ~93 tok/s, before any exchange cost. Inkling's TP3 paid ~2 ms per token
  for exchanges; at that cost (264 ms for 132 tokens) the bound falls to ~79 tok/s.
  Swift has a different layer count and layout, so its exchange cost is not
  known. Not measured.

## Where the no-GPU time goes (`gaps.py`)

Every gap between the union of the three devices' intervals, classified by the
device active before and after it:

| Transition | No-GPU time | Share | Gaps |
|---|---:|---:|---:|
| 2 -> 0 (between verify windows) | 283.4 ms | 49.5% | 85, mean 3.3 ms; ~10 of 11-34 ms make ~200 ms |
| inside one stage (2->2, 1->1, 0->0) | 216.2 ms | 37.7% | ~57,000, mean 1-6 us (kernel launch gaps) |
| stage hand-off (0->1, 1->2) | 61.3 ms | 10.7% | 165, mean ~370 us |

Each large gap ends with a 92 KB host-to-device copy to all three GPUs, then the
next window.

## Hypotheses tested (same 2,123-token prompt, 256 max tokens, plain runs)

`profile-serve.py <config> <ids> 256 - <out.json> [ENV=V ...]` with
`GEN_KEYS`, `ENGINE_EXTRA_ARGS`, `PERF_OUT`, `PERF_EVENT`. The engine's output
is reproducible run to run (greedy is the default of a bare `GEN`).

| Run | Tokens | Decode | Same tokens as base |
|---|---:|---:|---|
| base (service arguments) | 142 | 48.9 tok/s | - |
| `STRATA_SPLIT_OVERLAP=1` | 132 | 51.1 tok/s | no (different text) |
| `STRATA_P2P_HANDOFF=1` | 142 | 48.5 tok/s | yes |
| `STRATA_SPLIT_OVERLAP=1 STRATA_P2P_HANDOFF=1` | 142 | 49.7 tok/s | yes |
| `--ple-io mmap` | 142 | 48.3 tok/s | yes |
| `--ple-row-cache 16777216` | 142 | 48.8 tok/s | yes |
| `temperature=0.0` | 142 | 48.8 tok/s | yes |
| `temperature=1.0 top_p=0.95 top_k=20` (Pi's) | 139 | 50.7 tok/s | no (sampled) |

- Stage hand-off: overlap + P2P removes it for +1.6%. Overlap alone changed the
  generated text with the host-memory hand-off; not used.
- PLE n-gram table reads (`--ple-io direct` from SSD by default): no effect.
- Host-side sampling: no effect (greedy is not faster).
- `perf` (paranoid lowered to 1 for the run, restored to 4): the 20 CPU expert
  pool threads and the host thread spin-wait; the MTP drafter's host work is
  0.3% of samples. ~5,000 page faults on `experts.bin` during decode (cold
  experts read by the CPU pool through mmap).

## Conclusion

The layer split's no-GPU time is not one removable cost. The largest part, the
gaps between windows, is at most ~10% of decode (48.9 -> ~54 tok/s if it
vanished entirely); its cause was not isolated by the knobs above and would need
timers in the engine's window loop. Investigation stopped here by decision:
the bound does not justify the work.

Tensor parallelism through Strata's own `strata-split` was also examined
(`docs/swift-tp3-plan.md`): it builds for sm_60 after five small changes
(`out/benchmarks/2026-11-23-swift-decode-profile/strata-split-sm60.patch`, not
applied), but it needs either every expert in VRAM (51.8 GB > 51.5 GB) or a
page-locked host copy of every expert plus the embedding table (~80 GB > 60 GiB
RAM). Rejected for this host.
