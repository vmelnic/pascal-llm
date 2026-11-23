# Two real Pi agents on Swift at once (2026-11-24)

Question: does Swift 1.5 Flash-Next IQ3_S (service config: Strata 0.2.20 layer
split, MTP, `parallel=2`, 262,144 context) finish two agents' work sooner in
parallel than one after the other?

## Method

`run-agents.sh solo-a | solo-c | pair` (copy of
`benchmarks/2026-11-23-two-agents-real/run-agents.sh`): real Pi (project
`AGENTS.md`, skills, tools, `--thinking medium` from `ops/pi.sh`) in a fixed
clone of the repository (45ef286). Task A: explain `./ops/strata.sh start
inkling` in 5 points; task C: summarize `docs/hardware.md` in 5 points; both
read-only. One run per mode. Data is not versioned:
`out/benchmarks/2026-11-24-swift-two-agents/` (`results.jsonl`, `raw/` Pi
JSONL and server logs).

## Results

| Run | A | C | Total | Output tokens A + C | Decode per request |
|---|---:|---:|---:|---:|---:|
| A alone | 96.4 s | - | 96.4 s | 2,495 | 47.5-49.8 tok/s |
| C alone | - | 37.8 s | 37.8 s | 882 | 45.9-49.3 tok/s |
| one after the other | | | 134.2 s | 3,377 | |
| A + C at once | 92.1 s | 68.0 s | 92.1 s | 2,015 + 928 | 19-31 tok/s while both decode, 42.5 tok/s after C ends |

All runs: rc 0, `stop`, no tool errors. Expert cache 94-99.6% VRAM hits.

## Findings

- Two agents at once finish 31% sooner than one after the other. The second
  slot fills the layer split's idle GPUs (each P100 is idle 65-75% with one
  stream, `benchmarks/2026-11-23-swift-decode-profile/`).
- Each agent decodes slower while both run (19-31 instead of ~48 tok/s); the
  longer task barely lost time, the shorter one took longer.
- Sampled outputs differ in length between runs (A: 2,495 vs 2,015 tokens);
  one run per mode.
- Host RAM is at its limit with Swift and Gemma (about 1 GiB available); two
  long, fully populated contexts at once are not verified.
- Inkling on TP3 gained nothing from two agents
  (`benchmarks/2026-11-23-two-agents-real/`).
