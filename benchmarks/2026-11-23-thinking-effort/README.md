# Thinking effort in Pi: Inkling 2 Mini vs Swift on Strata (2026-11-23)

Question: what do `--thinking low`, `medium` and `xhigh` cost and buy on the two
Strata models, for one real agent task about this project?

## Method

- `run-effort.sh <label> <level>`: real Pi (project `AGENTS.md`, skills, tools)
  inside a fixed clone of this repository at 45ef286, so every run reads the
  same code. Task A: explain in 5 points what `./ops/strata.sh start inkling`
  does end to end (configuration files, port, GPUs, engine arguments), read
  only. One run per model and level, one at a time.
- Inkling 2 Mini SQ-20: `--tp 3 --parallel 1 --draft suffix`, effort passed
  through by the patched server (it was forced to 0 upstream). Swift 1.5
  Flash-Next IQ3_S: the service's standard config (`parallel=2`, MTP spec).
- Answers scored against 10 facts from the clone's `ops/strata.sh` and
  `ops/run-strata-server.sh`; every extra claim checked against the source.
- Data is not versioned: `out/benchmarks/2026-11-23-thinking-effort/`
  (`results.jsonl`, `evaluation.md`, `raw/` with Pi JSONL, server logs,
  metrics).

## Results

| Model, level | Wall | Turns / tools | Output tokens | Thinking chars | Decode | Facts | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Inkling low | 41.0 s | 2 / 2 | 934 | 225 | 59.2 tok/s | 10/10 | awkward Romanian |
| Inkling medium | 68.6 s | 3 / 3 | 2,536 | 3,222 | 59.9 tok/s | 10/10 | all config keys, failure path |
| Inkling xhigh | 173.5 s | 4 / 5 | 7,387 | 16,973 | 56.0 tok/s | 10/10 | found the placement check reading the Swift config (true) |
| Swift low | 75.8 s | 2 / 2 | 1,765 | 2,328 | 48.1 tok/s | 9/10 | 2 small errors (sync "copies models", "one SSH session") |
| Swift medium | 69.8 s | 3 / 3 | 1,309 | 421 | 48.3 tok/s | 10/10 | one stray CJK character |
| Swift xhigh | 180.9 s | 6 / 10 | 4,706 | 8,800 | 46.4 tok/s | 10/10 | most precise: file:line references, peer-service placement path |

Findings:

- Medium is enough on both models for this task: all 10 facts, ~70 s. xhigh
  takes ~2.5x longer and only adds depth (one real observation on each model).
- Inkling low was already fully correct in 41 s; Swift low made two small errors.
- Inkling thinks much more than Swift at the same level (xhigh: 16,973 vs 8,800
  thinking characters); Swift is the reasoning-efficient post-train.
- Inkling's decode is ~20% faster (56-60 vs 46-48 tok/s); with xhigh the two
  end up equal in wall time because Inkling generates more.
- Swift writes cleaner Romanian.

Defect found and fixed during the run (pascal-llm patch): at xhigh Inkling
quoted `<|message_model|>` from a file comment inside its thinking; the parser
took it as a new message and sent the rest of the thinking to the visible
answer. Inside thinking only an end of message or an answer channel now ends it
(`_THINK_TAGS`, test added). The invalid run is kept in
`raw/before-parser-fix/`; the rerun (same seed) produced the same text, split
correctly.

Decision: `ops/pi.sh` starts Pi with `--thinking medium` (override with
`PASCAL_PI_THINKING` or a later `--thinking`).
