# Suffix-lookup drafts on Inkling 2 Mini TP3 (2026-11-23)

Goal: raise Inkling single-agent decode towards 70-80 tok/s with model-free
suffix-lookup drafts (`--draft suffix`, `patches/strata-next/01-inkling-tp-decode.patch`).
Service `pascal-strata` on 8081, SQ-20, `--tp 3 --parallel 1`, F16 KV.

## Scripts

- `suffix-replay.py`: offline replay of the real Pi runs in
  `out/benchmarks/2026-11-23-two-agents-real/raw/` with the server's tokenizer
  and template and a port of the drafter; step costs from the TP3 bench. Runs on
  x99e, no GPU.
- `replay-turn.py <run.jsonl> <out.json> [default]`: sends the last assistant
  turn's prompt of a real run (greedy unless `default`), prints time, usage and
  the reply's SHA-256 prefix.
- `run-agents.sh solo-a`: the real Pi agent task A (as in the two-agent test).

Data is not versioned: `out/benchmarks/2026-11-23-suffix-drafts/` (replay
output, replies, `STRATA_SFX_TRACE` traces and timings, Pi JSONL, server logs).

## Results

| Measurement | No drafts | Suffix drafts |
|---|---:|---:|
| Replay of task A's last turn, greedy, 700 tokens | 58.1 tok/s | 62.9 (host verify), 63.4 (device verify) |
| Pi agent A, long answer (~11K prompt) | 55.8 tok/s | 58.7 tok/s |
| Pi agent A, tool-call turns | 57 tok/s | 61-71 tok/s |
| Pi agent A wall (different replies, 1,711 vs 1,159 tokens) | 59.4 s | 47.2 s |
| Verify step M=4 in service | - | 33.6 ms (plain step 17.0 ms) |

Drafts cover about a quarter of the tokens of an ordinary answer; with sampled
Pi requests (`temperature 1.0`) 52-55% of offered tokens are accepted. Plain
decode at the same depth is 56.1 tok/s in `--bench` (11,264-token prompt), so
the service adds no measurable overhead. 70-80 tok/s is not reachable this way.

The greedy reply with drafts diverged from the plain one after 339 tokens at a
near tie (top-2 gap 0.11): the M=4 verify kernels differ numerically from the
M=1 kernels (0.07 logit at identical state). Not a drafter defect.

SQ-15 was also measured as a faster pack: 20.5 tok/s against 44.6 for SQ-20 on
the layer split (its expert codecs do not load on the TP3 row split). Rejected.
