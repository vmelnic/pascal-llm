# Real Pi agents on Inkling 2 Mini TP3, two 262K slots (2026-11-23)

Goal: check whether two agents can work on Inkling at the same time, using real
multi-turn Pi tool loops rather than a single `hi`. The service was `pascal-strata`
on 8081: Strata 0.2.20 + `patches/strata-next/01-inkling-tp-decode.patch`,
`--tp 3 --parallel 2`, F16 KV, 262,144 tokens per slot. Gemma on P40 ran
throughout and was not touched.

## Method

- `run-agents.sh solo-a|solo-b|pair`: real Pi agents (project `AGENTS.md`,
  skills, read/bash tools, xhigh thinking) run inside a fresh clone of the
  repository, so they cannot change the working tree. Both tasks are read-only.
  - A: explain what `./ops/strata.sh start inkling` does, end to end.
  - B: tabulate every Swift decode tok/s value reported in `docs/`.
- `run-pi-pair.sh inkling`: the earlier two-session `hi` pair, used as the gate
  for the prefill fix.
- `sample.sh`: GPU, host RAM, service and `/health` every 60 s
  (`samples.csv`), plus server progress lines (`samples-engine.log`).
- Data is not versioned. The scripts write to `out/benchmarks/<this folder>/`
  on the workstation:
  - `raw/`: Pi JSONL, server logs, bench outputs and metrics;
  - `results.jsonl`: the runs after the fixes;
  - `results-before-fix.jsonl` and `raw/before-fix/`: the runs before the
    prefix-reuse and broadcast fixes;
  - `samples.csv` and `samples-engine.log`: the 60 s samples.

## Defects found and fixed (all in the pascal-llm patch)

1. **Prefill starved the other slot's decode.** Each step fed 1 decode token
   plus up to 511 prompt tokens, about 1.1 s, so the decoding slot got about
   1 tok/s. Fix: `--prefill-share 0.5`. While a slot decodes, each prefill step
   is followed by decode-only steps until the decoding slot has had half the
   time. Prefill chunks are unchanged.
2. **No prompt reuse across turns, even for one agent.** With tools, the
   server ends every prompt with an empty thinking turn,
   `<|message_model|><|content_thinking|><|end_message|>`, which the next
   turn's history does not contain. The engine's checkpoint at
   `prompt_len - 1` therefore never matched, and every turn re-read the whole
   history: 16 requests, 278,702 prompt tokens, 0 reused. Fix:
   `--tail-role-token 200001` (`<|message_model|>`), the option the main Strata
   engine already has. The checkpoint, which includes the ShortConv state and
   the SWA ring window, is saved in front of the last role token.
3. **TP3 deadlock in long prefill.** A broadcast (the owner of a global visit
   sends to the other ranks) of M=512 rows is 12 chunks over 4 rotating receive
   slots. The owner waits for nobody, so it could overwrite a slot that a slower
   rank had not read yet; that rank then spun forever. Observed after 15 turns:
   the prompt stuck at 512 of 34,206 tokens for 6 min, all three P100s at 100%
   utilisation and ~44 W. Fix: a one-float barrier exchange after every 3
   broadcast chunks, plus a 60 s exchange timeout that prints the epoch and
   traps instead of hanging.

Checks after the fixes:
- `--bench` P=1 tokens are identical to the earlier build (62.4 tok/s).
- Three 16,384-token prefills completed at 431 tok/s with no hang.

## Results

| Measurement | Before fixes | After fixes |
|---|---|---|
| `hi` pair: decode of the slot that decodes while the other reads its prompt | 14 tokens in 16 s (~1 tok/s) | ~27 tok/s (81 tokens in ~3 s); prefill-share fix only |
| Task A alone | 73.5 s, 2 turns, 1 tool call, no reuse | 59.4 s, 3 turns, 2 tool calls; 18,023 of 29,881 prompt tokens reused |
| Task B alone | engine deadlock after 13 turns / 12 tool calls at a 34,206-token prompt; 1,286 s, no answer | no deadlock: 79 turns, 78 tool calls (4 tool errors), context up to 86,960 tokens; stopped by the user after 5,271 s without an answer (agent loop) |
| Prompt reuse (service totals) | 0 of 16 requests, 0 of 278,702 tokens | 80 of 83 requests, 5,036,389 of 5,222,143 tokens (96.4%) |
| Turn time at 15-33K context | 34-95 s (full re-read at ~380 tok/s) | median 3.0 s, max 3.7 s (last 12 turns, ~86K context; median 199 new prompt tokens) |
| Reuse misses | every turn | 2 after the first request; the one at 86,960 tokens re-read everything in 189 s |
| Decode per turn | 50-59 tok/s | median 37.6 tok/s at ~86K context |
| VRAM P100 0/1/2 | 13,819 / 14,123 / 13,819 MiB | 13,817 / 14,145 / 13,829 MiB |
| Host RAM available (min) | 36.2 GiB | 35.4 GiB |
| Two agents at once (`pair`, A + C) | not run | see the next section: 112.4 s together vs 97.6 s one after the other |

## After the retry and slot fixes: two agents at once

Two more defects found by these runs, both fixed in the patch:

- **A retried turn replaced the good prompt checkpoint.** Pi retried one
  request after `Stream ended without finish_reason`. The retry resumed exactly
  at the checkpoint, the code then fell back to `prompt_len - 1`, and the next
  turn missed. That was the 86,960-token miss above (189 s). Now the
  checkpoint in front of the last role token is kept.
  `check-reuse-retry.py`: the retry reads 3 tokens, the next turn 44 of 718.
- **A request running alone always used slot 0.** On the Inkling engine the
  server's single-request path (`GEN`) used slot 0 even when the conversation
  lived in slot 1, overwriting the other conversation's cache. Now it takes the
  free slot whose cache matches its prompt. Effect: agent A's 4th turn
  (13,055 tokens) read 28.8 s from token 0 before the fix and 1,823 new tokens
  (reused 11,232) after.

Task C is a read-only summary of `docs/hardware.md`. It replaced B for the
pair because B loops.

| Run | A wall | C wall | Total | Output tokens A + C |
|---|---:|---:|---:|---:|
| one after the other (each alone) | 59.4 s | 38.2 s | 97.6 s | 1,711 + 816 |
| both at once, before the slot fix | 135.2 s | 80.4 s | 135.2 s | 1,873 + 1,400 |
| both at once, after the slot fix | 112.3 s | 83.1 s | 112.4 s | 1,896 + 1,490 |

Both agents answered correctly in every run, with no tool errors. Running
both at once was not faster than running them one after the other. Each
agent's first turn has to read its own ~7K-token prompt from scratch; the
second agent cannot reuse the first one's identical system prompt, and the
two reads take about 32 s one after the other (A's first turn: 47.6 s for 46
tokens). Only decoding overlaps, and the agents generated more tokens when run
together. On 3 x P100 one Inkling agent at a time gets the same work done as
two, and each answer arrives sooner.

## Open issues

- **Inkling loops on tool use.** In task B it repeated
  `grep ... docs/magnitude-swift-qualification.md` and paged its output in
  3-8 line slices for over an hour without answering. That is model behaviour,
  not engine speed.
- **Rare prompt-reuse misses.** Both causes found here are fixed (see the previous
  section); no other miss was seen.
- **No prompt sharing between conversations.** Two agents with the same ~7K
  system prompt each read it in full.
- **Fixed in the scripts after this run:**
  - Pi ignored the 30 min `timeout` (SIGTERM); `run-agents.sh` now uses
    `timeout -k 30`.
  - The sampler had two gaps (60 and 18 min) from a hanging `ssh`; every call
    is now bounded by `timeout 30`.

## Decision: back to one slot

The service runs `--parallel 1 --tail-role-token 200001`. Measured after the
restart (data: `raw/parallel1/`):

- one 262,144-token slot, global KV fully in VRAM on all three ranks (no host
  offload);
- VRAM 12,941 / 11,917 / 12,941 MiB, host RAM used 18 GiB;
- Pi `hi`: rc 0, 18.1 s, 7,001-token prompt in ~16 s, 75 tokens at 57.5 tok/s.
