# Two simultaneous Pi sessions on Strata, 262,144 tokens per slot (2026-11-23)

Question: can the `pascal-strata` service (port 8081, P100 0,1,2) serve two
independent agents at once with a 262,144-token context each, for Swift and
for Inkling? Gemma (`pascal-llm`, P40, port 8080) ran throughout and was not
touched.

## Method

1. Inkling engine check, `strata-inkling --bench --tp 3 --gpus 2,0,1
   --max-context 262144 --kv fp16 --draft off --prompt-len 512 --gen 128`,
   with `--parallel 1` and `--parallel 2`. With `STRATA_BENCH_SAME_PROMPT=1`
   both slots get the P=1 prompt; every slot must emit the P=1 greedy tokens.
2. Inkling service switched to `--parallel 2` (`ops/run-strata-server.sh`);
   Swift already ran with `parallel=2`.
3. `run-pi-pair.sh <model>`: two real Pi processes (`./ops/strata.sh pi
   --no-session --mode json`, project `AGENTS.md`, skills, tools) started in
   the same second with distinct prompts `hi, include exact cuvântul ALFA|BRAVO
   în răspuns`. One run per model.
4. `sample.sh`: every 60 s, GPU memory/utilisation, host RAM, service state,
   `/health` (`samples.csv`) and the server's progress lines
   (`samples-engine.log`), appended immediately so an interrupted run keeps
   its data. The Swift pair (30 s) fell between two samples; its memory comes
   from the snapshot after start.

Artifacts: `raw/` (bench outputs, Pi JSONL/stderr, server logs, memory
snapshots, `/health` after each pair), `results.jsonl` (one line per session).

## Results

| Measurement | Inkling 2 Mini SQ-20, TP3 | Swift 1.5 Flash-Next IQ3_S |
|---|---|---|
| Runtime | Strata 0.2.20 + pascal-llm TP3, `--parallel 2` | Strata 0.2.20 layer split, `parallel=2` |
| Slots x context | 2 x 262,144 (`/slots`) | 2 x 262,144 (`/slots`) |
| KV | F16; global KV 161,536 tokens VRAM + 100,608 mapped host tokens on the two ranks owning 3 global visits (2,358 MiB host each), all VRAM on the third | F16; 32,768-token GPU window per layer, remainder in host RAM |
| VRAM P100 0/1/2 after start (MiB of 16,384) | 13,811 / 14,115 / 13,811 (1 slot: 12,919 / 11,895 / 12,919) | 15,439 / 14,929 / 14,929 |
| Host RAM used / available after start | 23.4 / 37.3 GiB | 59.3 / 1.4 GiB, 2 GiB swap full |
| Engine bench, 1 slot | 62.4 tok/s decode, 468.6 tok/s prefill | - |
| Engine bench, 2 slots (same prompt) | 88.3 tok/s total (44.2 per slot), 474.2 tok/s prefill; both slots' 128 tokens identical to 1 slot | - |
| Pi prompt per session | 7,013 + 7,013 tokens, both read in full | 5,677; session B reused 5,655 cached tokens (22 read) |
| Prompt reading | A: 7,013 in ~16 s (~440 tok/s); B: done at 26 s, first token at 32 s | done at ~24 s for both (~240 tok/s) |
| Decode while both generate | ~36-37 tok/s per session (~73 total) | ~24-28 tok/s per session (~53 total) |
| Server per-request figure | A 6.3 tok/s, B 38.1 tok/s (120 tokens each) | A 24.8 tok/s (122), B 28.2 tok/s (124) |
| Pi wall A / B / pair | 36.4 / 36.1 / 36.5 s | 29.4 / 29.8 / 29.9 s |
| Exit, stop reason | rc 0, `stop`, both | rc 0, `stop`, both |
| Correct marker, no cross-session leak | yes, yes | yes, yes |
| Service after the pair | `/health` ok | `/health` ok |
| Single-session reference (Pi `hi`) | 58.9 tok/s, 18.2 s | 36.6 tok/s, 28.4 s |

## Findings

- Both models serve two isolated 262K-configured sessions through Pi. This is
  admission and isolation at a ~6-7K populated context, not populated 262K.
- Inkling: one session's prefill starves the other's decode. Session A
  produced 14 tokens in the 16 s while B read its 7,013-token prompt (about one
  token per 512-token prefill chunk), hence its 6.3 tok/s request figure. The
  two prompts share a ~7K prefix but Inkling read both in full; Swift reused it.
- Inkling's second slot spills 100,608 tokens per global visit to mapped host
  RAM on two P100s. Exact F16, but attention past position 161,536 reads host
  memory over PCIe; that depth was not measured.
- Swift leaves 1.4 GiB host RAM available with swap full while Gemma also runs;
  two fully populated 262K Swift sessions are unqualified on host RAM.

Not run (proposals): two populated 262K contexts; decode-priority / smaller
prefill chunks for Inkling fairness; prompt-prefix reuse across Inkling slots.
