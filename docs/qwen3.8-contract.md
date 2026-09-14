# Qwen3.8-27B execution contract

Status: official source geometry captured on 2026-09-13. The deployed artifact
is the approved `Qwen3.8-27B-UD-Q4_K_M.gguf`; the QPack figures below remain
the reference for the rejected custom-runtime feasibility work.

Source: `Qwen/Qwen3.8-27B` `config.json`.

```text
layers                         64
recurrent layers               48
full-attention layers          16
hidden size                    5,120
dense intermediate size        17,408
attention heads                24
KV heads                       4
attention head dimension       256
recurrent key heads            16 x 128
recurrent value heads          48 x 128
recurrent convolution width    4
vocabulary                     248,320
maximum positions              262,144
MTP layers                     1
```

The compact matrix payload uses FP4 E2M1 values plus one UE8M0 scale per 32
input values: `17/32` stored bytes per unpadded matrix value. Major derived
payloads, excluding record headers/alignment and small F32 tensors, are:

| Organ | Stored bytes | GiB |
|---|---:|---:|
| token embedding | 675,430,400 | 0.6290 |
| vocabulary head | 675,430,400 | 0.6290 |
| 64 dense SwiGLU MLPs | 9,091,153,920 | 8.4668 |
| 16 full-attention projections | 891,289,600 | 0.8301 |
| 48 recurrent projection sets | 2,954,833,920 | 2.7519 |
| base matrix subtotal | 14,288,138,240 | 13.3069 |
| one MTP matrix set | 225,607,680 | 0.2101 |

The validated QPack is 14,775,390,208 bytes including its declared records.
The manifest remains authoritative over these derived numbers.

Exact F16 request state at maximum population:

```text
target full-attention KV       16.0000 GiB
MTP full-attention KV           1.0000 GiB
recurrent target state        151.5000 MiB (F32)
```

The first tensor-parallel split aligns the 17,408 intermediate channels to
block-32 boundaries as `5,824 + 5,792 + 5,792`. Gate/up rows and matching down
columns remain co-owned, avoiding movement of the large intermediate vector.

## Serving contract

```text
runtime             upstream llama.cpp, commit 4a89937354190cef5a97baf8eeb17336105eb72d
weight artifact     UD-Q4_K_M GGUF, 16,464,440,224 bytes
compute devices     CUDA 0,1,2: three P100s, equal tensor split
P40                  excluded from hot path
KV                   F16 K and F16 V
context pool         262,144 positions shared by four unified slots
batching             continuous
reasoning            Pi default xhigh
speculative decode   embedded MTP, maximum three proposed tokens
MTP KV               F16 K and F16 V
service lifecycle    manual start; disabled at boot
```

The GGUF contains the model's MTP block. The pinned runtime supports it through
`draft-mtp`; deployment configuration now loads the embedded head completely
on the same three P100s and limits a speculative cycle to three proposed tokens.
The configuration change does not affect an already-running process. The
non-MTP populated-262K run started before this change remains the baseline;
MTP is not qualified until the next controlled service start reports a
speculative context and real acceptance/timing telemetry.
