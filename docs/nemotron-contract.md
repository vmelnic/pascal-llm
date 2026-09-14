# NVIDIA Nemotron-3.5-Lightning-30B-A3B serving contract

Status: configuration prepared, not started or qualified. The running Qwen
service was intentionally left untouched.

## Artifact identity

```text
repository          ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
HF revision         8a08a1c81dadcc75d35dbb96016cfd344b632e67d
main artifact       NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
main bytes          18,898,091,584
MTP artifact        mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
MTP bytes           1,155,907,520
architecture        nemotron_h_moe
weight format       official Q4_0 GGUF conversion
license             OpenMDW 1.1
```

## Artifact-declared architecture

```text
base blocks              52
routed-MoE blocks        26
recurrent SSM blocks     20
full-attention blocks     6
hidden width          2,688
attention heads          32
KV heads                  2
K/V width                128
routed experts           128
active experts             6
expert width           1,856
shared experts             1 x 3,712
native context     1,048,576
MTP layers                 1, stored in a separate artifact
```

## Prepared deployment

```text
compute devices      CUDA 0,1,2: three P100s
P40                  excluded from the hot path
split                 layer, equal proportions
context pool          262,144 positions shared by four slots
target KV             F16 K and F16 V
draft KV              F16 K and F16 V
speculation           external MTP artifact, at most three proposals
sampling              temperature 1.0, top-p 0.95
thinking              binary on/off in the embedded template
Pi default            xhigh means on, with the shared 16K reasoning cap
```

The pinned runtime has native `nemotron_h_moe` and external MTP support, but
explicitly does not support tensor split for this architecture. The common
launcher therefore accepts an optional external draft path declared by a
profile. Profiles with embedded MTP, including Qwen, retain their previous
arguments unchanged.

The embedded Jinja template declares `enable_thinking`, history-thinking
truncation, standard tool definitions and structured `<tool_call>` output. It
does not declare graded `reasoning_effort`; Pi therefore exposes only `off`
and `xhigh`, rather than pretending that intermediate effort levels have model
semantics. The 16K reasoning cap limits hidden reasoning, not the visible
answer, and leaves Pi's required answer reserve intact.

## Qualification boundary

The preparation is complete when the profile, launcher arguments and Pi model
catalog validate offline. Production claims require a later explicit model
switch and these ordered gates:

1. Initialize the target and external MTP model on the exact pinned binary.
2. Confirm F16 target/draft caches, four slots and layer placement from logs.
3. Complete one direct streaming `hi` without malformed reasoning output.
4. Restart Qwen through the unchanged embedded-MTP profile and repeat its
   direct gate.
5. Complete a real Pi tool loop with normal project instructions and skills.

Until then Nemotron is selectable and launch-ready, but unqualified.
