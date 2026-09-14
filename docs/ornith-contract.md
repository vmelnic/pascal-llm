# Ornith-1.5-35B-A3B serving contract

Status: base text integration qualified, 2026-09-13. Vision, MTP, populated
maximum context and concurrency remain separate gates.

## Source identity

```text
source model       ornith-ai/Ornith-1.5-35B-A3B-FP8
serving artifact   ornith-ai/Ornith-1.5-35B-A3B-GGUF
HF revision        12393612fd4f730ff5aadc23e9b8f9648aa49ceb
weight format      official Q4_K_M GGUF
artifact bytes     21,713,463,040
vision projector   official BF16 mmproj, 902,822,240 bytes
license            MIT
```

The source checkpoint uses compressed FP8/BF16 tensors, but Pascal GPUs do not
provide native FP8 execution. The deployed artifact must be described as
Q4_K_M. The projector is downloaded and validated with the text artifact; it
is not enabled until the text service passes and a real image request is
qualified.

## Architecture

```text
family                    Qwen3.5-compatible hybrid MoE
layers                    40
schedule                  3 Gated DeltaNet + 1 full attention, repeated
hidden size               2,048
attention heads           16 x 256
KV heads                  2 x 256
experts                   256
active experts/token      8
expert intermediate       512
native context            262,144
MTP layers                1
```

Runtime capability selection comes from the GGUF architecture and metadata;
common service code does not branch on the Ornith name.

## Initial deployment

```text
compute devices       CUDA 0,1,2: three P100s
P40                    excluded from the hot path
split                  tensor, equal thirds
KV                     F16 K and F16 V
context pool           262,144 positions shared dynamically
parallel slots         4
reasoning              enabled by the embedded template
MTP                    disabled; separate qualification not yet run
sampling               request-controlled; general default 0.6/0.95/20
```

The model card recommends temperature `0.6`, top-p `0.95`, top-k `20` for
general use and temperature `1.0` for published benchmark reproduction. Pi or
another OpenAI-compatible client may override sampling per request.

## Qualification boundary

Completion requires, in order:

1. Completed: candidate download through HF Xet.
2. Completed: exact revision, filenames, byte sizes and SHA-256 hashes
   validated before atomic promotion.
3. Completed: GGUF metadata accepted by the pinned `llama.cpp` binary.
4. Completed: direct streaming `hi` returned coherent visible output.
5. Completed: Qwen restarted through the common launcher and passed the same
   direct gate with MTP and F16 target/draft KV intact.
6. Completed: Ornith finished a real Pi read-tool call with project
   instructions enabled.

Vision, MTP, populated 262K, sustained concurrency and performance are separate
gates and must not be inferred from basic chat success.
