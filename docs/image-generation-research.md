# Image-generation models for the Pascal rig

Status: RealVisXL, Juggernaut and Animagine passed functional image gates on
the isolated P40 on 2026-09-14.

## Runtime and API decision

`llama.cpp` does not provide text-to-image inference: its diffusion example is
an experimental text-generation architecture. The sibling
`stable-diffusion.cpp` project already provides the required SDXL CUDA runtime
and native OpenAI-compatible `POST /v1/images/generations`,
`POST /v1/images/edits`, and `GET /v1/models` routes. This repository therefore
pins and operates that upstream server instead of implementing another image
protocol or diffusion runtime.

The deployed image service uses one complete checkpoint on physical CUDA
device `3`, the P40, with no CPU parameter offload or cross-GPU split. Runtime
inspection reports explicit FP32 parameter conversion and a tiled VAE. It
listens on the VPN/LAN only and has no authentication layer; OpenAI SDK clients
may provide a dummy local API key, but the endpoint must not be exposed
directly to the public Internet.

The original P100 baseline produced a valid 1216x832 Juggernaut RGB PNG in
120.864 seconds and peaked at 14,735 MiB. The current P40 path produced a
photorealistic sample in 189.574 seconds and peaked at 13,537 MiB. All three
P100s remained empty during that request.

The author's standard RealVisXL V5.0 checkpoint, not a Lightning derivative,
is now the Open WebUI default for photorealism. It produced a coherent
1216x832 image through native ComfyUI in 274.01 seconds at 50 DPM++ 2M/Karras
steps. Juggernaut remains the alternative cinematic checkpoint and Animagine
remains the anime/illustration checkpoint. The user selects among these in the
frontend; Qwen does not choose the diffusion model.

Animagine uses the optimized checkpoint and its official 28-step, CFG 5,
Euler Ancestral preset. Its official negative prompt is installed as the
profile default because the OpenAI Images schema has no top-level negative-
prompt field. User prompts should still use the model's documented tag format
and quality suffix.

Animagine produced a strong 832x1216 anime sample in 170.187 seconds on P100
and 211.642 seconds on P40. The P40 output passed visual inspection without
tile seams. CUDA diffusion Flash Attention remains rejected as an SM60 latency
optimization: an identical Juggernaut request remained unfinished after more
than 208 seconds, versus 111.84 seconds without it.

FP32 without VAE tiling is not admissible on P40: Juggernaut completed
sampling, then decode reached 24,187 MiB and failed allocation. The one
corrective change, upstream `--vae-tiling`, reduced the measured request peak
to 13,537 MiB and completed. DreamShaper was removed from the deployment after
its inspected output failed the visual-quality gate.

## Decision

The best use of this host is not to combine all 72 GiB into one heterogeneous
image graph. Keep the P100 domain available for text and run one complete image
profile on the P40:

| Device | Recommended model | Primary use |
|---|---|---|
| P100 0-2 | active LLM profile | tensor/layer-parallel text inference |
| P40 | RealVisXL V5.0, Juggernaut XL v9 or Animagine XL 4.0 | isolated realism, cinematic or anime worker |

Only one image checkpoint is resident at a time. Profile selection changes the
artifact; it does not introduce a model-family branch in the service.

## Hardware constraints that determine the choice

The installed devices are four separate memory and compute domains. The three
P100s have peer access to each other; the P40 has no CUDA P2P path to them. See
`hardware.md` for the measured topology.

The P100 PCIe has 16 GiB HBM2, up to 732 GB/s memory bandwidth, 18.7 TFLOP/s
FP16, and 9.3 TFLOP/s FP32.[^p100] The P40 has 24 GiB GDDR5X, 346 GB/s, 12
TFLOP/s FP32, and 47 TOPS INT8.[^p40] NVIDIA documents high-throughput native
FP16 on GP100, whereas GP102-class Pascal devices execute FP16 at a much lower
rate.[^mixed] P40 is therefore slower than the P100 FP16 baseline, but its
larger memory admits the complete FP32 graph and isolates image work from the
text GPUs.

For an unquantized model with `P` parameters, the raw weight lower bound is:

```text
FP16_weight_bytes = 2 * P
BF16_weight_bytes = 2 * P
```

Equal byte counts do not imply equal support. Pascal predates BF16 tensor
arithmetic. An official BF16-only checkpoint is therefore not accepted merely
because its bytes fit; it needs an explicit FP16 conversion and numerical
quality gate. CPU offload can recover capacity, but transfers components over
PCIe and is not a latency optimization. Diffusers describes sequential CPU
offload as especially slow and model offload as the less aggressive option.[^memory]

## Candidate evaluation

| Candidate | Official precision/path | Capacity on one P100 | Expected visual strength | Verdict |
|---|---|---:|---|---|
| RealVisXL V5.0 | standard SDXL FP16 checkpoint | yes by measured P40 execution | strong natural photorealism, including adult content | **default realism profile**; visual gate passed[^realvisxl] |
| Juggernaut XL v9 | SDXL, Diffusers FP16 | yes; model card states 8 GiB is comfortable | general photorealism and cinematic grading | **cinematic alternative**; review its additional paid-API licensing condition before commercial service[^juggernaut] |
| Animagine XL 4.0 | SDXL, F16 | yes | anime, illustration, character design, tag-controlled styles | **first choice for anime/cartoon**[^animagine] |
| SDXL Base 1.0 | official FP16 variant | yes | neutral baseline and compatibility oracle | keep as the reference pipeline, not necessarily the best final style[^sdxl] |
| SANA 1.6B 1024 | official FP16 transformer; VAE/text encoder must be BF16 or FP32 | official inference budget is 12 GiB | modern, efficient prompt-to-image alternative | **second experiment**: transformer FP16 plus FP32 VAE/text encoder; require an SM60 numerical gate[^sana-guide][^sana-models][^sana-diffusers] |
| PixArt-Sigma 0.6B | official FP16 | yes | efficient 1024/2K/4K generation | lightweight fallback; its own card notes limitations in photorealism, text and composition[^pixart] |
| Stable Diffusion 3.5 Medium | official BF16, three text encoders | not as a clean native Pascal path | good prompt following and typography | defer: below-24-GiB use relies on offload or removing/quantizing T5, which changes latency or quality[^sd35][^sd3docs] |
| FLUX.2 Klein 4B | official BF16, about 13 GiB VRAM | bytes fit; arithmetic contract does not | modern four-step image model | research-only until an FP16 conversion passes an independent quality gate on SM60[^flux2] |
| Z-Image Turbo | official BF16, 6B, under 16 GiB | bytes fit; arithmetic contract does not | strong realism, text and eight-step generation | research-only for the same reason; do not silently cast and claim support[^zimage] |
| FLUX.1 Schnell | 12B BF16 | raw weights alone are about 24 GB | strong general quality, 1-4 steps | reject for a P100; P40 capacity is marginal and its FP16 path is unsuitable[^flux1] |
| Qwen-Image | 20B MMDiT | no; raw 16-bit weights are about 40 GB | image text and editing | reject on this host without a quality-changing quantization/offload design[^qwenimage] |

## Why one isolated P40 worker beats a heterogeneous image job

Diffusion denoising is an ordered loop: each denoising step consumes the result
of the previous step. Splitting one transformer across GPUs may solve a memory
problem, but each step then pays synchronization and activation transfers. On
this host that is particularly unattractive across the P100/P40 boundary,
which lacks P2P.

For the selected SDXL pipelines, the complete FP32 graph and tiled VAE fit the
P40. One request therefore uses no collectives and leaves the P100 text domain
untouched. Additional image concurrency would require another complete worker;
it is not obtained by treating the four cards as one memory pool.

Frameworks such as xDiT do implement distributed parallelism for diffusion
transformers, but their published paths target newer accelerator classes and
do not establish SM60/SM61 support.[^xdit] They are relevant only if a future
model cannot fit one P100 and a full Pascal compatibility gate succeeds.

The P40 must not sit in the hot P100 denoising or text collective merely to
increase advertised VRAM. Its accepted role is the separately measured image
worker. It is slower per image than one P100, but it enables simultaneous text
and image residency without a P100/P40 boundary in either graph.

## Qualification plan

No download or service integration should start with all candidates. The
smallest useful sequence is:

1. Completed: qualify upstream SDXL on one P100 without BF16 kernels, CPU
   offload, NaN/Inf, or overflow beyond the 16 GiB physical limit.
2. Completed for transport: replace only the checkpoint with Juggernaut XL v9
   and Animagine XL 4.0 Opt. The runtime/API remained common.
3. Compare a fixed prompt set covering skin, hands, faces, text, wide cinematic
   scenes, low light, animation characters, prompt adherence and negative
   prompts. Preserve seeds and image metadata.
4. Open only if real usage requires it: measure queue behavior and warm P40
   latency under simultaneous Pi load. Do not run a benchmark campaign without
   an observed production bottleneck.
5. Only then test SANA 1.6B. Its independent oracle is the official pipeline on
   supported hardware or FP32 output, compared against the proposed FP16/FP32
   Pascal placement with fixed seeds and perceptual plus human review.

Success for one style does not establish success for another. Photorealism,
cinematic composition and anime/cartoon output should each retain a small
curated acceptance set.

## Operational recommendation

Expose one standard image-generation API with an artifact-selected
`realvisxl`, `juggernaut` or `animagine` profile. The complete request stays on
P40 device `3`; text stays on P100 devices `0,1,2`. Do not present the four
cards as a unified 72 GiB device.

## Sources

[^p100]: [NVIDIA Tesla P100 PCIe datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf)
[^p40]: [NVIDIA Tesla P40 datasheet](https://images.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf)
[^mixed]: [NVIDIA, Mixed-Precision Programming with CUDA 8](https://developer.nvidia.com/blog/?p=7311)
[^memory]: [Hugging Face Diffusers memory optimization guide](https://github.com/huggingface/diffusers/blob/main/docs/source/en/optimization/memory.md)
[^juggernaut]: [RunDiffusion Juggernaut XL v9 model card](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)
[^realvisxl]: [RealVisXL V5.0 model card](https://huggingface.co/SG161222/RealVisXL_V5.0)
[^animagine]: [Animagine XL 4.0 model card](https://huggingface.co/cagliostrolab/animagine-xl-4.0)
[^sdxl]: [Stability AI SDXL Base 1.0 model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[^sana-guide]: [NVIDIA SANA inference guide](https://github.com/NVlabs/Sana/blob/main/docs/sana.md)
[^sana-models]: [NVIDIA SANA model zoo](https://github.com/NVlabs/Sana/blob/main/docs/model_zoo.md)
[^sana-diffusers]: [Hugging Face Diffusers SANA pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/sana)
[^pixart]: [PixArt-Sigma XL 2 1024 model card](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
[^sd35]: [Stability AI Stable Diffusion 3.5 Medium model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
[^sd3docs]: [Hugging Face Diffusers Stable Diffusion 3 guide](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_3.md)
[^flux2]: [Black Forest Labs FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
[^zimage]: [Tongyi-MAI Z-Image Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
[^flux1]: [Black Forest Labs FLUX.1 Schnell model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
[^qwenimage]: [Qwen-Image official repository](https://github.com/QwenLM/Qwen-Image)
[^xdit]: [xDiT official repository](https://github.com/xdit-project/xDiT)
