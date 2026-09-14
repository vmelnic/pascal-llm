# Image-generation models for the Pascal rig

Status: researched on 2026-09-14; no model was downloaded or executed.

## Decision

The best use of this host for high-quality text-to-image generation is not to
combine all 72 GiB into one slow heterogeneous inference graph. Run complete
FP16 pipelines on individual P100s and use the three P100s as independent
workers:

| Worker | Recommended model | Primary use |
|---|---|---|
| P100 0 | Juggernaut XL v9 | photorealistic and cinematic images |
| P100 1 | Animagine XL 4.0 | anime and illustrated/cartoon images |
| P100 2 | DreamShaper XL Lightning | fast general and stylized images |
| P40 | queue capacity, encoding, or an explicitly measured FP32/INT8 experiment | not the default FP16 denoiser |

This is a research recommendation, not a measured result on this rig. The
first implementation gate should qualify the common SDXL pipeline on one P100,
then replicate it without model-family-specific service code.

## Hardware constraints that determine the choice

The installed devices are four separate memory and compute domains. The three
P100s have peer access to each other; the P40 has no CUDA P2P path to them. See
`hardware.md` for the measured topology.

The P100 PCIe has 16 GiB HBM2, up to 732 GB/s memory bandwidth, 18.7 TFLOP/s
FP16, and 9.3 TFLOP/s FP32.[^p100] The P40 has 24 GiB GDDR5X, 346 GB/s, 12
TFLOP/s FP32, and 47 TOPS INT8.[^p40] Its larger VRAM does not make it the
better diffusion worker: NVIDIA documents high-throughput native FP16 on GP100,
whereas GP102-class Pascal devices execute FP16 at a much lower rate.[^mixed]

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
| Juggernaut XL v9 | SDXL, Diffusers FP16 | yes; model card states 8 GiB is comfortable | strongest first candidate for photorealism and cinematic grading | **first choice for realism**; review its additional paid-API licensing condition before commercial service[^juggernaut] |
| Animagine XL 4.0 | SDXL, F16 | yes | anime, illustration, character design, tag-controlled styles | **first choice for anime/cartoon**[^animagine] |
| DreamShaper XL Lightning | SDXL, FP16, four-step example | yes | fast general art, fantasy and stylized output | **first choice for fast iteration**[^dreamshaper] |
| SDXL Base 1.0 | official FP16 variant | yes | neutral baseline and compatibility oracle | keep as the reference pipeline, not necessarily the best final style[^sdxl] |
| SANA 1.6B 1024 | official FP16 transformer; VAE/text encoder must be BF16 or FP32 | official inference budget is 12 GiB | modern, efficient prompt-to-image alternative | **second experiment**: transformer FP16 plus FP32 VAE/text encoder; require an SM60 numerical gate[^sana-guide][^sana-models][^sana-diffusers] |
| PixArt-Sigma 0.6B | official FP16 | yes | efficient 1024/2K/4K generation | lightweight fallback; its own card notes limitations in photorealism, text and composition[^pixart] |
| Stable Diffusion 3.5 Medium | official BF16, three text encoders | not as a clean native Pascal path | good prompt following and typography | defer: below-24-GiB use relies on offload or removing/quantizing T5, which changes latency or quality[^sd35][^sd3docs] |
| FLUX.2 Klein 4B | official BF16, about 13 GiB VRAM | bytes fit; arithmetic contract does not | modern four-step image model | research-only until an FP16 conversion passes an independent quality gate on SM60[^flux2] |
| Z-Image Turbo | official BF16, 6B, under 16 GiB | bytes fit; arithmetic contract does not | strong realism, text and eight-step generation | research-only for the same reason; do not silently cast and claim support[^zimage] |
| FLUX.1 Schnell | 12B BF16 | raw weights alone are about 24 GB | strong general quality, 1-4 steps | reject for a P100; P40 capacity is marginal and its FP16 path is unsuitable[^flux1] |
| Qwen-Image | 20B MMDiT | no; raw 16-bit weights are about 40 GB | image text and editing | reject on this host without a quality-changing quantization/offload design[^qwenimage] |

## Why replicas beat one four-GPU image job

Diffusion denoising is an ordered loop: each denoising step consumes the result
of the previous step. Splitting one transformer across GPUs may solve a memory
problem, but each step then pays synchronization and activation transfers. On
this host that is particularly unattractive across the P100/P40 boundary,
which lacks P2P.

For the selected SDXL pipelines, one complete job already fits one P100. Three
independent P100 workers therefore provide:

- three simultaneous jobs without cross-GPU collectives;
- independent realism, anime and fast-general model residency;
- failure isolation and simple queue scheduling;
- a clean path to batch or replica scaling.

Frameworks such as xDiT do implement distributed parallelism for diffusion
transformers, but their published paths target newer accelerator classes and
do not establish SM60/SM61 support.[^xdit] They are relevant only if a future
model cannot fit one P100 and a full Pascal compatibility gate succeeds.

The P40 should not sit in the hot denoising collective merely to increase the
advertised VRAM total. Plausible uses are a separate low-priority worker,
image/video encoding, safety or post-processing models, and cold component
residency. Each use still needs measurement; none is assumed faster.

## Qualification plan

No download or service integration should start with all candidates. The
smallest useful sequence is:

1. Qualify upstream SDXL FP16 on one P100 with SDXL Base as the compatibility
   oracle. Require no BF16 kernels, no CPU offload, no NaN/Inf, and peak device
   memory below the 16 GiB physical limit.
2. Replace only the SDXL checkpoint with Juggernaut XL v9, Animagine XL 4.0,
   and DreamShaper XL Lightning. The runtime/API remains common.
3. Compare a fixed prompt set covering skin, hands, faces, text, wide cinematic
   scenes, low light, animation characters, prompt adherence and negative
   prompts. Preserve seeds and image metadata.
4. Run one cold and three warm 1024x1024 jobs per checkpoint. Report load time,
   peak VRAM, seconds/image and failures separately. This is a deployment gate,
   not a general benchmark campaign.
5. Start three independent P100 workers and submit three concurrent jobs.
   Accept only if each output is correct and concurrent wall time improves over
   serial execution without host-RAM or PCIe collapse.
6. Only then test SANA 1.6B. Its independent oracle is the official pipeline on
   supported hardware or FP32 output, compared against the proposed FP16/FP32
   Pascal placement with fixed seeds and perceptual plus human review.

Success for one style does not establish success for another. Photorealism,
cinematic composition and anime/cartoon output should each retain a small
curated acceptance set.

## Operational recommendation

Expose one standard image-generation API with an artifact-selected profile,
for example `realistic`, `anime`, or `fast-general`. The scheduler assigns a
complete request to the P100 already holding that artifact. Do not present the
four cards as a unified 72 GiB device, and do not make the P40 a dependency for
the first working service.

## Sources

[^p100]: [NVIDIA Tesla P100 PCIe datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf)
[^p40]: [NVIDIA Tesla P40 datasheet](https://images.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf)
[^mixed]: [NVIDIA, Mixed-Precision Programming with CUDA 8](https://developer.nvidia.com/blog/?p=7311)
[^memory]: [Hugging Face Diffusers memory optimization guide](https://github.com/huggingface/diffusers/blob/main/docs/source/en/optimization/memory.md)
[^juggernaut]: [RunDiffusion Juggernaut XL v9 model card](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)
[^animagine]: [Animagine XL 4.0 model card](https://huggingface.co/cagliostrolab/animagine-xl-4.0)
[^dreamshaper]: [DreamShaper XL Lightning model card](https://huggingface.co/Lykon/dreamshaper-xl-lightning)
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
