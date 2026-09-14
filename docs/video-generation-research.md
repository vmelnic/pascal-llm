# Video-generation models for the Pascal rig

Status: researched on 2026-09-14; no model was downloaded or executed.

## Decision

There is no current frontier video model with an official, clean Pascal path
that turns this rig into a fast modern video generator. The useful options are
narrower:

1. **CogVideoX-2B FP16** is the first true text-to-video candidate. Its model
   card explicitly includes older GTX 1080 Ti execution and an FP16 model,
   making it the strongest evidence-backed Pascal candidate.[^cog-card]
2. **Stable Video Diffusion XT FP16** is the first image-to-video candidate.
   Generate a high-quality still with the image stack, then animate it.[^svd]
3. **AnimateDiff FP16** is the first stylized/cartoon short-loop candidate and
   can reuse SD 1.5 or SDXL image checkpoints.[^animatediff-docs]

Wan, LTX, SANA Video, FramePack and other recent frontier paths are primarily
BF16 and/or explicitly target newer NVIDIA generations. Their parameter bytes
may fit somewhere in the aggregate 72 GiB, but that does not establish a
working or fast SM60/SM61 execution path.

## Why video is materially harder than images

An image diffusion model denoises one spatial latent. A video model adds a
temporal dimension and retains activations for multiple frames. A simplified
latent workload scales as:

```text
work ~ denoising_steps * latent_height * latent_width * latent_frames
memory ~ weights + temporal_activations + attention/workspace
```

The exact constants depend on the architecture, but the multiplicative frame
dimension is real. More VRAM can enable more frames or a larger model; it does
not automatically reduce generation time. Splitting every denoising step across
cards also adds communication. This is why the P100-only P2P domain and the
P40 must be evaluated separately.

## Candidate evaluation

| Candidate | Task and official path | Pascal evidence | Verdict |
|---|---|---|---|
| CogVideoX-2B | text-to-video, FP16; official examples include a 720x480, 49-frame clip | model card explicitly states GTX 1080 Ti support; native SAT FP16 is listed at 18 GiB and Diffusers FP16 at 12.5 GiB | **first T2V experiment**; start with the official FP16 low-memory path, then test a two- or three-P100 distributed path only if it is genuinely supported[^cog-card][^cog-repo] |
| Stable Video Diffusion XT | image-to-video, FP16, 25 frames at 576x1024 | Diffusers documents CPU offload, forward chunking and decode chunking below 8 GiB | **first I2V experiment** on one P100; quality of the initial still is controlled separately[^svd][^svd-docs] |
| AnimateDiff | animation motion module over SD 1.5; SDXL beta is documented | official SDXL beta example reports about 13 GiB for 1024x1024x16, within one P100 | **first cartoon/stylized loop experiment**; lower confidence for photorealistic temporal consistency[^animatediff-repo] |
| Wan2.1 T2V 1.3B | text-to-video; official low-memory claim is 8.19 GiB | official Diffusers example uses BF16; no official FP16 Pascal quality result | defer until FP16 output passes an independent temporal-quality gate[^wan21] |
| Wan2.2 TI2V 5B | text/image-to-video, modern 720p path | official minimum is framed around a 24 GiB RTX 4090-class device | reject as the first Pascal path; P40 capacity is not equivalent to supported compute[^wan22] |
| LTX-Video 2B distilled | eight-step text/image-to-video | official configuration selects BF16 | defer; do not assume a silent FP16 cast preserves quality or kernel compatibility[^ltx][^ltx-config] |
| SANA Video | 2B 480p/720p and newer variants | official model zoo is BF16 | defer for the same unsupported-precision reason[^sanavideo] |
| FramePack | long-video working-memory design | repository states RTX 30/40/50 and requires FP16 plus BF16; GTX 10/20 are not tested | reject on P100/P40 without new upstream Pascal support[^framepack] |

## Recommended execution topology

### Text-to-video lane

Start CogVideoX-2B on one P100 using its official FP16 path. If the complete
pipeline does not fit without harmful offload, test the official distributed
path on two P100s. The third P100 remains free for image work or another job.

A three-P100 job is accepted only if end-to-end clip latency improves over the
best one- or two-P100 configuration. Capacity alone is not sufficient. Do not
include the P40 in the hot collective: it has no P2P with the P100s, much lower
memory bandwidth, and no comparable FP16 throughput.

### Image-to-video lane

Use the image service to create the key frame, then send that image to Stable
Video Diffusion XT on one P100:

```text
text prompt -> SDXL still image -> SVD XT temporal generation -> video encoder
```

This is not equivalent to native text-to-video, but it gives explicit control
over the main composition and style. It is the most defensible route to short
cinematic clips on this hardware.

### Cartoon and stylized lane

Use Animagine/SDXL for the reference image or checkpoint style, then AnimateDiff
for a short sequence. This route favors style consistency and loops over long,
physically coherent cinematic motion.

### P40 role

The P40 may run video encoding, auxiliary post-processing, a separate queued
job, or cold components if measurement proves a net win. It should not be
inserted into the frame-by-frame denoising path simply to use all installed
cards. CPU video encoding is also a valid baseline; encoding is not normally
the dominant neural inference cost.

## Qualification gates

The purpose of the first tests is to answer deployment questions, not to build
a broad benchmark collection.

1. Import and execute the upstream model on SM60 without BF16, unsupported
   FlashAttention, or a hidden CPU fallback.
2. Generate one fixed 5-6 second, 480p-class clip. Record model load, TTFF,
   total clip time, peak VRAM per GPU, host RAM, PCIe traffic and output FPS.
3. Repeat the same seed once warm. A warm result must not be reported as cold.
4. Inspect prompt adherence, identity/object persistence, geometry, motion
   continuity, flicker and visible temporal artifacts. A successfully encoded
   MP4 is not a quality pass.
5. Compare one P100, two P100s and three P100s only for CogVideoX if all use an
   official supported execution path. Stop adding cards as soon as wall time
   fails to improve.
6. Run two independent jobs on separate P100s. For multiple users, aggregate
   throughput and queue latency matter more than forcing one clip across every
   device.
7. Verify model license and derivative/API conditions before exposing any
   public or paid service.

No A100, H100, RTX 4090 or RTX 50-series timing should be scaled linearly and
presented as a P100 estimate. Tensor cores, precision support, memory systems
and kernels differ too much. The only truthful latency number for this rig is
the result of the real clip gate above.

## Deferred distributed frameworks

xDiT supports sequence, ring, tensor and pipeline parallel methods for many
diffusion transformers, including video architectures.[^xdit] It is useful
evidence that distributed DiT execution is possible, but it is not evidence
that its modern kernels support SM60 or that communication pays off over PCIe
on this host. It belongs after one working FP16 model, not before it.

## Sources

[^cog-card]: [CogVideoX-2B official model card](https://huggingface.co/zai-org/CogVideoX-2b)
[^cog-repo]: [CogVideo official repository](https://github.com/zai-org/CogVideo)
[^svd]: [Stable Video Diffusion XT model card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
[^svd-docs]: [Hugging Face Diffusers Stable Video Diffusion pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/svd)
[^animatediff-docs]: [Hugging Face Diffusers AnimateDiff pipeline](https://huggingface.co/docs/diffusers/api/pipelines/animatediff)
[^animatediff-repo]: [AnimateDiff official repository](https://github.com/guoyww/AnimateDiff)
[^wan21]: [Wan2.1 official repository](https://github.com/Wan-Video/Wan2.1)
[^wan22]: [Wan2.2 official repository](https://github.com/Wan-Video/Wan2.2)
[^ltx]: [LTX-Video official repository](https://github.com/Lightricks/LTX-Video)
[^ltx-config]: [LTX-Video 2B distilled official configuration](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.8-distilled.yaml)
[^sanavideo]: [NVIDIA SANA Video documentation](https://github.com/NVlabs/Sana/blob/main/docs/sana_video.md)
[^framepack]: [FramePack official repository](https://github.com/lllyasviel/FramePack)
[^xdit]: [xDiT official repository](https://github.com/xdit-project/xDiT)
