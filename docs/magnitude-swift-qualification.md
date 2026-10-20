# Magnitude / Swift qualification on X99E

Date: 2026-10-09; native CUDA checks continue on 2026-10-10 (host timezone). Active request: investigate and test
https://github.com/magnitudedev/magnitude with the user's existing Swift GSQ
and Heretic artifacts on X99E, excluding P40.

Completion condition: verify the published Linux runtime, discover and resolve
both exact local artifacts, establish the actual device and numerical policy,
and run real Pi with independent concurrent tool sessions if a compatible
runtime configuration can load. A healthy HTTP service or catalog speed
estimate alone does not qualify inference or production throughput. The user
subsequently requested removal of the filename-based MTP exclusion; that
source change and its original-filename discovery gate are part of this task.
The next user request also authorizes fixing the concrete loader failure.
The current scoped gate is the user-requested native CUDA port using installed
libraries. Full Swift/Pi release is withheld while the independent model
admission, per-device placement and exact-F16 policy gates remain outstanding.

## Current integrated status (2026-10-10)

The active request is faster original-model Magnitude inference on the three
P100s. No llama.cpp comparison runs are part of this gate. Installed CUDA
12.4, original GGUFs, exact DenseF16 history, maximum ten MTP proposals and
configured context 262144 remain; P40 is excluded. Older pending/refused
entries below are historical evidence, superseded by integrated outcomes.

Patches 01–24 load original GSQ across all three P100s and produce coherent
native generation at 8.200016 tok/s, versus 7.428979 with 01–21 and the native
01–17 baseline of 5.066839 tok/s. All 192 response tokens and MTP acceptance
(156 proposed, 101 accepted) match 01–21 exactly. This still fails the active
throughput objective. Attribution identifies projections as 88.06% of GPU
work. Actual CUDA reflection reports dense-expand GEMV at 206 registers per
thread, allowing only one 256-thread block on a P100 SM.

Patch 25 streams one weight k-block at a time while retaining the F32 fold
order. The integrated release build passed in 5m20s; binary SHA-256 is
`51037f6514f7423f50f60a22b74d2f9210ffe889d21cfda886519d7ed5a15f88`.
GSQ generated at 12.178975 tok/s with profiling disabled; all 192 response
choices match 01–24 exactly. The renewed end-to-end attribution completed;
its diagnostic process is stopped. Actual compiled
GEMV reflection now reports expansion 64 registers, output 59 and recurrent
projection 61, with no local spill. The completed real request establishes readiness, response identity and the
reported bounded-context throughput. It does not qualify populated long context
or the requested speed. The earlier diagnostic and production processes are
stopped.

Heretic reached READY in 23.50s and completed a coherent native request at
4.075757 tok/s. Two simultaneous real Pi sessions completed their independent
`read` tool loops successfully, with project instructions, normal skill
loading and tools enabled. Their elapsed times were 329.24s and 333.25s,
with final usage totals 4606 and 3942 tokens; this does not qualify a fully
populated 262144-token context. No cross-session answer contamination was
observed in this bounded gate. Cancellation, saturation and long-context
fairness remain unqualified. Heretic was then stopped before starting GSQ.

The 01–17 GSQ PID 205087 reached READY in 60.75s and is now stopped. Its warmed physical VRAM was
7373 / 6759 / 6559 MiB on P100s 0 / 1 / 2, and 0 MiB on P40. Original model
weights were imported on every stage; one warm-up forward traversed the three
F32 transfer edges. Its coherent 192-token generation passed at 5.066839 tok/s.
Two simultaneous real Pi read-tool sessions passed in 303.11s and 288.05s,
with final usage 4545 and 3922 tokens. Those bounded tool loops establish
neither the requested speed nor populated 262144-token concurrency.

## Explicit F16 history integration (2026-10-10)

Current gate after user steering: run the integrated executable and the exact
Swift artifact on X99E, then correct failures on that path. Separate GSQ and
legacy projection matrices were explicitly interrupted before completion;
neither is marked passed. The integrated release build uses the installed
CUDA 12.4 and exposes only the three P100s. No driver or toolkit is changed.

All 315 affected original-tensor import fixtures passed on CUDA in 394.93 s,
with exact payload/decode checks in packet, rows16 and mma16 layouts. This is
source import evidence, not generation or tok/s. The service now receives one
immutable explicit F16/MTP-10 policy for both assessment and resolution. The
complete service/all-target source check passed in 2m23s. The integrated
service initially refused MTP width 10 because the generic chain bound was 8.
Patch 09 raises that bound to 16 without charging the maximum when 10 is selected.
The rebuilt service advances to the physical history-capacity refusal below.

### Actual original-model runs on X99E

Both unchanged original GGUFs were invoked through `magnitude-engine`, with
`--device cuda --kv-codec dense-f16 --mtp-proposals 10 --context-tokens 262144`,
installed NVRTC 12.4, and `CUDA_VISIBLE_DEVICES=0,1,2`. Vulkan was disabled for
these processes. The active path selects one physical device; the visibility
list does not distribute weights or state across those cards.

The first actual GSQ CUDA load reached native preparation but rejected the
recurrent scan's ROWS=128 default (62,896 shared bytes versus P100's 49,152).
Patch 10 selects the previously qualified ROWS=32 form (47,152 bytes). The
second load passes that kernel's built-in census in 87.62 s and advances through
actual quantized projection units. It was explicitly stopped while still
preparing kernels, before full-model readiness, to follow the user's request
to avoid further test matrices. Its log is retained as
`out/magnitude-swift/swift-direct-initial-census-interrupted.log`; partial
preparation is not model-generation qualification.

The existing state-capacity check originally ran after this expensive kernel
preparation. Patch 11 moves the identical metadata-only state derivation before
opening the device or tuning, and carries that plan into graph construction.
No guard, precision setting, context limit or kernel-validation policy is
relaxed. The real Linux release build passed in 22.26 s.

| Original model | Native load time | Exit | Actual result |
| --- | ---: | ---: | --- |
| Swift GSQ IQ2_S MTP | 3.02 s | 1 | `one context exceeds the device domain's history capacity` |
| Swift Heretic i1 Q4_K_M | 3.07 s | 1 | Same physical history-capacity refusal |

The executable SHA-256 is
`19d418f54e8aa24bf5e4b1c1731a5a0ab9525f84319c9bfb111e4b041ad1ae77`.
Commands, artifact paths, policy, exit codes and logs are retained in
`out/magnitude-swift/swift-native-model-runs/results.json` and its sibling
logs; host copies are in
`/home/vm/pascal-llm/work/magnitude-build/swift-native-model-runs/`.
The actual service refusal is separately retained in
`out/magnitude-swift/evidence-swift-cuda-mtp10-capacity/`.

Target-plus-MTP exact F16 history alone is 18,253,611,008 B at 262,144 tokens,
more than one physical P100's 17,059,545,088 B before weights or workspace.
Three-device placement of weights, history, recurrent state and exact transfer
boundaries is still absent from Magnitude's active native executor. No model
generated tokens, so neither tok/s nor concurrent Pi qualification is claimed.
Owned test processes are stopped. Final host inspection reports all GPUs
visible, zero allocated VRAM, and P100 temperatures of 35/38/36 C; P40 was
excluded from inference throughout. Drivers, clocks and power limits were not
changed by this CUDA-port/run work.

The current request authorizes completing the original Swift models in Magnitude,
including source codecs, three-P100 execution and F16 history. The candidate now
has `dense-f16`: projected activations retain their dtype, checked attention binds
an independent history element, target and MTP state layouts derive F16 planes,
and native append/read and tuning inputs use those descriptors. The first source
check rejected a dense ABI leaf referenced from a shared helper also included by
affine entries. Entry-owned history type macros corrected that contract; the
complete executor/state check then passed.

The new mixed-dtype test uses the Swift attention geometry (4 KV heads, 6 query
heads per KV head, width 256), slab crossings, empty and fresh spans, exact F16
payload comparison including subnormal/signed-zero rounding, and a subsequent
read of appended history against an independent scalar F32 softmax. On P100,
all 1/8/9/17-row cases passed in 95.77 s; the CPU gate passed in 219.4 s.
These are kernel results, not model/Pi qualification.

The CPU test fixture failed twice before execution: first it supplied CUDA's
static `G`, then the local correction omitted CPU's required `I`. Editing was
stopped for an end-to-end reassessment. Both declarations require `P,S,I,U,F,N,NV`;
only CUDA requires static `KV,G`. The new plan derives the common specialization
once and adds the CUDA-only dimensions explicitly. No numerical tolerance or
production kernel is changed to fix these fixture errors. Original failure logs
must remain in the evidence directory alongside the corrected run.

Remaining request gates: all seven missing GSQ encodings with exact source-value
proof, separately fitting device placements and measured transfers, original
model generation, and concurrent real Pi tool loops. No full-model throughput
is claimed yet.

## Reproducible identities

- Investigated main: `b2f79f00f4aa8d4ca6ad97f37a8c457ddba4e8a9`.
- Tested release: `@magnitudedev/cli@0.2.6`, release source commit
  `38f0adb8f6470642aaa15f7aa46ac2d135359d23`.
- Linux archive: `magnitude-icn-base-linux-x64-gnu.tar.gz`, 340,704,150 B,
  SHA-256 `d106fb1b99c6381c3904d4b40e78ca968896de3c3bf5ac23209041e5e4cc2e33`.
  Size and hash were verified before extraction on X99E.
- Native build reported by the binary:
  `0.1.0+src.e387f90ada78d09c51e79a42`; release-mode Linux x86_64,
  CPU/CUDA/Vulkan backends, Rust 1.91.1.
- GSQ: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf`, 9,607,981,056 B,
  SHA-256 `6ce61b532e45d7113e2b48b8dff9c104ff521230fa119eb0c3e75a44fb9f1cbf`.
- Heretic: `Swift-1.5-Qwen3.8-27b-heretic.i1-Q4_K_M.gguf`, 16,810,715,936 B,
  SHA-256 `fc3b8413d6b1a707c40cb1df43d247a3de64237be7da15e3f04e345557cb3054`.

The isolated installation is `/home/vm/pascal-llm/work/magnitude-0.2.6/`.
Discovery fixtures use hard links to the original GGUF files; inode equality
is verified, no tensor bytes are rewritten, and no second model download is
requested. The service binds only `127.0.0.1:18843`, with separate inventory
and cache roots. The test controller terminates the temporary service.
Pi home settings and the existing llama.cpp service are not replaced.

## Evolving finding ledger

| Claim | Evidence | Impact / confidence | Required action |
|---|---|---|---|
| Runtime archive executes on X99E CPU | `version --json`, `doctor`, `/health` returned ready | Verified transport/bootstrap; not inference | Keep distinct from generation gate |
| Host initially could not expose NVIDIA GPUs; subsequently repaired | Kernel `7.0.0-38-generic`; no `/dev/nvidia*` or `/proc/driver/nvidia/version`; `modinfo nvidia`: module not found; `nvidia-smi` fails; PCI still enumerates three P100 and P40 | Initial external blocker; resolved after the user requested immediate GPU recovery | Current-kernel module and tracking metapackage installed; `nvidia-smi` and llama.cpp CUDA enumeration pass |
| Published CUDA runtime rejects Pascal even with a working driver | Release `inference/seismic/backends/cuda/src/profile.rs`: rejects compute capability below SM80; `inference/docs/compatibility.md`: kernels require BF16 MMA and `cp.async` | Verified in the published release; native CUDA port now passes the selected helper, attention, projection and recurrent gates below | Complete the remaining operator/model gates without advertising absent hardware instructions |
| Native CUDA port uses the installed libraries | Patches 03/04; all-three-P100 independent helper gate; 33 CUDA unit tests; prefill, dense projection and 27B recurrent host-oracle gates; rebuilt service identity below | Verified selected operators and installation; full-model/Pi execution remains separate | Quantized projection gate passes; retain the separate admission/placement/fidelity blockers |
| Vulkan detour was stopped at the user's request | Matching user-space libraries installed without kernel-driver changes; default headless GLX ICD failed; a process-local EGL ICD enumerated the P100 cards, but no model ran | Device enumeration only; no inference qualification | Follow the user's selected CUDA route; no further Vulkan work |
| Production selection is one device | Release `engine/executor/src/platform/selection.rs`: selects one `DeviceInfo`; service `configurations.rs`: one automatic device; load plan contains one allocation domain | Verified source; no three-P100 tensor parallelism in this active path | Implement and qualify multi-device placement/communication before replacing the three-P100 runtime |
| Target-plus-MTP recurrent metadata is now admitted | Patch 02; 15 family tests; real original-header test; both real load-plan requests advance beyond recurrent flags; Heretic completes CPU load | Verified correction of the 65-versus-64 adapter failure | Preserve target/head separation and malformed-suffix rejection |
| Stock GSQ original filename is excluded by discovery | `service/models/src/inventory.rs::is_execution_companion_name` excludes any filename component `mtp`, independent of whether the file is a complete target | Verified source; original `-mtp.gguf` is hidden by this heuristic | User-requested source fix removes `mtp` from filename-only exclusions; original file is retained |
| IQ2_S directory parsing is not executable weight support | After the metadata correction, original GSQ load-plan returns HTTP 409: `unsupported source encoding Iq2S`; artifact enum accepts it but the executable source/resident maps omit it | Verified executable-format limitation | Implement and numerically verify the exact decoder/import path; do not re-quantize the user's model |
| Default numerical policy differs from this project's gate | `engine/src/options.rs`: default KV codec `AffineK8V4`; Qwen35 decoder description specifies BF16 activations | Verified source; dense alone does not promise IEEE F16 KV | Explicit exact-F16 policy and numerical verification before production acceptance |

## Runtime evidence

The first isolated server run returned `/health` ready and `/api/v1/hardware`
with only `host-cpu`: 22 physical / 44 logical cores, 65,162,358,784 B RAM.
Enabled backends do not mean usable GPU devices. `doctor` reports engine
loading, not GPU compatibility.

The first 60-second discovery observation ended with
`reconciliationComplete: false`, no models and `/v1/models` empty. This is a
bounded observation, not proof that the weights are unsupported. The inventory
cache did inspect Heretic as architecture `qwen35`, Q4_K Medium, 27,320,697,856
parameters, context 262,144, one next-token-prediction layer, and text modality.
Reconciliation also hashes installed packages. A subsequent explicit refresh
returned `reconciliationComplete: true` with Heretic and the temporary GSQ
hard-link alias. Original GSQ load preview returned HTTP 404; Heretic load
preview returned HTTP 409, `unsupported_model`, with
`invalid per-layer recurrent flags`.

Both original GGUFs were inspected directly with the installed GGUF reader:
`qwen35.block_count = 65`, `nextn_predict_layers = 1`, and 65 boolean recurrent
flags. Magnitude calculates `layer_count = 65 - 1 = 64`, then insists the
flag array has 64 entries. The original arrays describe 64 trunk blocks plus
the MTP block. This verifies the concrete header-adapter mismatch without
changing model data. It precedes the separate IQ2_S import limitation.

## User-requested MTP filename fix

The source patch removes only the `mtp` filename token from the draft-name
heuristic. Explicit draft metadata (including Eagle3 and DFlash) still excludes
companions. No architecture-specific runtime branch, target weight rewrite,
quantization change or file rename is introduced. This discovery-layer change
does not fix the independently verified 65-versus-64 recurrent-flag mismatch
or introduce Pascal CUDA / multi-device support. The normative discovery
contract and focused inventory tests were updated together.

Patch: `patches/magnitude/01-discover-embedded-mtp-targets.patch`, against the
release source above. Source is staged on X99E under
`/home/vm/pascal-llm/work/magnitude-source/`. Its independent Rust toolchain is
under `work/magnitude-build/`; shell startup and existing runtime installs are
unchanged. Both focused Rust tests passed locally: original MTP-named complete
target discovery and continued explicit-companion exclusion. The X99E release build completed successfully in 6m13s with Rust 1.98.0.
The isolated patched service returned healthy; its original-filename discovery
test passed. Patched binary SHA-256:
`d8f0a72115320d6f132c606e23c83ce557df79a0c516af609e6be1ef879b3933`.
Its numerical engine build identifier is unchanged; the binary hash, compiler
version, source patch and release commit distinguish this service build.
The deployment uses regular copied runtime/catalog directories as required
by the installation validator, and a separate `installation-patched/`.

Raw release identity, probe script and HTTP responses are retained locally
under `out/magnitude-swift/`; raw host logs and responses are under the
isolated installation's `evidence/` and `evidence-retry/` directories.

## Scope and interpretation

Magnitude is currently an inference engine and service, not the earlier
browser-automation product associated with this repository name. Its Pi
connection configures models and a skill; the optional development extension
requires a supported desktop installation in the same graphical session.
An OpenAI-compatible HTTP interface is independently exposed by the engine.

The published CUDA comparison is a DGX Spark / Qwen3.6 35B-A3B 4-bit
64K-context run without speculative decoding. It is not evidence for Swift
27B on SM60 or for three-device PCIe tensor parallelism. No speedup or
image-recognition success is inferred from that comparison. Neither Swift
fixture includes a compatible visual projector in this test.

Primary source links:
- [Magnitude repository](https://github.com/magnitudedev/magnitude).
- [Pinned compatibility contract](https://github.com/magnitudedev/magnitude/blob/38f0adb8f6470642aaa15f7aa46ac2d135359d23/inference/docs/compatibility.md).
- [Pinned device selection](https://github.com/magnitudedev/magnitude/blob/38f0adb8f6470642aaa15f7aa46ac2d135359d23/inference/engine/executor/src/platform/selection.rs).
- [Published benchmark conditions](https://magnitude.dev/).

## Discovery-only baseline outcome

The patched X99E service completed reconciliation and returned exactly two
`Ready` discovery rows under the **original** Heretic and GSQ filenames.
`/v1/models` advertised both canonical original `hf:` IDs. The temporary
`-full.gguf` alias was absent. This passes the requested discovery fix gate
without renaming or changing either model.

Both load-plan requests and both actual instance-load requests returned
HTTP 409 / `unsupported_model` / `invalid per-layer recurrent flags`.
Consequently neither model generated tokens in Magnitude. Real Pi generation,
concurrent Pi qualification, image recognition and throughput are **not
qualified**; no tok/s or speedup is reported for this runtime. The header
mismatch, IQ2_S import support, exact F16 numerical policy and three-P100
execution remain separate work, not consequences of the filename fix.

The temporary Magnitude service was stopped after the gate. Existing Pi home
settings, model profiles and llama.cpp service configuration were not replaced.
The host's absent NVIDIA kernel module was observed but not modified.

Final evidence: `out/magnitude-swift/evidence-patched/` contains health,
hardware, discovery, advertised models, both load plans, both instance-load
responses, assessment snapshots and the server log. `x99e-build.log`,
`patched-binary.json`, focused local test logs and the durable source patch
bind the result to the actual source and binary.

## Target/MTP recurrent metadata correction

Patch `patches/magnitude/02-interpret-target-and-mtp-mixer-flags.patch` accepts
explicit recurrent flags for either target blocks or all stored blocks. It
validates every flag and requires any supplied MTP suffix to declare attention,
matching the already supported head semantics. Only target flags construct
target decoder blocks. Invalid array lengths, non-Boolean flags and recurrent
MTP heads remain errors. No tensor bytes or runtime architecture branches change.
The model-input design contract and `docs/feasibility.md` were updated with
this admission gate and unchanged weight/KV/compute/communication equations.

All 15 family interpretation tests passed, including complete target/head
geometry equality between implicit, target-only and target-plus-MTP metadata.
The separate opt-in real-header test also passed against byte-identical header
prefixes extracted from both original GGUF files: each binds 64 target blocks
and one MTP block. Full payload admission is deliberately not claimed by a
header test. Reproduce with `MAGNITUDE_MTP_GGUF_HEADERS` set to a platform path
list and the `local_mtp_headers` test filter plus `--ignored --nocapture`.

The X99E release build completed in 27.93 seconds. Its native build is
`0.1.0+src.03b1f3adf4efde152dcda76b`; executable SHA-256 is
`25c78c7f89574afb93138c6971e8508b78f70bb361fc4f3af6d48db04c513d75`.
The isolated installation declaration was regenerated from the executable's
actual native-build identity. Raw header test output is in
`out/magnitude-swift/real-header-tests.log`; the new host probe uses
`evidence-header/`, keeping the discovery-only baseline evidence intact.

The new service reconciliation completed with both original filenames. Heretic
load-plan returned HTTP 200 at 262,144 context positions, selecting host CPU
and charging 18,404,534,155 B under the service's default numerical policy.
GSQ load-plan returned HTTP 409 with `unsupported source encoding Iq2S`.
The old recurrent-flag error is absent from both requests. The actual Heretic
instance request advanced through device selection/opening and weight loading,
but the controller's 30-second request limit expired while loading; its
`loading` progress is not successful residency. The temporary service and
worker process group were terminated. The request timeout is a test-controller
limit, not an observed model-loader rejection.

A separate direct engine loader diagnostic avoids repeating inventory hashing
and explicitly selects CPU, 262,144 positions, MTP depth two and dense KV.
Its BF16 model activation/KV semantics are recorded explicitly: this is loader
qualification, not the project's exact-F16 generation gate. No generation is
requested. The direct engine shares the worker/protocol used by the service.

Direct inspection of all 866 GSQ tensors establishes that the missing support
is broader than the first `Iq2S` error. The artifact is mixed-format: 315 tensors,
6,733,721,600 stored bytes, use `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS` or `Q2_K`, all absent from Magnitude's executable encoding maps.
The other tensors use supported F32/BF16/IQ3_S/IQ4_XS/Q4_K/Q6_K formats.
Fixing one unsupported enum would merely expose the next unsupported codec;
header adaptation does not supply the decoders, resident representations and
numerical kernels for these seven encodings. Counts and exact stored-byte
accounting are retained in `out/magnitude-swift/gsq-encodings.json`.

## Latest measured outcome

The header-adapter correction is deployed on X99E and the original-filename
discovery gate still passes for both models. Heretic's direct CPU loader
completed in 124.08 seconds and `/health` returned `ready: true`, context
262,144, vocabulary 248,320. The diagnostic used dense **BF16** KV and MTP
depth two; it executed no generation request. Kernel preparation took 90.64
seconds including 87.89 seconds tuning. These timings are loader costs, not
tok/s, populated-context performance or Pi qualification. The controller
terminated the diagnostic process after collecting readiness.

GSQ passes family/header interpretation but its executable plan rejects its
unsupported mixed-format weights. All seven missing encodings are documented
above. Both complete original artifacts remain unchanged. Full-model Pascal CUDA execution, three-device placement, exact F16 KV and real concurrent
Pi generation remain unqualified in Magnitude. No change to the active
llama.cpp service or Pi settings was made by these fixes. The later driver
recovery requested by the user is recorded below.

Latest evidence: `out/magnitude-swift/evidence-header/`,
`out/magnitude-swift/evidence-engine-load/`, `header-binary.json`,
`header-tests.log`, `real-header-tests.log`, `gsq-encodings.json`, and the admission
patches 01/02 under `patches/magnitude/`.

## Subsequent user-requested NVIDIA recovery

The user requested immediate GPU recovery after noticing absent monitoring.
PCI still enumerated all three P100 cards and P40. The running kernel was
7.0.0-38-generic, while the installed NVIDIA module package covered only
7.0.0-31-generic. APT history records installation of the new kernel by
`/usr/bin/unattended-upgrade` on 2026-10-08 at 10:20:43; the host booted it
on 2026-10-09 at 12:32:31. The missing module was observed before the
Magnitude admission changes; those changes did not alter the driver.

Recovery installed the three current-kernel module/object/signature packages
and `linux-modules-nvidia-580-server-generic`, without upgrading or removing
existing packages. The tracking metapackage depends on the distribution's
current kernel-specific NVIDIA module. Modules loaded without a reboot.
`nvidia-smi` reports driver 580.178.04 and all four cards; temperatures were
34/38/35/32 C, 0 MiB allocated. Existing llama.cpp with
`CUDA_VISIBLE_DEVICES=0,1,2 --list-devices` successfully enumerates CUDA0/1/2
as the three P100 cards. No model was started and P40 was not allocated.
The user-level `pascal-llm.service` remains inactive as found.


## Existing-library CUDA port (2026-10-09)

The user rejected the Vulkan direction and asked to adapt Magnitude to the
libraries already installed. `SEISMIC_NVRTC_DIRECTORY=/usr/lib/x86_64-linux-gnu`
is an existing engine-development setting, not a new loader search path.
It binds NVRTC 12.4 (`libnvrtc.so.12.4.127`) and the matching builtins on X99E.
No CUDA toolkit replacement, driver replacement or model restart is needed.
The native Rust projection process's mapped libraries independently confirm
`libnvrtc.so.12.4.127`, `libnvrtc-builtins.so.12.4.127` and driver
`libcuda.so.580.178.04` (`sm60-native-library-maps.log`).

The candidate native prelude preserves the register-fragment and copy-completion
ABI with SM60-compatible software BF16 RNE, F16 pair conversion, blocking copies,
convergent matrix gathers and F32 FMA matrix products. SM80 hardware paths remain
selected on SM80. Integer paths preserve modulo-32-bit accumulation. Source/model
names do not select an execution branch. Projection's four-byte copies and BF16
packed code construction also need corresponding ISA alternatives.

Independent helper gate: `patches/magnitude/qualify-sm60-helpers.py` compiles
through installed NVRTC and launches through `libcuda.so.1` on P100 ordinal 0,
UUID `50fd03dc-7d0c-e635-6586-be91b456758e`, rejecting any non-SM60 device.
The gate subsequently passes on all three P100 cards (the same prelude hash).
The first run passed 362,144 BF16 RNE cases, packed BF16/F16 pairs and BF16
widening; 1,536 outputs each for F16, BF16 and signed-I8 matrix products;
448 shared matrix-fragment values; 1,024 copy/padding/peer-visibility bytes;
4,096 integer reduction/dot results and 128 readonly load values. The F32
`tanhf` fallback's maximum absolute error against the F64 host tanh oracle
was 5.960464477539063e-8. Matrix outputs match independent row-major host
F32 `fmaf` accumulation, not an emulation of the register shuffle implementation.
Tensor-core association and full-model accuracy remain separate gates.

The first CUDA Rust run passed all 33 library tests, including SM60 probe source
legality and the physical 4 KiB parameter limit despite a newer driver. The
candidate suppresses hardware matrix capabilities and matrix characterization
on Pascal; software helpers must not be reported as tensor-core hardware.
Native operator tests require `SEISMIC_TEST_BACKEND=cuda`, so failed device
opening cannot silently become a skipped success. Prefill attention passes all
three selected configurations against its independent host/portable oracle on
P100 (maximum gated error 1.95e-3 under the unchanged existing tolerance).
The first dense-weight projection run reaches real CUDA execution and then
rejects `dense_expand_gemm_small`: declared shared bytes 93,184 exceed the
reported physical limit 49,152. This is a resource failure, not a numerical
failure. The build's earlier AppleDouble transfer failure was corrected without
bypassing the native parity check.

After that second failure, implementation stopped for an end-to-end resource
reassessment. The new plan retains the multistage candidates and adds generic
32-row, single-stage projection candidates: 37,376 B declared shared memory for
two weight streams, 20,992 B for one, including double-buffered coefficient
records. Grid row bands follow the selected tile, and a single stage synchronizes
before reuse and before multiplication. Projection accumulation/epilogue recipes
and stored weights are unchanged. The first new-plan test stopped before GPU
execution because its explicit fixture did not supply the new `ROW_TILE`/`PIPE`
parameters. One fixture correction supplied them for both row bands, preserving
all oracles and tolerances. The corrective retest passes dense expansion and
output for 1/2/3/8/12/16/17/40/64/128 rows, across the existing mapping set;
1 test passed in 236.90 seconds (not throughput). The quantized-weight
projection gate also passes Q4_K/Q5_K/Q6_K/Q8 gate/up expansion for all ten row
counts and the existing mapping set, including the diagnostic INT8 path, with
unchanged host-oracle tolerances: 1 test passed in 1,126.09 seconds. This includes
NVRTC compilation of many variants and is not a throughput measurement. No
model tok/s or production Pi claim is made.

The original small recurrent test reaches and numerically passes its four step
mappings, then rejects its width-32 chunk before launch: the existing native
chunk `where` clause admits only width 128/256. This is an upstream fixture/entry
geometry mismatch, independent of Pascal, not evidence of a GPU numerical
failure. A focused test uses the admitted width 128 with 32 scan rows and
16/32/48/64 value heads, including the Swift 27B geometry, with the unchanged
host oracle and catalog tolerances. It passes in 79.52 seconds; 27B gated RMS
relative error is 1.513e-3, delta-state RMS relative error 2.816e-4. This scan
uses `3*(96*128 + 32*32 + 1680) + (32/16)*16*17*4 = 47,152 B`
shared memory. The larger 64/128-row scan candidates do not fit P100 at this
width and were not qualified; no recurrent kernel behavior was changed.

Evidence: helper JSON files for P100 ordinals 0/1/2;
`out/magnitude-swift/sm60-cuda-tests.log`, `sm60-attention-tests.log`,
`sm60-projection-tests-retry.log`, `sm60-projection-tiles.log`, and
`sm60-projection-tiles-retry.log`, `sm60-quantized-projection-tests.log`,
`sm60-recurrent-tests.log`, and `sm60-recurrent-32-tests.log`. Patches 01 through 05 replay cleanly against the
pinned release; all 34 affected source files match the candidate. The proof
and hashes are retained in `out/magnitude-swift/sm60-patch-replay.json`.
Existing llama.cpp user service remains inactive. P40 is excluded by explicit
CUDA visibility in runtime tests.


### Rebuilt service artifact and installed-library integration

The CUDA candidate service builds successfully on X99E with Rust 1.98.0 in
5m34s. Its native build is `0.1.0+src.6ded17a8715072bf21f78a7a`, binary
SHA-256 `1ad0cdc8c5e49a03fa3d0c903e35af9d3a3c39426a9e1a2dfd7572f4ec8bfe0a`.
It is staged at `work/magnitude-0.2.6/installation-cuda-sm60/` with its own
matching `installation.json`. The regular runtime directory links directly to
X99E's existing `libnvrtc.so.12.4.127` and matching builtins 12.4.127; it does
not copy or use the release's bundled NVRTC 12.9. The loader already supports
both this installation layout and the explicit development directory.

`doctor` passes. A bounded service run on loopback returns the matching native
build and exactly the three P100 CUDA devices from `/api/v1/hardware`; Vulkan
ICD discovery is disabled for this process and P40 is excluded. The probe loads
zero models and terminates the service in a `finally` block. Enumeration is
only an integration check, separate from the numerical operator gates and
full-model admission. The pre-existing header-only installation is retained.

Evidence: `out/magnitude-swift/evidence-cuda-sm60/`,
`sm60-service-build.log`, `sm60-service-probe.log`, and `sm60_service_probe.py`.
All 34 affected remote source hashes match the clean five-patch replay. The
candidate does not implement missing GSQ formats, three-device allocation or
an exact-F16 KV policy. No generation, model tok/s or concurrent Pi qualification
is claimed by this CUDA port.


Final cleanup check: all four cards report 0 MiB allocated, temperatures
36/40/38/31 C. Port 18843 has no listener, the temporary test processes have
exited, and the pre-existing `pascal-llm.service` remains inactive. No power,
clock, kernel-driver or Pi-setting changes were made by the CUDA port.


### Independent IEEE F16 history policy (2026-10-10)

Patch 06 adds explicit `dense-f16` history independently of model activation
precision. BF16 query/key/value projections remain BF16; target and enabled
MTP/separate-draft history planes use IEEE F16 when requested. Layout charging,
execution binding, kernel specialization, cache identity and serialized policy
all retain this distinction. The legacy `dense` policy remains tied to activation
precision. A 17-attention-layer target-plus-MTP history is exactly 69,632 bytes
per populated token, 18,253,611,008 bytes at 262,144. Service policy wiring and
physical three-device admission remain outstanding.

The independent mixed-history test uses Swift's 4 KV heads, 6 queries per KV
head and width 256, pre-existing F16 history not representable in BF16, slab
crossings, causal fresh spans, untaken destinations, signed zero and subnormal
append boundaries. CPU and CUDA P100 runs pass for 1/8/9/17 rows. Whole history
payloads are exact byte comparisons, including untouched rows; subsequent
reads are checked against independent ordered F32 softmax. Output envelopes
are unchanged from upstream. CPU test: 219.40 s; P100 test: 95.77 s. These
include compilation and are not tok/s. All 74 state unit tests and the executor
attention tuning identity test pass. Strict CUDA legacy BF16 prefill passes its
three configurations in 51.27 s. A same-named test without a CUDA selection on
macOS skipped device work and is not CUDA evidence.

The first width-256 prefill candidate requested 81,984 shared bytes, exceeding
P100's 49,152 limit. The generic key tile now permits 16/32 columns; 16 columns
and register queries fit in 32,832 bytes. Native allocation, declaration and
specialization agree. No tolerance, driver, clock or power setting changed.
CPU fixture corrections supplied the actual backend static set and PARTS
parameter; they did not alter an oracle. Evidence is retained in
`out/magnitude-swift/dense-f16-*.log` with rejected attempts separately archived.
No complete-model numerical or concurrent Pi result is claimed.

### GSQ source representation gate (2026-10-10, integration in progress)

The seven missing codecs have a lossless resident plan, four-bit palette
indices with F32 group scales or original hierarchical Q2_K coefficients.
The installed GGML C decoder is ABI checked using its exported block/type size
functions. Library revision `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`, shared
library SHA-256 `7c2abdfcff7a966fdb2e68306ace62f940cec648ebd9d05253bac6c07dec33fe`.
All 41,287,680 sampled values across all 315 affected tensors match bit for bit.
Each codec also matches on 16,252,928 synthetic values covering all finite F16
base-scale payloads, including signed zero and subnormals. The initial fixture
mistook IQ2_S's type ordinal for IQ3_S; the exported ABI size check rejected it
before numerical comparison. The ordinal was corrected from the actual ggml
header; no artifact or reference was changed.

These checks qualify source-format conversion arithmetic, not native import or
matrix execution. The seven formats occupy 14,616,330,240 proposed resident
bytes; charging every artifact tensor conservatively gives 18,858,921,984 bytes.
Per-card capacity equations and a concrete layer partition are recorded in
`docs/feasibility.md`. No aggregate-memory allocation or throughput claim is
made. The generic registry and CUDA conversion/projection integration is being
qualified with 315 independent original-tensor fixtures. A build-tree parity
failure rejected a CUDA-only helper file; the helper is folded into the existing
CUDA import implementation, preserving the parity check without exceptions.
Evidence: `out/magnitude-swift/gsq-palette-reference.json`,
`gsq-resident-capacity.json`, `gsq-registry.log` (13 tests pass), and the retained
independent fixture exporter. Full source/native tests are still pending.

The six-patch source state replays cleanly from the pinned release and matches
all 64 affected files (`out/magnitude-swift/dense-f16-patch-replay.json`). The
proof explicitly names the qualified state before pending GSQ edits, rather
than conflating it with the evolving CUDA codec candidate. Engine CLI and
service binaries pass `cargo check --bins` with the explicit F16 policy's
serialized form. The staged service executable still corresponds to patches
01–05 and has not been replaced by these source changes.

Read-only CUDA peer capability queries confirm all six directed links between
P100 0/1/2 support direct peer access. This is capability evidence, not a
bandwidth or transfer-latency result; no peer context was enabled by the query.
API semantics were checked against NVIDIA's CUDA 12.4 driver documentation:
https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PEER__ACCESS.html


## Native three-P100 implementation gate (2026-10-10)

The user explicitly requested fixing the native executor to use all three
P100s. The candidate now places contiguous decoder stages from the artifact's
semantic weight roles and state layout. Entry/readout/MTP remain on the first
P100. Each stage owns its weights, exact F16 history, recurrent bank slabs,
graph families and intermediate arena on its actual device. Stage boundaries
wait for the producer and copy exact hidden-row bytes into admitted storage on
the receiver through CUDA peer copies. Existing foreign-binding checks remain
intact. P40 is excluded by explicit selectors and `CUDA_VISIBLE_DEVICES=0,1,2`.

The admission authority observes all three physical allocation domains and
checks each before a logical accounting claim. State/bank growth publishes only
after all owners allocate; failed unpublished allocation rolls back. The
existing session scheduler and per-session transactions remain authoritative.
Allocation census reports each physical device separately. Source equations
and physical budgets are in `docs/feasibility.md`.

Local complete `magnitude-engine-cli` source checks passed. Source patches
01–12 replayed in an isolated fresh Git repository; patch 13 checked against
that result. An initial deployment omitted the module exports; the full source
manifest corrects that transport failure. The Linux release build passed in 5m21s. Binary SHA-256 is
`f3681d2199f121ee4307793ec7ff0fa0a475f296b73ba278cac55c7889b03e31`.
The first original Swift GSQ attempt (PID 164314) used three explicit P100
selectors, DenseF16, MTP 10 and configured context 262144. Physical placement
and state planning passed before the projection formation failure recorded
below. No throughput, populated 256K
context or concurrent Pi qualification is claimed by compilation.


The first actual three-device Swift GSQ load (PID 164314, binary above) passed
physical placement/state planning but stopped during native `dense_output`
preparation: the GEMV bias fold required group width 32, while the original
Q2_K weights use width 16. Patches 01–13 had not qualified complete projections.
Patch 14 computes per-16-code activation sums with the same all-ones fragment
operation and folds the bias at the declared group width. Existing 32-code
sum order is preserved. The original GGUF and all quantization parameters stay
unchanged. The corrective Linux release rebuild passed in 5m20s. Binary SHA-256 is
`6b5cad9a2b1bc0f3afefbde16ccaeff55ef4ba0ce7d95dc961313c277f585b1a`.
The unchanged original GSQ rerun (PID 168254) is preparing native kernels;
the complete-model and concurrent Pi gates remain pending. Formation now
proceeds past the earlier assertion. This does not yet qualify the Q2_K
projection numerically or demonstrate three-device generation.


The patch-14 actual GSQ attempt stopped before target-weight import in native
startup qualification: `gguf_iq1_s -> iq1g32@mma16` differed from the registered
repack reference on the startup's arbitrary-byte fixture. The run is archived
as `out/magnitude-swift/swift-gsq-three-p100-repack-iq1-failure.log`. The source
fixture contains half NaNs at packet indices 19 (`0x7e59`) and 30 (`0xfcd7`).
This is a candidate explanation, not a verified verdict. The change loop
stopped inference edits after the second real-load failure and reopened the
end-to-end import/qualification contract. A focused reproduction of the exact
startup fixture will identify differing resident words before any correction.
Original-artifact sample passes do not waive this startup guard. No model
weights were loaded on the peers and no three-device generation is claimed.


The focused exact-startup diagnostic reproduced 48 differing bytes each for
IQ1_S, IQ2_XS and IQ3_XXS; the other four bindings matched. The first differing
words are CUDA `0x7fffffff` versus registry `0x7fc00000`, both NaNs. This is
the language reference's defined canonical NaN, not a changed oracle or
tolerance. Patch 15 canonicalizes only NaN results of IQ scale arithmetic;
all finite/Inf/signed-zero result bits follow the original expression. The
regression now asserts complete resident-byte equality for every binding of
the failed startup fixture, including padding. The initial diagnostic's test
exit zero meant it completed observations, not that unequal payloads passed
qualification. Corrected native and full-model gates are pending.


Patch 15's corrected strict native regression passed all seven source/resident
bindings with zero differing bytes in 2.47s. Evidence:
`out/magnitude-swift/repack-startup-corrected.log` (the exact build duration
is authoritative there). Heretic's original patch-14 load (PID 182719) then
reached the native attention guard and refused `history_key` because its
qualification fixture still allocated the activation dtype. Patch 16 supplies
the selected DenseF16 dtype to that existing fixture, covering both target
and MTP callers. The guard is preserved.

The same typed-interface reassessment found that the native decoder graph's
hidden residual port is F32, not BF16 projection activation. Patch 16 makes
physical receiver buffers match this actual F32 ABI without a cast. The
quantitative boundary prerequisite was corrected before inference edits in
`docs/feasibility.md`: H=5120 requires 20,480*B bytes per boundary. The original
BF16 boundary assumption is invalidated. Release rebuild and actual model
readiness/generation/Pi gates remain pending. No model currently has imported
weights on the three P100s.


Patches 01–16 replay into an isolated fresh release checkout and match all
113 files of the deployed source manifest bit for bit. Evidence:
`out/magnitude-swift/native-three-device-source-replay.json`. The corrected
Linux release build passed in 5m20s; executable SHA-256
`678c8378eb38d7732c54cf7ea3d057ab39096e394565429d4d252ce0d9b14947`.
The unchanged original GSQ rerun (PID 191027) has begun preparation with the
same three P100 selectors, DenseF16, MTP 10 and configured context 262144.
Readiness, physically imported weights, generation and concurrent Pi gates
remain pending.


The patches 01–16 original GSQ load (PID 191027) passed repack qualification
then refused `embedding_rows`: the newly supported `iq2g16` had no nonzero
semantic fixture. Heretic's same-source rerun (PID 193268) prepared its native
programs in 6.73s, then refused graph argument 14: generic dense history ports
were still declared with `block.activation`. A reassessment after this second
real Heretic failure traced the history contract through preparation, kernel
bundles, checked/native graph entries and port creation. Patch 17 now carries
the history element explicitly through that entire path for every caller.
No foreign-port check, comparison or semantic guard was removed.

The newly supported resident formats also have valid GGUF unit packets. IQ1
uses source grid point 1029 (all zero grid entries), positive delta and scale
8; IQ2/IQ3 use their all-8/all-4 grid point zero; Q2_K uses unit local/global
scales and code one. The existing independent host-decoded unit-fixture test
now covers all six new resident formats and layouts and passed in 10.97s.
The initial source build missed one workspace-accounting enum destructure;
its local corrective match ignores the new non-allocation element field.
The Linux release rebuild then passed in 47.33s. All 114 source manifest files
match a clean replay of patches 01–17. Executable SHA-256:
`32a50aa50b2d30a9cb1385b9b4f25304088c16ae2842d3585b3056272401fe46`.

Original Heretic PID 198128 reached native READY in 23.50s on all three P100s,
DenseF16, MTP 10, context **configured** 262144. Actual warmed device memory
was 6635 / 5861 / 6011 MiB; P40 was 0 MiB. Its warm-up traversed all three
physical owners, with three exact 20,480-byte F32 peer transfers. A real
192-token request returned coherent English, HTTP 200: decode 47,107.813195ms,
**4.075757 tok/s**, prompt 36 tokens/1546.491141ms, total wall 49.371970s.
MTP proposed 689 tokens and accepted 121 (17.56%). Steady transfer observations
for 8-/16-row batches were about 41–62us per edge, so these transfer timings
do not explain the low total decode throughput. No kernel-level bottleneck
verdict is implied. Evidence: `swift-heretic-three-p100-direct.json` in the
ignored evidence directory and the host's native Heretic run log.

Both concurrent real Pi CLI sessions completed successfully from the real
repository with normal project context, skill discovery and tools enabled,
using an explicit qualification provider extension through an SSH tunnel.
Each called `read` on its own requested document and produced the correct
independent answer. Session A took 329.2382s; B took 333.2541s. Their final
usage totals were 4606 and 3942 tokens. These are wall times and actual small
contexts, not Pi decode-only rates or populated 262144-token qualification.
Evidence: `out/magnitude-swift/swift-heretic-pi-three-p100-results.json`, each
session's `events.jsonl` and persisted Pi session. Home Pi settings were not
changed. The native run log and identity are archived as
`swift-heretic-three-p100-qualified.log` and
`swift-heretic-three-p100-run.json` in that evidence directory.

After stopping the owned Heretic process, the original GSQ PID 205087 reached
READY in 60.75s using the same executable and numerical policy. Its stages
are blocks 0..19 / 19..42 / 42..64. Actual warmed VRAM is 7373 / 6759 /
6559 MiB, P40 0 MiB. This is physical residency, not aggregate-memory fit.
It imported 851 distinct target weights in 7.72s and ran a warm-up across all
three actual owners. Generation and simultaneous real Pi gates are pending.


The original GSQ native request returned HTTP 200 and coherent English:
36 prompt tokens, 192 generated tokens, no cache reuse. Native decode took
37,893.448413ms (**5.066839 tok/s**), prompt processing 1567.575238ms,
and request wall time 40.160406s. MTP proposed 650 tokens and accepted 125
(19.23%). Evidence: `out/magnitude-swift/swift-gsq-three-p100-direct.json`.
Two simultaneous real Pi sessions are now running on this exact resident
model with the same normal instructions/skills/tool policy used for Heretic.

A read-only GPU sample during those real Pi requests showed P100 clocks
1328MHz SM / 715MHz HBM on all three cards, temperatures 43 / 56 / 45 C,
and VRAM 7789 / 6839 / 6635 MiB. One sampled stage was compute-active at
100% utilization; the other stages were idle at that instant, consistent with
the implemented sequential decoder stages. This is not a simultaneous
three-card tensor-parallel implementation or a kernel-level bottleneck
measurement. No clock, driver, power or model fidelity changes were made.


Both original GSQ concurrent real Pi sessions completed independent `read`
tool loops successfully: A 303.1104s, B 288.0525s, final usage totals 4545
and 3922 tokens. This is a bounded short-context tool gate, not populated
262K or target throughput qualification. The owned GSQ process and tunnel
were stopped at the user's request, releasing all GPU memory. The renewed
active request is strictly Magnitude throughput. Patch 18 enables opt-in
CUDA event attribution on real sealed-model graph submissions. It retains
the existing numerical guards and uses installed CUDA; diagnostic events
are not production speed evidence. No comparison run of another backend
is part of the renewed gate.


Patch 18's installed-CUDA release build passed in 1m05s, SHA-256
`a7c993b1a71e61be57bc84aea465b949ec7d02b25699230c5300600f07b61337`.
The original GSQ attribution run PID 216630 reached READY in 60.97s and
completed a coherent 64-token real request. Its diagnostic events recorded
11.729617s of GPU work including prefill, 88.14% in projection families and
59.95% in FFN. Event-instrumented response speed is not production evidence.
The process was stopped after recording attribution. Evidence:
`out/magnitude-swift/swift-gsq-cuda-attribution.log`, `-response.json` and
`-summary.json`. Source-derived work counts and the selected SIMT change
were added to `docs/feasibility.md` before inference edits.

Patch 19 evaluates actual GEMV rows directly on SM60, over identical resident
codes and scales, with F32 group sums and existing scale/bias folds. It
preserves the row/batch association across all M and uses fixed register
indices. Its focused CUDA qualification covers original GSQ, Q4K/Q6K and
IQ4 packets, M=1/3/11, two K-part layouts, independent F64 numerical bounds
and exact row/batch bits. This focused numerical gate is pending. The first
release build preceded the final register-index adjustment; its success is
not qualification of the final candidate. Patch 20 additionally permits
request-local adaptive MTP under the unchanged maximum ten; numerical state
and target verification remain authoritative. All 116 source-manifest files
match a clean replay of patches 01–20. Production throughput remains pending.


The first focused GEMV test build refused its fixture because the independent
`project` helper takes F32 inputs and accumulates its reference in F64; the
fixture incorrectly supplied an F64 vector. The single corrective patch
passes the actual BF16-expanded/F32-readable values directly to that unchanged
reference. The retry built in 6.01s and is running, without changing numerical
bounds. No numerical result is yet claimed for the GEMV change.

Patch 21 shares MMA operands between four independent outputs and preserves
each output's ascending-K F32 FMA sequence. The installed-CUDA helper gate
passed in 0.8630s, including 1536 exact F16 and 1536 exact BF16 output bits
against scalar `fmaf`. Matrix resource reflection is 32 registers and zero
local bytes, versus the old 48-register helper. This is component evidence,
not production tok/s. Evidence: `out/magnitude-swift/sm60-reused-operands.json`.
All 116 source-manifest files match a clean replay of patches 01–21.


The focused native Pascal GEMV gate passed in 507.49s: all ten original
source format cases, M=1/3/11, two K-part layouts, independent unchanged
numerical bounds and exact single-row/batch output bits. Evidence:
`out/magnitude-swift/pascal-gemv-numerical-retry.log`. This qualifies the
changed GEMV arithmetic; it does not establish model throughput. The final
integrated rebuild also contains the independently exact patch-21 MMA helper
and opt-in per-session MTP controller. Its real model gate is next.

The final integrated release build passed in 5m21s; binary SHA-256 is
`867ca79e0d9c1f60c3169763a05e4b64cf654edeb5f42d9deb905cdfccd20e41`.
Original GSQ PID 237246 is preparing newly keyed kernels with profiling
disabled and adaptive MTP enabled under the unchanged maximum ten. Startup
qualification has not yet reached READY; no optimized throughput is claimed.

The existing generation suite reports 23 passes and three DFlash failures.
A focused rerun using the exact pre-patch-20 `mtp.rs` reproduces the same
three failures (two missing draft effects and one checkpoint assertion).
The adaptive source was restored byte for byte afterwards. These are existing
separate-draft fixture failures; they do not qualify or reject the active
chained-MTP model path. Evidence: `out/magnitude-swift/`
`adaptive-mtp-existing-generation.log` and
`generation-before-adaptive-dflash.log`.

The 01–21 GSQ candidate reached serving in 1053.82s after first-time source
qualification and compilation. All three P100s hold the native model;
warmed residency is 7721/6773/6573 MiB, with P40 at zero. The unchanged
192-token request completed HTTP 200 with coherent text, 36 prompt tokens,
no prompt cache, 1528.530ms prefill and 25844.735ms decode:
**7.428979 tok/s**, wall 28.198764s, draft 156 and accepted 101. This is a
46.62% increase over the native baseline but fails the requested throughput.
Evidence: `out/magnitude-swift/swift-gsq-pascal-speed-direct.json` and
`swift-gsq-pascal-speed-production.log`. No llama.cpp run was used.
The renewed plan attributes this candidate's remaining GPU work before
selecting the next correction; real concurrent Pi qualification remains
pending until useful production throughput is achieved.

The renewed 64-token installed-CUDA attribution records 10.092834s of GPU
work including prefill/warm-up, with projections 8.887360s (88.06%). Evidence:
`out/magnitude-swift/swift-gsq-pascal-speed-profile.log`, its `-summary.json`
and `-response.json`. The diagnostic PID 250356 was stopped before the next
model candidate.

Patch 22 exposes typed packet operand pairs and directly widened SIMT codes.
Its first exact gate exposed the old F16 byte permutation bug: (1,0) became
(1,1). The focused GPU reproducer confirmed BF16 (1,0), old F16 (1,1), and
SIMT (1,0). Patch 23 corrects selector 0x4140 to 0x4240. The next gate passed
both exhaustive integer references but refused a native Dense<Bf16> NaN
payload: old 0x7fe10000 versus extra-rounding 0x7fff0000. Editing stopped for
a source-element/operand-interface reassessment. Patch 24 preserves native
dense widening, while retaining activation rounding for other source types.
No NaN exception or oracle relaxation was used. The renewed exact gate
passed 393216 packet comparisons and 1966080 integer-reference comparisons
in 1.215s. Evidence: `out/magnitude-swift/simt-operands-native-exact.json`
and the two preceding failed gate logs.

The integrated 01–24 release build passed in 5m36s; SHA-256
`cb868029ec8be6a40b58aeaa776513eb6e9e245b21310a4d869472bc36ffdaf1`.
Original GSQ PID 264944 is preparing source-keyed kernels, with profiling
disabled and adaptive MTP enabled. Its real model throughput is pending.
All 116 source files match clean replay of patches 01–24.

Read-only CUDA reflection of actual cached kernels reveals dense-expand
GEMV uses 206 registers/thread and GEMV16 224, with no local spills. A
256-thread, 206-register block alone charges 52736 registers before allocation
granularity; P100's 65536-register SM cannot admit a second such block.
Gated-delta step uses 87 registers/thread in its selected <2,4> form, with no
spills. This is an occupancy constraint, not yet a throughput verdict or a
new source change. Evidence: `out/magnitude-swift/native-hot-resources.json`
and `native-step-resources.json`. The current real candidate gate precedes
any subsequent correction to retained packet data.

## Streamed-packet corrective candidate (2026-10-10)

The profile-disabled 01–24 GSQ gate completed at 8.200015990 tok/s:
prompt evaluation 1557.295ms, decode 23414.588ms, 192 output tokens.
Complete response choices match 01–21; proposed/accepted counts are 156/101.
Evidence: `swift-gsq-direct-simt-direct.json`,
`swift-gsq-direct-simt-equivalence.json`, and
`swift-gsq-direct-simt-production.log` under `out/magnitude-swift/`.
This is a failed performance gate. Its selected local corrective candidate is
patch 25, based on the verified register occupancy constraint rather than a
new precision or model representation.

The 01–25 release build passed in 5m20s. Original GSQ is loading on PID 284518
with profiling off and original max-ten adaptive MTP policy. Actual cached
kernel reflection reports dense-expand GEMV 64 registers (previously 206),
dense-output 59 (previously 107), and gated-delta-project 61 (previously 126).
There is no local spill. The wider GEMV16 expansion reports 72 registers.
Evidence: `native-streamed-gemv-resources.json`. Integrated startup and the
same real 192-token request remain pending; no speed result is attributed to
this candidate yet. If the local correction misses the throughput gate,
editing stops for an end-to-end reassessment and a new plan.

### Actual 01–25 throughput verdict

Original GSQ reached native READY after 1921.28s from start. The identical
192-token prompt with profiling disabled measured **12.178974729 tok/s**,
prompt 1509.652ms, decode 15764.874ms, wall 18.103648s. Full response choices,
usage and MTP counters exactly match 01–24: 156 proposed, 101 accepted.
Evidence: `swift-gsq-streamed-gemv-{direct,equivalence,production,run}`
artifacts in `out/magnitude-swift/`. Warmed P100 allocations were
7765/6813/6609MiB; P40 was 0MiB. Actual SM clocks during the request were
1328MHz, temperatures 40/45/41C. No driver/power settings were changed.

This corrective candidate improves native throughput by 48.52% over 01–24
but **still fails the requested throughput gate**. Editing stopped for an
end-to-end reassessment. The production PID 284518 was stopped, and cached
01–25 diagnostic PID 303250 starts with CUDA attribution enabled. Its timing
is diagnostic, not a production throughput result. The next change is not
selected until the new attribution establishes the remaining cost.

### Renewed26 addressing component and counter availability

Patches01–25 remain the12.178975tok/s reference. Renewed attribution totals
6.726132GPU-seconds including prefill/warm-up, projections82.94% and recurrent
step12.80%. Single-row slab addressing is the first implementation in the
renewed end-to-end plan, with exact existing arithmetic for other slab sizes.
The installed-CUDA independent pointer-table oracle passed4636672 comparisons
in0.12952s, including both pointer words, single/multiple rows and offsets
larger than32bits. SourceSHA25e582f8c08576d3fc4c43695a5e14d3d8d4b1d18bbbfb140c3170860798f8f5.
The01–26 isolated patch replay matches all117 changed source files. Release
build passed in 5m34s. Integrated original GSQ generation measured
13.350953387 tok/s (192 tokens, 14.380995457 seconds decode), with exact
choices, usage and 156/101 MTP counters matching 01–25. Binary SHA-256 is
`ce0c5fec6b71403deef716ab4c23fd0007b726a68fcc45291b91a565f3b7e7de`.
Profiling was disabled. The active throughput objective remains unmet.

Installed nvprof12.4.127 was invoked on the cached01–25 engine for one pure
expansion invocation, with only device0 counters selected. It terminated
with`CUDA profiling error` and no collected events/metrics. This attempt is
unqualified; no bandwidth/counter claim is derived from it. Its owned process
exited; allGPU memory was released. Existing CUDA-driver/NVRTC address gate
then passed, with no driver, library or power changes. Evidence:
`swift-gsq-streamed-gemv-counters-failed.csv` and the preserved run metadata.
The earlier real event attribution remains usable.

### Shared-activation paired SIMT candidate (01–27)

Patch 27 shares activation reads and bias sums between paired projections
whose resident group widths match. Each stream retains its original F32
product/reduction/fold order; unequal widths use the existing route.
Single/two-row loop specializations cover actual native batch classes.
The fragment exact-bit gate passed 131072 comparisons in 38.07 seconds on
installed CUDA 12.4 / P100. The Linux integrated build passed in 5m21s, binary SHA-256
`20365ba70dfd956028e2023351308a992b979c20e3a4d39b6d25e5771db40526`. Source replay matches all 117 changed files. Model throughput and
concurrent real Pi qualification remain pending.

### Verified cold-compilation tuning budget loss

Actual installed-cache records show projection searches expiring before
alternatives are measured. The current `gated_delta_project` record
`12fb9bf26bdb...` reports 91.863835197 seconds forming programs,
0.077770758 seconds measuring, zero search allowance, one default candidate
and an incomplete factored search. A prior expansion record
`9b7cf23e0c06...` contains a census with no candidate records. At inspection,
all ten current touched projection records held only the default candidate.
This is concrete failure to select kernels for Pascal, rather than evidence
that the available launch configurations are equally fast. Kernel artifacts
occupy 521679905 bytes, below the existing 1 GiB cache budget; eviction is not
the cause of this cold compilation. No cache contents have been edited.

The selected configuration correction, after the initial 01–27 model gate,
is to back up and invalidate only current default-only projection tuning
records with no search allowance. Keep compiled artifacts, numerical bounds,
original weights, F16 history and all actually selected/validated alternatives.
Reloading then measures existing native alternatives from the warmed cache.
The real same-request throughput and concurrent Pi remain the acceptance gate.
The cache labels some factored CUDA results `backend=metal` due to a literal
in the generic factored-result serializer; actual candidate identity and
installed-device records are CUDA/SM60. That label is not used as evidence of
a Metal run or changed device selection.

The original 01–27 GSQ run reached READY in 1915.50 seconds and measured
13.914127535 tok/s on the same actual 192-token request (13.798924835
seconds decode). Choices, usage and 156/101 MTP counters match 01–26 exactly.
The throughput objective still fails. Its owned PID 334201 is stopped.
The selected correction backed up 84 current default-only projection tuning
records and invalidated those records only; compiled CUDA artifacts and
previously measured alternatives remain. The unchanged binary is reloading
with all diagnostics disabled as PID 352506. Warm selection and generation
are pending; no improvement is attributed to retuning before measurement.

The warm correction reached READY after 4300.27 seconds. Its local-tunnel
request timed out; a direct request executed on x99e subsequently returned
HTTP 200, 192 tokens, 13.918248361 tok/s (13.794839338 seconds decode), and
15.956833211 seconds wall time. Choices and usage exactly match the first
01–27 request, with unchanged 156/101 MTP counters. The speed gate fails
again; stop editing and reassess before another implementation. Cache CUDA
artifacts total 847142166 bytes, still below the 1 GiB budget.

Source reassessment confirms that artifact keys exclude requested symbol
sets, so symbol sets in the key are not the cache problem. Instead, a normal
60-second census/search clock leaves individual search shares below the
47–95-second all-variant formation. Patch 28 adds explicit extended offline
selection, with a fresh 600-second search budget, separate per-unit census
allowance, and initial variant formation excluded from measurement time.
Default-only incomplete stored choices are reconsidered; genuine measured
alternatives and numerical acceptance limits are retained. The exact
capacity/traffic and preparation equations were recorded before source
changes. Build and actual request acceptance are pending.

Patch 28 built in 1 minute 4 seconds; binary SHA-256 is
`8ce4e62585df17141309ee71ad53ea1242b9838f63f96471470746c18b7b05fb`.
The previous owned process exited after SIGTERM; PID 450866 is preparing the
original GSQ model with extended selection and production diagnostics off.
The isolated patch replay matches all 119 tracked source files. Observed
searches now measure 22–487 configurations and retain numerical rejections;
some expansion bindings select KSPLIT 8 after validation. This is selection
evidence, not a throughput claim. A `gated_delta_project` INT8 parameter of 1
does not quantize this model's activations: its A/B dense weight bindings make
the existing all-packed-weight `quantizable` condition false, so the authored
kernel still uses the 16-bit path. Cache modification times are refreshed on
read for LRU accounting, so they must not be treated as result-write times.

The actual extended-selection gate returned HTTP 200, 192 tokens at
14.006926902 tok/s (13.707503533 seconds decode), with 16.029490767 seconds
wall time. READY took 1186.93 seconds. The speed gate fails. Choices differ
from 01–27 and MTP counters become 153/99; passing the existing bounded
numerical tests is not an unchanged-quality qualification. No Pi or
full-context acceptance is inferred from this request.

The sole selected correction, patch 29, stages every normalized CUDA GEMV
row once, including M=1, using the same `form_row` and 256-thread arithmetic.
All six consumers' CUDA scratch/stage/shared-memory declarations change
together. Attention/recurrent stage dependency waits remain present.
`MAGNITUDE_NATIVE_REFERENCE=1` prepares authored defaults and ignores tuning
alternatives while qualifying this exact movement, without changing numerical
limits, weights, activation/KV formats or configured max MTP 10. Capacity,
useful traffic, arithmetic and occupancy equations were recorded before
inference edits. The isolated replay matches all 119 tracked source files.
Build and actual model acceptance are pending.

Patch 29 built in 5m22s; binary SHA-256 `c471e978a2a13efb59b1b85cb54e5139bae4490af7f61e16c94cd7ed6659d5ff`. The actual GSQ request returned HTTP 200, 192 tokens at 13.067738432 tok/s, with 14.692672416 seconds decode and 17.023245672 seconds wall time. Choices and usage exactly match the original 01–27 request; MTP counters return to 156/101. This second speed failure stops edits for end-to-end reassessment; concurrent Pi and the full populated context are not qualified.

The 29 diagnostic run took 57.13 seconds to serving and observed target Verify batches M=2 (18) and M=4 (11). Projection events account for approximately 83% of 6.449172430 total seconds including warm-up and prefill. Diagnostic rate is not production qualification. Patch 30 selects hierarchical, lossless IQ2 coefficients and dedicated M2/M4 native launches. The first exhaustive metadata inspection established finite, signed-zero and Inf bit equality, but identified changed NaN payloads after GPU multiplication. The corrective decoder retains the canonical resident NaN; strict exact-bit retest and full source imports are required.

The strict IQ2 coefficient retest passed 75497472 exact-bit comparisons on P100 / installed NVRTC 12.4 in 0.981748897 seconds, including all binary16 base patterns and 16 original odd scales through GEMV block and staged GEMM-record accessors. The previous NaN discrepancy is fixed; no numerical limits were changed. Source replay for patch 30 matches its 17 affected files. Exact original IQ2 fixture imports and real model throughput remain pending.

Patch 30 final integrated build passed in 334.539250724 seconds. The actual startup-byte gate then failed: seven previous bindings matched exactly, but CUDA `repack_weight.cu` rejected the new IQ2H binding as unregistered. No model was started. This second failure after the NaN correction stops edits for end-to-end reassessment. Registry and resident decoder alone do not complete the native import path. Plan 31 must update admission, odd coefficient/factor writes, packet-plane writes and row/mma16 padding together before retesting.

Patch 31 completed the CUDA source adapter and passed the integrated build in 320.552522776 seconds, binary SHA-256 `2ae7421b5b7e96b3c71a8005b4e77ce1a0513f0571c044f2685059be267959b3`. The actual startup physical-byte gate passed all ten bindings (including three new IQ2H routes) in 4.91 seconds, with no differing bytes; its 301.672375574-second command wall time includes compilation. All 167 original GSQ IQ2 fixtures then passed 4360704 exact decoded-value comparisons across packet/rows16/mma16, in 233.44 test seconds (234.019414530 command seconds). All 18 source files match the uploaded manifest. Owned PID 660125 is preparing the unchanged original GSQ on explicit three P100 with original F16 history, max MTP 10 and diagnostics disabled. Actual throughput is pending.

The actual patch-31 model gate is blocked in preparation, with three CUDA compilation workers active for over 20 minutes and no complete residency. A four-second privileged read-only CPU profile (48088 samples, no lost samples) locates the work in installed NVRTC 12.4.127. No driver, library, power or host permission changes were made. Captured owned production attention source has zero gate width GR=0. Its original M2 formation timed out after 20 seconds; excluding statically empty matrix segments compiled the same production source in 0.545515292 seconds. This is the first actual preparation failure of plan 31. Its sole corrective patch 32 excludes all static empty GEMV/GEMM segments and the zero-row default GEMV body before instantiation. No nonempty operation or model format changes. Rebuilt original-model acceptance is pending.

Patch 32 integrated build passed in 321.624152218 seconds, binary SHA-256 `d24acfc0d2f71d005467f80df281be044b2afa532ac519df4c7d981956429792`. Owned old loader 660125 exited after SIGTERM and its original logs/run record are preserved. New loader PID 665198 is preparing the original GSQ on the same three P100, max MTP 10, reference arithmetic and diagnostics disabled. All 119 tracked source files exactly match the isolated replay of patches 01–32. Actual READY, residency and throughput remain pending.

Patch 32 reached serving in 231.41 seconds (programs 196.93 seconds, graph/resource planning 22.25 seconds, target imports 7.50 seconds, warm-up 364 ms). The physical stage weights are 6263590272, 5331454592 and 5467174912 bytes; resident target weights are 16713367552 bytes plus 348852224 head bytes. Observed GPU residency after the request is 6993/6135/6235 MiB on P100s 0/1/2, and zero on excluded P40. The actual request returned HTTP 200, 192 tokens at 13.663491064564354 tok/s, prompt 1504.603875 ms, decode 14052.045637 ms, wall 16.389253556 seconds and MTP counters 156/101. Choices and usage exactly match patch 29. Loading is corrected; the speed gate fails. This second failure ends the 31/32 change loop: stop edits, reassess the actual projection execution and choose a new complete plan. No concurrent Pi speed or fully populated context qualification is inferred.

New plan 33 implements independent small-batch row scheduling across all six native projection families and twelve M2/M4 launches. The focused actual P100 gate passed 10768 exact fragment and logical-epilogue comparisons in 13.755066342 seconds across twelve source formats, including both IQ2H variants and mixed group widths. The comparison kernel holds both old/new computations and is not a production resource/performance measurement. Source replay matches all 119 tracked files for patches 01–33. Integrated build and original-model speed remain pending; the current 32 server remains loaded during rebuilding.

Patch 33 built in 321.513620320 seconds, binary SHA-256 `006b566f61e4877710b466e7b22ba0aa5bb860c5e214c2f2bc7f4ee81593f13e`, and reached serving in 250.19 seconds on the same three P100. Its actual request returns 192 tokens at 5.678221135430781 tok/s (33813.406597 ms decoding, 36.131128764 seconds wall). Choices/usage exactly match 32 and MTP counters remain 156/101. This is the first speed failure of plan 33. Selected sole correction 34 interleaves logical row CTAs per output weight tile instead of traversing the full matrix separately for each row. No weight, numerical, KV or device-placement change; acceptance is pending.

Patch 34 built in 321.412669528 seconds, binary SHA-256 `1c617721329b73bdfc6f55ae7ea11b9319068d73eed2ce6042b529fa22ab9df1`. The original GSQ request returns 192 tokens at 10.542328842549137 tok/s (18212.294728 ms decoding, 20.548452885 seconds wall), with exact patch-32 choices/usage and 156/101 MTP counters. Interleaving recovers part of the regression but remains below the earlier fused-row 13.663491 rate. This second speed failure closes plan 33/34. Stop inference edits and reassess the actual projection path with focused device-event timings and exact output comparisons; no production speed or concurrent Pi qualification is inferred.

Plan 35 restores fused row launches and selects only measured exact native schedules. Independent projection probes use the actual captured binding dimensions and synthetic resident bytes; they are diagnostic evidence, not original-model qualification. Same-group paired B=2 lookahead measures 0.3728 versus 0.4331 ms; same-group B=4 interleaving 0.4077 versus 0.4415 ms; mixed-group B=2 interleaving 0.3374 versus 0.5149 ms. Every probe compares original F32 pair outputs exactly. Rejected mantissa-conversion and CTA warp-count variants are not in the production source. The selected fragment gate passed 10768 exact comparisons across twelve formats and both mixed group pairs in 27.865169438 seconds on P100/NVRTC 12.4; combined old/new comparison resources are not production resource claims. All 119 tracked files match the isolated patches 01–35 replay. Integrated build and original-model rate remain pending.

Patch 35 integrated build passed in 322.735436906 seconds, binary SHA-256 `af9c9d1c3e51d31f2ee74a2be76713884911d32cd8c5045e807739bc250bafc2`. Owned PID 716989 reaches serving in 251.06 seconds on three P100. Original GSQ returns 192 tokens at 14.385283143632181 tok/s (13346.973993 ms decoding, 15.687882488 seconds wall), with exact patch-32 choices/usage and unchanged 156/101 MTP counters. This improves the verified fused baseline by 5.28%, but the requested throughput is still unmet. Preserve this measured base and reassess end-to-end for concurrent projection row sharding across the three devices; no full-context or concurrent Pi speed qualification is inferred. Read-only driver queries confirm CUDA virtual memory-management support (attribute 102=1) on each of the three P100, with no driver or host changes. No projection sharding implementation or gain is claimed yet.

The initial parallel direct-output fixture's exact-coverage claim is invalidated: it retained the prior single-device reference and its CTA-based offset was interpreted as a 16-row tile-group offset. The corrected explicit three-root/two-gather graph uses tile_group_begin=2*CTA_begin and poisons every output before parallel execution. Both actual three-P100 gates now pass every F32 output (49152 values at B=2, 98304 at B=4, for each group pairing). Same-group single/parallel event medians are 0.3705/0.1846 ms (B=2), 0.4072/0.2036 ms (B=4); mixed-group 0.3373/0.1634 ms and 0.5046/0.2791 ms. Graph completion includes the two peer 2D output copies. Fixture storage is 110037020 bytes per GPU with deliberately duplicated synthetic weights; production row-shard placement, input copies, capacity and full runtime integration remain unqualified. The active original model remains patch35 at14.385283 tok/s.

The canonical-master-only VMM fixture preserves original-local F32 values but regresses parallel same-group to 0.669/0.825 ms due to peer reads at 2 MiB boundary pages. Its master-only single-device 15–19 ms rates are not an original-local speed baseline. The sole physical-placement correction provides partial device-local virtual views and clones only overlapping boundary pages, copying them once after canonical upload. The corrected same-group native gate passes every original-local F32 value and poisoned full coverage, measuring 0.185/0.205 ms. The mixed-group gate also adds two graph input copies, poisons peer activations and all outputs, and passes every F32 value at B=2/4, with event medians 0.172472/0.290906 ms versus original-local 0.336907/0.506381 ms. This includes two input and two output memcpy nodes plus three simultaneous native kernels. Physical fixture weight charges are 37748736/39845888/44040192 bytes, with 12582912 boundary bytes copied once. Complete requested fixture allocations are 42600476/44697628/48891932 bytes. No full model or production integration is inferred: active source remains patches01–35 and original GSQ14.385283 tok/s.

Plan 36 integrated build fails first on a missing CString import, then after that sole correction on an unqualified Tensor reference in executor physical accounting. Stop this change loop. The end-to-end reassessment finds the CUDA driver, backend, runtime and facade compile; the remaining executor expression incorrectly uses an unimported facade type. New plan 37 completes checked, allocation-deduplicated physical accounting for both representation and scale through the public Tensor interface, then rebuilds and runs the original model with the integrated three-device projection path. No model throughput gain is inferred from the prototype or compilation.

Plan 37 first build reaches the engine and identifies a placement boundary mismatch: opened public devices are Device clones, not Rc<Device>. Its sole correction accepts the existing Device slice and borrows those devices for physical-cost inspection. No ownership, allocation or numerical change. Retest required.

Plan 37 retest integrated build passes in 47.19 seconds, binary SHA-256 `88eacf37f60510f92ab6e6e5f218f314b86eea5bb51590b1ddb95b1f2c8aa713`. All 126 tracked source files exactly match replay of patches 01–37. Preserve the measured patch-35 binary by copying its live process executable and verifying its SHA before stopping owned PID 716989. New PID 755257 loads original GSQ with MAGNITUDE_PARALLEL_PROJECTIONS=1, explicit three-P100 selectors, exact F16 KV, original precision, reference kernels and max MTP 10. Actual READY and throughput remain pending.

Plan 37 second gate fails after importing 851 distinct target weights in 24.41 seconds: warm-up compares physical ResidentTarget bytes against logical ResidencyStore bytes and reports `head residency cache omits a target allocation`. This closes plan 37 after its sole compile correction; stop and reassess cache/family/domain accounting. New plan 38 preserves logical import progress and canonical byte interfaces, but counts unique physical representation/scale allocations across the entire residency cache and peers. Both optional holding and charge reconciliation use that same resident cache interface. Actual model inference has not yet run.

Patch 38 integrated build passes in 48.35 seconds, binary SHA-256 `964c87e46fe61dd738d858dbdbea4c6d84a66bc67ff7adad3e4a8aac0331ae3f`. Source replay matches all 126 files. New original-GSQ loader PID 761975 uses the same three-P100 parallel projection settings; prior failed logs and the measured patch-35 executable are preserved. Actual forward acceptance remains pending.

Patch 38 original GSQ reaches serving in 74.60 seconds (warm-up 566 ms) and returns HTTP 200, 192 tokens at 17.966034031957285 tok/s (10686.832701 ms decode, 13.668571966 seconds wall). Choices and usage exactly match patch 35; MTP remains 156/101. Residency is 8059/8521/8675 MiB with P40 zero. The speed gate still fails. Diagnose the actual parallel fork/join projection intervals before choosing a correction: the previous launch-detail profile disables graphs and cannot attribute this new parallel execution.

Patch 39 full diagnostic build passes in 1m09s, binary SHA-256 `492f6eebf2a1918ad347a88ca3c8d0abf9489005edcb45e0bc90650b0cd6b4ca`. Original GSQ reaches READY in about 75 seconds, then completes the 64-token original prefix and MTP 51/34 through actual parallel fork/join projections. Per-graph events have explicit ends after replay, excluding later host graph construction. The diagnostic adds per-entry driver/event work; its 7.691445 tok/s is not production throughput. Non-graph entry intervals still end at the next mark and can contain host graph-construction gaps; do not rank their kernel execution from those values. The printed parallel_graph boolean identifies the enabled diagnostic mode, not proof that every launch is a parallel graph. Dedicated projection GEMV/GEMM intervals are used for the next correction.

Direct remote output stores retain exact poisoned coverage but regress paired M2/M4 to 0.265/0.395 ms versus gathers 0.199/0.214 ms, and are rejected. Four-warp gathers pass 835584 exact F32 values across M1/2/4/11/16; M2/M4 improve to 0.175/0.202 ms, whereas M16 regresses to 1.28 ms and keeps eight warps. Narrow checked plane addressing plus four warps passes the same-group and mixed-group exact gates (each 835584 values), with M2/M4 medians about 0.171/0.199 and 0.164/0.233 ms. Original master-only VMM single-device rates are peer-read diagnostics, never speed baselines. Patch 40 selects smaller blocks only for dedicated M2/M4 entries across all six projection families and twelve DSL geometries, retaining KSPLIT association and the wider addressing fallback. Full build and original-model speed are pending. Read-only GPU queries show 250 W default/max limits, 715 MHz maximum memory clocks and recent 1328 MHz maximum SM clocks on all three P100; no tuning/driver changes were made.

Patch 40 integrated build passes in 5m36s, binary SHA-256 `f1d45ab094f1ebc6b83bc7292bf92c2a3e924b1e9ed86cc360501374e10018e2`. All 126 tracked source files match isolated patches 01–40. Owned diagnostic PID 769318 stops after preserving its measured logs; new original-GSQ PID 778057 loads with production diagnostics disabled, exact F16 KV, reference arithmetic, adaptive max-MTP10 and true three-P100 projection sharding. Actual READY and rate are pending.

Patch 40 reaches serving in 271.18 seconds and returns HTTP 200, 192 tokens at 17.904092848741108 tok/s (10723.804977 ms decoding, 13.744964192 seconds wall). Choices/usage exactly match 38 and MTP remains 156/101. Smaller-block probe gains do not improve the end-to-end rate. This second speed failure closes the 38/40 performance loop; stop inference edits and reassess production replay construction, scheduling, copies and kernels together. New plan 41 measures per-launch CUDA event nodes inside the complete production replay graph, plus cache-construction CPU time and whole-submission GPU time. The earlier diagnostic reconstructed one graph per projection and contaminated non-graph intervals; it cannot identify the complete production bottleneck. No target-speed, Heretic or concurrent-Pi qualification is inferred.

Plan 41 first build rejects a mutable graph-frontier update while the context-current guard still borrows self. Its sole correction restores the previous context before mutating the host frontier; event dependencies and timing are unchanged. Rebuild retest required. No production process was stopped.

Plan 41 corrected build passes in 1m08s (SHA `e61adce8bbc3ed7a35e263fca5b6d47cf77fb0d199d5d618c35089717a41e95c`), and complete production-replay instrumentation completes the original 64-token prefix. Exact per-launch graph spans total 4.154259274 seconds; 84 of 510 submissions construct graphs, costing 1.675995142 CPU seconds. Whole-submission GPU spans start before capture and therefore also include host construction gaps; do not add them to CPU construction as independent times. Recurrent step accounts for 0.742128768 event seconds and draft_rows 0.773708106 including a cold 327.8 ms event. Concurrent real Pi sessions inject project instructions/skills/tools, but both fail before any read-tool loop: a 17825792-byte temporary allocation has only 16025585 quota bytes available. Their CLI exit zero is not PASS. The worker labels this budget exhaustion DeviceLost and unloads; no hardware/driver failure is established. This second gate failure closes plan 41 after its sole compile correction. New plan 42 must bound and reuse parallel temporary storage before further throughput claims or recurrent tuning.

Patch 42 integrated build passes in 1m04s, binary SHA-256 `b7ef1e7ad3fc4b1eca82658d91fc65f125b6a4944325607da690fbb979e1ca1e`. All 126 source files match patches 01–42 replay. The per-peer arena key is the unique source DeviceIdentity, not hardware UUID, keeping separate CUDA queues distinct. Input regions and unique output regions occupy disjoint aligned spans in one frame; every projection fork/join and source-stream ordering serialize reuse. Old graph generations retain their own arena Arc, and failed growth preserves the existing cache. New original-GSQ PID 792093 loads with profiling disabled. Actual 192-token output/rate and both real Pi tool loops remain pending.

Patch 42 original GSQ returns HTTP 200, 192 tokens at 18.554402012598306 tok/s (10347.948690 ms decode, 13.111322352 seconds wall), with exact patch-40 choices/usage and 156/101 MTP. Both real concurrent Pi sessions now complete their independent read-tool loops and normal final answers in 172.264094/160.171894 seconds. Initial prompts are 3024/3034 tokens, with 15.504836/17.903961 tok/s generation; after each tool result, cache_n is exactly its own previous total (3065/3089), and generation is 18.525901/17.606193 tok/s. Completion IDs are disjoint, paths/answers differ correctly, no assistant errors occur, and both tools succeed. Source system sections inject the real project AGENTS.md. Skill discovery is enabled by the normal CLI, but the existing user/project Pi catalog has no loaded skills; no skill invocation is qualified. This corrects the earlier unbounded-temporary Pi failure. Requested speed and fully populated context remain unqualified.

Focused recurrent gate 43 first build rejects nested inner Rust documentation comments from an included test-file wrapper. Its sole correction appends the focused ignored test to the existing verified CUDA recurrent fixture, removing the owned wrapper. Production inference code is unchanged; exact mapping/state/timing retest required.

Focused recurrent gate 43 passes after the wrapper correction: build 6.54 seconds, exact state/window/tape/gated output comparisons and DRAM-rotated mapping measurements in 12.29 seconds at active NK16/NV48/W128/C4, BF16 inputs and partial stop/tape publication. M2 medians for (ROWS,WARPS)=(2,4),(4,4),(2,8),(4,8) are 109.236/87.376/95.584/69.382 microseconds; M4 142.688/117.664/133.472/91.684 microseconds. These fixture rates do not replace production timings, whose tape/version workload differs. Patch 44 selects the existing exact 4-row/8-warp CUDA mapping by reordering declared defaults, with no kernel or tensor-storage change. Original-model and real Pi speed remain pending.

Patch 44 integrated build passes in 5m22s, binary SHA-256 `323a382b6fe14270762d79ffc6103fed45a8e7abdc055117bca52840772c02d8`. Original GSQ PID 826254 reaches serving in 74.38 seconds and returns 192 tokens at 20.678389085 tok/s (9285.055969 ms decode, 12.072021017 seconds wall). A focused repeated request with cache_prompt=false, preserving a separate artifact, measures 22.316196130 tok/s (8603.616803 ms decode, 10.471243521 seconds wall); KV prompt cache remains zero. Both responses and usage exactly match patch42 with 156/101 MTP. Requested speed remains unmet. Reassessment identifies unsupported MTP operators using canonical masters of row-sharded weights: draft_rows and head_logits_rows are outside the six-family parallel capture descriptor. The head loader consumes the same placed ResidencyStore, so its large combine/readout tensors follow the sharded importer too. Plan 45 extends explicit operator descriptors and row controls without changing their arithmetic; non-projection table parameters must retain the full canonical read view rather than partial output-row aliases.

Patch 45 built successfully (5m21s), binary SHA-256 `b71fcc8f6e9ee1ca61583eb3364b4a11bc41f7945bc9cc80181c05ee52bf70cf`. Original GSQ reached serving in75.05 seconds on three P100. The first192-token request failed HTTP500 after1.808660249 seconds with CUDA illegal memory access700; the worker unloaded. No patch-45 rate or Pi qualification exists. This is the first failure of this change; isolate the offending operation using installed Compute Sanitizer before its sole corrective patch. Read-only queries show all four GPUs present and excluded P40 at zero residency.

The filtered instrumented run reached READY but did not return its request before stopping; expected missing-symbol API reports exhausted its print limit, so no precise sanitizer verdict is claimed. Reachable source establishes the partial-view defect independently: native/head.rs caps the draft vocabulary at65536, native_head.rs takes the leading rows, projection.rs derives cuts from that view, while RowShardPlan owns cuts of248320 rows. Patch46 is the sole correction: refuse local-alias fork/join unless each weight view covers its entire sealed row allocation at offset0. Partial views retain their original full-canonical source execution; complete target and combine matrices remain distributed. Rebuilt model retest is pending.

Patch46 builds in1m05s, SHA-256`a7292f4fc235c86caa00b97ae96ddf8f0057cf20792d3d47a64121451b15e7b4`; original GSQ reaches serving in74.59 seconds. Requests return HTTP200 and exactly the prior choices/usage and156/101 MTP counters, but rates regress to7.403654/7.571625 tok/s. This second failure closes45/46. Reassess: distributing the entire draft entry makes each32-output CTA independently decode/normalize the arbitrary embedding row through the canonical multi-owner view. New plan47 separates that original fixed-order preparation from output projection, preparing once per input row on the authored source then copying the exact F32 joined values for parallel local combine reads. No new performance claim.

Patch47's first integrated build fails the authored-native ABI validator: non-launch-scoped entries do not generate SEISMIC_FORMING guards. The sole correction48 removes those guards and defines both unscoped CUDA launch symbols, retaining identical argument words, scratch, arithmetic, and DSL launches. This follows seismic/build/src/lib.rs native_abi_symbols and runtime/native/abi.rs render_source_with. Rebuild and actual-model gates remain pending. A separate diagnostic of constexpr four-column scheduling for single-weight M4 preserves outputs but regresses0.247→0.276ms; rejected and absent from production source.

Patch48 builds in5m22s, SHA-256`d4c11ace4d1a6f5c9e3b4451840989c3b779fa34f5266e5ac348999ab058dccc`, READY75.14 seconds. First/warm original192 requests measure7.426055/7.591135 tok/s with exact prior choices/usage and156/101 MTP counters. This second47/48 failure closes its loop. Splitting preparation does not recover the regression; the repeated-preparation explanation is not established as its main cause. New plan49 records actual projection view/admission metadata and complete replay intervals before another performance edit. No Pi qualification or speed-target PASS.

Plan49's actual ABI trace and complete warm replay spans establish the regression: attention gate is a zero-byte[0,5120] view of the query's12288-row allocation. The nonempty-view guard incorrectly disables the entire parallel attention projection. Warm M2/M4 medians are11.096896/13.827280ms, accounting for5.362061241 of8.658886457 recorded kernel seconds. All other full view extents/bytes match their plans. Plan50 ignores zero-byte weight views in admission while retaining every nonempty prefix/slice check. No arithmetic change or fidelity exception.

Patch50 builds in1m05s, SHA-256`2c2adbd7cc3139bed46dccb1b15336be42cc84e61d591aec98f7f5b5b3858dde`; READY73.37 seconds. Original192 requests measure24.162965 and26.383662 tok/s, exactly matching44 choices/usage and156/101 MTP counters. Warm rate improves18.23% over44's22.316196. Two concurrent realPi sessions are running; production acceptance and requested speed remain pending/unmet. Read-only telemetry during real inference shows1328MHz on all three P100,40/48/45°C and P40 at zero residency.

Real concurrentPi50 PASS: both independent sessions have normal injected project AGENTS.md and skill discovery (existing catalog empty), one successful read tool and a normal final answer. Durations128.48/119.44s. First tool-selection rates11.5336/14.3840 tok/s overlap long prefills; final-answer rates23.9791/21.5466. Second requests cache exactly their own first totals3065/3089; response ids and tool paths remain distinct. This qualifies functional concurrency, not the requested throughput or fully populated262144-token context. Artifact: out/magnitude-swift/pi-production50-validation.json.

Actual51 confirms the selected4-row/8-warp recurrent mapping atB1 forM1/M2/M4; five host-mapped controls are offsets0/256/512/768/1024 of276480-byte upload backing. Production median0.295552ms. Plan52 stages only read mutable mapped spans locally once per replay, using checked ranges and bounded queue-owned scratch; actual effect is pending. Its first Rust build rejects Display formatting for DeviceId in a diagnostic line. Sole correction53 uses Debug formatting; no execution equation or behavior changes. Rebuild and actual gates remain pending.

Patch53 builds in1m04s, SHA-256`60f2ee99d25a12ed0a7aba41f84f4cf0a9b3f293efce62bc52531ac505bfa63a`; original GSQ reaches serving in74.59 seconds with mapped upload staging enabled and diagnostics off. First/warm192 requests measure27.796682/31.098078 tok/s, with exact50 choices/usage and156/101 MTP counters. Warm improves17.87% over50 and39.35% over44. Real concurrentPi53 is running. This reaches the earlier30tok/s lower range but does not fulfill the later request to exceed reported llama40–60; no completed-task claim.

Real concurrentPi53 PASS: both sessions have injected project AGENTS.md, normal skill discovery (catalog empty), successful read tool and normal final answer, with own-cache3065/3089 and distinct ids. Durations126.48/128.49s. First selections17.4522/25.5486 tok/s overlap prefills; final answers28.1443/26.7417 tok/s. Requested speed beyond reported40–60 remains unmet; original Heretic on the final path and fully populated262144 context remain unqualified.

Inspection-only guard54 probe passes all835584 F32 outputs on3P100; constant bound prototype improves M2 median0.172016→0.168174ms andM4 0.200309→0.196886ms. No production source change or model-rate claim. Inspection-only single lookahead55 matches saved F32 outputs; M2 0.213661→0.198555ms, butM4 regresses0.247136→0.329549ms. Broad lookahead is rejected; onlyM2 is a possible later selected correction. Preserve53's actual31.098078 rate and real2Pi functional gate while qualifying the same final path on original Heretic.

Original Heretic.i1-Q4_K_M (16810715936-byte source) loads with the same53 binary and explicit three P100, exact F16 KV, context262144 configured, maxMTP10 and staged mapped uploads. READY54.19 seconds; resident target16540401664 bytes. First/warm original192 requests returnHTTP200 at27.538987/28.774896 tok/s with153/99 MTP counters and identical responses between them. Original source file and format are preserved. Real concurrentPi Heretic53 is running; no final speed/Pi/full-context PASS is inferred. GSQ53's31.098078 rate and realPi functional qualification remain preserved.

Real concurrentPi Heretic53 PASS with project AGENTS.md and normal skill discovery (catalog empty), successful reads and final answers. Durations129.48/130.48s; initial19.7746/26.0142 and final25.8890/24.1112 tok/s. Both cache their own totals3065/3089 with distinct response ids/tool paths. Functional gates now pass for both original models on53. Requested higher speed remains unmet. Inspection-only metadata56 passes835584 F32 outputs on threeP100: caching authored packed coefficient blocks over fourK chunks improves pairedM2 median0.172016→0.144210ms, butM4 regresses0.200309→0.210810ms. OnlyM2 is selected for further qualification; no source implementation or model-rate gain is claimed.

Plan54 integrates onlyM2 packed coefficient-block reuse after both same/mixed three-P100 exact gates and the13-format212992 coefficient-bit gate. M4 regression is excluded; dense/other widths retain original behavior. Source replay matches all128 files. Integrated build and actual model rate are pending;53 remains the qualified functional base for both Swift models.

Patch54 builds in5m21s, SHA-256`32dc9be9ea5de787a26106f18794db84fdeb62d7314e16b83057c2a4dfebc3fc`. Original Heretic reaches READY in60.024642s; first/warm192 requests returnHTTP200 at28.920546/30.312022 tok/s, with exact53 choices/usage and153/99 MTP counters. Warm improves5.341900% over53. Original GSQ54 is loading on explicit threeP100; GSQ54 throughput and realPi54 are pending. Requested higher40–60 speed and fully populated262144 context remain unqualified.

Original GSQ54 READY270.058741s (new CUDA programs), first/warm192HTTP20028.201420/31.563914 tok/s, exact53 choices/usage and156/101 MTP counters. The approximately1.50% warm difference is small; Heretic's5.34% increase is more substantial. Two real Pi sessions on GSQ54 are active; current throughput remains below the requested higher40–60 rate.

Real concurrentPi GSQ54 PASS: normal injected project AGENTS.md and skill discovery (catalog empty), successful independent read tools and normal final answers. Durations127.682/119.643s; initial11.9075/14.7517 and final26.1631/27.2368 tok/s. Own-cache totals3065/3089 and distinct request ids. This confirms functionality and isolation; it does not establish an improvement over53's concurrent rates or the requested throughput.

Real concurrentPi Heretic54 PASS: both independent sessions retain injected project instructions and normal skill discovery (catalog empty), successful reads and normal final answers. Durations130.493/120.426s; initial13.2460/15.7923 and final25.8471/27.5493 tok/s. Own-session cache and disjoint response ids pass. Both original models are functionally qualified on54. The requested throughput remains unmet; next attribution uses the same54 binary with mapped-upload staging ON and opt-in existing replay diagnostics. Host CPU time7.32s/~7.81wall on a GSQ192 request cannot be interpreted as CPU compute: CUDA's default event synchronization can spin while waiting.

Same54 binary warm originalHeretic64 diagnostic, with mapped staging enabled: dense_output M2 median0.217728ms/sum0.373770176s, dense_expand M2 0.203328ms/0.343050080s; M4 expand/output0.352544/0.335248ms. Recurrent step median0.082496ms, resolving51's0.295552ms mapped-input bottleneck. There are595 submissions, graph-construction0.074418798s, CPU encode0.390694626s; event spans and whole-submission intervals overlap and are not additive. New plan55 probes code-size effects of the four-chunk metadata-loop unrolling before any new production edit.

Plan55 loop-size probe closed after the sole correction: all835584 poisoned F32 values match, but rolled M2 is0.235395ms with48 local bytes versus baseline0.145334ms/zero local bytes. Explicit IQ2H word selection removes local storage, yet still regresses to0.157133ms (80registers versus87baseline). Neither variant enters production. This second speed failure ends that direction. The actual54 original models and concurrent Pi gates remain qualified; next work must reassess the feed-forward data path rather than continue unroll edits.

Row-half Plan56 has two fixture compilation failures. The initial interpretation blamed lowercasegroup(), but exact NVRTC carets identify uppercaseGROUPS, an authored operator buffer macro expandingShape::GROUPS. No row-half inference ran or speed was measured. End-to-end reassessment isolates the standalone fixture from that macro under newPlan57; production54 and its qualified gates remain unchanged.

Down-projection prototype baseline initially fails its own context with700. Compute-sanitizer identifies a16-byte code read atblock112; address arithmetic identifies the old2800-byte stride. Generated semantic QUERY_WEIGHT bindings still describeIQ2H32 while the formalPARAM_2 fixture describesQ4K/9792. Original Heretic remainsREADY with all P100s visible, P40zero. Binding directly to updated formal metadata corrects the fixture. Complete original Q4K/N5120/K17408 baseline then PASS all348160 F32 values, M2 median0.175531ms/56registers/zero local bytes. Row-half comparison is pending; no production source past54.

Corrected original-dimension Q4K down row-half gate passes348160 F32 values but M2 regresses0.175531→0.194440ms despite56→40registers and zero local bytes. Row halves are rejected for both paired and down shapes. No inference source past54. NewPlan59 combines54's cached coefficients with35's raw-code lookahead, which54 currently bypasses; exact same/mixed/Q4K fork/join gates precede any integration.

Further diagnostics remain unintegrated: raw lookahead60 paired0.146822 anddown0.189507ms versus0.145334/0.175531baseline; exact two-stage products61 0.258245/0.302990ms versus0.175531 despite348160 exact outputs. Decoded-weight experiments explicitly change FP32 association and are not fidelity-qualified: scalarrow62 1.067282/0.936094ms, mma16fragment63 0.212136ms versus0.175531. None is selected. Normal54 originalHeretic is restored READY as ownedPID976209 with diagnostics OFF, sameSHA32dc9be9…3fc, explicit3P100, F16KV/maxMTP10. OriginalGSQ54 remains measured31.563914 andHeretic54 30.312022 tok/s; both normalrealPi54 gates pass. Requested higher40–60 remains unmet. No driver/library/power/homePi/P40 change.
