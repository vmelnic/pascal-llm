# Magnitude Swift admission and native Pascal CUDA fixes

Base: `magnitudedev/magnitude` release source
`38f0adb8f6470642aaa15f7aa46ac2d135359d23` (`@magnitudedev/cli@0.2.6`).
Apply these patches in order to a clean checkout:

1. `01-discover-embedded-mtp-targets.patch`: discovers complete target GGUFs
   whose original filename contains MTP; explicit draft metadata remains
   authoritative.
2. `02-interpret-target-and-mtp-mixer-flags.patch`: accepts recurrent flags
   for target blocks or target-plus-head blocks, validates the head suffix,
   and keeps target/head geometry separate.
3. `03-native-sm60-cuda.patch`: preserves the native CUDA helper ABI on Pascal
   with SIMT matrix products, software BF16 RNE and synchronous copies; admits
   SM60 with honest physical limits and without advertising tensor instructions.
   The existing NVRTC directory setting selects the host's installed CUDA 12.4.
4. `04-fit-native-projections-to-shared-memory.patch`: makes row tile and pipeline
   depth generic launch parameters, adding a synchronized 32-row single-stage
   candidate within P100's shared-memory budget. Explicit operator fixtures use
   the complete launch specialization; numerical oracles and tolerances are
   unchanged.
5. `05-qualify-bounded-recurrent-scan.patch`: exercises the existing 32-row
   recurrent scan at width 128, including the 27B value-head geometry, with the
   independent host model and the existing numerical bounds. No runtime policy
   or kernel is changed by this test patch.

6. `06-dense-f16-history.patch`: introduces explicit IEEE F16 history independently
   of activation precision, propagates it into target/MTP/separate-draft layouts
   and native bindings, and bounds wide-head prefill with a generic key tile.
   CPU and strict P100 mixed-history tests compare writable KV bytes exactly.

7. `07-exact-gsq-cuda-codecs.patch`: imports the seven missing importance-grid
   and Q2_K source encodings without requantization, using exact palettes and
   scales. Native imports passed on original samples from all 315 affected
   tensors. Projection qualification was interrupted to prioritize the user's
   requested integrated model run; it is not reported as passed.
8. `08-explicit-service-numerical-policy.patch`: passes one immutable selected
   history codec and proposal width into assessment and actual load resolution.
   The service accepts `--kv-codec dense-f16 --mtp-proposals 10`.
9. `09-bound-mtp-chains-at-16.patch`: permits the requested ten proposals under
   a generic bounded chain width; allocations still follow the selected width.
10. `10-bounded-cuda-recurrent-default.patch`: selects the 47,152-byte scan
    default instead of the 62,896-byte form rejected by the actual P100 load.
11. `11-check-state-capacity-before-kernels.patch`: derives the same physical
    state-capacity plan before device opening and kernel preparation, and
    carries it into graph construction. Impossible loads fail promptly.

12. `12-explicit-cuda-peer-copy.patch`: adds an exact CUDA peer-copy operation
    with distinct physical allocations, access permits, completed producer
    fences and checked source/destination ranges. Foreign graph bindings remain
    rejected.
13. `13-native-physical-layer-placement.patch`: adds explicit ordered device
    selection, artifact-derived contiguous decoder stages, physically owned
    weights/history/recurrent slabs and graph pools, exact hidden-row transfers,
    independent per-device admission and allocation reporting. The CLI accepts
    repeated `--peer-device cuda:<uuid>` options. Original GGUFs, F16 KV and the
    selected MTP proposal width are preserved. Local complete CLI source checks
    pass; the Linux release build passed and both original models reached native READY on three P100s.

14. `14-biased-sixteen-code-gemv.patch`: corrects the actual original-GSQ load
    failure for biased 16-code groups. The GEMV computes group-width-matched
    activation sums; the existing 32-code fold keeps its original order.
    Actual full-model preparation remains the qualification gate.

15. `15-canonical-import-nan.patch`: preserves the registry's canonical F32
    NaN payload in exact IQ palette imports. The actual startup arbitrary-byte
    fixture exposed the difference from CUDA's arithmetic NaN payload. Finite
    values, infinities, signed zeros, original GGUFs and the strict startup
    comparison are unchanged. The exact failed-startup regression is included;
    its corrected native exact-byte result passed for all seven bindings. The
    complete-model startup passed with patches 01–17.

16. `16-correct-history-and-boundary-bindings.patch`: supplies the selected
    DenseF16 representation to the existing native attention qualification
    buffers, for target and MTP callers. Physical boundary receivers now use
    the actual F32 residual ABI declared by the generic decoder block graph,
    independently of BF16 projection activations. Peer copies still preserve
    exact bytes and foreign-binding checks. Both original models reached READY with patches 01–17.

17. `17-propagate-history-and-gsq-fixtures.patch`: carries the dense history
    element with the attested kernels and both native/checked graph entries,
    so every target/MTP/draft history port follows the same typed kernel ABI.
    Adds valid source-format unit packets for the six new resident codecs.
    The extended existing unit-fixture test passed for all layouts. Original
    Heretic reached READY and completed a coherent 192-token native request
    across all three P100s and two concurrent real Pi tool loops. GSQ reached READY; its generation/Pi gates are pending.

18. `18-attribute-real-cuda-generation.patch`: opt-in installed-CUDA event
    attribution on real sealed-model submissions. Diagnostic events disable
    replay and do not qualify production speed. Original GSQ attribution
    identifies the projection families as 88.14% of recorded GPU time.
19. `19-native-simt-gemv.patch`: replaces compatibility MMA inside Pascal
    GEMV with actual-row SIMT dot products over the unchanged native packets,
    fixed group reductions and the existing F32 scale/bias fold. Uses fixed
    C-fragment register indices and retains the epilogue/cross-part reduction.
    The focused native gate checks independent projection bounds and exact
    row/batch invariance for the original formats; passed in 507.49s.
20. `20-session-adaptive-mtp.patch`: an opt-in acceptance controller scoped
    to each MTP request. Configured maximum ten and physical state capacity
    remain; actual proposal widths contract after rejected prefixes and grow
    after three fully accepted rounds. Target verification/sampling and full
    vocabulary remain. Selected only with `MAGNITUDE_ADAPTIVE_MTP=1`.

21. `21-reuse-pascal-matrix-operands.patch`: shares fragment operands among
    four independent matrix outputs while preserving every output's exact
    ascending-K F32 FMA sequence. Installed-CUDA helper qualification passed
    exact F16/BF16 matrix bits against scalar `fmaf`; matrix register count
    fell from 48 to 32, with zero local bytes. Production effect is pending.

22. `22-direct-simt-operands.patch`: keeps exact packed-weight codes widened
    in F32 for SIMT; the same packet decoders expose a typed operand pair.
    Tensor-core operands and F32 group folds retain their existing behavior.
23. `23-correct-f16-byte-operand.patch`: fixes the existing signed-byte F16
    permute selector. The independent GPU integer oracle exposed (1,0)
    becoming (1,1); the corrected decoder returns the source pair exactly.
24. `24-preserve-native-dense-operands.patch`: preserves original dense bits
    when source and activation elements match; other dense sources retain
    activation rounding. Exact checks include dense NaN payloads. The
    installed-CUDA gate passed 393216 packet-bit and 1966080 integer-reference
    comparisons in 1.215s. Integrated production throughput is pending.

25. `25-stream-pascal-gemv-packets.patch`: consumes one k-block through the
    existing code/coefficient interfaces instead of retaining two four-block
    superblocks for each weight stream. F32 fold and cross-part orders remain.
    Actual P100 expansion registers fall from 206 to 64 with no local spill;
    the real original GSQ request measured12.178975tok/s with all192 response
    choices matching01–24. This still misses the requested throughput.

26. `26-shortcut-single-row-slabs.patch`: removes runtime-wide division and
    remainder when one logical row occupies a slab. Other slab sizes retain
    their existing addresses. Installed CUDA passed 4636672 exact address
    comparisons; original GSQ measured 13.350953 tok/s with exact choices
    and acceptance counters matching patch 25.

The patches include normative design updates and focused Rust tests. The
second patch includes an opt-in test against locally supplied real GGUF
headers. `MAGNITUDE_MTP_GGUF_HEADERS` is a platform-separated path list.
The CUDA patch adapts the existing backend. Patches 01–11 add the missing GSQ
source codecs and explicit F16 policy, but do not implement multi-device model
placement. Patches 12–13 implement native physical placement; component checks do not establish complete-model qualification.

From a patched checkout, with Rust 1.98.0 available:

```sh
cargo +1.98.0 test --locked --manifest-path inference/Cargo.toml \
  -p magnitude-family-qwen35 --test interpretation
cargo +1.98.0 test --locked --manifest-path inference/Cargo.toml \
  -p magnitude-family-qwen35 --test interpretation local_mtp_headers \
  -- --ignored --nocapture
cargo +1.98.0 build --release --locked --manifest-path inference/Cargo.toml \
  -p magnitude-service-server --bin magnitude-inference
```

After staging the rebuilt executable in an isolated installation, regenerate
`installation.json`'s `nativeBuild` from that executable's `version --json`.
The validator requires regular runtime/catalog directories. Retain the binary
SHA-256, release commit, patches and actual build identity with measurements.

The X99E CUDA candidate is staged at
`/home/vm/pascal-llm/work/magnitude-0.2.6/installation-cuda-sm60/`.
Its regular `runtime/` directory links to the existing matched host NVRTC 12.4
and builtins files; the packaged NVRTC 12.9 is not used by this candidate.
`cuda-sm60.env` records the explicit library directory and the three-P100
visibility. It disables Vulkan discovery for this process. No model service
or Pi configuration is changed by staging the candidate.

Current host results and remaining qualification gates are recorded in
`docs/magnitude-swift-qualification.md`. A header test or successful load plan
does not qualify generation, exact F16 KV or concurrent production Pi use.

The earlier patches 01–11 single-device attempts invoked both unchanged Swift
models with CUDA 12.4, F16 history, MTP 10 and configured context 262144. Both
refused the single-P100 history-capacity check. Those processes are stopped;
their commands and evidence remain in
`out/magnitude-swift/swift-native-model-runs/results.json` (binary SHA-256
`19d418f54e8aa24bf5e4b1c1731a5a0ab9525f84319c9bfb111e4b041ad1ae77`).

The standalone native executable is built at
`/home/vm/pascal-llm/work/magnitude-source/inference/target/release/magnitude-engine`.
The 01–17 baseline SHA-256 was
`32a50aa50b2d30a9cb1385b9b4f25304088c16ae2842d3585b3056272401fe46`.
Both original models passed native generation and two concurrent real Pi
read-tool loops. Baseline decode rates were 4.075757 tok/s for Heretic and
5.066839 tok/s for GSQ; these processes are stopped.

The integrated 01–21 GSQ binary SHA-256 is
`867ca79e0d9c1f60c3169763a05e4b64cf654edeb5f42d9deb905cdfccd20e41`.
Its profile-disabled generation measured 7.428979 tok/s with maximum MTP ten
and request-local adaptive proposals. Warmed VRAM was 7721/6773/6573 MiB,
with P40 excluded. The requested throughput remains unmet. Patches 22–24
passed the exact operand-bit gate; integrated GSQ measured 8.200016 tok/s
and all 192 response tokens match 01–21. The patch-25 corrective candidate
built successfully; original GSQ measured12.178975tok/s with identical
192-token choices and unchanged156/101 MTP counters. End-to-end reassessment
precedes any further code change.
Context 262144 is configured; actually populated long-context concurrency
is not qualified. Run identities and complete evidence are retained under
`out/magnitude-swift/` and in `docs/magnitude-swift-qualification.md`.
Readiness, throughput and concurrent Pi qualification are not implied by a
release build.


On X99E, the independent native helper gate uses the installed libraries and
rejects a selected device unless it is SM60:

```sh
python3 patches/magnitude/qualify-sm60-helpers.py \
  --prelude work/magnitude-source/inference/seismic/runtime/src/native/cuda_prelude.cuh \
  --device 0 --output out/magnitude-swift/sm60-helpers.json
```

Run the selected native operator tests with explicit CUDA admission; this avoids
silently skipping a broken CUDA device. P40 is not visible to these processes:

```sh
export SEISMIC_NVRTC_DIRECTORY=/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0
export SEISMIC_TEST_BACKEND=cuda
cargo +1.98.0 test --release --locked --manifest-path inference/Cargo.toml \
  -p seismic-cuda --lib
cargo +1.98.0 test --release --locked --manifest-path inference/Cargo.toml \
  -p magnitude-kernels --test cuda_projection \
  cuda_dense_weights_match_host_model -- --exact --nocapture
cargo +1.98.0 test --release --locked --manifest-path inference/Cargo.toml \
  -p magnitude-kernels --test cuda_projection \
  cuda_dense_expand_matches_host_model -- --exact --nocapture
cargo +1.98.0 test --release --locked --manifest-path inference/Cargo.toml \
  -p magnitude-kernels --test cuda_recurrent \
  cuda_chunk_32_rows_matches_the_host_model -- --exact --nocapture
```

When transferring sources from macOS, disable AppleDouble metadata
(`COPYFILE_DISABLE=1 tar --no-xattrs ...`); the native tree-parity gate correctly
rejects unexpected `._*` source files. Do not bypass that gate.

27. `27-share-paired-simt-activations.patch`: shares the activation traversal
    and bias sum between equal-group-width paired SIMT projections while
    preserving each stream's exact product and scale/bias fold order.
    Unequal widths retain the independent route. Native fragment-bit
    verification passed 131072 comparisons; original GSQ measured 13.914128 tok/s
    with exact 192-token choices and counters matching patch 26. The
    requested throughput remains unmet; warmed configuration selection
    is the current correction.
    `qualify-paired-simt.py` verifies twelve decoders, mixed pairs and actual
    one/two/multiple-row traversal against the unchanged independent route.

28. `28-preserve-native-tuning-measurement-budget.patch`: explicit
    `MAGNITUDE_EXTENDED_TUNING=1` offline selection with a separate 600-second
    search clock, per-unit census allowance, and initial all-variant
    compilation excluded from measurement time. Reconsiders default-only
    incomplete stored results while retaining measured alternatives and
    numerical checks. The warm correction to 27 measured 13.918248 tok/s
    and failed the speed gate. Patch 28 measured 14.006927 tok/s and changed choices; the speed gate failed.

29. `29-stage-single-row-normalization.patch`: forms a normalized CUDA GEMV
    row once through the existing staging launch, including a single row.
    Updates all six CUDA consumers' scratch allocation and shared-memory
    declarations, retaining dependency waits and identical normalization
    arithmetic. Generic `MAGNITUDE_NATIVE_REFERENCE=1` prepares authored
    defaults directly while qualifying exact implementation changes.
    Extended selection measured 14.006927 tok/s and changed token choices;
    its speed gate failed. Its sole correction measured 13.067738 tok/s with the original choices restored; the speed gate failed again.

30. `30-lossless-iq2-small-row-projections.patch`: factors IQ2 resident
    coefficients into one exact F32 factor per original 256-value packet
    and unchanged unsigned odd local scales. The existing four-bit palette
    code bytes stay unchanged; the import registry and portable decoder
    describe the new storage ABI. CUDA matrix placement selects the compact
    forms; other backends retain the existing forms. Updates all four native
    matrix slots, GEMV, staged GEMM and lookup accessors, import/reference
    fixtures and native unit patterns. Adds dedicated M=2/M=4 launches to the
    six measured hot projection entries, retaining the original KSPLIT and
    floating-point association. `qualify-iq2-hierarchy.py` passed 75497472
    exact coefficient-bit comparisons on P100, including canonical NaNs.
    Native original-model imports and throughput are pending; no speed or
    production qualification follows from the coefficient gate alone.

31. `31-complete-cuda-iq2-hierarchical-import.patch`: completes the native CUDA source adapter for the IQ2H forms introduced by patch 30. Retains original grid/sign/code decoding, writes exact unsigned odd coefficients and the shared F32 factor with canonical NaNs, and updates both packet and row/mma16 physical writes. Patch 30 startup import failed before model load; source admission and payload writes were omitted in its hand-written CUDA adapter. The integrated build, ten startup physical-byte bindings and all 167 original IQ2 fixtures (4360704 exact decoded comparisons) passed. Original-model preparation then stalled in NVRTC on a static zero-row attention segment; patch 32 addresses that load failure.

32. `32-exclude-empty-attention-segments.patch`: excludes statically zero-width attention matrix segments from native GEMV/GEMM formation and removes the default zero-row GEMV body. Nonempty operations, dependencies, arithmetic and storage remain unchanged. The owned original GSQ loader spent over 20 minutes in installed NVRTC; its captured M2 source timed out after 20 seconds, whereas static empty-segment guards compiled it in 0.545515292 seconds. The complete rebuilt original GSQ reached serving in 231.41 seconds with residency on all three P100. Its actual 192-token request measured 13.663491 tok/s and exact patch-29 choices/usage. Loading passes; the speed gate fails. The 31/32 loop ends for reassessment.

33. `33-native-row-parallel-projections.patch`: gives each M=2/M=4 native projection activation row independent CUDA blocks. Generic source/epilogue row views retain original logical indexing, scale/bias association, four-lane reductions and K-part boundaries. Updates all six entry families and twelve launch grids together, including attention zero-fill. `qualify-row-parallel.py` passed 10768 exact fragment/epilogue comparisons on P100 across twelve formats and mixed group widths in 13.755 seconds. Actual original GSQ reaches serving in 250.19 seconds but measures 5.678221 tok/s with exact patch-32 choices/usage; first speed failure. Concurrent Pi is not qualified.

34. `34-interleave-small-row-weight-tiles.patch`: interleaves logical activation-row blocks per weight tile by swapping grid axes and updating generic tile selection, row views and attention zero-fill together. All twelve launches change; arithmetic/storage/device stage placement remain unchanged. Sole correction of plan 33. Exact fragment/epilogue gate and integrated build passed; original GSQ measured 10.542329 tok/s with exact choices/usage. Second speed failure closes this loop.

35. `35-selected-native-projection-schedule.patch`: removes the rejected row-parallel grids/views and restores fused projections. Interleaves independent B=2 single-stream and B=4 paired column chains while preserving each result's exact FMA/group/reduction order. For paired nondense equal-group B=2, overlaps one next 64-value raw/metadata block with current computation; other classes retain original streaming. Selects measured native probe gains without model-family branches, weight changes or new global buffers. `qualify-native-column-schedule.py` passes 10768 exact comparisons across twelve formats and mixed widths. Original GSQ measures 14.385283 tok/s with exact earlier choices/usage; the requested speed remains unmet.

36. `36-concurrent-native-projection-row-shards.patch`: introduces immutable row-sharded CUDA matrices with an explicit canonical import view and partial device-local views. Maps physical pages on their owners and clones only overlapping boundary pages. Captures projection kernels on three CUDA contexts with explicit input copies, pitched output gathers and a fork/join dependency frontier. Per-module row controls retain the original kernel argument ABI and floating-point association. Physical permits, publication fences and graph allocations remain explicit. The poisoned native P100 prototype passes exact outputs at M=1/2/4/11/16; it is diagnostic evidence, not full-model throughput. Integrated build initially fails on a missing import and then an executor type reference; no production gain is qualified.

37. `37-checked-physical-allocation-accounting.patch`: completes allocation-deduplicated, checked physical charges through the public tensor interface, including scales, and accepts the actual opened Device slice at the placement boundary. Full CLI build passes in 47.19 seconds after the sole placement-type correction. Original GSQ is loading with parallel projections on three P100; real model throughput and concurrent Pi remain pending.

38. `38-physical-residency-cache-census.patch`: keeps canonical import progress while counting each unique physical representation/scale allocation once across all residency owners. Aligns cache charges with family/domain physical accounting so target/head tied allocations use consistent units. Integrated build passes in 48.35 seconds; original-model forward and throughput remain pending.

39. `39-attribute-parallel-projection-graphs.patch`: diagnostic mode measures actual captured projection fork/join graphs using existing CUDA events, including root completion of peer branches. Records explicit graph ends and retains diagnostic graph allocations through batch completion. Does not use these instrumented rates as production throughput or non-graph marks as pure kernel timing. Full diagnostic original GSQ request passes.

40. `40-selected-sharded-small-batch-geometry.patch`: selects max(4, KSPLIT) warps for dedicated M2/M4 CUDA projections, updating six source helpers, twelve launch grids and thread counts together. Keeps KSPLIT partitions, reduction association, epilogues and larger batches unchanged. Adds checked u32 plane quotient addressing with the original u64 fallback and u64 final offsets. Three-P100 poisoned native probes match original F32 values for both same/mixed group pairings. Full-model throughput remains pending.

41. `41-profile-complete-production-replays.patch`: records CUDA event nodes around authored launches and complete parallel projection fork/join paths inside the full replay graph. Measures graph-construction CPU time separately; whole-submission GPU spans also include prelaunch capture gaps and must not be added independently to CPU construction. The short original-model diagnostic passes; concurrent real Pi exposes unbounded retained temporary allocations and unloads on quota exhaustion. No production PASS follows from CLI exit zero.

42. `42-bound-parallel-projection-scratch.patch`: reuses disjoint input/output frames in a growing arena per source CUDA queue and destination peer. Queue identities and explicit fork/join frontiers preserve isolation and execution order; cached graphs retain earlier generations whose geometric capacities sum to less than twice the largest arena. Each allocation retains physical-device grants. Full build passes in 1m04s; original model and concurrent Pi acceptance remain pending.

43. `43-qualify-active-pascal-recurrent-mappings.patch`: focused ignored CUDA test qualifies all four authored step mappings at NK16/NV48/W128/C4 and BF16 M2/M4 with a partial stop/tape publication. Compares gated output, window, state and tape bits, then times sixteen DRAM-rotated argument sets. Passes exact comparisons in 12.29 seconds; 4-row/8-warp medians improve109→69 and143→92 microseconds.

44. `44-select-exact-recurrent-row-mapping.patch`: selects the measured existing 4-row/8-warp CUDA default through declared parameter order. All arithmetic, storage and other backends remain unchanged. Original GSQ measures20.678389 tok/s on first192 and22.316196 with replay graphs already constructed and KV prompt cache disabled; choices/usage match42 exactly. Requested speed remains unmet.

45. `45-distribute-native-mtp-projections.patch`: extends explicit native projection capture to draft combine and head logits. Draft blocks receive immutable output-row controls before their original normalization barriers; each output retains original F32 association. Explicit weight ports use partial local aliases, while arbitrary embedding-table lookups retain full canonical read views. Conditioning, token/status and features copies, outputs and joins use bounded peer scratch. The first original-model request fails with CUDA illegal memory access; patch46 addresses a partial-vocabulary view mismatch.

46. `46-guard-partial-projection-views.patch`: rejects local-alias capture when a projection weight view has a different row partition from its backing allocation. The original canonical execution remains for prefixes/slices. Original GSQ output and MTP counters match44, but rates regress to7.404/7.572 tok/s, closing45/46 for reassessment.

47. `47-prepare-draft-row-once.patch`: preserves the draft entry's fixed256-thread normalization order in one source preparation CTA per input row, then copies its exact joined F32 scratch and projects independent local output shards. Removes repeated arbitrary embedding lookups from peer projection CTAs. Actual original-model speed and concurrent Pi qualification are pending.

48. `48-bind-unscoped-draft-launches.patch`: the sole correction of47's failed ABI build defines both unscoped launch symbols without unsupported forming guards. Scratch, argument ABI, calculations and row ownership are unchanged. Model acceptance is pending.

49. `49-trace-projection-view-admission.patch`: opt-in once-per-shape diagnostics record the actual weight-view row/byte/offset contract against its physical allocation plan on graph construction. No production arithmetic or transfers change. Used after47/48 retained the throughput regression.

50. `50-admit-empty-projection-weight-views.patch`: excludes zero-byte views from partition compatibility checks, restoring parallel attention whose empty gate aliases query backing. Nonempty prefix/slice safety remains. Original GSQ measures24.163/26.384 tok/s with exact44 choices and MTP counters; requested speed is still unmet. ConcurrentPi gate is running.

51. `51-trace-active-cuda-geometry.patch`: opt-in sealed dimensions, actual block/grid and mapped input metadata. Confirms recurrent4-row/8-warp,B1 plus five host-mapped controls on the production path. No arithmetic changes.

52. `52-stage-mapped-cuda-upload-views.patch`: opt-in exact local snapshots of mutable mapped upload spans before replay. Merges only overlapping/adjacent ranges, preserves queue/session ownership with bounded retained arenas and copy/launch/peer-join ordering, and propagates root pointer overrides into parallel projections. Actual speed and Pi acceptance pending.

53. `53-format-upload-frame-device-id.patch`: sole correction of52's Rust formatting error uses Debug for DeviceId in the optional log. No execution behavior changes.

54. `54-cache-exact-m2-coefficient-blocks.patch`: selects authored packed coefficient-block reuse over fourK chunks only forM2 nondense projections with complete superblocks. Retains original partitions, decoding and F32 association; other widths/dense paths remain. Thirteen-format212992 coefficient-bit and same/mixed three-P100 output gates pass; integrated model acceptance pending.
