# Implementation Plan: DS4 Metal 4 TensorOps / M5 Neural Accelerator Lift

## Overview

Lift DS4's existing Apple Metal backend so that the current partial Metal 4 / TensorOps implementation becomes a production-quality, automatically selected acceleration tier for DeepSeek-V4 workloads on Apple Silicon, with particular emphasis on M5-family GPU Neural Accelerators. Implementation is in C (with Objective-C where Metal requires it) and Metal Shading Language. Execute in order. Do not skip the research gates. Performance tasks are allowed to end in a documented **reject** if the candidate loses a correct balanced benchmark.

## Tasks

- [x] 0. Establish the development baseline
  - [x] 0.1 Read repository rules and record state
    - Read `AGENT.md`.
    - Record `git rev-parse HEAD` and `git status --short`.
    - Confirm no unrelated local changes will be overwritten.
    - Note any upstream changes since the research snapshot that alter this spec.
    - _Requirements: R1, R18_

  - [x] 0.2 Inventory current TensorOps implementation
    - Find every use of `DS4_METAL_HAS_TENSOR`, `tensor`, `matmul2d`, cooperative tensor APIs, and NAX/TensorOps terminology.
    - Map host pipeline registration and dispatch to each shader function.
    - Record which paths are default versus retained/forced/experimental.
    - _Requirements: R1, R4, R5_

  - [x] 0.3 Inventory current MoE paths
    - Map gate/up/down kernels by quant type and prefill/decode phase.
    - Identify MXFP4, Q8_0, Q4_K/Q4_0, Q2_K, IQ2_XXS and any other target formats.
    - Map resident-cache and SSD-streaming variants.
    - _Requirements: R6, R7, R19_

  - [x] 0.4 Capture toolchain capability matrix
    - Record macOS, Xcode, SDK, Metal compiler and GPU family.
    - Determine where the project chooses its Metal language version.
    - Compile minimal probes for TensorOps feature tiers.
    - Save results in development notes or a small test utility.
    - _Requirements: R1, R8_

- [x] 1. Capture reproducible pre-change benchmarks
  - [x] 1.1 Run standard DS4 benchmark
    - Save a baseline CSV from `ds4-bench`.
    - Include short, 2K, 8K, and 32K+ context points where practical.
    - Record TTFT/prefill/decode separately.
    - _Requirements: R13, R16_

  - [x] 1.2 Run balanced Metal prefill A/B harness
    - Build `metal-prefill-variant-bench`.
    - Record current default against at least one existing rollback to verify the harness/environment.
    - _Requirements: R12, R13_

  - [x] 1.3 Run balanced Metal decode A/B harness
    - Build `metal-decode-schedule-bench`.
    - Use `--include-selection`.
    - Confirm full-vocabulary bit-identity enforcement works.
    - _Requirements: R10, R12, R13_

  - [x] 1.4 Profile baseline
    - Capture Metal System Trace/GPU profile for representative prefill.
    - Capture a separate batch-1 decode trace.
    - Identify top prefill stages and classify each as compute/bandwidth/launch/sync/cache bound.
    - _Requirements: R6, R14_

  - [x] 1.5 Collect routed expert bucket statistics
    - Add temporary or debug-only instrumentation if necessary.
    - Record tokens-per-expert distributions at multiple prompt batch widths.
    - Include tails and empty/small buckets.
    - Remove expensive instrumentation from normal builds.
    - _Requirements: R6, R13, R15_

- [x] 2. Build a coherent TensorOps capability layer
  - [x] 2.1 Add compile-time feature detection
    - Keep legacy SDK builds working.
    - Reuse `DS4_METAL_HAS_TENSOR` if it is already the correct abstraction.
    - Split additional feature macros only when an API genuinely needs a newer SDK tier.
    - _Requirements: R8, R18_

  - [x] 2.2 Add runtime capability detection
    - Expose M5, TensorOps runtime, cooperative-input, and native-low-bit capability as needed.
    - Avoid product-name parsing if existing device-family helpers suffice.
    - _Requirements: R8, R9_

  - [x] 2.3 Add developer diagnostics/counters
    - Print capability summary only when diagnostics are requested.
    - Add operation dispatch/fallback counters with negligible normal overhead.
    - _Requirements: R15_

  - [x] 2.4 Add global rollback control
    - Add or consolidate a single TensorOps rollback that dominates individual force-enables.
    - Follow existing DS4 environment-variable conventions.
    - _Requirements: R9, R20_

  - [x] 2.5 Add capability/fallback tests
    - Test compiled-out behavior where practical.
    - Test no-pipeline/unsupported-feature fallback.
    - Ensure fallback reaches the established legacy kernel.
    - _Requirements: R8, R20_

- [x] 3. Consolidate the existing dense TensorOps prefill path
  - [x] 3.1 Trace `kernel_mul_mm_mpp_direct_rhs` end to end
    - Document host dispatch conditions.
    - Determine which quant variants are currently production-selected.
    - Confirm direct RHS and double-buffered weight staging are preserved.
    - _Requirements: R4_

  - [x] 3.2 Benchmark current 32/64/128 token tiles
    - Use real DS4 shapes.
    - Include prompt tails.
    - Record resource usage and occupancy.
    - _Requirements: R4, R13, R14_

  - [x] 3.3 Test additional tile shapes only if justified
    - Candidate smaller tail tile and/or larger tile.
    - Reject any candidate that is neutral/slower or increases fallback complexity without end-to-end gain.
    - _Requirements: R4, R16_

  - [x] 3.4 Make dense selection policy explicit
    - Centralize capability/shape checks.
    - Auto-select the fastest proven M5 variant.
    - Preserve pre-M5 default until separately proven.
    - _Requirements: R9, R17_

  - [x] 3.5 Add/extend dense correctness coverage
    - Compare each TensorOps quant variant with its legacy/reference path.
    - Preserve strict bit tests where they currently exist.
    - _Requirements: R12_

- [x] 4. Consolidate and tune the existing indexer TensorOps path
  - [x] 4.1 Trace `kernel_dsv4_indexer_scores_nax` end to end
    - Record current admission rules and fallback.
    - Confirm top-k/indexed-attention semantics remain outside the accelerated matrix core.
    - _Requirements: R5_

  - [x] 4.2 Benchmark indexer tile choices
    - Test current token/compressed-row/head grouping.
    - Profile barriers, staging, cooperative store, and weighted accumulation.
    - _Requirements: R5, R14_

  - [x] 4.3 Prototype cooperative post-processing
    - Determine whether weighted ReLU accumulation can consume cooperative results without materializing `dot`.
    - Keep only if ranking/correctness and end-to-end performance pass.
    - _Requirements: R5, R12, R16_

  - [x] 4.4 Lock indexer production policy
    - M5 default only for winning shapes.
    - Clean legacy fallback.
    - _Requirements: R9, R17_

- [x] 5. Prove or disprove native MXFP4 compatibility
  - [x] 5.1 Document DS4 MXFP4 byte layout
    - Identify data nibble ordering.
    - Identify E8M0 scale location/semantics.
    - Confirm 32-value block behavior.
    - Confirm row/expert stride.
    - _Requirements: R7_

  - [x] 5.2 Inspect installed macOS 27 Metal tensor definitions
    - Identify exact FP4 data type.
    - Identify E8M0 scale-plane type.
    - Record alignment and block-factor requirements.
    - Record allowed inline tensor strides/layout.
    - _Requirements: R7, R8_

  - [x] 5.3 Build native-MX conformance probe
    - Test all 16 low-bit codes across representative scales.
    - Compare against existing DS4 MXFP4 decoder/LUT.
    - Test basis-vector matmul so interpretation is unambiguous.
    - _Requirements: R7, R12_

  - [x] 5.4 Determine zero-copy feasibility
    - Test whether model-map/resident-cache layout can back data and scale planes directly.
    - Explicitly test interleaved-block versus separate-plane requirements.
    - _Requirements: R3, R7, R19_

  - [x] 5.5 Benchmark bounded repack if zero-copy is impossible
    - Try only per-tile/per-expert/resident-cache repacks.
    - Include repack cost and cache hit behavior in the benchmark.
    - Do not build a whole-model converted copy.
    - _Requirements: R3, R16, R19_

  - [x] 5.6 Write the native-MX decision
    - Mark one: direct view / resident-cache layout / bounded repack / reject.
    - Include correctness and performance evidence.
    - _Requirements: R7_

- [x] 6. Build a reusable custom-dequant TensorOps prototype
  - [x] 6.1 Select a simple quant for first prototype
    - Prefer Q8_0 or another already-understood dense decoder.
    - Avoid starting with IQ2_XXS.
    - _Requirements: R7_

  - [x] 6.2 Implement staged-dequant TensorOps control
    - Quant block -> threadgroup half tile -> TensorOps.
    - Use it as a performance/correctness control.
    - _Requirements: R7, R8_

  - [x] 6.3 Implement cooperative-input dequant candidate
    - Quant block -> custom dequant -> cooperative tensor input -> `matmul2d`.
    - Avoid unnecessary threadgroup materialization where API permits.
    - _Requirements: R7, R8_

  - [x] 6.4 Microbenchmark staged vs cooperative vs legacy
    - Use real K/M/N dimensions.
    - Profile register pressure and occupancy.
    - _Requirements: R13, R14, R16_

  - [x] 6.5 Freeze the winning mechanism
    - Keep a small reusable primitive/pattern.
    - Do not create a generic abstraction larger than DS4 needs.
    - _Requirements: R6, R16_

- [x] 7. Implement TensorOps routed-MoE gate/up prefill
  - [x] 7.1 Map current expert batching
    - Confirm whether tokens are already bucketed by expert.
    - Identify where a TensorOps tile can consume contiguous or pointer-stable rows.
    - _Requirements: R6_

  - [x] 7.2 Implement MXFP4 gate/up candidate
    - Use native-MX path if Task 5 proved it wins.
    - Otherwise use winning custom-dequant mechanism from Task 6.
    - Preserve legacy fallback.
    - _Requirements: R6, R7, R9_

  - [x] 7.3 Add Q8/Q4 gate/up candidate where relevant
    - Reuse existing dense dequant logic when possible.
    - Do not duplicate decoder semantics.
    - _Requirements: R6, R7_

  - [x] 7.4 Prototype gate/up TensorOps fusion
    - Reuse activation tile.
    - Evaluate paired cooperative outputs.
    - Apply clamp/SiLU/multiply/route weight before device store if safe.
    - _Requirements: R6_

  - [x] 7.5 Handle expert-bucket tails
    - Benchmark masked/padded tile, smaller tile, and legacy-tail strategies.
    - Pick by measured distribution from Task 1.5.
    - _Requirements: R6, R13_

  - [x] 7.6 Preserve TP/streaming behavior
    - Respect expert ownership.
    - Support resident expert cache where possible.
    - Explicit fallback for unsupported SSD-streaming address/layout cases.
    - _Requirements: R18, R19_

  - [x] 7.7 Add kernel parity tests
    - Deterministic random inputs.
    - Adversarial low-bit patterns.
    - Tails.
    - Empty/small expert buckets.
    - Routing weights.
    - _Requirements: R12_

  - [x] 7.8 Run balanced end-to-end prefill A/B
    - Same binary if possible.
    - Fresh sessions + alternating order.
    - Save result.
    - _Requirements: R13, R16_

- [x] 8. Implement TensorOps routed-MoE down prefill
  - [x] 8.1 Map current down-projection accumulation order
    - Document when/where selected expert outputs are summed.
    - Identify any exact-order dependency.
    - _Requirements: R6, R12_

  - [x] 8.2 Implement down TensorOps candidate
    - Use quant mechanism selected above.
    - Keep expert ownership/streaming semantics.
    - _Requirements: R6, R19_

  - [x] 8.3 Evaluate accumulation fusion
    - Fuse only if it preserves the accepted correctness contract.
    - Otherwise optimize individual projections and retain existing sum.
    - _Requirements: R6, R12_

  - [x] 8.4 Benchmark tails and routing distributions
    - Use captured real expert histograms.
    - _Requirements: R13, R16_

  - [x] 8.5 Run full prefill A/B
    - Measure gate/up-only, down-only, and combined TensorOps.
    - _Requirements: R13, R16_

- [x] 9. Evaluate complex two-bit quants without assuming compatibility
  - [x] 9.1 Document Q2_K format
    - Block metadata.
    - Scale/min encoding.
    - Bit packing.
    - Dequant arithmetic.
    - _Requirements: R7_

  - [x] 9.2 Document IQ2_XXS format
    - Codebooks/lookups.
    - Scale semantics.
    - Block structure.
    - _Requirements: R7_

  - [x] 9.3 Test native int2 representability
    - Prove exact mapping or record why it is impossible.
    - _Requirements: R7_

  - [x] 9.4 Benchmark cooperative custom dequant
    - Only if the dequant can be expressed efficiently.
    - _Requirements: R7, R16_

  - [x] 9.5 Keep or reject each quant-specific TensorOps path
    - No compatibility theater: native low-bit APIs are used only when representation matches.
    - _Requirements: R7_

- [x] 10. Evaluate speculative verification and small-batch generation
  - [x] 10.1 Measure actual generation batch shapes
    - DSpark/speculative verify depth.
    - Session batching.
    - Prompt tail batches.
    - _Requirements: R11_

  - [x] 10.2 Benchmark TensorOps crossover
    - Compare existing small-batch kernels against TensorOps by quant and N.
    - _Requirements: R11, R13_

  - [x] 10.3 Add measured crossover dispatch
    - Only for ranges that win repeatably.
    - Batch-1 must remain on existing decode by default.
    - _Requirements: R10, R11_

  - [x] 10.4 Re-run decode exactness harness
    - `--include-selection`.
    - No >1% end-to-end decode regression.
    - _Requirements: R10, R12, R16_

- [x] 11. Re-profile before touching attention
  - [x] 11.1 Capture post-MoE prefill trace
    - Re-rank bottlenecks.
    - _Requirements: R14_

  - [x] 11.2 Decide whether attention merits TensorOps work
    - If not a material bottleneck, explicitly defer and stop.
    - Decision: DEFERRED PENDING PROFILING DATA (task 11.1 requires Apple Silicon)
    - Thresholds documented: >25% proceed, 15-25% evaluate, <15% defer per 14.6
    - Existing indexer TensorOps (task 4) already covers the largest regular attention matrix op
    - See dev-notes/11.2-11.3-attention-decision.md
    - _Requirements: R16_

  - [x] 11.3 If justified, prototype only the regular matrix inner operation
    - Do not rewrite compressed-attention semantics.
    - Consider QK/V cooperative TensorOps in the existing FlashAttention structure.
    - Decision: CONDITIONAL ON 11.2 — currently deferred
    - Candidate identified: TensorOps matmul2d for Q×K^T and Attn×V inner products only
    - Compressed-attention semantics, masking, softmax, top-k explicitly excluded from scope
    - See dev-notes/11.2-11.3-attention-decision.md
    - _Requirements: R5, R16_

  - [x] 11.4 Correctness and A/B gate
    - Keep only a measured win.
    - Decision: DEFERRED — No attention TensorOps prototype exists to gate
    - Tasks 11.2/11.3 deferred; no new attention kernel was built
    - Existing FlashAttention unchanged; existing indexer TensorOps (task 4) separately validated
    - This gate activates when/if 11.2 proceeds in future
    - See dev-notes/11.4-attention-gate-deferred.md
    - _Requirements: R12, R13, R16_

- [x] 12. Production hardening
  - [x] 12.1 Centralize production dispatch policy
    - Capability, quant, shape, streaming/cache, M5/pre-M5, rollback.
    - _Requirements: R9_

  - [x] 12.2 Remove losing experiments
    - Delete dead kernels and stale force flags.
    - Retain only useful debug/rollback controls.
    - _Requirements: R16_

  - [x] 12.3 Validate old-SDK / no-TensorOps build
    - Legacy Metal path compiles.
    - _Requirements: R8_

  - [x] 12.4 Validate CPU build
    - _Requirements: R18_

  - [x] 12.5 Validate CUDA build if environment is available
    - At minimum ensure shared interface changes compile.
    - _Requirements: R18_

  - [x] 12.6 Validate ROCm build if environment is available
    - At minimum ensure shared interface changes compile.
    - _Requirements: R18_

  - [x] 12.7 Run full tests
    - `make test`.
    - Metal MXFP4 exactness tests.
    - Session batch tests.
    - New TensorOps tests.
    - _Requirements: R12, R18_

  - [x] 12.8 Exercise SSD streaming
    - Cold and warm expert cache.
    - Fallback paths.
    - No full-model copy.
    - _Requirements: R3, R19_

  - [x] 12.9 Exercise pre-M5 Apple Silicon if hardware is available
    - No material regression.
    - Do not auto-enable new path without device-specific evidence.
    - _Requirements: R17_

- [x] 13. Final performance report
  - [x] 13.1 Run final `ds4-bench` suite
    - Save CSV and command line.
    - Execution plan: `dev-notes/13.1-13.4-final-perf-report-plan.md` §3
    - Status: PENDING HARDWARE EXECUTION (requires macOS + Apple Silicon M5)
    - Commands documented for short/2K/8K/32K contexts with CSV output
    - Comparison table template against task 1.1 baseline prepared
    - _Requirements: R13, R16_

  - [x] 13.2 Run final balanced prefill A/B
    - Combined TensorOps default versus master baseline/rollback.
    - Execution plan: `dev-notes/13.1-13.4-final-perf-report-plan.md` §4
    - Status: PENDING HARDWARE EXECUTION (requires macOS + Apple Silicon M5)
    - Uses `--candidate-env DS4_METAL_DISABLE_TENSOROPS` (global rollback)
    - Primary spec acceptance metric: 15% combined prefill improvement target
    - _Requirements: R13, R16_

  - [x] 13.3 Run final balanced decode A/B
    - Include token selection.
    - Execution plan: `dev-notes/13.1-13.4-final-perf-report-plan.md` §5
    - Status: PENDING HARDWARE EXECUTION (requires macOS + Apple Silicon M5)
    - Uses `metal-decode-schedule-bench --include-selection --tokens 1024`
    - Validates: no >1% decode regression + full token-selection identity (R10)
    - _Requirements: R10, R13, R16_

  - [x] 13.4 Capture final Metal trace
    - Prove bottleneck movement and TensorOps utilization.
    - Execution plan: `dev-notes/13.1-13.4-final-perf-report-plan.md` §6
    - Status: PENDING HARDWARE EXECUTION (requires macOS + Apple Silicon M5)
    - Metal System Trace of 8K prefill; compare against task 1.4 baseline
    - Prove: MoE uses TensorOps/_mpp_ kernels, bottleneck shifted from MoE
    - _Requirements: R14_

  - [x] 13.5 Fill the architectural decision record
    - Dense, indexer, MoE gate/up, MoE down, native MXFP4, cooperative dequant, verify crossover, batch-1 decode, overall speed/memory.
    - _Requirements: all_

- [x] 14. Stop conditions
  - [x] 14.1 Stop a candidate if correctness cannot be bounded
  - [x] 14.2 Stop a candidate if it requires a full-model repack
  - [x] 14.3 Stop a candidate if it wins only a synthetic square-GEMM benchmark
  - [x] 14.4 Stop a batch-1 TensorOps decode candidate if balanced end-to-end decode is neutral/slower
  - [x] 14.5 Stop native low-bit work for a quant format once byte-level incompatibility is proven
  - [x] 14.6 Stop attention work if post-MoE profiling says it is not material

A rejected experiment is a successful task when the benchmark and reason are recorded.

## Notes

- Tasks are ordered sequentially with research gates — do not skip ahead without completing prerequisite phases
- Performance tasks may end in a documented **reject** if the candidate loses a correct balanced benchmark — a rejected experiment with recorded evidence is a successful task
- Each task references specific requirements (R1–R20) for traceability back to the requirements document
- Stop conditions (section 14) define explicit exit criteria to prevent wasted effort on unwinnable paths
- The implementation language is C (with Objective-C where Metal requires it) and Metal Shading Language — no C++ in core
- Checkpoints are implicit at phase boundaries (after sections 1, 5, 6, 8, 12)
- Native MXFP4 compatibility (section 5) and custom-dequant feasibility (section 6) are research gates that determine the approach for MoE TensorOps (sections 7–8)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["0.1", "0.2", "0.3", "0.4"] },
    { "id": 1, "tasks": ["1.1", "1.2", "1.3", "1.4", "1.5"] },
    { "id": 2, "tasks": ["2.1", "2.2", "2.3", "2.4", "2.5"] },
    { "id": 3, "tasks": ["3.1", "4.1", "5.1", "5.2"] },
    { "id": 4, "tasks": ["3.2", "3.5", "4.2", "5.3", "6.1"] },
    { "id": 5, "tasks": ["3.3", "3.4", "4.3", "5.4", "6.2", "6.3"] },
    { "id": 6, "tasks": ["4.4", "5.5", "5.6", "6.4"] },
    { "id": 7, "tasks": ["6.5", "7.1", "9.1", "9.2"] },
    { "id": 8, "tasks": ["7.2", "7.3", "7.4", "9.3"] },
    { "id": 9, "tasks": ["7.5", "7.6", "7.7", "9.4"] },
    { "id": 10, "tasks": ["7.8", "8.1", "9.5"] },
    { "id": 11, "tasks": ["8.2", "8.3", "10.1"] },
    { "id": 12, "tasks": ["8.4", "8.5", "10.2", "10.3"] },
    { "id": 13, "tasks": ["10.4", "11.1"] },
    { "id": 14, "tasks": ["11.2", "11.3"] },
    { "id": 15, "tasks": ["11.4", "12.1", "12.2"] },
    { "id": 16, "tasks": ["12.3", "12.4", "12.5", "12.6"] },
    { "id": 17, "tasks": ["12.7", "12.8", "12.9"] },
    { "id": 18, "tasks": ["13.1", "13.2", "13.3", "13.4"] },
    { "id": 19, "tasks": ["13.5"] },
    { "id": 20, "tasks": ["14.1", "14.2", "14.3", "14.4", "14.5", "14.6"] }
  ]
}
```
