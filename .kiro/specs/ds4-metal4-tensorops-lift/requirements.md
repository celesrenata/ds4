# DS4 Metal 4 TensorOps / M5 Neural Accelerator Lift — Requirements

## 1. Purpose

Lift DS4's existing Apple Metal backend so that the current partial Metal 4 / TensorOps implementation becomes a production-quality, automatically selected acceleration tier for DeepSeek-V4 workloads on Apple Silicon, with particular emphasis on M5-family GPU Neural Accelerators.

This is **not** a Core ML conversion project and **not** a request to route the DS4 model through the Apple Neural Engine (ANE). In this spec, "Neural Accelerator" means the matrix/tensor acceleration hardware integrated into M5 GPU shader cores and exposed to custom Metal kernels through Metal Tensor APIs / Metal Performance Primitives (MPP) TensorOps.

The implementation must build on the TensorOps code already present in `main`, not replace it with a parallel backend.

## 2. Baseline repository facts

The implementation agent must verify these against the checked-out revision before modifying code:

- DS4 is a DeepSeek-V4-specialized C inference engine, not a generic GGUF runtime.
- The production Apple path is the whole-model Metal graph.
- Model loading must remain mmap-backed by default.
- C is preferred; Objective-C is allowed where Metal requires it; C++ must not be introduced.
- Correctness is higher priority than benchmark wins.
- `metal/dense.metal` already contains a `DS4_METAL_HAS_TENSOR` TensorOps prefill implementation using `matmul2d`, tensor views, cooperative accumulation, and 32/64/128 token variants.
- `metal/dsv4_misc.metal` already contains a TensorOps/NAX indexer-score path.
- `metal/moe.metal` contains quantized routed-MoE kernels and MXFP4 support but, at the research snapshot, has no `matmul2d` TensorOps implementation.
- DS4 already includes balanced Metal A/B benchmark harnesses for prefill and decode.
- M5 and pre-M5 Apple Silicon detection already exists in the GPU interface.
- Current code has substantial exact/bit-identical fusion work; new work must preserve that engineering discipline.

If any of those facts changed upstream, update the design before implementation rather than forcing this spec onto stale code.

---

## 3. User stories

### 3.1 M5 performance

As an M5-family DS4 user, I want compute-dense prefill work to use TensorOps so that the GPU Neural Accelerators are exercised instead of leaving dense matrix throughput on the table.

### 3.2 Existing DS4 behavior

As a DS4 user, I want the acceleration to be transparent so that model files, prompts, server APIs, sampling behavior, GGUF loading, SSD streaming, and existing command-line workflows do not need to change.

### 3.3 Portability

As an Apple Silicon user on hardware without M5 Neural Accelerators, I want DS4 to retain a correct and fast legacy Metal path, and to use TensorOps on older GPUs only where measured performance justifies it.

### 3.4 Developer diagnosability

As a DS4 developer, I want explicit capability detection, path counters, rollback controls, and reproducible A/B benchmarks so that I can prove which TensorOps paths are active and whether they are actually faster.

### 3.5 Quantized MoE acceleration

As a DS4 user running quantized DeepSeek-V4, I want routed expert matmuls to benefit from TensorOps without expanding the whole model into F16 or copying the entire GGUF into a second allocation.

---

## 4. Functional requirements

### R1 — Repository and SDK discovery

WHEN implementation begins  
THE SYSTEM SHALL read `AGENT.md`, inspect the current Metal backend, record the checked-out commit, and inventory every existing TensorOps path before editing kernels.

WHEN the local Apple toolchain is available  
THE SYSTEM SHALL record macOS version, Xcode/Command Line Tools version, macOS SDK version, Metal compiler version, supported Metal language version, and relevant GPU family capabilities.

WHEN a TensorOps feature depends on a particular SDK or OS revision  
THE SYSTEM SHALL prove availability by compiling and, where necessary, running a minimal capability probe rather than inferring support from the machine model alone.

### R2 — Terminology and backend scope

WHEN this feature is documented or exposed in diagnostics  
THE SYSTEM SHALL refer to the M5 GPU hardware as the GPU Neural Accelerator / TensorOps acceleration and SHALL NOT claim that arbitrary DS4 Metal kernels execute on the separate Apple Neural Engine.

WHEN evaluating Core ML, Core AI, or Metal ML command-encoder integration  
THE SYSTEM SHALL treat those as out-of-scope unless a measured prototype demonstrates a zero-copy, dynamically routable, GGUF-compatible advantage over inline TensorOps.

### R3 — Preserve DS4 model loading

WHEN TensorOps is enabled  
THE SYSTEM SHALL preserve DS4's mmap-backed model-loading behavior.

WHEN TensorOps consumes mapped model weights  
THE SYSTEM SHALL prefer pointer-backed / inline tensor views or existing resident expert-cache buffers over whole-model repacking.

WHEN a native quantized tensor format requires layout or alignment incompatible with DS4's on-disk layout  
THE SYSTEM SHALL fall back to custom dequantization or the legacy kernel unless the required repack is bounded, cached, and demonstrated to improve end-to-end performance after including repack cost.

THE SYSTEM SHALL NOT allocate an F16 copy of the full DeepSeek-V4 model.

### R4 — Existing dense TensorOps path

WHEN an eligible dense prefill matrix multiplication is encountered  
THE SYSTEM SHALL consider the existing `kernel_mul_mm_mpp_direct_rhs` family before adding a new dense implementation.

WHEN changing that path  
THE SYSTEM SHALL preserve or improve its direct-RHS design unless benchmark evidence proves another memory strategy faster.

WHEN token tile sizes are selected  
THE SYSTEM SHALL benchmark representative `N` tiles and choose by measured shape-specific performance rather than a single hard-coded assumption.

### R5 — Existing indexer TensorOps path

WHEN ratio-4 compressed-attention indexer score construction is eligible  
THE SYSTEM SHALL preserve the current TensorOps/NAX indexer semantics, including causal visibility, top-k compatibility, and the full required score domain.

WHEN modifying indexer TensorOps  
THE SYSTEM SHALL compare it against both the legacy tiled implementation and the current TensorOps implementation before replacing either.

### R6 — Routed-MoE TensorOps lift

WHEN profiling shows routed-MoE matrix work is a material portion of M5 prefill time  
THE SYSTEM SHALL implement and benchmark TensorOps candidates for the routed expert gate/up and down projections.

WHEN gate and up projections use the same activation tile  
THE SYSTEM SHALL evaluate a fused TensorOps design that reuses the activation and applies SwiGLU and route weighting before unnecessary device-memory round trips.

WHEN a cooperative-tensor result can be consumed by the next operation without an intermediate device-memory store  
THE SYSTEM SHALL evaluate that fusion and retain it only if correctness and end-to-end benchmarks pass.

WHEN expert routing produces irregular token counts  
THE SYSTEM SHALL evaluate expert bucketing/compaction, dynamic tile sizes, or existing DS4 routing structures before assuming that one TensorOps GEMM shape is optimal.

### R7 — Quantization compatibility research

WHEN adding a native quantized TensorOps path  
THE SYSTEM SHALL first produce a byte-level compatibility result covering data encoding, scale encoding, block size, row stride, alignment, and transpose semantics.

WHEN DS4 MXFP4 is evaluated against Metal's native MX low-bit tensor support  
THE SYSTEM SHALL explicitly verify E2M1 value encoding, E8M0 scale encoding, 32-value block semantics, scale-plane block factors, sign/zero behavior, and row alignment using the installed SDK.

WHEN Q2_K or IQ2_XXS is evaluated  
THE SYSTEM SHALL NOT assume that "2-bit" Metal tensor support is representation-compatible merely because both formats use roughly two bits per weight.

WHEN direct native representation is not compatible  
THE SYSTEM SHALL evaluate custom DS4 dequantization into a cooperative tensor before falling back to threadgroup-memory staging.

### R8 — OS capability tiers

WHEN running on an OS/toolchain with basic TensorOps but without cooperative-tensor matmul inputs  
THE SYSTEM SHALL use only TensorOps paths that are supported by that capability tier.

WHEN cooperative tensors are valid matmul inputs  
THE SYSTEM SHALL permit custom dequantization directly into cooperative tensors where this removes threadgroup round trips.

WHEN native low-bit integer or floating tensor formats are available  
THE SYSTEM SHALL enable them only for DS4 formats proven representation-compatible.

WHEN compiled on a toolchain that lacks the needed tensor headers/features  
THE SYSTEM SHALL build the legacy Metal backend successfully without source edits.

### R9 — Runtime dispatch and rollback

WHEN DS4 starts on Apple Silicon  
THE SYSTEM SHALL derive a runtime TensorOps capability record independently from the compile-time capability.

WHEN an operation is dispatched  
THE SYSTEM SHALL choose among native-quant TensorOps, custom-dequant TensorOps, existing TensorOps, and legacy Metal according to operation shape, quant type, OS capability, hardware, and benchmark-derived policy.

WHEN a TensorOps path is unsupported or rejected  
THE SYSTEM SHALL fall back without changing user-visible model behavior.

THE SYSTEM SHALL provide developer rollback/force controls following the repository's existing environment-variable conventions.

THE SYSTEM SHALL NOT require ordinary users to set an environment variable to obtain the fastest proven M5 path.

### R10 — Single-token decode

WHEN the batch dimension is one and profiling shows the operation is memory-bandwidth-bound  
THE SYSTEM SHALL retain the existing specialized matvec path unless TensorOps wins a balanced end-to-end benchmark.

WHEN a TensorOps single-token decode experiment does not produce a statistically repeatable improvement  
THE SYSTEM SHALL remove or leave it disabled rather than increasing production complexity.

### R11 — Batched verification / small-batch generation

WHEN DS4 executes speculative verification, multi-session batching, or another generation path with a matrix-shaped token batch  
THE SYSTEM SHALL benchmark TensorOps independently from single-token decode.

WHEN the batch crosses a measured TensorOps crossover point  
THE SYSTEM SHALL be able to select the TensorOps variant without affecting the single-token fast path.

### R12 — Correctness

WHEN replacing an existing kernel whose benchmark contract is bit-identical  
THE SYSTEM SHALL first attempt to preserve bit identity.

WHEN TensorOps necessarily changes floating-point reduction order  
THE SYSTEM SHALL NOT silently weaken existing tests.

WHEN bit identity is not achievable  
THE SYSTEM SHALL add an explicit numerical-equivalence test with a documented rationale, calibrated error envelope, token-selection check, and end-to-end quality regression gate before the candidate may become default.

WHEN MXFP4 native or custom TensorOps dequantization is implemented  
THE SYSTEM SHALL test all representable nibble values and a representative set of E8M0 scales against the existing DS4 decoder.

WHEN a TensorOps candidate changes selected experts, top-k compressed rows, token selection, NaN/Inf behavior, or deterministic benchmark outputs beyond the approved equivalence contract  
THE SYSTEM SHALL fail the acceptance gate.

### R13 — Benchmark methodology

WHEN comparing a candidate against control  
THE SYSTEM SHALL use the same engine, warmed sessions, alternating run order, and fresh-session discipline already used by DS4's Metal A/B harnesses.

WHEN reporting performance  
THE SYSTEM SHALL separate:
- kernel microbenchmark time,
- layer/stage time,
- prefill tokens/s,
- time to first generated token,
- decode tokens/s,
- long-context slope,
- model-load/repack overhead,
- memory footprint.

WHEN an optimization is retained  
THE SYSTEM SHALL show repeatable benefit across more than one prompt/batch shape relevant to the target operation.

### R14 — Profiling

WHEN performance work starts  
THE SYSTEM SHALL capture a Metal System Trace / GPU profile or equivalent counters for the baseline and candidate.

THE SYSTEM SHALL use profiling to distinguish compute-bound, bandwidth-bound, launch-bound, cache-bound, and synchronization-bound stages.

WHEN TensorOps is expected to use M5 Neural Accelerators  
THE SYSTEM SHALL gather the best available evidence that the TensorOps operation is executing through the intended hardware path.

### R15 — Diagnostics

WHEN developer diagnostics are enabled  
THE SYSTEM SHALL report:
- TensorOps compiled: yes/no,
- TensorOps runtime available: yes/no,
- M5-family detected: yes/no,
- native low-bit capability tier,
- dense TensorOps dispatch count,
- indexer TensorOps dispatch count,
- MoE TensorOps dispatch count,
- TensorOps fallback counts grouped by reason.

WHEN diagnostics are disabled  
THE SYSTEM SHALL avoid adding per-token logging or hot-path synchronization.

### R16 — Performance acceptance

WHEN the feature is enabled by default on the target M5 system  
THE SYSTEM SHALL have no statistically significant regression greater than 1% in the established end-to-end decode benchmark.

WHEN TensorOps is used for a production prefill path  
THE SYSTEM SHALL demonstrate a repeatable improvement over the immediately preceding production path for the shapes it claims.

THE PROJECT SHOULD TARGET at least a 15% end-to-end M5 prefill improvement on representative DeepSeek-V4 prompts after all enabled TensorOps work is combined.

THE PROJECT SHOULD TARGET at least a 1.25x speedup in the individual compute-dense kernels migrated to TensorOps, unless end-to-end profiling shows the kernel is no longer material.

A production TensorOps path that improves a microbenchmark but regresses total prefill, TTFT, memory pressure, or load time SHALL NOT become default.

### R17 — Non-M5 Apple Silicon

WHEN running on M1-M4 Apple Silicon  
THE SYSTEM SHALL preserve the current default performance within normal benchmark noise.

WHEN TensorOps fallback shaders outperform the legacy path on a pre-M5 device  
THE SYSTEM MAY enable the TensorOps path only after the same correctness and A/B gates pass for that device class.

### R18 — Non-Metal backends

WHEN building CPU, CUDA, or ROCm configurations  
THE SYSTEM SHALL compile and behave as before unless a shared interface change is strictly required.

WHEN a shared interface changes  
THE SYSTEM SHALL provide stubs or compatible implementation for non-Metal builds.

### R19 — SSD streaming and resident expert cache

WHEN SSD streaming is active  
THE SYSTEM SHALL either consume the existing mapped/resident expert representation directly or explicitly fall back to the established streaming kernel.

WHEN the resident expert cache is active  
THE SYSTEM SHALL evaluate whether cached expert buffers can be exposed as inline/native tensor inputs without extra copies.

THE SYSTEM SHALL NOT make TensorOps support contingent on loading all experts into RAM.

### R20 — Failure behavior

WHEN TensorOps pipeline creation, capability probing, or runtime dispatch fails  
THE SYSTEM SHALL emit a concise diagnostic in developer/debug mode and continue through the legacy Metal path when safe.

WHEN a required TensorOps path would return incorrect results  
THE SYSTEM SHALL fail closed to legacy Metal rather than continue with approximate or partially initialized output.

---

## 5. Explicit non-goals

- Rewriting DS4 around Core ML, Core AI, MLX, MPSGraph, or a generic graph compiler.
- Converting the full GGUF to another model format.
- Replacing all Metal kernels with TensorOps.
- Forcing TensorOps for single-token decode.
- Removing existing exact fast paths solely to make the architecture look uniform.
- Adding C++ to the core project.
- Reworking CUDA/ROCm for feature parity in the same change.
- Treating Apple Neural Engine and M5 GPU Neural Accelerators as interchangeable terms.
- Assuming every low-bit GGUF quant maps to a native Metal low-bit tensor.
- Keeping benchmark experiments in production if they do not win.

---

## 6. Definition of done

This spec is complete when:

1. Existing TensorOps dense and indexer paths are inventoried, tested, and incorporated into one coherent runtime capability/dispatch policy.
2. The primary routed-MoE prefill bottleneck has at least one production-quality TensorOps implementation on M5, or a benchmark-backed written finding demonstrates why the existing kernel remains faster.
3. MXFP4 native-tensor compatibility on macOS 27 is conclusively proven or disproven.
4. Custom-dequant-to-cooperative-tensor feasibility is proven for at least one non-native DS4 quant format or rejected with benchmark evidence.
5. Single-token decode remains regression-free.
6. Batched verify/small-batch TensorOps crossover is measured.
7. Existing `make test` and relevant Metal tests pass.
8. New kernel-level correctness tests pass.
9. Balanced A/B benchmarks and end-to-end `ds4-bench` results are saved.
10. The fastest proven M5 paths auto-enable without user tuning and retain rollback controls for development.
