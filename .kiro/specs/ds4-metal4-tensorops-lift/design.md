# DS4 Metal 4 TensorOps / M5 Neural Accelerator Lift — Technical Design

> Recommended Kiro workflow: **Design-First Feature Spec**. This project is constrained by an existing low-level architecture, strict correctness rules, memory-layout constraints, and performance targets. Do not regenerate this as a greenfield requirements-first design.

## 1. Executive summary

DS4 already contains meaningful Metal 4 / TensorOps work. The correct project is therefore not "add ANE support" and not "port DS4 to a new framework." The project is to turn a partially integrated TensorOps acceleration layer into a coherent production tier, then extend it to the most important remaining matrix-heavy region: routed MoE prefill.

The central design principle is:

**Keep the model in DS4's existing representation, move only the mathematically appropriate operations onto TensorOps, fuse around cooperative tensors where it saves memory traffic, and let benchmark evidence decide every production dispatch.**

The highest-confidence opportunities are:

1. Stabilize/generalize the existing dense prefill TensorOps path.
2. Stabilize/generalize the existing ratio-4 indexer TensorOps path.
3. Add TensorOps to routed-MoE prefill, prioritizing MXFP4 and Q8/Q4-compatible shapes.
4. On macOS 27, test zero-copy or near-zero-copy native MXFP4 tensor views using low-bit floating data plus E8M0 scale planes.
5. On macOS 26.3+, test custom dequantization into cooperative tensors for DS4 quants that do not map natively.
6. Treat speculative verification and other batched generation separately from batch-1 decode.
7. Leave batch-1 decode on today's hand-tuned matvec/fused kernels unless it wins end-to-end.

---

## 2. Research snapshot and why it changes the original issue

### 2.1 DS4 current state

At the research snapshot (`main` around commit `84cc882352757baf628a1776badf7cc54d584e28`), the repository already contains:

- `metal/dense.metal`
  - Manual generation matvec kernels.
  - Fused decode projections.
  - Small-batch prompt kernels.
  - A `DS4_METAL_HAS_TENSOR` path named around `kernel_mul_mm_mpp_direct_rhs`.
  - TensorOps `matmul2d`.
  - A cooperative destination tensor.
  - Direct RHS reads from device memory.
  - Double-buffered weight-tile staging.
  - F16, Q8_0, Q4_0, and Q4_K variants.
  - token-tile variants at 32, 64, and 128.

- `metal/dsv4_misc.metal`
  - Existing simdgroup-matrix indexer scoring.
  - A retained `DS4_METAL_HAS_TENSOR` / NAX indexer score path.
  - TensorOps used on the regular 128-deep token x compressed-row score tile while preserving the later top-k/indexed-attention semantics.

- `metal/moe.metal`
  - Routed MoE kernels.
  - Q2_K, Q4_K, Q5, Q6, Q8-class and IQ2_XXS-related paths.
  - MXFP4 support with a 32-value block constant.
  - MXFP4 E2M1 values and E8M0 scale conversion/LUT machinery.
  - At the snapshot, no `matmul2d` match in this file.

- `speed-bench/README.md`
  - a balanced Metal decode schedule benchmark with bit-identical full-vocabulary logit checking;
  - a balanced Metal prefill variant benchmark using fresh sessions, warmup, ABBA/BAAB ordering, and bit-identical final logits.

- `ds4_gpu.h`
  - M5 vs pre-M5 Apple Silicon detection.
  - tensor-resident command lifetime.
  - mmap/model-map APIs.
  - resident/streaming expert cache support.
  - test flags and multiple Metal-specific controls.

This means issue #14's original proposal is partially implemented upstream. New work must extend today's architecture instead of recreating an older prototype.

### 2.2 Apple platform facts that matter

Use Apple primary documentation and the installed SDK as the source of truth.

Current Apple guidance says:

- TensorOps is the low-level API for custom ML matrix operations inside Metal shaders.
- On M5, TensorOps can use the Neural Accelerator integrated into each GPU shader core.
- The same TensorOps source remains portable to older Apple GPUs, where optimized shader implementations are used.
- LLM prefill is a canonical dense compute-bound target for the M5 Neural Accelerators.
- LLM inference overall can still be memory-bandwidth-bound, making quantization and data movement first-class concerns.
- macOS 26.3 added cooperative tensors as inputs to matmul, enabling custom dequantization without a mandatory threadgroup-memory round trip.
- macOS 26.4 added native 4-bit and 8-bit integer tensor types.
- macOS 27 expands low-bit support to floating formats and 2-bit integers and adds multi-plane scale support including FP8 E8M0 block scaling / MX formats.
- Metal tensors can be created inline from existing device pointers, which is essential for DS4's mmap/resident-buffer architecture.
- Cooperative tensors permit post-matmul work while values remain distributed in fast per-thread state, reducing device-memory materialization.
- Tile shape must be tuned for the real matrix dimensions.
- Metal System Trace / GPU profiling should determine whether a stage is compute, bandwidth, launch, or synchronization limited.

### 2.3 Critical terminology

There are two separate Apple concepts:

- **Apple Neural Engine (ANE):** a separate inference engine typically reached through higher-level ML frameworks / schedulers.
- **M5 GPU Neural Accelerators:** matrix acceleration blocks inside GPU shader cores, directly usable by custom Metal TensorOps.

This project targets the second.

Trying to move DS4 wholesale into Core ML or a Metal ML command encoder would give up much of DS4's custom quantization, expert routing, mmap layout, low-level fusion, and dynamic graph control. It is not the first-choice architecture.

---

## 3. Architectural constraints

### 3.1 Preserve memory representation

The default path must not:

- decompress the whole model,
- create a second full-size weight copy,
- require model conversion,
- require a custom GGUF variant solely for Metal,
- defeat SSD streaming.

TensorOps inputs should come from, in priority order:

1. existing model-map/model-cache GPU-accessible storage through inline tensor views;
2. existing resident expert cache buffers;
3. a bounded per-expert/per-tile repack cache if and only if it pays for itself;
4. threadgroup/cooperative temporary tiles;
5. legacy kernel fallback.

### 3.2 Preserve backend shape

Do not build a second graph engine. Add TensorOps as an implementation choice underneath current DS4 GPU operations.

Conceptually:

```text
DS4 graph operation
    |
    +-- capability + shape + quant dispatch
            |
            +-- native quantized TensorOps
            |
            +-- custom-dequant TensorOps
            |
            +-- existing TensorOps
            |
            +-- existing legacy Metal kernel
```

### 3.3 Compile-time versus runtime capability

Keep two layers distinct.

Compile-time capability answers:

- Does the SDK provide `<metal_tensor>` / MPP TensorOps?
- Does this Metal compiler accept the required language/API features?
- Were the TensorOps kernels compiled into the library?

Runtime capability answers:

- What Apple GPU family is active?
- Is this M5-class hardware?
- What OS/API feature tier is available?
- Can the required pipeline be created?
- Does this operation's shape/quant satisfy alignment and layout requirements?

Do not use `#ifdef` as a substitute for runtime dispatch.

---

## 4. Phase 0 — mandatory research and baseline

No kernel rewrite begins until this phase is complete.

### 4.1 Repository inventory

Record:

```text
git rev-parse HEAD
git status --short
```

Read:

- `AGENT.md`
- issue #14
- `metal/dense.metal`
- `metal/moe.metal`
- `metal/dsv4_misc.metal`
- `metal/flash_attn.metal`
- relevant host-side pipeline registration/dispatch code
- `ds4_gpu.h`
- `speed-bench/README.md`
- existing Metal exactness tests
- recent M5/MXFP4-related commits

Produce a table:

| Operation | Quant | Phase | Current kernel | TensorOps already? | Dominant shape | Default/experimental |
|---|---|---|---|---|---|---|
| dense projection | F16 | prefill | ... | yes | MxKxN | ... |
| dense projection | Q8_0 | prefill | ... | yes | ... | ... |
| indexer scoring | F16/F32 staging | prefill | ... | yes | token x comp x 128 | ... |
| routed gate/up | MXFP4 | prefill | ... | no | routed | ... |
| routed down | MXFP4 | prefill | ... | no | routed | ... |
| routed gate/up | Q2_K/IQ2/etc | prefill/decode | ... | no | ... | ... |
| single-token dense | Q8/F16 | decode | manual | no | matvec | default |
| verify block | ... | generation | ... | ? | small matrix | ... |

The table must use actual checked-out names, not names copied blindly from this design.

### 4.2 Toolchain matrix

Record:

- hardware model and GPU core count;
- `system_profiler SPHardwareDataType`;
- macOS version;
- Xcode version;
- active developer directory;
- macOS SDK version;
- Metal compiler version;
- Metal language standard used by the project;
- availability of relevant tensor data-type symbols in the active SDK.

Create tiny compile probes for:

1. basic `tensor` / `matmul2d`;
2. cooperative tensor as matmul input;
3. 4/8-bit integer tensor types;
4. low-bit float and E8M0 scale-plane declarations;
5. candidate MXFP4 tensor-inline declaration.

Delete throwaway probes or move lasting ones into a deliberate capability test.

### 4.3 Baseline performance

Before edits, save:

- `ds4-bench` CSV across context sizes;
- short prompt prefill;
- 8K-ish prefill;
- 32K/64K if practical;
- 128 decode tokens;
- `metal-prefill-variant-bench`;
- `metal-decode-schedule-bench --include-selection`;
- one Metal System Trace for prefill;
- one trace for decode.

Capture:
- prompt t/s;
- generation t/s;
- time to first generated token;
- peak/steady memory;
- model load time;
- per-stage profile if existing profile flags support it.

Do not compare a cold control against a warm candidate.

---

## 5. Capability model

Add the smallest capability abstraction consistent with the current backend style.

Conceptually:

```c
typedef struct {
    bool tensorops_compiled;
    bool tensorops_runtime;
    bool m5_neural_accelerator;
    bool coop_tensor_input;
    bool native_int4_int8;
    bool native_lowbit_float;
    bool native_e8m0_scale_plane;
    bool native_mx_candidate;
} ds4_metal_tensorops_caps;
```

This is pseudocode, not a required ABI.

### 5.1 Detection rules

- `tensorops_compiled`: set from build/compiled feature availability.
- `tensorops_runtime`: require successful pipeline/function availability.
- `m5_neural_accelerator`: use existing DS4 M5 detection, not product-name string parsing if avoidable.
- OS/API tiers: use availability checks supported by the project's Objective-C/Metal setup.
- Native quant flags: require both compiled symbols and runtime support.

### 5.2 Diagnostics

Provide one debug dump, e.g.:

```text
Metal TensorOps:
  compiled                  yes
  runtime                   yes
  M5 neural accelerator     yes
  cooperative input         yes
  native int4/int8          yes
  native low-bit float      yes
  E8M0 scale plane          yes
```

Also maintain low-overhead counters:

```text
dense_tensorops_dispatches
indexer_tensorops_dispatches
moe_gate_up_tensorops_dispatches
moe_down_tensorops_dispatches
verify_tensorops_dispatches

fallback_shape
fallback_quant
fallback_alignment
fallback_os
fallback_pipeline
fallback_streaming
```

Counters should be disabled or nearly free in normal runs.

---

## 6. Dense prefill: consolidate, do not rewrite

### 6.1 Existing design to preserve

The existing direct-RHS TensorOps kernel has the right general shape:

- output tile `64 x NR1`;
- K step around 32;
- double-buffered weight staging;
- RHS activation exposed directly from device memory;
- cooperative F32 accumulation;
- current token tiles 32/64/128;
- custom dequant of quantized weights into the staged tile.

The code comments explicitly record that staging both operands in threadgroup memory was slower and that direct RHS was the useful variant.

Treat this as empirical project knowledge.

### 6.2 Work to do

1. Identify every host dispatch that can use this existing path.
2. Confirm which variants are currently default, force-only, or retained experiments.
3. Make capability/shape policy explicit.
4. Benchmark tile dimensions:
   - `NR1=16` only if useful for tails,
   - 32,
   - 64,
   - 128,
   - optionally 256 only if compiler/resource usage remains sane.
5. Benchmark K tile 32 versus 64 where dequant structure permits.
6. Keep the current direct RHS unless a candidate beats it.
7. Add tail handling only if it reduces fallback cost enough to matter.

Do not create a generic template framework that obscures the existing kernel.

---

## 7. Indexer TensorOps: retain semantics, tune the regular matrix core

DeepSeek-V4 ratio-4 layers build indexer scores over compressed rows, then select up to the model's configured top-k for attention.

The existing TensorOps indexer correctly focuses on the regular part of the problem: token x compressed-row dot products over 128 dimensions.

Keep:
- causal visibility logic;
- weighted ReLU accumulation semantics;
- top-k stage;
- indexed-attention stage;
- current compressed-row representation.

Investigate:
- token tile `TM`;
- compressed-row tile `TN`;
- one-head versus paired-head processing;
- K step;
- execution SIMD group count;
- whether Q or K should remain staged;
- whether one operand can be a direct inline tensor;
- cooperative-result postprocessing to avoid the temporary `dot` store if mapping allows weighted ReLU accumulation in-place.

A fusion that changes top-k ordering because of floating-point drift is not acceptable without an explicit numerical-equivalence decision.

---

## 8. Routed MoE — primary new TensorOps target

### 8.1 Why MoE is the priority

DeepSeek-V4 Flash is a routed MoE model. Repeating expert gate/up/down projections across many layers means a TensorOps gap in `metal/moe.metal` can dominate prefill even if dense attention-side projections are already accelerated.

Do not assume this from architecture alone: confirm with Metal trace/stage profiling.

### 8.2 First map the current routed layout

Before coding, answer:

- How many experts are selected per token for the target model?
- How are selected token/expert pairs represented?
- Are tokens already sorted or bucketed by expert for prefill?
- Does the current MXFP4 prefill path process one token/slot, multiple slots, or compacted expert batches?
- What are the actual `M`, `N`, `K` values for gate, up, and down?
- How many tokens per expert are typical at batch sizes 32, 128, 512, 2K, 8K?
- What tail distribution exists?
- How does tensor parallelism alter ownership?
- How does SSD streaming / resident expert caching alter pointer stability?
- Which operations are compute-bound versus weight-bandwidth-bound?

Save a histogram of tokens-per-expert for representative prompts. TensorOps tile selection must be based on that distribution.

### 8.3 Candidate A — MXFP4 native TensorOps on macOS 27

This is the highest-upside hypothesis and must be tested first on a macOS 27-capable toolchain.

DS4 currently exposes clues consistent with an MX format:
- block size 32;
- E2M1-like 4-bit values;
- E8M0 scale representation.

Apple's current Metal tensor stack has:
- low-bit floating tensor formats on macOS 27;
- E8M0 block scale planes;
- MX-style multi-plane tensors;
- inline tensor construction from existing pointers.

However, "looks compatible" is not proof.

#### Mandatory compatibility proof

Create a small test that compares:

```text
DS4 block bytes
   -> existing DS4 MXFP4 decode
   -> reference F32/F16 values

same bytes interpreted as proposed Metal inline MX tensor
   -> TensorOps against identity/basis or controlled vectors
   -> recovered result
```

Verify:
- nibble ordering;
- E2M1 code mapping;
- signed zero;
- subnormal/zero scale cases;
- E8M0 exponent semantics;
- one scale per 32 weights;
- row stride;
- block stride;
- pointer alignment;
- required scale-plane alignment;
- transpose mode;
- whether data and scale planes can be represented as views into the existing interleaved block layout.

#### Important layout question

If DS4 stores each 32-weight block as:

```text
[scale][packed 4-bit values]
```

while Metal's multi-plane tensor expects logically separate data and scale planes, a truly zero-copy native view may not be possible.

Do not solve that by repacking the entire model.

Evaluate, in this order:

1. Can inline tensor strides/auxiliary plane descriptors express the interleaving?
2. Can a lightweight gather/repack tile be fused into a custom cooperative input?
3. Can existing resident expert-cache construction create a native-MX-friendly cached layout once per hot expert?
4. Is a per-layer/per-expert repack amortized at realistic prompt sizes?
5. If none wins end-to-end, keep custom dequant.

The native-MX path is optional. A negative result is a valid deliverable.

### 8.4 Candidate B — custom dequantization into cooperative TensorOps input

For formats without a direct native mapping:

```text
mapped quant blocks
   -> DS4 dequant in shader
   -> cooperative tensor
   -> matmul2d
   -> cooperative output
   -> activation/fusion
   -> final store
```

This is preferred over:

```text
quant blocks
   -> dequant
   -> threadgroup tile
   -> tensor load
   -> matmul
```

when the installed TensorOps API permits cooperative tensors as matmul inputs and profiling confirms lower data-movement cost.

Start with a format whose decoder is simple and already proven in `dense.metal`, such as Q8_0, to validate the mechanism before tackling Q2_K/IQ2_XXS.

### 8.5 Candidate C — threadgroup-dequant TensorOps

Use this as:
- compatibility tier for older supported OS versions;
- baseline against cooperative input;
- fallback if cooperative mapping/register pressure is poor.

It may still outperform manual SIMD-group matrix code because the matmul itself reaches the M5 Neural Accelerator.

### 8.6 Gate/up fusion

Ideal dataflow:

```text
activation tile
   | \
   |  \ 
   v   v
 gate  up        (TensorOps)
   \   /
    \ /
 cooperative gate/up results
      |
   clamp / SiLU / multiply / route weight
      |
      v
     mid
```

Questions Kiro must answer experimentally:

- Can two cooperative destination tensors from matching descriptors be indexed with a compatible per-thread mapping?
- Can gate and up be run back-to-back while the shared activation stays hot?
- Is storing only `mid` valid for the production graph, or are gate/up values needed elsewhere/debugging?
- Does fusing route weight here preserve the existing arithmetic order closely enough?
- Is register pressure lower/higher than writing gate/up to device memory?

Retain diagnostic materialization only in a debug path if production does not need it.

### 8.7 Down projection

The down projection has a different routing shape. Avoid forcing the gate/up kernel design onto it.

Evaluate:
- per-expert token bucket GEMM;
- batched selected-slot layout;
- direct route-weight incorporation before down;
- accumulation across selected experts;
- whether current DS4 performs a later sum that can be fused without changing reduction order.

If expert accumulation order is correctness-sensitive, preserve the existing order and optimize only each expert projection.

---

## 9. Q2_K and IQ2_XXS policy

Native Metal `int2` support does **not** imply native compatibility with DS4 `Q2_K` or `IQ2_XXS`.

These formats can include:
- nontrivial scale/min packing;
- group/block metadata;
- codebooks or lookup decoding;
- layout semantics beyond a raw signed/unsigned two-bit integer tensor.

Therefore:

1. Document the exact DS4 block structure.
2. Determine whether a native tensor can represent it exactly.
3. If not, do not build a misleading "native int2" path.
4. Test cooperative custom dequant.
5. Compare against the current specialized kernel.

For complex quants, the dequant cost itself may dominate enough that TensorOps cannot help at small `N`. Let measurements decide.

---

## 10. Attention / FlashAttention

Apple demonstrates TensorOps as a building block for custom FlashAttention, but DS4 already has highly specialized attention for compressed DeepSeek-V4 state.

Do not start by replacing `flash_attn.metal`.

After dense + indexer + MoE work:

1. Profile attention prefill again.
2. Identify regular QK/V matrix tiles that remain material.
3. Consider TensorOps only for those sub-operations.
4. Keep DS4's compressed-history, raw-window, indexer, causal, and cache semantics.
5. Prefer a surgical TensorOps inner loop to an attention rewrite.

Potential candidate:
- prefill QK tile -> cooperative scores -> softmax in cooperative/thread-local mapping -> V matmul.

Only proceed if attention remains a top prefill bottleneck.

---

## 11. Decode strategy

### 11.1 Batch-1 decode

Expected characteristics:
- repeated weight streaming;
- low arithmetic intensity;
- many matrix-vector operations;
- launch/synchronization sensitivity;
- current code already has significant fused dispatch work.

Therefore the default decision rule is:

```text
if batch == 1:
    legacy/manual fused decode stays default
    unless balanced A/B proves TensorOps faster end-to-end
```

Do not optimize for theoretical Neural Accelerator utilization if memory bandwidth dominates.

### 11.2 Speculative verification / small batches

This is separate.

For verification batches or session batching:
- identify `N` distribution;
- benchmark existing `mul_mv_ext` / matrix paths against TensorOps;
- find crossover `N`;
- use TensorOps only beyond a measured crossover.

Possible policy table generated at development time:

| Quant | Operation | N 1 | N 2-4 | N 5-8 | N 9-32 | N >32 |
|---|---|---|---|---|---|---|
| Q8 | dense | legacy | legacy | measured | TensorOps? | TensorOps |
| F16 | dense | legacy | measured | TensorOps? | TensorOps | TensorOps |
| MXFP4 MoE | ... | legacy | ... | ... | ... | ... |

Do not hard-code this illustrative table without benchmarks.

---

## 12. Dispatch design

Add a small operation-level selector rather than scattering M5 checks through every graph call.

Conceptual API:

```c
enum ds4_metal_mm_impl {
    DS4_METAL_MM_LEGACY,
    DS4_METAL_MM_TENSOROPS_STAGED,
    DS4_METAL_MM_TENSOROPS_COOP_DEQUANT,
    DS4_METAL_MM_TENSOROPS_NATIVE_QUANT,
};

ds4_metal_mm_impl ds4_metal_choose_mm_impl(
    operation_kind op,
    tensor_type weight_type,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    runtime_mode mode,
    streaming_state streaming);
```

This need not literally be a global function if current architecture has a better local dispatch mechanism.

Selection inputs:
- compile/runtime capability;
- device family;
- OS feature tier;
- quant type;
- alignment;
- transpose/layout;
- token count;
- expert bucket size;
- streaming/cache state;
- quality mode;
- TP mode;
- experimental rollback flags.

Selection output should be stable and cheap.

### 12.1 Default policy

- M5 + eligible prefill: fastest measured TensorOps path.
- pre-M5: current legacy default unless local benchmark data justifies TensorOps.
- batch-1 decode: legacy.
- unsupported quant/layout: legacy.
- debug force flags: can override where safe.
- rollback flags always win over force flags, consistent with existing repository patterns.

---

## 13. Pipeline creation and build integration

### 13.1 Do not make old SDKs fail to compile

Retain conditional compilation around TensorOps source.

If the repository currently concatenates/includes shader sources through a shared generated source path, keep that mechanism.

The build must support:
- legacy Apple Metal build with an SDK lacking new TensorOps types;
- enhanced build with current SDK;
- CPU build;
- CUDA build;
- ROCm build.

### 13.2 Pipeline registry

Register TensorOps pipelines lazily or with the project's existing pipeline-cache strategy.

Avoid:
- per-token pipeline creation;
- repeated function-constant compilation;
- pipeline lookup through expensive string/dictionary work in hot decode paths.

Use existing fast pipeline lookup conventions.

---

## 14. Memory and caching design

### 14.1 Mmap path

For dense weights already exposed through a Metal buffer or mapped range:
- prefer inline tensor over host-created copy;
- ensure storage mode and address lifetime match command lifetime;
- prove alignment.

### 14.2 Resident expert cache

This is the most promising place to support an alternative native-MX layout if the on-disk layout cannot be expressed directly.

If a hot expert is copied into a resident cache anyway, optionally create:

```text
resident expert cache entry
  + original/legacy view
  + optional tensor-native packed view
```

Only if:
- extra memory is bounded by the existing cache budget or an explicit sub-budget;
- creation is asynchronous/amortized;
- cache hit rate makes it worthwhile;
- SSD streaming still functions.

### 14.3 Scratch

Track:
- threadgroup bytes;
- cooperative tensor/register pressure;
- occupancy;
- temporary device buffers.

A TensorOps kernel that causes enough register/threadgroup pressure to cut occupancy may lose despite faster matmul hardware.

---

## 15. Correctness strategy

### 15.1 Three correctness classes

**Class A — bit-identical required**

Use for:
- paths whose current harness explicitly requires full-vocabulary bit identity;
- router selection where tiny drift may change selected experts;
- top-k compressed-row selection if score drift changes ordering;
- decode fast paths intended as exact substitutes.

**Class B — numerically equivalent with strict contract**

Allowed only when TensorOps reduction order makes bit identity structurally impossible.

Required evidence:
- kernel output comparison over deterministic random and adversarial inputs;
- max absolute error;
- max relative error;
- ULP distribution where useful;
- no NaN/Inf regressions;
- end-to-end full logit comparison;
- same selected token across test corpus;
- same selected experts/top-k rows unless the design explicitly proves downstream equivalence.

Do not invent the tolerance in advance. Calibrate it against:
1. existing DS4 accepted precision transitions;
2. legacy GPU vs reference behavior;
3. accumulation type and expected error;
4. end-to-end quality.

Then freeze the tolerance in the test with a comment explaining its derivation.

**Class C — ranking approximation**

Only existing intentionally approximate/ranking-only paths may use this class. Do not introduce a new approximation merely to make TensorOps faster without an explicit design update.

### 15.2 Native MXFP4 conformance test

Extend or complement `tests/test_mxfp4_metal.c`.

Test matrix:
- all 16 E2M1 codes;
- scale bytes including 0, tiny, normal, large;
- positive/negative zero;
- row crossing;
- multiple blocks;
- odd output rows/tails;
- transposed/non-transposed interpretation if applicable.

Compare:
- existing LUT/scalar decode;
- candidate native Metal tensor matmul against a basis vector;
- candidate custom cooperative dequant.

---

## 16. Benchmark design

### 16.1 Existing harnesses are the standard

Extend, do not bypass:
- `speed-bench/metal_prefill_variant_bench`
- `speed-bench/metal_decode_schedule_bench`
- `ds4-bench`

Use rollback variables so control and candidate share one binary when practical.

### 16.2 Add a focused TensorOps matrix benchmark

Create a small Metal benchmark only if existing harnesses cannot isolate enough signal.

It should exercise real DS4 dimensions and quant blocks, not synthetic square GEMMs only.

Matrix cases should include:
- dense F16;
- Q8_0;
- Q4_K;
- MXFP4;
- Q2_K or IQ2_XXS if candidate exists;
- token tiles matching real prefill;
- MoE expert bucket sizes from real routing histograms.

Report:
- median;
- p10/p90 or min/max;
- effective GB/s;
- effective output rate;
- warmup count;
- dispatch count.

### 16.3 End-to-end acceptance matrix

At minimum:

| Scenario | Metric |
|---|---|
| short prompt | prefill t/s + TTFT |
| 2K context | prefill t/s |
| 8K context | prefill t/s |
| 32K context | prefill t/s |
| long context if practical | prefill slope |
| batch-1 128-token decode | t/s |
| speculative verification | verify throughput |
| SSD streaming | prefill/decode + load stalls |
| resident expert cache warm | prefill |
| resident expert cache cold | load + prefill |

Record thermals/power state enough to avoid comparing throttled runs.

### 16.4 Keep/reject rule

A candidate becomes production default only if:
- correctness passes;
- end-to-end target operation improves repeatably;
- memory/load costs do not erase the gain;
- fallback is clean;
- code complexity is proportional to the win.

A candidate that loses is removed or left behind a clearly marked research-only compile/debug path, not silently accumulated as dead production complexity.

---

## 17. Profiling plan

For each major stage:

1. baseline trace;
2. candidate trace;
3. annotate:
   - GPU occupancy,
   - memory bandwidth,
   - cache behavior,
   - threadgroup memory,
   - dispatch gaps,
   - synchronization,
   - TensorOps/Neural Accelerator activity if exposed;
4. compare arithmetic intensity.

Expected hypotheses:

| Stage | Initial hypothesis |
|---|---|
| large dense prefill | compute-heavy enough for TensorOps |
| routed MoE prefill | mixed compute + weight bandwidth; likely high value |
| ratio-4 indexer score | regular 128-deep matrix core; existing TensorOps useful |
| batch-1 decode matvec | bandwidth/launch bound; TensorOps uncertain |
| verify block | more compute reuse; TensorOps more promising |
| long-context attention | may be memory/reduction dominated |

These are hypotheses, not conclusions.

---

## 18. Environment controls

Follow current repository naming style. Prefer rollback controls over permanent user knobs.

Suggested semantic controls; exact names may be adjusted to match current conventions:

```text
DS4_METAL_DISABLE_TENSOROPS
DS4_METAL_DISABLE_M5_TENSOROPS_DENSE
DS4_METAL_DISABLE_M5_TENSOROPS_INDEXER
DS4_METAL_DISABLE_M5_TENSOROPS_MOE
DS4_METAL_DISABLE_M5_TENSOROPS_NATIVE_MX
DS4_METAL_DISABLE_M5_TENSOROPS_VERIFY

DS4_METAL_ENABLE_TENSOROPS_DIAGNOSTICS
```

Optional research-only force flags:

```text
DS4_METAL_FORCE_TENSOROPS_DENSE
DS4_METAL_FORCE_TENSOROPS_MOE
DS4_METAL_FORCE_TENSOROPS_NATIVE_MX
```

Do not ship dozens of shape-specific public knobs. Shape tuning belongs in internal dispatch tables/functions.

---

## 19. Source-level implementation map

Kiro must revalidate names on current `main`, but likely touch points are:

### `metal/dense.metal`
- consolidate current MPP/TensorOps prefill path;
- capability-safe variants;
- optional cooperative-input dequant experiments;
- verify/small-batch candidates.

### `metal/dsv4_misc.metal`
- current NAX indexer;
- potential cooperative post-processing/tile tuning.

### `metal/moe.metal`
- primary new work;
- native MXFP4 candidate;
- custom dequant cooperative input;
- gate/up fusion;
- down projection;
- preserve existing exact decode kernels.

### host Metal backend / pipeline registry
- compile feature detection;
- runtime capability model;
- pipeline registration;
- operation selector;
- diagnostics/counters;
- rollback flags.

### `ds4_gpu.h`
- only minimal shared declarations/capability/test hooks;
- keep non-Apple stubs where required.

### `ds4.c`
- only graph-level dispatch/fusion wiring that cannot live in backend;
- avoid spreading Metal API specifics into model logic.

### `tests/`
- native MXFP4 compatibility;
- TensorOps capability/fallback;
- kernel parity;
- end-to-end exact/equivalence.

### `speed-bench/`
- reuse A/B harnesses;
- add focused TensorOps microbench only if needed.

---

## 20. Implementation sequencing

### Milestone A — understand and make current TensorOps explicit

No new MoE math yet.

Deliver:
- baseline;
- capability probe;
- path inventory;
- current dense/indexer selection understood;
- diagnostics;
- rollback;
- tests.

### Milestone B — MXFP4 feasibility spike

Two branches/experiments:

**B1 native MX**
- compile on macOS 27 SDK;
- prove representation;
- benchmark direct mapped/cache layout;
- reject if repack dominates.

**B2 custom cooperative dequant**
- prove custom dequant -> cooperative input -> TensorOps;
- start with a simple quant if MX interleaving is awkward;
- benchmark versus staged TensorOps and legacy.

Choose the winning mechanism(s).

### Milestone C — production MoE gate/up

- integrate routing-aware token/expert batching;
- TensorOps gate/up;
- activation/route-weight fusion where safe;
- exact/equivalence tests;
- A/B prefill.

### Milestone D — production MoE down

- TensorOps candidate;
- preserve expert accumulation semantics;
- A/B.

### Milestone E — small-batch generation

- determine crossover;
- integrate verify path only if useful;
- batch-1 remains untouched by default.

### Milestone F — optional attention work

Only if post-MoE profiling says attention is now a material remaining bottleneck.

### Milestone G — productionization

- defaults;
- fallback;
- test matrix;
- docs/comments;
- remove losing experiments;
- final benchmark CSV/traces.

---

## 21. Research questions Kiro must answer, not guess

1. Where is `DS4_METAL_HAS_TENSOR` defined and how does the current build select the required Metal language version?
2. Which existing TensorOps kernels are actually enabled by default on M5 today?
3. What are the exact host-side dispatch conditions for `kernel_mul_mm_mpp_direct_rhs`?
4. What are the exact host-side dispatch conditions for `kernel_dsv4_indexer_scores_nax`?
5. Which quant formats dominate the user's target GGUF and the repository's standard `ds4flash.gguf`?
6. Which operations dominate prefill on M5 after current August 2026 optimizations?
7. Can macOS 27 Metal express DS4's MXFP4 bytes and E8M0 scale bytes without repacking?
8. If scale and data are interleaved on disk, can inline tensor planes express that stride pattern?
9. What alignment does Metal require for FP4 data and E8M0 scale planes?
10. Does native MXFP4 TensorOps use F32 or lower-precision accumulation for the desired descriptor?
11. Does a custom cooperative input allow DS4's existing dequant routine to feed `matmul2d` directly?
12. What cooperative tensor element mapping is guaranteed between two gate/up matmuls?
13. Does gate/up cooperative fusion reduce device traffic enough to overcome register pressure?
14. How are routed tokens grouped by expert today?
15. What is the token-count distribution per expert at 32/128/512/2K/8K prompt batches?
16. What tile dimensions maximize M5 throughput for DS4's real expert matrices?
17. Does TensorOps help MXFP4 if the workload remains weight-bandwidth-bound?
18. Where is the crossover between legacy matvec/small-batch kernels and TensorOps?
19. Does speculative verification create enough batch width to cross it?
20. Can existing resident expert-cache buffers host a tensor-native layout with bounded memory?
21. How does SSD streaming interact with inline tensor pointer lifetime?
22. Do any TensorOps reduction-order differences alter expert selection, indexer top-k, or sampled tokens?
23. Which existing tests already encode a tolerance model versus strict bit identity?
24. Can the TensorOps kernels remain compiled out cleanly on older SDKs?
25. Does enabling TensorOps on M1-M4 help or hurt enough to warrant a separate policy?
26. Which paths still dominate after MoE is accelerated?

Every unanswered item is a research task or an explicit deferred item, not an invitation to assume.

---

## 22. Primary research sources

The implementation should prioritize these sources in this order:

1. The checked-out DS4 source and `AGENT.md`.
2. Apple Metal headers in the active SDK.
3. Apple Metal Performance Primitives Programming Guide current to the installed Xcode.
4. Apple WWDC26: **Optimize custom machine learning operations with Metal tensors**.
5. Apple Tech Talk: **Accelerate your machine learning workloads with the M5 and A19 GPUs**.
6. Apple sample: **Running inline ML operations in a shader with Metal 4**.
7. DS4 GitHub issue #14.
8. Recent DS4 commits touching MXFP4, M5, prefill, indexer, decode scheduling, and Metal verification.
9. Kiro's own Design-First Feature Spec rules only for workflow structure, not technical truth.

Third-party benchmarks may inspire hypotheses but may not override the checked-out code, installed SDK, or measured local results.

---

## 23. Final architectural decision record template

At the end, add a concise record to the implementation PR or project notes:

```text
Target hardware:
OS / SDK:
Baseline commit:

Dense prefill:
  chosen path:
  speedup:
  correctness:

Indexer:
  chosen path:
  speedup:
  correctness:

MoE gate/up:
  chosen path:
  quant formats:
  speedup:
  correctness:

MoE down:
  chosen path:
  speedup:
  correctness:

Native MXFP4:
  compatible: yes/no
  zero-copy: yes/no
  repack required:
  decision:

Cooperative dequant:
  formats:
  decision:

Batch-1 decode:
  decision:
  delta:

Verify batch:
  crossover:
  delta:

Overall:
  short prefill:
  8K prefill:
  32K prefill:
  TTFT:
  decode:
  memory:
```

The feature is successful when this record makes it obvious *where* the gain came from and which experiments were deliberately rejected.
