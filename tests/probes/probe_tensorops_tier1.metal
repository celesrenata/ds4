// DS4 TensorOps Tier 1 Probe — Basic tensor / matmul2d / cooperative destination
// Requires: macOS 26.0+ SDK
// Usage: xcrun metal -c probe_tensorops_tier1.metal -o /dev/null
//
// Tests: <metal_tensor>, <MetalPerformancePrimitives/MetalPerformancePrimitives.h>,
//        tensor inline construction, matmul2d, cooperative destination tensor,
//        dextents, slice, store.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void ds4_probe_tier1(
        device half *A [[buffer(0)]],
        device half *B [[buffer(1)]],
        device float *C [[buffer(2)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    (void)tiitg;

    // Inline tensor from device pointer with dynamic extents
    auto tA = tensor(A, dextents<int32_t, 2>(32, 64));
    auto tB = tensor(B, dextents<int32_t, 2>(32, 64), array<int, 2>({1, 32}));

    // matmul2d with cooperative F32 accumulation
    matmul2d<matmul2d_descriptor(32, 64, 32, false, true, true,
                 matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    // Zero-init cooperative tensor
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    // Slice and run
    auto sA = tA.slice(0, 0);
    auto sB = tB.slice(0, 0);
    mm.run(sA, sB, cT);

    // Store result
    auto tC = tensor(C, dextents<int32_t, 2>(64, 32), array<int, 2>({1, 64}));
    cT.store(tC.slice(0, 0));
}
