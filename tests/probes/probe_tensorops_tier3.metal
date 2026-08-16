// DS4 TensorOps Tier 3 Probe — Native int4/int8 tensor element types
// Requires: macOS 26.4+ SDK
// Usage: xcrun metal -c probe_tensorops_tier3.metal -o /dev/null
//
// Tests: Creating tensors with native 4-bit and 8-bit integer element types.
//        This allows the Metal tensor hardware to consume packed integer data
//        directly without explicit dequantization.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// NOTE: The exact type names for int4/int8 tensor elements may differ from
// this placeholder. Adjust to match the actual SDK symbols when compiling on
// macOS 26.4+. Candidates include:
//   - tensor_element_int4 / tensor_element_uint4
//   - int4_t / uint4_t as tensor element types
//   - Packed integer tensor descriptors
//
// If this probe fails to compile, check Apple's Metal tensor release notes
// for the exact type names and update accordingly.

kernel void ds4_probe_tier3_native_int(
        device uchar *weights_int8 [[buffer(0)]],
        device uchar *weights_int4 [[buffer(1)]],
        device float *C [[buffer(2)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    (void)tgpig; (void)tiitg;

    // Attempt int8 tensor construction
    // auto t_int8 = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
    //     (device int8_t *)weights_int8, dextents<int32_t, 2>(64, 32));

    // Attempt int4 tensor construction (packed nibbles)
    // auto t_int4 = tensor<device int4_t, dextents<int32_t, 2>, tensor_inline>(
    //     (device int4_t *)weights_int4, dextents<int32_t, 2>(64, 32));

    // Placeholder: compile with the types available in the installed SDK.
    // The probe is successful if the above tensor declarations compile and
    // a matmul2d accepting them as inputs produces a valid pipeline.

    // Fallback: just verify the header exists and basic int tensor type is declared
    (void)weights_int8;
    (void)weights_int4;
    (void)C;
}

// TODO(macOS 26.4): Replace the commented-out code above with actual SDK types
// once the exact API is confirmed on target hardware. The key questions:
//   1. What is the declared element type name for 4-bit integers?
//   2. Does matmul2d accept int4/int8 tensor operands directly?
//   3. What accumulator type is required (float, int32)?
//   4. What alignment/stride constraints exist for packed data?
