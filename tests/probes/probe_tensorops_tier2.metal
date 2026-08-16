// DS4 TensorOps Tier 2 Probe — Cooperative tensor as matmul INPUT
// Requires: macOS 26.3+ SDK
// Usage: xcrun metal -c probe_tensorops_tier2.metal -o /dev/null
//
// Tests: Using a cooperative tensor (from custom dequantization) as an input
//        operand to matmul2d, eliminating the threadgroup-memory round trip.
//        This is the key enabler for custom-dequant TensorOps on non-native formats.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// Simulated custom dequantization: fill a cooperative tensor from scalar code
// then use it as a matmul input.
kernel void ds4_probe_tier2_coop_input(
        device half *A [[buffer(0)]],
        device half *B [[buffer(1)]],
        device float *C [[buffer(2)]],
        threadgroup half *shmem [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    (void)tgpig;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;

    // Standard destination matmul descriptor
    matmul2d<matmul2d_descriptor(32, NR0, NK, false, true, true,
                 matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    // Get a cooperative tensor to use as RIGHT INPUT (the tier 2 feature)
    // On macOS 26.3+, cooperative tensors can be passed as matmul inputs.
    // Template: <LeftElementType, RightElementType, DestinationElementType>
    auto coop_weights = mm.template get_right_input_cooperative_tensor<half, half, float>();

    // Simulate custom dequant: each thread fills its cooperative slots
    for (uint16_t i = 0; i < coop_weights.get_capacity(); ++i) {
        if (coop_weights.is_valid_element(i)) {
            // In real code: dequantize from quantized block here
            coop_weights[i] = (half)0.5h;
        }
    }

    // LHS from device memory (activations)
    auto tB = tensor(B, dextents<int32_t, 2>(32, NK));

    // Destination cooperative tensor
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(coop_weights), float>();

    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    // Run matmul with cooperative input (THIS is the tier 2 feature test)
    auto sB = tB.slice(0, 0);
    mm.run(sB, coop_weights, cT);

    // Store
    auto tC = tensor(C, dextents<int32_t, 2>(NR0, 32));
    cT.store(tC.slice(0, 0));
}
