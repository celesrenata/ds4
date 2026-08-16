/* DS4 Native-MX Conformance Probe — Task 5.3
 *
 * Purpose:
 *   Establish the complete reference expectation for native-MX tensor
 *   compatibility testing. Since macOS 26.5 SDK has NO FP4/E2M1 type,
 *   NO E8M0 scale-plane, and NO multi-plane MX tensor support, this test
 *   runs only the CPU reference path now and provides stub sections for the
 *   actual Metal native-MX path when macOS 27 SDK becomes available.
 *
 * Tests:
 *   1. All 16 E2M1 low-bit codes across representative E8M0 scales.
 *   2. Comparison against the DS4 MXFP4 decoder (LUT consistency).
 *   3. Basis-vector matmul reference to prove interpretation is unambiguous.
 *   4. Nibble-ordering convention documentation + test expectation.
 *
 * Build (reference-only, no Metal dependency):
 *   cc -std=c11 -O2 -o test_mxfp4_native_mx_probe tests/test_mxfp4_native_mx_probe.c -lm
 *
 * Validates: R7, R12
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * DS4 MXFP4 format definitions (from metal/moe.metal and tests/test_mxfp4_dot.c)
 * --------------------------------------------------------------------------- */

#define QK_MXFP4 32u

typedef struct {
    uint8_t e;              /* E8M0 shared scale exponent */
    uint8_t qs[QK_MXFP4 / 2u]; /* 32 × E2M1 nibbles packed into 16 bytes */
} block_mxfp4;

/* E2M1 value table: bit layout [S][E1 E0][M0] */
static const float mxfp4_values[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,   /* codes 0–7 (positive) */
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,   /* codes 8–15 (negative) */
};

/* Alternative independent reference using ldexp arithmetic */
static float e2m1_decode_independent(uint8_t code) {
    /* E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
     * code layout: [S][E1 E0][M0]
     *   S   = code >> 3
     *   E   = (code >> 1) & 0x3
     *   M   = code & 0x1
     *
     * If E==0 (subnormal): value = (-1)^S × 0.5 × M = (-1)^S × {0, 0.5}
     * If E>0 (normal):     value = (-1)^S × 2^(E-1) × (1 + 0.5*M)
     */
    const int sign = (code >> 3) & 1;
    const int E = (code >> 1) & 3;
    const int M = code & 1;
    float val;
    if (E == 0) {
        val = M ? 0.5f : 0.0f;
    } else {
        val = ldexpf(1.0f + 0.5f * (float)M, E - 1);
    }
    return sign ? -val : val;
}

/* E8M0 scale conversion (matches DS4 production code) */
static float e8m0_to_f32(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Independent E8M0 reference using ldexp.
 * DS4 convention: e=0 maps to 2^(-127) (float bit pattern 0x00400000),
 * which is the smallest positive subnormal float with only the implicit
 * mantissa bit set. For e≥1, the value is 2^(e-127). */
static float e8m0_to_f32_independent(uint8_t e) {
    if (e == 0) return ldexpf(1.0f, -127);
    return ldexpf(1.0f, (int)e - 127);
}

/* DS4 MXFP4 dot product — the canonical reference decoder */
static float dot_mxfp4_reference(const block_mxfp4 *blocks,
                                 const float *y, uint32_t n_elem) {
    float sum = 0.0f;
    const uint32_t n_blocks = n_elem / QK_MXFP4;
    for (uint32_t ib = 0; ib < n_blocks; ib++) {
        const float d = e8m0_to_f32(blocks[ib].e);
        for (uint32_t j = 0; j < QK_MXFP4 / 2u; j++) {
            const uint8_t q = blocks[ib].qs[j];
            sum += d * mxfp4_values[q & 0x0fu] * y[ib * QK_MXFP4 + j];
            sum += d * mxfp4_values[q >> 4u]   * y[ib * QK_MXFP4 + j + QK_MXFP4 / 2u];
        }
    }
    return sum;
}

/* ---------------------------------------------------------------------------
 * Test 1: All 16 E2M1 codes match independent decode across representative scales
 * --------------------------------------------------------------------------- */

static int test_e2m1_all_codes(void) {
    int failures = 0;

    /* Representative E8M0 scale bytes covering the full range */
    static const uint8_t test_scales[] = {
        0,     /* minimum (2^-126) */
        1,     /* 2^-126 */
        64,    /* 2^-63 */
        100,   /* 2^-27 */
        126,   /* 2^-1 = 0.5 */
        127,   /* 2^0  = 1.0 (unity) */
        128,   /* 2^1  = 2.0 */
        150,   /* 2^23 */
        200,   /* 2^73 */
        254,   /* 2^127 (maximum) */
    };
    const int n_scales = (int)(sizeof(test_scales) / sizeof(test_scales[0]));

    printf("Test 1: All 16 E2M1 codes × %d representative scales\n", n_scales);

    for (int si = 0; si < n_scales; si++) {
        const uint8_t e = test_scales[si];
        const float scale_lut = e8m0_to_f32(e);
        const float scale_ind = e8m0_to_f32_independent(e);

        /* Verify E8M0 converters agree */
        if (scale_lut != scale_ind) {
            printf("  FAIL: E8M0 scale mismatch for e=%u: LUT=%g independent=%g\n",
                   e, scale_lut, scale_ind);
            failures++;
            continue;
        }

        for (uint8_t code = 0; code < 16; code++) {
            const float val_lut = mxfp4_values[code];
            const float val_ind = e2m1_decode_independent(code);

            /* Compare unscaled E2M1 values (handle ±0 specially) */
            if (val_lut == 0.0f && val_ind == 0.0f) {
                /* Both zero — check sign bit consistency */
                uint32_t bits_lut, bits_ind;
                memcpy(&bits_lut, &val_lut, sizeof(bits_lut));
                memcpy(&bits_ind, &val_ind, sizeof(bits_ind));
                if (bits_lut != bits_ind) {
                    printf("  FAIL: Zero sign mismatch for code 0x%x: "
                           "LUT=0x%08x independent=0x%08x\n",
                           code, bits_lut, bits_ind);
                    failures++;
                }
            } else if (val_lut != val_ind) {
                printf("  FAIL: E2M1 value mismatch for code 0x%x: "
                       "LUT=%g independent=%g\n", code, val_lut, val_ind);
                failures++;
            }

            /* Verify scaled product */
            const float scaled_lut = scale_lut * val_lut;
            const float scaled_ind = scale_ind * val_ind;
            if (scaled_lut != scaled_ind && !(scaled_lut == 0.0f && scaled_ind == 0.0f)) {
                printf("  FAIL: Scaled value mismatch for e=%u code=0x%x: "
                       "LUT=%g independent=%g\n", e, code, scaled_lut, scaled_ind);
                failures++;
            }
        }
    }

    printf("  %s (%d failures across %d combinations)\n",
           failures == 0 ? "PASS" : "FAIL", failures, n_scales * 16);
    return failures;
}

/* ---------------------------------------------------------------------------
 * Test 2: DS4 dot product decoder consistency with element-wise decode
 * --------------------------------------------------------------------------- */

static int test_decoder_consistency(void) {
    int failures = 0;
    printf("Test 2: DS4 decoder consistency (dot vs element-wise)\n");

    /* For each E8M0 scale and each E2M1 code, construct a single-block weight
     * with only that code set, dot against a unit vector at the correct
     * position, and verify the result matches the expected decoded value. */

    static const uint8_t test_scales[] = { 0, 1, 64, 126, 127, 128, 200, 254 };
    const int n_scales = (int)(sizeof(test_scales) / sizeof(test_scales[0]));

    for (int si = 0; si < n_scales; si++) {
        const uint8_t e = test_scales[si];
        const float scale = e8m0_to_f32(e);

        for (uint8_t code = 0; code < 16; code++) {
            /* Test low-nibble position: place code in qs[0] low nibble,
             * which corresponds to element 0 in the block. */
            {
                block_mxfp4 block;
                memset(&block, 0, sizeof(block));
                block.e = e;
                block.qs[0] = code; /* low nibble = element 0, high nibble = 0 */

                float y[QK_MXFP4];
                memset(y, 0, sizeof(y));
                y[0] = 1.0f; /* basis vector at element 0 */

                const float got = dot_mxfp4_reference(&block, y, QK_MXFP4);
                const float expected = scale * mxfp4_values[code];

                if (got != expected && !(got == 0.0f && expected == 0.0f)) {
                    printf("  FAIL: Low-nibble dot mismatch e=%u code=0x%x: "
                           "got=%g expected=%g\n", e, code, got, expected);
                    failures++;
                }
            }

            /* Test high-nibble position: place code in qs[0] high nibble,
             * which corresponds to element 16 in the block. */
            {
                block_mxfp4 block;
                memset(&block, 0, sizeof(block));
                block.e = e;
                block.qs[0] = (uint8_t)(code << 4u); /* high nibble = element 16 */

                float y[QK_MXFP4];
                memset(y, 0, sizeof(y));
                y[QK_MXFP4 / 2u] = 1.0f; /* basis vector at element 16 */

                const float got = dot_mxfp4_reference(&block, y, QK_MXFP4);
                const float expected = scale * mxfp4_values[code];

                if (got != expected && !(got == 0.0f && expected == 0.0f)) {
                    printf("  FAIL: High-nibble dot mismatch e=%u code=0x%x: "
                           "got=%g expected=%g\n", e, code, got, expected);
                    failures++;
                }
            }
        }
    }

    printf("  %s (%d failures across %d probes)\n",
           failures == 0 ? "PASS" : "FAIL", failures, n_scales * 16 * 2);
    return failures;
}

/* ---------------------------------------------------------------------------
 * Test 3: Basis-vector matmul reference
 *
 * Constructs a multi-block MXFP4 weight "matrix" (K rows × DIM columns) and
 * multiplies by standard basis vectors to recover individual decoded values.
 * This proves that the dot-product interpretation is unambiguous: each basis
 * vector e_i selects exactly the decoded value at position i.
 * --------------------------------------------------------------------------- */

#define PROBE_DIM 64u  /* 2 blocks of 32 elements */
#define PROBE_ROWS 16u

static int test_basis_vector_matmul(void) {
    int failures = 0;
    printf("Test 3: Basis-vector matmul (PROBE_DIM=%u, PROBE_ROWS=%u)\n",
           PROBE_DIM, PROBE_ROWS);

    const uint32_t blocks_per_row = PROBE_DIM / QK_MXFP4;
    block_mxfp4 weights[PROBE_ROWS * (PROBE_DIM / QK_MXFP4)];

    /* Fill with a deterministic pattern covering all 16 codes */
    uint32_t state = 0xDEADBEEFu;
    for (uint32_t r = 0; r < PROBE_ROWS; r++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_mxfp4 *blk = &weights[r * blocks_per_row + b];
            state = state * 1664525u + 1013904223u;
            blk->e = (uint8_t)(100u + (state >> 24) % 55u); /* E8M0 in [100,154] */
            for (uint32_t j = 0; j < QK_MXFP4 / 2u; j++) {
                state = state * 1664525u + 1013904223u;
                blk->qs[j] = (uint8_t)(state >> 24);
            }
        }
    }

    /* For each row and each element position, construct a basis vector and
     * verify that the dot product recovers the expected decoded element. */
    for (uint32_t r = 0; r < PROBE_ROWS; r++) {
        const block_mxfp4 *row_blocks = &weights[r * blocks_per_row];

        for (uint32_t elem = 0; elem < PROBE_DIM; elem++) {
            float basis[PROBE_DIM];
            memset(basis, 0, sizeof(basis));
            basis[elem] = 1.0f;

            const float got = dot_mxfp4_reference(row_blocks, basis, PROBE_DIM);

            /* Compute expected value by manual element decode */
            const uint32_t block_idx = elem / QK_MXFP4;
            const uint32_t elem_in_block = elem % QK_MXFP4;
            const block_mxfp4 *blk = &row_blocks[block_idx];
            const float scale = e8m0_to_f32(blk->e);

            uint8_t nibble_code;
            if (elem_in_block < QK_MXFP4 / 2u) {
                /* Low nibble: elements 0–15 → qs[elem_in_block] & 0x0F */
                nibble_code = blk->qs[elem_in_block] & 0x0Fu;
            } else {
                /* High nibble: elements 16–31 → qs[elem_in_block - 16] >> 4 */
                nibble_code = blk->qs[elem_in_block - QK_MXFP4 / 2u] >> 4u;
            }
            const float expected = scale * mxfp4_values[nibble_code];

            if (got != expected && !(got == 0.0f && expected == 0.0f)) {
                printf("  FAIL: row=%u elem=%u got=%g expected=%g "
                       "(block=%u elem_in_block=%u nibble=0x%x)\n",
                       r, elem, got, expected, block_idx, elem_in_block,
                       nibble_code);
                failures++;
                if (failures > 10) {
                    printf("  ... too many failures, stopping\n");
                    goto done;
                }
            }
        }
    }

done:
    printf("  %s (%d failures across %u probes)\n",
           failures == 0 ? "PASS" : "FAIL", failures, PROBE_ROWS * PROBE_DIM);
    return failures;
}

/* ---------------------------------------------------------------------------
 * Test 4: Nibble-ordering convention test
 *
 * Proves that DS4's split-half convention is correctly modeled:
 *   qs[j] low nibble  → element j      (j = 0..15)
 *   qs[j] high nibble → element j + 16 (j = 0..15)
 *
 * This is the critical ordering question for native-MX compatibility:
 * if Metal uses sequential nibble ordering (element 2k in low nibble of byte k,
 * element 2k+1 in high nibble of byte k), a reorder would be required.
 * --------------------------------------------------------------------------- */

static int test_nibble_ordering_convention(void) {
    int failures = 0;
    printf("Test 4: Nibble-ordering convention (split-half)\n");

    /* Construct a block where each code is unique and position-dependent,
     * then verify extraction matches the split-half convention exactly. */
    block_mxfp4 block;
    block.e = 127; /* scale = 1.0 */

    /* Low nibbles (elements 0–15): code = j % 8 (positive values only) */
    /* High nibbles (elements 16–31): code = 8 + (j % 8) (negative values) */
    for (uint32_t j = 0; j < 16u; j++) {
        const uint8_t lo = (uint8_t)(j % 8u);        /* positive codes 0–7 */
        const uint8_t hi = (uint8_t)(8u + j % 8u);   /* negative codes 8–15 */
        block.qs[j] = (uint8_t)(lo | (hi << 4u));
    }

    float y[QK_MXFP4];

    /* Verify element 0 through basis vector at position 0 */
    memset(y, 0, sizeof(y));
    y[0] = 1.0f;
    float got = dot_mxfp4_reference(&block, y, QK_MXFP4);
    float expected = mxfp4_values[0]; /* code 0 = 0.0 */
    if (got != expected) {
        printf("  FAIL: element 0 = %g (expected %g)\n", got, expected);
        failures++;
    }

    /* Verify element 5 through basis vector at position 5 */
    memset(y, 0, sizeof(y));
    y[5] = 1.0f;
    got = dot_mxfp4_reference(&block, y, QK_MXFP4);
    expected = mxfp4_values[5]; /* code 5 = 3.0 */
    if (got != expected) {
        printf("  FAIL: element 5 = %g (expected %g)\n", got, expected);
        failures++;
    }

    /* Verify element 16 through basis vector at position 16
     * (high nibble of qs[0]) */
    memset(y, 0, sizeof(y));
    y[16] = 1.0f;
    got = dot_mxfp4_reference(&block, y, QK_MXFP4);
    expected = mxfp4_values[8]; /* code 8 = -0.0 */
    /* -0.0 × 1.0 = -0.0, but sum starting from 0.0f may become +0.0 */
    if (got != expected && !(got == 0.0f && expected == 0.0f)) {
        printf("  FAIL: element 16 = %g (expected %g)\n", got, expected);
        failures++;
    }

    /* Verify element 21 through basis vector at position 21
     * qs[5] high nibble → code 8 + (5%8) = 13 → -3.0 */
    memset(y, 0, sizeof(y));
    y[21] = 1.0f;
    got = dot_mxfp4_reference(&block, y, QK_MXFP4);
    expected = mxfp4_values[13]; /* code 13 = -3.0 */
    if (got != expected) {
        printf("  FAIL: element 21 = %g (expected %g)\n", got, expected);
        failures++;
    }

    /* Cross-check: all elements via unit vector should sum to a known value */
    for (uint32_t i = 0; i < QK_MXFP4; i++) y[i] = 1.0f;
    got = dot_mxfp4_reference(&block, y, QK_MXFP4);
    /* Sum: (0 + 0.5 + 1 + 1.5 + 2 + 3 + 4 + 6) × 2 = 36.0 for low half
     * High half: all negative → (-0 + -0.5 + -1 + -1.5 + -2 + -3 + -4 + -6) × 2 = -36.0
     * Total: 36.0 + (-36.0) = 0.0 */
    expected = 0.0f;
    if (got != expected) {
        printf("  FAIL: all-ones sum = %g (expected %g)\n", got, expected);
        failures++;
    }

    /* Document the convention for native-MX comparison */
    printf("  DS4 nibble convention:\n");
    printf("    qs[j] & 0x0F → element j      (j = 0..15)\n");
    printf("    qs[j] >> 4   → element j + 16 (j = 0..15)\n");
    printf("  Split-half: low nibbles = elements 0-15, "
           "high nibbles = elements 16-31\n");
    printf("  %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

/* ---------------------------------------------------------------------------
 * Test 5: Multi-block row matmul with mixed scales
 *
 * Exercises the full decode path across block boundaries where scales differ,
 * using known activation vectors. This is the reference expectation that any
 * native-MX implementation must match exactly.
 * --------------------------------------------------------------------------- */

#define MATMUL_DIM 128u  /* 4 blocks */
#define MATMUL_ROWS 4u

static int test_multiblock_matmul(void) {
    int failures = 0;
    printf("Test 5: Multi-block row matmul (DIM=%u, ROWS=%u)\n",
           MATMUL_DIM, MATMUL_ROWS);

    const uint32_t blocks_per_row = MATMUL_DIM / QK_MXFP4;
    block_mxfp4 weights[MATMUL_ROWS * (MATMUL_DIM / QK_MXFP4)];

    /* Construct weights with deliberately different scales per block */
    static const uint8_t block_scales[4] = { 120, 127, 130, 140 };
    for (uint32_t r = 0; r < MATMUL_ROWS; r++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_mxfp4 *blk = &weights[r * blocks_per_row + b];
            blk->e = block_scales[b];
            for (uint32_t j = 0; j < QK_MXFP4 / 2u; j++) {
                /* Pattern: code cycles through all 16 values */
                const uint8_t lo = (uint8_t)((r + b + j) & 0x0Fu);
                const uint8_t hi = (uint8_t)((r * 3u + b * 5u + j * 7u) & 0x0Fu);
                blk->qs[j] = (uint8_t)(lo | (hi << 4u));
            }
        }
    }

    /* Activation vector: linearly spaced */
    float x[MATMUL_DIM];
    for (uint32_t i = 0; i < MATMUL_DIM; i++) {
        x[i] = (float)((int)i - 64) / 32.0f;
    }

    /* Compute reference output (simple matrix-vector multiply) */
    float ref_output[MATMUL_ROWS];
    for (uint32_t r = 0; r < MATMUL_ROWS; r++) {
        ref_output[r] = dot_mxfp4_reference(
            &weights[r * blocks_per_row], x, MATMUL_DIM);
    }

    /* Verify by recomputing element-wise */
    for (uint32_t r = 0; r < MATMUL_ROWS; r++) {
        float sum = 0.0f;
        for (uint32_t elem = 0; elem < MATMUL_DIM; elem++) {
            const uint32_t blk_idx = elem / QK_MXFP4;
            const uint32_t elem_in_blk = elem % QK_MXFP4;
            const block_mxfp4 *blk = &weights[r * blocks_per_row + blk_idx];
            const float scale = e8m0_to_f32(blk->e);
            uint8_t nibble_code;
            if (elem_in_blk < QK_MXFP4 / 2u) {
                nibble_code = blk->qs[elem_in_blk] & 0x0Fu;
            } else {
                nibble_code = blk->qs[elem_in_blk - QK_MXFP4 / 2u] >> 4u;
            }
            sum += scale * mxfp4_values[nibble_code] * x[elem];
        }
        const float tol = 2.0e-5f * fmaxf(1.0f, fabsf(sum));
        if (fabsf(ref_output[r] - sum) > tol) {
            printf("  FAIL: row %u: dot=%g element_wise=%g diff=%g\n",
                   r, ref_output[r], sum, fabsf(ref_output[r] - sum));
            failures++;
        }
    }

    printf("  Reference output: [%g, %g, %g, %g]\n",
           ref_output[0], ref_output[1], ref_output[2], ref_output[3]);
    printf("  %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

/* ---------------------------------------------------------------------------
 * Stub: Native Metal MX path (macOS 27+ SDK required)
 *
 * When DS4_METAL_SDK_NATIVE_LOWBIT is defined (macOS 27 SDK available):
 *   - Construct a native MX tensor from the same weight bytes
 *   - Perform matmul2d against the basis vectors
 *   - Compare results against the CPU reference computed above
 *
 * Key questions to answer when macOS 27 SDK is available:
 *   1. What is the exact type name? (fp4_e2m1? float4b_e2m1_format? ...)
 *   2. Does Metal use sequential nibble ordering or split-half?
 *   3. Can interleaved [scale][data] blocks be expressed via stride descriptors?
 *   4. What alignment does the scale plane require?
 *   5. Does e8m0 byte 0 produce 2^(-126) or zero?
 * --------------------------------------------------------------------------- */

#if defined(DS4_METAL_SDK_NATIVE_LOWBIT)

/*
 * TODO(macOS 27): Implement the following:
 *
 * 1. Create Metal device and command queue
 * 2. Upload weight bytes to a Metal buffer
 * 3. Construct a native MX tensor (data plane + E8M0 scale plane)
 *    using the actual macOS 27 API
 * 4. For each of the 16 E2M1 codes × representative scales:
 *    a. Construct a single-code weight block
 *    b. Create a native MX tensor from it
 *    c. Matmul against a known activation (basis vector or unit)
 *    d. Read back the result
 *    e. Compare against mxfp4_values[code] * e8m0_to_f32(scale)
 * 5. For the multi-block test:
 *    a. Use the same weight matrix as test_multiblock_matmul
 *    b. Matmul against the same activation vector
 *    c. Compare results against ref_output[]
 *
 * Critical conformance checks:
 *   - Nibble ordering: does element N map to the same nibble position?
 *   - Scale semantics: does e=0 produce the same value as DS4?
 *   - Zero handling: is code 0x8 treated as -0.0 or +0.0?
 *   - Block alignment: can 17-byte interleaved blocks be used, or must
 *     the data be repacked into separate planes?
 *
 * If any of these differ, document the difference and its implications
 * for the zero-copy vs bounded-repack decision in task 5.4.
 */

static int test_native_mx_conformance(void) {
    printf("Test 6: Native Metal MX conformance (macOS 27 SDK)\n");
    printf("  TODO: implement when SDK types are available\n");
    return 0;
}

#else

static int test_native_mx_conformance(void) {
    printf("Test 6: Native Metal MX conformance — SKIPPED\n");
    printf("  Reason: macOS 26.5 SDK has NO FP4/E2M1 type, NO E8M0 scale-plane,\n");
    printf("          NO multi-plane MX tensor support (Tier 4 absent)\n");
    printf("  Status: BLOCKED — requires macOS 27 SDK\n");
    printf("  Available types: int4b_format, uint4b_format (integer only)\n");
    printf("  These are representation-incompatible with E2M1 floating-point.\n");
    return 0; /* Not a failure — blocked by SDK version */
}

#endif /* DS4_METAL_SDK_NATIVE_LOWBIT */

/* ---------------------------------------------------------------------------
 * Main
 * --------------------------------------------------------------------------- */

int main(void) {
    int total_failures = 0;

    printf("=== DS4 Native-MX Conformance Probe ===\n");
    printf("SDK status: macOS 26.5 — Tier 4 (FP4/E8M0/MX) NOT AVAILABLE\n");
    printf("Running: CPU reference path only\n\n");

    total_failures += test_e2m1_all_codes();
    printf("\n");
    total_failures += test_decoder_consistency();
    printf("\n");
    total_failures += test_basis_vector_matmul();
    printf("\n");
    total_failures += test_nibble_ordering_convention();
    printf("\n");
    total_failures += test_multiblock_matmul();
    printf("\n");
    total_failures += test_native_mx_conformance();

    printf("\n=== Summary ===\n");
    if (total_failures == 0) {
        printf("All reference tests PASS\n");
        printf("Native-MX conformance: BLOCKED (macOS 27 SDK required)\n");
    } else {
        printf("%d reference test failure(s)\n", total_failures);
    }

    printf("\n=== Nibble-Ordering Question (for macOS 27) ===\n");
    printf("DS4 convention:  split-half\n");
    printf("  low nibbles  (qs[j] & 0x0F) → elements 0–15\n");
    printf("  high nibbles (qs[j] >> 4)    → elements 16–31\n");
    printf("Metal convention: UNKNOWN until Tier 4 APIs are available\n");
    printf("If Metal uses sequential (elem 2k in low, elem 2k+1 in high):\n");
    printf("  → nibble reorder required for zero-copy\n");
    printf("If Metal uses split-half (same as DS4):\n");
    printf("  → nibble ordering is compatible (still need plane separation)\n");

    return total_failures != 0 ? 1 : 0;
}
