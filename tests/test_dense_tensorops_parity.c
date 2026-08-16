/*
 * test_dense_tensorops_parity.c — Dense TensorOps vs legacy kernel parity (Task 3.5).
 *
 * For each production TensorOps quant variant (F16, Q8_0, Q4_0, Q4_K):
 *   - Generates deterministic random weight matrices and activation vectors.
 *   - Computes the reference result using the legacy Metal kernel (--quality mode).
 *   - Computes the TensorOps result using the production M5 path.
 *   - Compares outputs element-by-element.
 *
 * Tests multiple token counts: 32, 64, 128, 192 (split-prefix), 256
 * Tests alignment edge cases: various in_dim/out_dim combinations.
 *
 * Correctness contract per R12:
 *   - F16:  Bit-identical (same F16→F32 dequant, same reduction order in both paths).
 *   - Q8_0: Bit-identical for 32-aligned token counts on matching dispatch shapes.
 *   - Q4_0: Bit-identical for 32-aligned token counts.
 *   - Q4_K: Bit-identical for 32-aligned token counts.
 *
 * When TensorOps necessarily changes reduction order (e.g. split-prefix remainder),
 * we allow a numerically equivalent result with documented tolerance.
 *
 * Validates: Requirements R12
 */

#ifdef __APPLE__

#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* =========================================================================
 * Quant type constants (match ds4_metal.m enum).
 * =========================================================================
 */
#define DS4_QTYPE_Q4_0  2u
#define DS4_QTYPE_Q8_0  8u
#define DS4_QTYPE_Q4_K 12u

static int g_failures;
static int g_tests_run;
static int g_tests_skipped;

#define TEST(cond, ...) do { \
    g_tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: "); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, " (%s:%d)\n", __FILE__, __LINE__); \
        g_failures++; \
    } \
} while (0)

#define SKIP(reason) do { \
    g_tests_skipped++; \
    fprintf(stderr, "  SKIP: %s\n", reason); \
    return; \
} while (0)

/* Linker stub. */
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static uint64_t align_page(uint64_t bytes) {
    const uint64_t page = (uint64_t)getpagesize();
    return (bytes + page - 1u) / page * page;
}

/* =========================================================================
 * F16 helpers.
 * =========================================================================
 */
static uint16_t f32_to_f16(float f) {
    /* Simple software F32→F16 conversion for test generation. */
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 31u) & 1u;
    int32_t exp = (int32_t)((bits >> 23u) & 0xffu) - 127;
    uint32_t mantissa = bits & 0x7fffffu;

    uint16_t h;
    if (exp > 15) {
        h = (uint16_t)((sign << 15u) | 0x7c00u); /* inf */
    } else if (exp < -14) {
        /* subnormal or zero */
        int shift = -14 - exp;
        mantissa = (mantissa | 0x800000u) >> (shift + 13);
        h = (uint16_t)((sign << 15u) | (mantissa & 0x3ffu));
    } else {
        uint32_t hexp = (uint32_t)(exp + 15);
        uint32_t hmant = (mantissa + 0x1000u) >> 13u; /* round to nearest */
        if (hmant & 0x400u) { hexp++; hmant = 0; }
        h = (uint16_t)((sign << 15u) | (hexp << 10u) | (hmant & 0x3ffu));
    }
    return h;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15u) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10u) & 0x1fu;
    uint32_t mant = (uint32_t)(h & 0x3ffu);
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31u;
        } else {
            /* subnormal */
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1u; exp--; }
            mant &= 0x3ffu;
            bits = (sign << 31u) | ((uint32_t)(exp + 127 - 15) << 23u) | (mant << 13u);
        }
    } else if (exp == 0x1fu) {
        bits = (sign << 31u) | 0x7f800000u | (mant << 13u);
    } else {
        bits = (sign << 31u) | ((uint32_t)(exp + 127 - 15) << 23u) | (mant << 13u);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* =========================================================================
 * Deterministic weight generators.
 * =========================================================================
 */

/* Fill F16 weight matrix: out_dim rows × in_dim columns (row-major F16). */
static void fill_f16_weights(uint16_t *weights, uint32_t in_dim,
                             uint32_t out_dim, uint32_t seed) {
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 17u + i * 23u + (o ^ i) * 3u +
                                 seed * 29u) % 67u) - 33;
            weights[(uint64_t)o * in_dim + i] = f32_to_f16((float)v / 96.0f);
        }
    }
}

/* Fill Q8_0 weight matrix: out_dim rows, each row = (in_dim/32) blocks of 34 bytes. */
static void fill_q8_0_weights(uint8_t *weights, uint32_t in_dim,
                              uint32_t out_dim, uint32_t seed) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u +
                                     seed * 29u + ((o + seed) ^ k) * 5u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = f32_to_f16(amax / 127.0f);
            const float scale = f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

/* Fill Q4_0 weight matrix: out_dim rows, each row = (in_dim/32) blocks of 18 bytes.
 * Block format: F16 scale (2 bytes) + 16 nibble-pair bytes (32 nibbles). */
static void fill_q4_0_weights(uint8_t *weights, uint32_t in_dim,
                              uint32_t out_dim, uint32_t seed) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 18u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            /* Generate plausible 4-bit values (-8..7 range). */
            float amax = 0.0f;
            float vals[32];
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 13u + k * 19u + seed * 7u +
                                     (o ^ k ^ seed)) % 15u) - 7;
                vals[i] = (float)v / 8.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = f32_to_f16(amax / 7.0f);
            memcpy(row + b * 18u, &scale_bits, sizeof(scale_bits));
            uint8_t *qs = row + b * 18u + 2u;
            const float scale = f16_to_f32(scale_bits);
            for (uint32_t i = 0; i < 16; i++) {
                int q0 = scale != 0.0f ? (int)lrintf(vals[i * 2] / scale) + 8 : 8;
                int q1 = scale != 0.0f ? (int)lrintf(vals[i * 2 + 1] / scale) + 8 : 8;
                if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
                if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
                qs[i] = (uint8_t)(q0 | (q1 << 4u));
            }
        }
    }
}

/* Fill Q4_K weight matrix: out_dim rows.
 * Block size = 256 elements, block bytes = 144.
 * Layout per block (256 values): 2×F16 (d, dmin) + 12 bytes K-scale + 128 nibble bytes.
 * For test purposes we generate plausible block content. */
static void fill_q4_k_weights(uint8_t *weights, uint32_t in_dim,
                              uint32_t out_dim, uint32_t seed) {
    const uint32_t blocks_per_row = in_dim / 256u;
    const uint64_t row_bytes = (uint64_t)blocks_per_row * 144u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = row + (uint64_t)b * 144u;
            /* d (F16), dmin (F16) at offset 0..3. */
            const float d_val = 0.01f + 0.001f * (float)((o * 3u + b * 7u + seed) % 20u);
            const float dmin_val = 0.005f + 0.0005f * (float)((o + b * 11u + seed * 3u) % 15u);
            uint16_t d_bits = f32_to_f16(d_val);
            uint16_t dmin_bits = f32_to_f16(dmin_val);
            memcpy(block + 0, &d_bits, 2);
            memcpy(block + 2, &dmin_bits, 2);
            /* K-scale bytes (12 bytes at offset 4..15). */
            for (uint32_t i = 0; i < 12; i++) {
                block[4 + i] = (uint8_t)((o * 5u + b * 3u + i * 13u + seed) % 64u);
            }
            /* Quantized nibbles (128 bytes at offset 16..143). */
            for (uint32_t i = 0; i < 128; i++) {
                block[16 + i] = (uint8_t)((o * 7u + b * 11u + i * 17u + seed * 5u) % 256u);
            }
        }
    }
}

/* Fill activation vectors deterministically. */
static void fill_activations(float *x, uint32_t n_tok, uint32_t in_dim,
                             uint32_t seed) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i) +
                                 seed * 31u) % 71u) - 35;
            x[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
}

/* =========================================================================
 * Row byte calculators (match ds4_metal.m).
 * =========================================================================
 */
static uint64_t q8_0_row_bytes(uint32_t in_dim) {
    return ((uint64_t)(in_dim + 31u) / 32u) * 34u;
}

static uint64_t q4_0_row_bytes(uint32_t in_dim) {
    return ((uint64_t)(in_dim + 31u) / 32u) * 18u;
}

static uint64_t q4_k_row_bytes(uint32_t in_dim) {
    return ((uint64_t)in_dim / 256u) * 144u;
}

static uint64_t f16_row_bytes(uint32_t in_dim) {
    return (uint64_t)in_dim * sizeof(uint16_t);
}

/* =========================================================================
 * Parity test: run quality(true) → legacy result, quality(false) → TensorOps.
 * Compare and report.
 * =========================================================================
 */

typedef struct {
    const char *quant_name;
    uint32_t    quant_type;     /* 0 = F16 (uses ds4_gpu_matmul_f16_tensor) */
    uint32_t    in_dim;
    uint32_t    out_dim;
    uint32_t    n_tok;
    uint32_t    seed;
    /* Whether bit-identity is expected (vs numerically equivalent). */
    bool        expect_bit_identical;
    /* Tolerance for numerically equivalent (ignored if bit-identical expected). */
    float       max_abs_tol;
    float       max_rel_tol;
} parity_test_case;

static void run_parity_test(const parity_test_case *tc) {
    fprintf(stderr, "  %s in=%u out=%u tok=%u seed=%u ...",
            tc->quant_name, tc->in_dim, tc->out_dim, tc->n_tok, tc->seed);

    /* Compute row bytes and weight allocation. */
    uint64_t row_bytes_val;
    if (tc->quant_type == 0) {
        row_bytes_val = f16_row_bytes(tc->in_dim);
    } else if (tc->quant_type == DS4_QTYPE_Q8_0) {
        row_bytes_val = q8_0_row_bytes(tc->in_dim);
    } else if (tc->quant_type == DS4_QTYPE_Q4_0) {
        row_bytes_val = q4_0_row_bytes(tc->in_dim);
    } else if (tc->quant_type == DS4_QTYPE_Q4_K) {
        row_bytes_val = q4_k_row_bytes(tc->in_dim);
    } else {
        fprintf(stderr, " unknown quant\n");
        g_failures++;
        return;
    }

    const uint64_t weight_bytes = (uint64_t)tc->out_dim * row_bytes_val;
    const uint64_t weight_alloc = align_page(weight_bytes);
    const uint64_t x_bytes = (uint64_t)tc->n_tok * tc->in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)tc->n_tok * tc->out_dim * sizeof(float);
    const uint64_t out_count = (uint64_t)tc->n_tok * tc->out_dim;

    void *weights_raw = NULL;
    if (posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) != 0) {
        fprintf(stderr, " alloc fail\n");
        g_failures++;
        return;
    }
    memset(weights_raw, 0, (size_t)weight_alloc);

    /* Fill weights. */
    if (tc->quant_type == 0) {
        fill_f16_weights((uint16_t *)weights_raw, tc->in_dim, tc->out_dim, tc->seed);
    } else if (tc->quant_type == DS4_QTYPE_Q8_0) {
        fill_q8_0_weights((uint8_t *)weights_raw, tc->in_dim, tc->out_dim, tc->seed);
    } else if (tc->quant_type == DS4_QTYPE_Q4_0) {
        fill_q4_0_weights((uint8_t *)weights_raw, tc->in_dim, tc->out_dim, tc->seed);
    } else if (tc->quant_type == DS4_QTYPE_Q4_K) {
        fill_q4_k_weights((uint8_t *)weights_raw, tc->in_dim, tc->out_dim, tc->seed);
    }

    /* Allocate GPU tensors. */
    ds4_gpu_tensor *x_tensor = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out_ref_tensor = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *out_test_tensor = ds4_gpu_tensor_alloc(out_bytes);
    if (!x_tensor || !out_ref_tensor || !out_test_tensor) {
        fprintf(stderr, " tensor alloc fail\n");
        g_failures++;
        ds4_gpu_tensor_free(x_tensor);
        ds4_gpu_tensor_free(out_ref_tensor);
        ds4_gpu_tensor_free(out_test_tensor);
        free(weights_raw);
        return;
    }

    /* Fill activations. */
    float *x_host = malloc((size_t)x_bytes);
    float *out_ref_host = malloc((size_t)out_bytes);
    float *out_test_host = malloc((size_t)out_bytes);
    if (!x_host || !out_ref_host || !out_test_host) {
        fprintf(stderr, " host alloc fail\n");
        g_failures++;
        free(x_host); free(out_ref_host); free(out_test_host);
        ds4_gpu_tensor_free(x_tensor);
        ds4_gpu_tensor_free(out_ref_tensor);
        ds4_gpu_tensor_free(out_test_tensor);
        free(weights_raw);
        return;
    }

    fill_activations(x_host, tc->n_tok, tc->in_dim, tc->seed);

    /* Upload inputs. */
    if (!ds4_gpu_tensor_write(x_tensor, 0, x_host, x_bytes)) {
        fprintf(stderr, " upload fail\n");
        g_failures++;
        goto cleanup;
    }
    if (!ds4_gpu_set_model_map(weights_raw, weight_alloc)) {
        fprintf(stderr, " model map fail\n");
        g_failures++;
        goto cleanup;
    }

    /* --- Reference: quality mode (legacy kernel) --- */
    ds4_gpu_set_quality(true);
    memset(out_ref_host, 0xcc, (size_t)out_bytes);
    if (!ds4_gpu_tensor_fill_f32(out_ref_tensor, -999.0f, out_count)) {
        fprintf(stderr, " ref fill fail\n");
        g_failures++;
        goto cleanup;
    }

    int ref_ok;
    if (tc->quant_type == 0) {
        ref_ok = ds4_gpu_matmul_f16_tensor(out_ref_tensor, weights_raw,
                                           weight_alloc, 0, tc->in_dim,
                                           tc->out_dim, x_tensor, tc->n_tok);
    } else if (tc->quant_type == DS4_QTYPE_Q8_0) {
        ref_ok = ds4_gpu_matmul_q8_0_tensor(out_ref_tensor, weights_raw,
                                            weight_alloc, 0, tc->in_dim,
                                            tc->out_dim, x_tensor, tc->n_tok);
    } else {
        ref_ok = ds4_gpu_matmul_quant_tensor(out_ref_tensor, weights_raw,
                                             weight_alloc, 0, tc->quant_type,
                                             tc->in_dim, tc->out_dim,
                                             x_tensor, tc->n_tok);
    }
    if (!ref_ok) {
        fprintf(stderr, " ref dispatch fail\n");
        g_failures++;
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(out_ref_tensor, 0, out_ref_host, out_bytes)) {
        fprintf(stderr, " ref read fail\n");
        g_failures++;
        goto cleanup;
    }

    /* --- Test: production TensorOps path --- */
    ds4_gpu_set_quality(false);
    memset(out_test_host, 0xdd, (size_t)out_bytes);
    if (!ds4_gpu_tensor_fill_f32(out_test_tensor, -999.0f, out_count)) {
        fprintf(stderr, " test fill fail\n");
        g_failures++;
        goto cleanup;
    }

    int test_ok;
    if (tc->quant_type == 0) {
        test_ok = ds4_gpu_matmul_f16_tensor(out_test_tensor, weights_raw,
                                            weight_alloc, 0, tc->in_dim,
                                            tc->out_dim, x_tensor, tc->n_tok);
    } else if (tc->quant_type == DS4_QTYPE_Q8_0) {
        test_ok = ds4_gpu_matmul_q8_0_tensor(out_test_tensor, weights_raw,
                                             weight_alloc, 0, tc->in_dim,
                                             tc->out_dim, x_tensor, tc->n_tok);
    } else {
        test_ok = ds4_gpu_matmul_quant_tensor(out_test_tensor, weights_raw,
                                              weight_alloc, 0, tc->quant_type,
                                              tc->in_dim, tc->out_dim,
                                              x_tensor, tc->n_tok);
    }
    if (!test_ok) {
        fprintf(stderr, " test dispatch fail\n");
        g_failures++;
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(out_test_tensor, 0, out_test_host, out_bytes)) {
        fprintf(stderr, " test read fail\n");
        g_failures++;
        goto cleanup;
    }

    /* Restore quality for subsequent tests. */
    ds4_gpu_set_quality(false);

    /* --- Compare --- */
    {
        const bool bit_identical =
            memcmp(out_ref_host, out_test_host, (size_t)out_bytes) == 0;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        uint64_t max_abs_idx = 0;
        uint64_t nan_count = 0;

        for (uint64_t i = 0; i < out_count; i++) {
            if (!isfinite(out_test_host[i]) || !isfinite(out_ref_host[i])) {
                nan_count++;
                continue;
            }
            const float err = fabsf(out_test_host[i] - out_ref_host[i]);
            if (err > max_abs_err) {
                max_abs_err = err;
                max_abs_idx = i;
            }
            const float ref_mag = fabsf(out_ref_host[i]);
            if (ref_mag > 1e-8f) {
                const float rel = err / ref_mag;
                if (rel > max_rel_err) max_rel_err = rel;
            }
        }

        TEST(nan_count == 0,
             "%s in=%u out=%u tok=%u: %llu NaN/Inf values",
             tc->quant_name, tc->in_dim, tc->out_dim, tc->n_tok,
             (unsigned long long)nan_count);

        if (tc->expect_bit_identical) {
            TEST(bit_identical,
                 "%s in=%u out=%u tok=%u: expected bit-identical, "
                 "max_abs=%g max_rel=%g at idx=%llu",
                 tc->quant_name, tc->in_dim, tc->out_dim, tc->n_tok,
                 max_abs_err, max_rel_err,
                 (unsigned long long)max_abs_idx);
            if (bit_identical) {
                fprintf(stderr, " BIT-IDENTICAL ✓\n");
            } else {
                fprintf(stderr, " MISMATCH max_abs=%g max_rel=%g\n",
                        max_abs_err, max_rel_err);
            }
        } else {
            TEST(max_abs_err <= tc->max_abs_tol,
                 "%s in=%u out=%u tok=%u: max_abs=%g exceeds tol=%g at idx=%llu",
                 tc->quant_name, tc->in_dim, tc->out_dim, tc->n_tok,
                 max_abs_err, tc->max_abs_tol,
                 (unsigned long long)max_abs_idx);
            TEST(max_rel_err <= tc->max_rel_tol,
                 "%s in=%u out=%u tok=%u: max_rel=%g exceeds tol=%g",
                 tc->quant_name, tc->in_dim, tc->out_dim, tc->n_tok,
                 max_rel_err, tc->max_rel_tol);
            fprintf(stderr, " max_abs=%g max_rel=%g %s\n",
                    max_abs_err, max_rel_err,
                    (max_abs_err <= tc->max_abs_tol &&
                     max_rel_err <= tc->max_rel_tol) ? "OK" : "FAIL");
        }
    }

cleanup:
    free(x_host);
    free(out_ref_host);
    free(out_test_host);
    ds4_gpu_tensor_free(x_tensor);
    ds4_gpu_tensor_free(out_ref_tensor);
    ds4_gpu_tensor_free(out_test_tensor);
    free(weights_raw);
}

/* =========================================================================
 * Test case tables.
 *
 * Token counts: 32, 64, 128 exercise the three tile sizes (NR1).
 *               192 exercises split-prefix on Q8_0 (numerically equiv).
 *               256 exercises the 128-token tile with 2 threadgroups.
 *
 * Dimension choices:
 *   - in_dim must be 64-aligned for Q8/Q4 TensorOps, 32-aligned for F16.
 *   - out_dim must be 64-aligned for all TensorOps paths.
 *   - Q4_K requires in_dim divisible by 256.
 * =========================================================================
 */

/* F16 parity tests: bit-identical expected for 32-aligned token counts. */
static const parity_test_case f16_cases[] = {
    { "F16", 0, 128,  64,  32, 1, true, 0, 0 },
    { "F16", 0, 128,  64,  64, 2, true, 0, 0 },
    { "F16", 0, 128,  64, 128, 3, true, 0, 0 },
    { "F16", 0, 256, 128,  32, 4, true, 0, 0 },
    { "F16", 0, 256, 128,  64, 5, true, 0, 0 },
    { "F16", 0, 256, 128, 128, 6, true, 0, 0 },
    { "F16", 0, 256, 128, 256, 7, true, 0, 0 },
    { "F16", 0, 2048, 512, 64, 8, true, 0, 0 },
    { "F16", 0, 2048, 512, 128, 9, true, 0, 0 },
};

/* Q8_0 parity tests: bit-identical for aligned counts; numerically equiv for split-prefix. */
static const parity_test_case q8_0_cases[] = {
    { "Q8_0", DS4_QTYPE_Q8_0,  64,  64,  32, 10, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0,  64,  64,  64, 11, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0,  64,  64, 128, 12, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0, 128,  64,  32, 13, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0, 128,  64,  64, 14, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0, 128,  64, 128, 15, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0, 128, 128, 256, 16, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0, 256, 128,  32, 17, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0, 256, 128,  64, 18, true, 0, 0 },
    { "Q8_0", DS4_QTYPE_Q8_0, 2048, 512, 64, 19, true, 0, 0 },
    /* Split-prefix: 192 tokens where (192 % 32) == 0 but exercises a different
     * tile selection path.  This should still be bit-identical since 192 is
     * 32-aligned and goes through a 64-token tile selection. */
    { "Q8_0", DS4_QTYPE_Q8_0, 128,  64, 192, 20, true, 0, 0 },
};

/* Q4_0 parity tests: bit-identical for all 32-aligned token counts. */
static const parity_test_case q4_0_cases[] = {
    { "Q4_0", DS4_QTYPE_Q4_0,  64,  64,  32, 30, true, 0, 0 },
    { "Q4_0", DS4_QTYPE_Q4_0,  64,  64,  64, 31, true, 0, 0 },
    { "Q4_0", DS4_QTYPE_Q4_0,  64,  64, 128, 32, true, 0, 0 },
    { "Q4_0", DS4_QTYPE_Q4_0, 128,  64,  32, 33, true, 0, 0 },
    { "Q4_0", DS4_QTYPE_Q4_0, 128, 128, 256, 34, true, 0, 0 },
    { "Q4_0", DS4_QTYPE_Q4_0, 256, 128,  64, 35, true, 0, 0 },
    { "Q4_0", DS4_QTYPE_Q4_0, 2048, 512, 64, 36, true, 0, 0 },
};

/* Q4_K parity tests: in_dim must be 256-aligned, out_dim 64-aligned. */
static const parity_test_case q4_k_cases[] = {
    { "Q4_K", DS4_QTYPE_Q4_K, 256,  64,  32, 40, true, 0, 0 },
    { "Q4_K", DS4_QTYPE_Q4_K, 256,  64,  64, 41, true, 0, 0 },
    { "Q4_K", DS4_QTYPE_Q4_K, 256,  64, 128, 42, true, 0, 0 },
    { "Q4_K", DS4_QTYPE_Q4_K, 256, 128, 256, 43, true, 0, 0 },
    { "Q4_K", DS4_QTYPE_Q4_K, 512, 128,  32, 44, true, 0, 0 },
    { "Q4_K", DS4_QTYPE_Q4_K, 512, 128,  64, 45, true, 0, 0 },
    { "Q4_K", DS4_QTYPE_Q4_K, 2048, 512, 64, 46, true, 0, 0 },
};

/* =========================================================================
 * main
 * =========================================================================
 */
int main(int argc, char **argv) {
    (void)argc; (void)argv;

    fprintf(stderr, "test_dense_tensorops_parity: initializing Metal...\n");

    if (!ds4_gpu_init()) {
        fprintf(stderr, "test_dense_tensorops_parity: no Metal device, skipping.\n");
        return 0;
    }

    /* Check whether TensorOps is available on this device. */
    const int tensorops_available = ds4_gpu_tensorops_runtime();
    if (!tensorops_available) {
        fprintf(stderr,
                "test_dense_tensorops_parity: TensorOps runtime unavailable, "
                "skipping parity tests (legacy path only).\n");
        ds4_gpu_cleanup();
        return 0;
    }

    fprintf(stderr,
            "test_dense_tensorops_parity: TensorOps runtime active, "
            "running A/B parity tests...\n\n");

    /* F16 tests. */
    fprintf(stderr, "=== F16 TensorOps vs Legacy ===\n");
    for (size_t i = 0; i < sizeof(f16_cases) / sizeof(f16_cases[0]); i++) {
        run_parity_test(&f16_cases[i]);
    }

    /* Q8_0 tests. */
    fprintf(stderr, "\n=== Q8_0 TensorOps vs Legacy ===\n");
    for (size_t i = 0; i < sizeof(q8_0_cases) / sizeof(q8_0_cases[0]); i++) {
        run_parity_test(&q8_0_cases[i]);
    }

    /* Q4_0 tests. */
    fprintf(stderr, "\n=== Q4_0 TensorOps vs Legacy ===\n");
    for (size_t i = 0; i < sizeof(q4_0_cases) / sizeof(q4_0_cases[0]); i++) {
        run_parity_test(&q4_0_cases[i]);
    }

    /* Q4_K tests. */
    fprintf(stderr, "\n=== Q4_K TensorOps vs Legacy ===\n");
    for (size_t i = 0; i < sizeof(q4_k_cases) / sizeof(q4_k_cases[0]); i++) {
        run_parity_test(&q4_k_cases[i]);
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "test_dense_tensorops_parity: %d tests, %d failures, %d skipped\n",
            g_tests_run, g_failures, g_tests_skipped);

    ds4_gpu_cleanup();

    if (g_failures) {
        fprintf(stderr, "test_dense_tensorops_parity: FAILED\n");
        return 1;
    }
    fprintf(stderr, "test_dense_tensorops_parity: PASSED\n");
    return 0;
}

#else /* !__APPLE__ */

#include <stdio.h>

int main(void) {
    fprintf(stderr, "test_dense_tensorops_parity: not Apple, skipping.\n");
    return 0;
}

#endif /* __APPLE__ */
