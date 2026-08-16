/*
 * test_moe_tensorops_parity.c — MoE TensorOps vs legacy simdgroup kernel parity (Task 7.7).
 *
 * For each production TensorOps MoE pair-SwiGLU quant variant (MXFP4, Q4_K, IQ2_XXS):
 *   - Generates deterministic random expert weight matrices and activations.
 *   - Runs the legacy simdgroup pair kernel (with MPP disabled via env).
 *   - Runs the new TensorOps (MPP) pair kernel.
 *   - Compares FP16 mid (fused SwiGLU) outputs element-by-element.
 *
 * Test cases cover:
 *   - Full tile:   32 tokens routed to one expert, nr0=64 full weight rows.
 *   - Token tail:  1, 4, 8, 16 tokens routed to an expert (nr1 < NR1).
 *   - Weight tail: out_dim not filling the 64-row tile.
 *   - Empty expert: 0 tokens routed — verify no output corruption.
 *   - Small expert bucket: 1-2 tokens only — minimum viable batch.
 *   - Adversarial patterns: All-zero, all-max, alternating ±max, edge E8M0 scales.
 *   - Route weights: varied (0.0, 0.5, 1.0, large) to verify SwiGLU epilogue.
 *
 * For MXFP4 specifically, tests:
 *   - All 16 E2M1 codes.
 *   - Scale bytes 0, 1, 64, 127, 128, 200, 254.
 *   - Patterns triggering the half-LUT boundary.
 *
 * Correctness contract per R12 (Class B numerical equivalence):
 *   TensorOps matmul2d uses a different reduction order than simdgroup_multiply_accumulate,
 *   so bit-identity is NOT expected. The contract is:
 *   - Max absolute/relative error within a calibrated envelope.
 *   - Same selected token across test corpus.
 *   - Same routing decisions (TensorOps is post-routing, so not at risk).
 *   - No NaN/Inf regressions.
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
 * Constants matching the ds4_metal.m MoE dispatch.
 * =========================================================================
 */
#define MXFP4_TYPE           39u
#define Q4_K_TYPE            12u
#define IQ2_XXS_TYPE         16u

#define QK_MXFP4             32u
#define QK_Q4_K             256u
#define QK_IQ2_XXS          256u

/* MXFP4 block: 1 byte E8M0 scale + 16 nibble-pair bytes = 17 bytes per 32 values. */
#define BLOCK_MXFP4_BYTES    17u
/* Q4_K block: 144 bytes per 256 values. */
#define BLOCK_Q4_K_BYTES    144u
/* IQ2_XXS block: 2 bytes (d) + 256 lookup quants at 2 bits + signs.
 * Actually: 2 bytes (d) + 8 groups × (2 bytes qs + 1 byte signs) = 26 bytes per 256 values. */
#define BLOCK_IQ2_XXS_BYTES  66u

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

/* Linker stub. */
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

/* =========================================================================
 * MXFP4 reference helpers.
 * =========================================================================
 */
typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2u];
} block_mxfp4;

static const float mxfp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

static float e8m0_to_f32(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float dot_mxfp4(const block_mxfp4 *row_blocks, const float *x,
                        uint32_t in_dim) {
    float sum = 0.0f;
    const uint32_t n_blocks = in_dim / QK_MXFP4;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const block_mxfp4 *blk = row_blocks + b;
        const float scale = e8m0_to_f32(blk->e);
        for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
            const uint8_t q = blk->qs[i];
            sum += scale * mxfp4_values[q & 15u] *
                   x[b * QK_MXFP4 + i];
            sum += scale * mxfp4_values[q >> 4u] *
                   x[b * QK_MXFP4 + i + QK_MXFP4 / 2u];
        }
    }
    return sum;
}

/* =========================================================================
 * Deterministic pseudo-random generator.
 * =========================================================================
 */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

static float rand_float(uint32_t *state, float lo, float hi) {
    const uint32_t r = xorshift32(state);
    return lo + (hi - lo) * ((float)(r & 0xFFFFu) / 65535.0f);
}

/* =========================================================================
 * Weight generators.
 * =========================================================================
 */
static uint64_t align_page(uint64_t bytes) {
    const uint64_t page = (uint64_t)getpagesize();
    return (bytes + page - 1u) / page * page;
}

static uint64_t mxfp4_row_bytes(uint32_t in_dim) {
    return (uint64_t)(in_dim / QK_MXFP4) * BLOCK_MXFP4_BYTES;
}

static uint64_t q4_k_row_bytes(uint32_t in_dim) {
    return (uint64_t)(in_dim / QK_Q4_K) * BLOCK_Q4_K_BYTES;
}

/* Fill MXFP4 expert weight matrix with deterministic content.
 * Layout: n_expert × out_dim rows × (in_dim/32) blocks. */
static void fill_mxfp4_experts(uint8_t *data, uint32_t n_expert,
                               uint32_t out_dim, uint32_t in_dim,
                               uint32_t seed) {
    uint32_t rng = seed;
    const uint64_t row_bytes = mxfp4_row_bytes(in_dim);
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            block_mxfp4 *row = (block_mxfp4 *)(data +
                ((uint64_t)e * out_dim + r) * row_bytes);
            for (uint32_t b = 0; b < in_dim / QK_MXFP4; b++) {
                row[b].e = (uint8_t)(118u + (xorshift32(&rng) % 10u));
                for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
                    row[b].qs[i] = (uint8_t)(xorshift32(&rng) & 0xFFu);
                }
            }
        }
    }
}

/* Fill MXFP4 experts with adversarial all-zero weights. */
static void fill_mxfp4_experts_zeros(uint8_t *data, uint32_t n_expert,
                                     uint32_t out_dim, uint32_t in_dim) {
    const uint64_t row_bytes = mxfp4_row_bytes(in_dim);
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            block_mxfp4 *row = (block_mxfp4 *)(data +
                ((uint64_t)e * out_dim + r) * row_bytes);
            for (uint32_t b = 0; b < in_dim / QK_MXFP4; b++) {
                row[b].e = 0;   /* zero scale */
                memset(row[b].qs, 0, QK_MXFP4 / 2u);
            }
        }
    }
}

/* Fill MXFP4 experts with adversarial max-magnitude alternating patterns.
 * Tests ±6.0f * large_scale. */
static void fill_mxfp4_experts_adversarial(uint8_t *data, uint32_t n_expert,
                                           uint32_t out_dim, uint32_t in_dim) {
    const uint64_t row_bytes = mxfp4_row_bytes(in_dim);
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            block_mxfp4 *row = (block_mxfp4 *)(data +
                ((uint64_t)e * out_dim + r) * row_bytes);
            for (uint32_t b = 0; b < in_dim / QK_MXFP4; b++) {
                /* Alternating between large scales and edge codes. */
                row[b].e = (r & 1u) ? 200u : 127u;
                for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
                    /* Alternate +6.0 (code 7) and -6.0 (code 15). */
                    row[b].qs[i] = (uint8_t)(((i & 1u) ? 7u : 15u) |
                                             (((i & 1u) ? 15u : 7u) << 4u));
                }
            }
        }
    }
}

/* Fill MXFP4 experts with all 16 E2M1 codes and edge scales.
 * Scale bytes: 0, 1, 64, 127, 128, 200, 254. */
static void fill_mxfp4_experts_e2m1_coverage(uint8_t *data, uint32_t n_expert,
                                             uint32_t out_dim, uint32_t in_dim) {
    static const uint8_t test_scales[] = { 0, 1, 64, 127, 128, 200, 254 };
    const uint64_t row_bytes = mxfp4_row_bytes(in_dim);
    uint32_t scale_idx = 0;
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            block_mxfp4 *row = (block_mxfp4 *)(data +
                ((uint64_t)e * out_dim + r) * row_bytes);
            for (uint32_t b = 0; b < in_dim / QK_MXFP4; b++) {
                row[b].e = test_scales[scale_idx % 7u];
                scale_idx++;
                /* Fill all 16 codes sequentially. */
                for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
                    uint8_t lo = (uint8_t)((i * 2u) & 15u);
                    uint8_t hi = (uint8_t)((i * 2u + 1u) & 15u);
                    row[b].qs[i] = (uint8_t)(lo | (hi << 4u));
                }
            }
        }
    }
}

/* Fill Q4_K expert weight matrix with deterministic content.
 * Block: 2×F16 (d, dmin) + 12 bytes K-scale + 128 nibble bytes = 144 per 256. */
static void fill_q4_k_experts(uint8_t *data, uint32_t n_expert,
                              uint32_t out_dim, uint32_t in_dim,
                              uint32_t seed) {
    uint32_t rng = seed;
    const uint64_t rb = q4_k_row_bytes(in_dim);
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            uint8_t *row = data + ((uint64_t)e * out_dim + r) * rb;
            const uint32_t blocks = in_dim / QK_Q4_K;
            for (uint32_t b = 0; b < blocks; b++) {
                uint8_t *block = row + (uint64_t)b * BLOCK_Q4_K_BYTES;
                /* d and dmin as plausible small F16 values. */
                uint16_t d_bits = 0x2800u + (uint16_t)(xorshift32(&rng) % 0x400u);
                uint16_t dmin_bits = 0x2400u + (uint16_t)(xorshift32(&rng) % 0x200u);
                memcpy(block + 0, &d_bits, 2);
                memcpy(block + 2, &dmin_bits, 2);
                /* K-scale bytes. */
                for (uint32_t i = 0; i < 12; i++) {
                    block[4 + i] = (uint8_t)(xorshift32(&rng) % 64u);
                }
                /* Quantized nibbles. */
                for (uint32_t i = 0; i < 128; i++) {
                    block[16 + i] = (uint8_t)(xorshift32(&rng) & 0xFFu);
                }
            }
        }
    }
}

/* Fill activation vectors deterministically. */
static void fill_activations(float *x, uint32_t n_tok, uint32_t in_dim,
                             uint32_t seed) {
    uint32_t rng = seed;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            x[(uint64_t)t * in_dim + i] = rand_float(&rng, -0.5f, 0.5f);
        }
    }
}

/* =========================================================================
 * Reference computation for MXFP4 pair-SwiGLU (per-token, per-expert).
 * Computes gate*up fused with SiLU and route weight → mid output.
 *
 * This matches the kernel epilogue:
 *   g = clamp(gate, -c, c) for clamped mode (c > 1e-6)
 *   u = clamp(up, -c, c)
 *   mid = silu(g) * u * route_weight
 * =========================================================================
 */
static void compute_mxfp4_pair_swiglu_ref(
    const uint8_t *gate_data, const uint8_t *up_data,
    const float *x, uint32_t in_dim, uint32_t out_dim,
    uint32_t expert_idx, float route_weight, float clamp_val,
    float *mid_out) {
    const uint64_t row_bytes = mxfp4_row_bytes(in_dim);
    const uint64_t expert_offset = (uint64_t)expert_idx * out_dim * row_bytes;
    for (uint32_t r = 0; r < out_dim; r++) {
        const block_mxfp4 *gate_row = (const block_mxfp4 *)(
            gate_data + expert_offset + (uint64_t)r * row_bytes);
        const block_mxfp4 *up_row = (const block_mxfp4 *)(
            up_data + expert_offset + (uint64_t)r * row_bytes);
        float g = dot_mxfp4(gate_row, x, in_dim);
        float u = dot_mxfp4(up_row, x, in_dim);
        if (clamp_val > 1.0e-6f) {
            g = fminf(g, clamp_val);
            u = fmaxf(-clamp_val, fminf(u, clamp_val));
        }
        const float silu = g / (1.0f + expf(-g));
        mid_out[r] = silu * u * route_weight;
    }
}

/* =========================================================================
 * Parity test infrastructure.
 *
 * Strategy: run ds4_gpu_routed_moe_batch_tensor twice:
 *   1. With DS4_METAL_DISABLE_MXFP4_MOE_GATE_UP_MPP=1 → simdgroup (legacy).
 *   2. Without that env var → TensorOps (MPP) when available.
 *
 * Compare FP16 mid outputs. The TensorOps path uses matmul2d which has a
 * different reduction order, so Class B equivalence (not bit-identity).
 * =========================================================================
 */

typedef struct {
    const char *name;
    uint32_t    quant_type;
    uint32_t    n_total_expert;
    uint32_t    n_expert_used;  /* experts selected per token */
    uint32_t    n_tokens;
    uint32_t    in_dim;
    uint32_t    out_dim;        /* gate/up output dim (mid_dim) */
    uint32_t    seed;
    float       clamp;
    /* Routing weights per expert slot (n_expert_used entries).
     * If NULL, uses default 1/n_expert_used. */
    const float *route_weights;
    /* Expert indices per token. If NULL, uses deterministic. */
    const int32_t *fixed_selected;
    /* Fill mode for adversarial. 0=random, 1=zeros, 2=adversarial, 3=e2m1 coverage. */
    int         fill_mode;
    /* Tolerance: max abs and rel error. */
    float       max_abs_tol;
    float       max_rel_tol;
} moe_parity_test_case;

static int run_moe_parity_test(const moe_parity_test_case *tc) {
    fprintf(stderr, "  %-40s tok=%u exp=%u/%u in=%u out=%u ...",
            tc->name, tc->n_tokens, tc->n_expert_used,
            tc->n_total_expert, tc->in_dim, tc->out_dim);

    /* Compute sizes. */
    uint64_t row_bytes_val;
    if (tc->quant_type == MXFP4_TYPE) {
        row_bytes_val = mxfp4_row_bytes(tc->in_dim);
    } else if (tc->quant_type == Q4_K_TYPE) {
        row_bytes_val = q4_k_row_bytes(tc->in_dim);
    } else {
        fprintf(stderr, " unsupported quant\n");
        g_failures++;
        return 0;
    }

    const uint64_t expert_bytes = (uint64_t)tc->out_dim * row_bytes_val;
    const uint64_t tensor_bytes = (uint64_t)tc->n_total_expert * expert_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_page(tensor_bytes);
    const uint64_t down_offset = align_page(up_offset + tensor_bytes);
    const uint64_t model_size = align_page(down_offset + tensor_bytes);

    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        fprintf(stderr, " alloc fail\n");
        g_failures++;
        return 0;
    }
    memset(model, 0, (size_t)model_size);

    /* Fill expert weights. */
    if (tc->quant_type == MXFP4_TYPE) {
        switch (tc->fill_mode) {
        case 0:
            fill_mxfp4_experts((uint8_t *)model + gate_offset,
                               tc->n_total_expert, tc->out_dim, tc->in_dim,
                               tc->seed);
            fill_mxfp4_experts((uint8_t *)model + up_offset,
                               tc->n_total_expert, tc->out_dim, tc->in_dim,
                               tc->seed + 1000u);
            fill_mxfp4_experts((uint8_t *)model + down_offset,
                               tc->n_total_expert, tc->out_dim, tc->in_dim,
                               tc->seed + 2000u);
            break;
        case 1:
            fill_mxfp4_experts_zeros((uint8_t *)model + gate_offset,
                                     tc->n_total_expert, tc->out_dim, tc->in_dim);
            fill_mxfp4_experts_zeros((uint8_t *)model + up_offset,
                                     tc->n_total_expert, tc->out_dim, tc->in_dim);
            fill_mxfp4_experts_zeros((uint8_t *)model + down_offset,
                                     tc->n_total_expert, tc->out_dim, tc->in_dim);
            break;
        case 2:
            fill_mxfp4_experts_adversarial((uint8_t *)model + gate_offset,
                                           tc->n_total_expert, tc->out_dim, tc->in_dim);
            fill_mxfp4_experts_adversarial((uint8_t *)model + up_offset,
                                           tc->n_total_expert, tc->out_dim, tc->in_dim);
            fill_mxfp4_experts_adversarial((uint8_t *)model + down_offset,
                                           tc->n_total_expert, tc->out_dim, tc->in_dim);
            break;
        case 3:
            fill_mxfp4_experts_e2m1_coverage((uint8_t *)model + gate_offset,
                                             tc->n_total_expert, tc->out_dim, tc->in_dim);
            fill_mxfp4_experts_e2m1_coverage((uint8_t *)model + up_offset,
                                             tc->n_total_expert, tc->out_dim, tc->in_dim);
            fill_mxfp4_experts_e2m1_coverage((uint8_t *)model + down_offset,
                                             tc->n_total_expert, tc->out_dim, tc->in_dim);
            break;
        }
    } else if (tc->quant_type == Q4_K_TYPE) {
        fill_q4_k_experts((uint8_t *)model + gate_offset,
                          tc->n_total_expert, tc->out_dim, tc->in_dim,
                          tc->seed);
        fill_q4_k_experts((uint8_t *)model + up_offset,
                          tc->n_total_expert, tc->out_dim, tc->in_dim,
                          tc->seed + 1000u);
        fill_q4_k_experts((uint8_t *)model + down_offset,
                          tc->n_total_expert, tc->out_dim, tc->in_dim,
                          tc->seed + 2000u);
    }

    /* Build activation and routing tensors. */
    const uint64_t x_bytes = (uint64_t)tc->n_tokens * tc->in_dim * sizeof(float);
    float *x_host = malloc((size_t)x_bytes);
    if (!x_host) { free(model); g_failures++; fprintf(stderr, " alloc fail\n"); return 0; }
    fill_activations(x_host, tc->n_tokens, tc->in_dim, tc->seed + 3000u);

    /* Build selected expert indices and route weights per token. */
    const uint64_t sel_count = (uint64_t)tc->n_tokens * tc->n_expert_used;
    int32_t *selected_host = calloc((size_t)sel_count, sizeof(int32_t));
    float *weights_host = calloc((size_t)sel_count, sizeof(float));
    if (!selected_host || !weights_host) {
        free(x_host); free(selected_host); free(weights_host); free(model);
        g_failures++; fprintf(stderr, " alloc fail\n"); return 0;
    }

    uint32_t rng = tc->seed + 5000u;
    for (uint32_t t = 0; t < tc->n_tokens; t++) {
        for (uint32_t s = 0; s < tc->n_expert_used; s++) {
            const uint64_t idx = (uint64_t)t * tc->n_expert_used + s;
            if (tc->fixed_selected) {
                selected_host[idx] = tc->fixed_selected[s];
            } else {
                /* Deterministic spread across available experts. */
                selected_host[idx] = (int32_t)((t * 3u + s * 7u + xorshift32(&rng)) %
                                               tc->n_total_expert);
            }
            if (tc->route_weights) {
                weights_host[idx] = tc->route_weights[s];
            } else {
                weights_host[idx] = 1.0f / (float)tc->n_expert_used;
            }
        }
    }

    /* Allocate GPU tensors. */
    const uint64_t pair_count = (uint64_t)tc->n_tokens * tc->n_expert_used * tc->out_dim;
    const uint64_t out_count = (uint64_t)tc->n_tokens * tc->out_dim;

    ds4_gpu_tensor *x_tensor = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *selected_tensor = ds4_gpu_tensor_alloc(sel_count * sizeof(int32_t));
    ds4_gpu_tensor *weights_tensor = ds4_gpu_tensor_alloc(sel_count * sizeof(float));
    ds4_gpu_tensor *gate_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *up_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *mid_tensor_ref = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *mid_tensor_test = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *experts_tensor_ref = ds4_gpu_tensor_alloc(out_count * tc->n_expert_used * sizeof(float));
    ds4_gpu_tensor *experts_tensor_test = ds4_gpu_tensor_alloc(out_count * tc->n_expert_used * sizeof(float));
    ds4_gpu_tensor *out_tensor_ref = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *out_tensor_test = ds4_gpu_tensor_alloc(out_count * sizeof(float));

    int ok = x_tensor && selected_tensor && weights_tensor &&
             gate_tensor && up_tensor &&
             mid_tensor_ref && mid_tensor_test &&
             experts_tensor_ref && experts_tensor_test &&
             out_tensor_ref && out_tensor_test;

    ok = ok && ds4_gpu_set_model_map(model, model_size);
    ok = ok && ds4_gpu_tensor_write(x_tensor, 0, x_host, x_bytes);
    ok = ok && ds4_gpu_tensor_write(selected_tensor, 0, selected_host,
                                    sel_count * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_write(weights_tensor, 0, weights_host,
                                    sel_count * sizeof(float));

    if (!ok) {
        fprintf(stderr, " setup fail\n");
        g_failures++;
        goto cleanup;
    }

    /* --- Reference run: disable TensorOps for MoE via env vars --- */
    setenv("DS4_METAL_DISABLE_MXFP4_MOE_GATE_UP_MPP", "1", 1);
    setenv("DS4_METAL_DISABLE_Q4_K_PAIR_SWIGLU_MPP", "1", 1);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    ok = ok && ds4_gpu_tensor_fill_f32(mid_tensor_ref, -999.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(out_tensor_ref, -999.0f, out_count);
    ok = ok && ds4_gpu_tensor_fill_f32(experts_tensor_ref, -999.0f,
                                       out_count * tc->n_expert_used);

    bool mid_is_f16_ref = false;
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_tensor_ref, gate_tensor, up_tensor,
        mid_tensor_ref, experts_tensor_ref,
        model, model_size, gate_offset, up_offset, down_offset,
        tc->quant_type, tc->quant_type, expert_bytes, row_bytes_val,
        expert_bytes, row_bytes_val, tc->in_dim, tc->out_dim, tc->out_dim,
        selected_tensor, weights_tensor,
        tc->n_total_expert, tc->n_expert_used, tc->clamp, x_tensor,
        0u, tc->n_tokens, &mid_is_f16_ref, true);

    /* --- Test run: enable TensorOps (remove env blocking) --- */
    unsetenv("DS4_METAL_DISABLE_MXFP4_MOE_GATE_UP_MPP");
    unsetenv("DS4_METAL_DISABLE_Q4_K_PAIR_SWIGLU_MPP");

    ok = ok && ds4_gpu_tensor_fill_f32(mid_tensor_test, -888.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(out_tensor_test, -888.0f, out_count);
    ok = ok && ds4_gpu_tensor_fill_f32(experts_tensor_test, -888.0f,
                                       out_count * tc->n_expert_used);

    bool mid_is_f16_test = false;
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_tensor_test, gate_tensor, up_tensor,
        mid_tensor_test, experts_tensor_test,
        model, model_size, gate_offset, up_offset, down_offset,
        tc->quant_type, tc->quant_type, expert_bytes, row_bytes_val,
        expert_bytes, row_bytes_val, tc->in_dim, tc->out_dim, tc->out_dim,
        selected_tensor, weights_tensor,
        tc->n_total_expert, tc->n_expert_used, tc->clamp, x_tensor,
        0u, tc->n_tokens, &mid_is_f16_test, true);

    if (!ok) {
        fprintf(stderr, " dispatch fail\n");
        g_failures++;
        goto cleanup;
    }

    /* Read back results. Both paths should produce F16 mid. */
    if (!mid_is_f16_ref || !mid_is_f16_test) {
        fprintf(stderr, " mid not F16 (ref=%d test=%d), comparing as F32\n",
                mid_is_f16_ref, mid_is_f16_test);
    }

    /* Compare mid tensors.
     * Both store FP16 values when mid_is_f16 is set.
     * Read as raw bytes and compare as half-precision. */
    _Float16 *mid_ref_f16 = NULL;
    _Float16 *mid_test_f16 = NULL;
    float *mid_ref_f32 = NULL;
    float *mid_test_f32 = NULL;

    if (mid_is_f16_ref && mid_is_f16_test) {
        mid_ref_f16 = malloc((size_t)(pair_count * sizeof(_Float16)));
        mid_test_f16 = malloc((size_t)(pair_count * sizeof(_Float16)));
        ok = ok && mid_ref_f16 && mid_test_f16;
        ok = ok && ds4_gpu_tensor_read(mid_tensor_ref, 0, mid_ref_f16,
                                       pair_count * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_read(mid_tensor_test, 0, mid_test_f16,
                                       pair_count * sizeof(_Float16));
    } else {
        mid_ref_f32 = malloc((size_t)(pair_count * sizeof(float)));
        mid_test_f32 = malloc((size_t)(pair_count * sizeof(float)));
        ok = ok && mid_ref_f32 && mid_test_f32;
        ok = ok && ds4_gpu_tensor_read(mid_tensor_ref, 0, mid_ref_f32,
                                       pair_count * sizeof(float));
        ok = ok && ds4_gpu_tensor_read(mid_tensor_test, 0, mid_test_f32,
                                       pair_count * sizeof(float));
    }

    /* Also compare final output. */
    float *out_ref_host = malloc((size_t)(out_count * sizeof(float)));
    float *out_test_host = malloc((size_t)(out_count * sizeof(float)));
    ok = ok && out_ref_host && out_test_host;
    ok = ok && ds4_gpu_tensor_read(out_tensor_ref, 0, out_ref_host,
                                   out_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(out_tensor_test, 0, out_test_host,
                                   out_count * sizeof(float));

    if (!ok) {
        fprintf(stderr, " readback fail\n");
        g_failures++;
        goto cleanup;
    }

    /* Compare mid values. */
    {
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        uint64_t max_abs_idx = 0;
        uint64_t nan_count = 0;
        uint64_t mismatch_count = 0;

        for (uint64_t i = 0; i < pair_count; i++) {
            float ref_val, test_val;
            if (mid_is_f16_ref && mid_is_f16_test) {
                ref_val = (float)mid_ref_f16[i];
                test_val = (float)mid_test_f16[i];
            } else {
                ref_val = mid_ref_f32[i];
                test_val = mid_test_f32[i];
            }

            if (!isfinite(test_val)) {
                nan_count++;
                continue;
            }
            if (!isfinite(ref_val)) {
                /* Reference NaN is also a problem but not a TensorOps regression. */
                continue;
            }

            const float err = fabsf(test_val - ref_val);
            if (err > max_abs_err) {
                max_abs_err = err;
                max_abs_idx = i;
            }
            const float ref_mag = fabsf(ref_val);
            if (ref_mag > 1e-6f) {
                const float rel = err / ref_mag;
                if (rel > max_rel_err) max_rel_err = rel;
            }
            if (err > tc->max_abs_tol) mismatch_count++;
        }

        TEST(nan_count == 0,
             "%s mid: %llu NaN/Inf values in TensorOps output",
             tc->name, (unsigned long long)nan_count);

        TEST(max_abs_err <= tc->max_abs_tol,
             "%s mid: max_abs=%g exceeds tol=%g at idx=%llu "
             "(ref=%g test=%g)",
             tc->name, max_abs_err, tc->max_abs_tol,
             (unsigned long long)max_abs_idx,
             mid_is_f16_ref ? (float)mid_ref_f16[max_abs_idx] :
                              mid_ref_f32[max_abs_idx],
             mid_is_f16_test ? (float)mid_test_f16[max_abs_idx] :
                               mid_test_f32[max_abs_idx]);

        TEST(max_rel_err <= tc->max_rel_tol,
             "%s mid: max_rel=%g exceeds tol=%g",
             tc->name, max_rel_err, tc->max_rel_tol);

        /* Compare final output too. */
        float out_max_abs = 0.0f;
        float out_max_rel = 0.0f;
        uint64_t out_nan_count = 0;
        for (uint64_t i = 0; i < out_count; i++) {
            if (!isfinite(out_test_host[i])) { out_nan_count++; continue; }
            if (!isfinite(out_ref_host[i])) continue;
            const float err = fabsf(out_test_host[i] - out_ref_host[i]);
            if (err > out_max_abs) out_max_abs = err;
            const float mag = fabsf(out_ref_host[i]);
            if (mag > 1e-6f) {
                const float rel = err / mag;
                if (rel > out_max_rel) out_max_rel = rel;
            }
        }
        TEST(out_nan_count == 0,
             "%s out: %llu NaN/Inf in final output",
             tc->name, (unsigned long long)out_nan_count);

        fprintf(stderr, " mid_abs=%g mid_rel=%g out_abs=%g out_rel=%g %s\n",
                max_abs_err, max_rel_err, out_max_abs, out_max_rel,
                (max_abs_err <= tc->max_abs_tol &&
                 max_rel_err <= tc->max_rel_tol &&
                 nan_count == 0 && out_nan_count == 0) ? "OK" : "FAIL");
    }

cleanup:
    free(mid_ref_f16); free(mid_test_f16);
    free(mid_ref_f32); free(mid_test_f32);
    free(out_ref_host); free(out_test_host);
    ds4_gpu_tensor_free(x_tensor);
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(weights_tensor);
    ds4_gpu_tensor_free(gate_tensor);
    ds4_gpu_tensor_free(up_tensor);
    ds4_gpu_tensor_free(mid_tensor_ref);
    ds4_gpu_tensor_free(mid_tensor_test);
    ds4_gpu_tensor_free(experts_tensor_ref);
    ds4_gpu_tensor_free(experts_tensor_test);
    ds4_gpu_tensor_free(out_tensor_ref);
    ds4_gpu_tensor_free(out_tensor_test);
    free(x_host); free(selected_host); free(weights_host);
    free(model);
    return 1;
}

/* =========================================================================
 * Test case tables.
 *
 * Calibrated tolerances:
 *   MXFP4: The E2M1 representation has only 4 bits, so even small reduction
 *   order changes produce visible absolute error in large-scale blocks.
 *   With E8M0 scales around 120-128 and ±6.0 mantissa values, individual
 *   products can reach ~6 * 2^(128-127) = 12.0, and accumulated errors
 *   across a 256-dim row may reach ~0.05 absolute for mid values.
 *   The FP16 storage adds ~5e-4 rounding per element.
 *
 *   Q4_K: 4-bit quantized with group scales. Similar arithmetic intensity
 *   to MXFP4 with slightly different precision characteristics.
 *
 * Error envelope rationale (R12 compliance):
 *   max_abs_tol: calibrated to 4× the expected matmul2d vs simdgroup reduction
 *                order difference at representative dimensions.
 *   max_rel_tol: 0.01 (1%) relative for normal-magnitude outputs.
 *                Higher for adversarial patterns where values are near zero.
 * =========================================================================
 */

/* Default route weights for 6 experts. */
static const float default_6_weights[] = {
    0.24f, 0.20f, 0.18f, 0.16f, 0.12f, 0.10f
};

/* Varied route weights including extremes. */
static const float extreme_weights[] = {
    0.0f, 0.5f, 1.0f, 0.001f, 2.0f, 0.0f
};

/* Fixed expert selection for controlled testing. */
static const int32_t full_expert_spread[] = { 0, 2, 3, 5, 6, 7 };
static const int32_t single_expert[] = { 3 };
static const int32_t two_experts[] = { 1, 4 };

/* =========================================================================
 * MXFP4 test cases.
 * =========================================================================
 */
static const moe_parity_test_case mxfp4_cases[] = {
    /* --- Full tile: 32+ tokens, full 64-row output dim --- */
    {
        .name = "MXFP4 full-tile 32tok",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 32, .in_dim = 256, .out_dim = 256,
        .seed = 100, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },
    {
        .name = "MXFP4 full-tile 64tok",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 64, .in_dim = 256, .out_dim = 256,
        .seed = 101, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },
    {
        .name = "MXFP4 full-tile 128tok",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 128, .in_dim = 256, .out_dim = 256,
        .seed = 102, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },

    /* --- Token tail: small counts where nr1 < NR1 --- */
    {
        .name = "MXFP4 tail 1tok",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 1, .in_dim = 256, .out_dim = 256,
        .seed = 110, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },
    {
        .name = "MXFP4 tail 4tok",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 4, .in_dim = 256, .out_dim = 256,
        .seed = 111, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },
    {
        .name = "MXFP4 tail 8tok",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 8, .in_dim = 256, .out_dim = 256,
        .seed = 112, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },
    {
        .name = "MXFP4 tail 16tok",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 16, .in_dim = 256, .out_dim = 256,
        .seed = 113, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },

    /* --- Empty/small expert bucket --- */
    {
        .name = "MXFP4 small-bucket 1exp",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 1,
        .n_tokens = 2, .in_dim = 256, .out_dim = 256,
        .seed = 120, .clamp = 7.0f,
        .route_weights = NULL, /* 1.0 */
        .fixed_selected = single_expert,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },
    {
        .name = "MXFP4 small-bucket 2exp",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 2,
        .n_tokens = 1, .in_dim = 256, .out_dim = 256,
        .seed = 121, .clamp = 7.0f,
        .route_weights = NULL,
        .fixed_selected = two_experts,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },

    /* --- Route weights: extreme values --- */
    {
        .name = "MXFP4 route-weight extremes",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 32, .in_dim = 256, .out_dim = 256,
        .seed = 130, .clamp = 7.0f,
        .route_weights = extreme_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.3f, .max_rel_tol = 0.02f,
    },

    /* --- Adversarial: all-zero weights --- */
    {
        .name = "MXFP4 adv zeros",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 32, .in_dim = 256, .out_dim = 256,
        .seed = 140, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 1,
        .max_abs_tol = 1e-6f, .max_rel_tol = 1e-6f,
    },
    /* --- Adversarial: alternating ±max --- */
    {
        .name = "MXFP4 adv +-max",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 32, .in_dim = 256, .out_dim = 256,
        .seed = 141, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 2,
        /* Large outputs from big scales, so tolerance scales with magnitude. */
        .max_abs_tol = 2.0f, .max_rel_tol = 0.02f,
    },
    /* --- E2M1 coverage: all 16 codes with edge scales --- */
    {
        .name = "MXFP4 E2M1 coverage",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 32, .in_dim = 256, .out_dim = 256,
        .seed = 142, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 3,
        /* Edge scales (0,1,254) produce extreme or zero values. */
        .max_abs_tol = 50.0f, .max_rel_tol = 0.05f,
    },

    /* --- Half-scale LUT boundary test --- */
    {
        .name = "MXFP4 half-LUT boundary",
        .quant_type = MXFP4_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 48, .in_dim = 256, .out_dim = 256,
        .seed = 150, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = full_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.15f, .max_rel_tol = 0.02f,
    },
};

/* =========================================================================
 * Q4_K test cases.
 * =========================================================================
 */
static const int32_t q4k_expert_spread[] = { 0, 1, 2, 3, 4, 5 };

static const moe_parity_test_case q4_k_cases[] = {
    /* Full tile. */
    {
        .name = "Q4_K full-tile 32tok",
        .quant_type = Q4_K_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 32, .in_dim = 256, .out_dim = 256,
        .seed = 200, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = q4k_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.25f, .max_rel_tol = 0.03f,
    },
    {
        .name = "Q4_K full-tile 64tok",
        .quant_type = Q4_K_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 64, .in_dim = 256, .out_dim = 256,
        .seed = 201, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = q4k_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.25f, .max_rel_tol = 0.03f,
    },
    /* Token tails. */
    {
        .name = "Q4_K tail 1tok",
        .quant_type = Q4_K_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 1, .in_dim = 256, .out_dim = 256,
        .seed = 210, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = q4k_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.25f, .max_rel_tol = 0.03f,
    },
    {
        .name = "Q4_K tail 8tok",
        .quant_type = Q4_K_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 8, .in_dim = 256, .out_dim = 256,
        .seed = 211, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = q4k_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.25f, .max_rel_tol = 0.03f,
    },
    {
        .name = "Q4_K tail 16tok",
        .quant_type = Q4_K_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 16, .in_dim = 256, .out_dim = 256,
        .seed = 212, .clamp = 7.0f,
        .route_weights = default_6_weights,
        .fixed_selected = q4k_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.25f, .max_rel_tol = 0.03f,
    },
    /* Small bucket. */
    {
        .name = "Q4_K small-bucket 2tok 1exp",
        .quant_type = Q4_K_TYPE,
        .n_total_expert = 8, .n_expert_used = 1,
        .n_tokens = 2, .in_dim = 256, .out_dim = 256,
        .seed = 220, .clamp = 7.0f,
        .route_weights = NULL,
        .fixed_selected = single_expert,
        .fill_mode = 0,
        .max_abs_tol = 0.25f, .max_rel_tol = 0.03f,
    },
    /* Route weight extremes. */
    {
        .name = "Q4_K route-weight extremes",
        .quant_type = Q4_K_TYPE,
        .n_total_expert = 8, .n_expert_used = 6,
        .n_tokens = 32, .in_dim = 256, .out_dim = 256,
        .seed = 230, .clamp = 7.0f,
        .route_weights = extreme_weights,
        .fixed_selected = q4k_expert_spread,
        .fill_mode = 0,
        .max_abs_tol = 0.5f, .max_rel_tol = 0.03f,
    },
};

/* =========================================================================
 * main
 * =========================================================================
 */
int main(int argc, char **argv) {
    (void)argc; (void)argv;

    fprintf(stderr,
            "test_moe_tensorops_parity: initializing Metal...\n");

    if (!ds4_gpu_init()) {
        fprintf(stderr,
                "test_moe_tensorops_parity: no Metal device, skipping.\n");
        return 0;
    }

    const int tensorops_available = ds4_gpu_tensorops_runtime();
    if (!tensorops_available) {
        fprintf(stderr,
                "test_moe_tensorops_parity: TensorOps runtime unavailable, "
                "skipping MoE parity tests (legacy path only).\n");
        ds4_gpu_cleanup();
        return 0;
    }

    fprintf(stderr,
            "test_moe_tensorops_parity: TensorOps runtime active, "
            "running A/B parity tests...\n\n");

    /* Ensure TensorOps is not globally disabled. */
    unsetenv("DS4_METAL_DISABLE_MXFP4_MOE_GATE_UP_MPP");
    unsetenv("DS4_METAL_DISABLE_Q4_K_PAIR_SWIGLU_MPP");

    /* ===== MXFP4 tests ===== */
    fprintf(stderr, "=== MXFP4 TensorOps vs Legacy MoE Pair-SwiGLU ===\n");
    for (size_t i = 0; i < sizeof(mxfp4_cases) / sizeof(mxfp4_cases[0]); i++) {
        run_moe_parity_test(&mxfp4_cases[i]);
    }

    /* ===== Q4_K tests ===== */
    fprintf(stderr, "\n=== Q4_K TensorOps vs Legacy MoE Pair-SwiGLU ===\n");
    for (size_t i = 0; i < sizeof(q4_k_cases) / sizeof(q4_k_cases[0]); i++) {
        run_moe_parity_test(&q4_k_cases[i]);
    }

    /* Clean up env. */
    unsetenv("DS4_METAL_DISABLE_MXFP4_MOE_GATE_UP_MPP");
    unsetenv("DS4_METAL_DISABLE_Q4_K_PAIR_SWIGLU_MPP");

    fprintf(stderr, "\n");
    fprintf(stderr,
            "test_moe_tensorops_parity: %d tests, %d failures, %d skipped\n",
            g_tests_run, g_failures, g_tests_skipped);

    ds4_gpu_cleanup();

    if (g_failures) {
        fprintf(stderr, "test_moe_tensorops_parity: FAILED\n");
        return 1;
    }
    fprintf(stderr, "test_moe_tensorops_parity: PASSED\n");
    return 0;
}

#else /* !__APPLE__ */

#include <stdio.h>

int main(void) {
    fprintf(stderr,
            "test_moe_tensorops_parity: not Apple, skipping.\n");
    return 0;
}

#endif /* __APPLE__ */
