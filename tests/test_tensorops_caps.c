/*
 * test_tensorops_caps.c — TensorOps capability layer unit tests (Task 2.5).
 *
 * Tests compile-time macros, runtime fallback behavior, rollback control,
 * and counter semantics.  Runs on any platform; Metal-specific behavior is
 * tested only when __APPLE__ + Metal device is available.
 */

#include "ds4_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define TEST(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        g_failures++; \
    } \
} while (0)

/* Stub required by the linker when ds4_metal.m pulls ds4 log helpers. */
#ifndef __APPLE__
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
#endif

/* =========================================================================
 * 1. Compile-time macro sanity (all platforms).
 * =========================================================================
 */
static void test_compile_time_macros(void) {
    /* Macros must be defined as integers (0 or 1). */
    int sdk_tensorops = DS4_METAL_SDK_TENSOROPS;
    int sdk_coop      = DS4_METAL_SDK_COOP_INPUT;
    int sdk_int       = DS4_METAL_SDK_NATIVE_INT;
    int sdk_lowbit    = DS4_METAL_SDK_NATIVE_LOWBIT;

    TEST(sdk_tensorops == 0 || sdk_tensorops == 1,
         "DS4_METAL_SDK_TENSOROPS must be 0 or 1");
    TEST(sdk_coop == 0 || sdk_coop == 1,
         "DS4_METAL_SDK_COOP_INPUT must be 0 or 1");
    TEST(sdk_int == 0 || sdk_int == 1,
         "DS4_METAL_SDK_NATIVE_INT must be 0 or 1");
    TEST(sdk_lowbit == 0 || sdk_lowbit == 1,
         "DS4_METAL_SDK_NATIVE_LOWBIT must be 0 or 1");

    /* Tier ordering: higher tiers imply lower tiers were available at SDK time. */
    if (sdk_lowbit)  TEST(sdk_int,       "lowbit implies native_int SDK");
    if (sdk_int)     TEST(sdk_coop,      "native_int implies coop_input SDK");
    if (sdk_coop)    TEST(sdk_tensorops, "coop_input implies basic tensorops SDK");

#ifndef __APPLE__
    /* On non-Apple builds, all macros should be 0. */
    TEST(sdk_tensorops == 0, "non-Apple: SDK_TENSOROPS must be 0");
    TEST(sdk_coop == 0,      "non-Apple: SDK_COOP_INPUT must be 0");
    TEST(sdk_int == 0,       "non-Apple: SDK_NATIVE_INT must be 0");
    TEST(sdk_lowbit == 0,    "non-Apple: SDK_NATIVE_LOWBIT must be 0");
#endif
}

/* =========================================================================
 * 2. Non-Apple fallback (inline stubs return 0).
 * =========================================================================
 */
#ifndef __APPLE__
static void test_non_apple_stubs(void) {
    TEST(ds4_gpu_tensorops_compiled() == 0,              "stub: compiled");
    TEST(ds4_gpu_tensorops_runtime() == 0,               "stub: runtime");
    TEST(ds4_gpu_tensorops_m5_neural_accelerator() == 0, "stub: m5");
    TEST(ds4_gpu_tensorops_coop_input() == 0,            "stub: coop_input");
    TEST(ds4_gpu_tensorops_native_int4_int8() == 0,      "stub: native_int");
    TEST(ds4_gpu_tensorops_native_lowbit_float() == 0,   "stub: lowbit_float");
    TEST(ds4_gpu_tensorops_native_e8m0_scale() == 0,     "stub: e8m0");
    TEST(ds4_gpu_tensorops_counter_dense() == 0,         "stub: ctr_dense");
    TEST(ds4_gpu_tensorops_counter_indexer() == 0,       "stub: ctr_indexer");
    TEST(ds4_gpu_tensorops_counter_moe_gate_up() == 0,   "stub: ctr_moe_gu");
    TEST(ds4_gpu_tensorops_counter_moe_down() == 0,      "stub: ctr_moe_down");
    TEST(ds4_gpu_tensorops_counter_verify() == 0,        "stub: ctr_verify");
    TEST(ds4_gpu_tensorops_counter_fallback_shape() == 0,     "stub: fb_shape");
    TEST(ds4_gpu_tensorops_counter_fallback_quant() == 0,     "stub: fb_quant");
    TEST(ds4_gpu_tensorops_counter_fallback_alignment() == 0, "stub: fb_align");
    TEST(ds4_gpu_tensorops_counter_fallback_os() == 0,        "stub: fb_os");
    TEST(ds4_gpu_tensorops_counter_fallback_pipeline() == 0,  "stub: fb_pipe");
    TEST(ds4_gpu_tensorops_counter_fallback_streaming() == 0, "stub: fb_stream");

    /* These should be no-ops without crashing. */
    ds4_gpu_tensorops_print_caps();
    ds4_gpu_tensorops_print_counters();
    ds4_gpu_tensorops_reset_counters();
}
#endif

/* =========================================================================
 * 3. Apple / Metal tests — only run when a device is available.
 * =========================================================================
 */
#ifdef __APPLE__
/* Forward declare ds4_log_is_tty if needed by the .m linkage. */
bool ds4_log_is_tty(FILE *fp);
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void test_apple_rollback(void) {
    /*
     * ds4_gpu_init() has already been called (or not, if no Metal device).
     * We test that the rollback semantics work:
     *
     * When DS4_METAL_DISABLE_TENSOROPS is set:
     *   - ds4_gpu_tensorops_runtime() must return 0
     *   - ds4_gpu_mpp_available() (via runtime) must return 0
     *
     * We can't easily reinitialize here, so just verify the relationship:
     * if runtime is active, rollback is not. If rollback is active, runtime
     * must be zero.
     */
    const char *env = getenv("DS4_METAL_DISABLE_TENSOROPS");
    const int rollback_requested = (env && (strcmp(env, "1") == 0 ||
                                            strcasecmp(env, "true") == 0 ||
                                            strcasecmp(env, "yes") == 0 ||
                                            strcasecmp(env, "on") == 0));
    if (rollback_requested) {
        TEST(ds4_gpu_tensorops_runtime() == 0,
             "rollback: runtime must be 0 when DISABLE_TENSOROPS is set");
    }

    /* compiled() reflects the SDK, not runtime state. */
    int compiled = ds4_gpu_tensorops_compiled();
    TEST(compiled == 0 || compiled == 1, "compiled must be 0 or 1");
}

static void test_apple_counter_lifecycle(void) {
    /* Reset counters and verify they read zero. */
    ds4_gpu_tensorops_reset_counters();
    TEST(ds4_gpu_tensorops_counter_dense() == 0, "reset: dense == 0");
    TEST(ds4_gpu_tensorops_counter_indexer() == 0, "reset: indexer == 0");
    TEST(ds4_gpu_tensorops_counter_moe_gate_up() == 0, "reset: moe_gu == 0");
    TEST(ds4_gpu_tensorops_counter_moe_down() == 0, "reset: moe_down == 0");
    TEST(ds4_gpu_tensorops_counter_verify() == 0, "reset: verify == 0");
    TEST(ds4_gpu_tensorops_counter_fallback_shape() == 0, "reset: fb_shape == 0");
    TEST(ds4_gpu_tensorops_counter_fallback_quant() == 0, "reset: fb_quant == 0");
    TEST(ds4_gpu_tensorops_counter_fallback_alignment() == 0, "reset: fb_align == 0");
    TEST(ds4_gpu_tensorops_counter_fallback_os() == 0, "reset: fb_os == 0");
    TEST(ds4_gpu_tensorops_counter_fallback_pipeline() == 0, "reset: fb_pipe == 0");
    TEST(ds4_gpu_tensorops_counter_fallback_streaming() == 0, "reset: fb_stream == 0");
}

static void test_apple_capability_coherence(void) {
    /*
     * If runtime is enabled, compiled must also be true.
     * Higher-tier capabilities require runtime to be active.
     */
    int runtime = ds4_gpu_tensorops_runtime();
    int compiled = ds4_gpu_tensorops_compiled();

    if (runtime) {
        TEST(compiled, "runtime implies compiled");
    }
    if (ds4_gpu_tensorops_coop_input()) {
        TEST(runtime, "coop_input implies runtime");
    }
    if (ds4_gpu_tensorops_native_int4_int8()) {
        TEST(runtime, "native_int implies runtime");
    }
    if (ds4_gpu_tensorops_native_lowbit_float()) {
        TEST(runtime, "lowbit_float implies runtime");
    }
    if (ds4_gpu_tensorops_native_e8m0_scale()) {
        TEST(ds4_gpu_tensorops_native_lowbit_float(),
             "e8m0 implies lowbit_float");
    }
}
#endif

/* =========================================================================
 * main
 * =========================================================================
 */
int main(int argc, char **argv) {
    (void)argc; (void)argv;

    fprintf(stderr, "test_tensorops_caps: running...\n");

    test_compile_time_macros();

#ifdef __APPLE__
    /* Try to initialize Metal; not fatal if no device (CI without GPU). */
    int init_ok = ds4_gpu_init();
    if (!init_ok) {
        fprintf(stderr, "test_tensorops_caps: no Metal device; skipping runtime tests\n");
    }

    test_apple_rollback();
    test_apple_counter_lifecycle();
    test_apple_capability_coherence();

    if (init_ok) ds4_gpu_cleanup();
#else
    test_non_apple_stubs();
#endif

    if (g_failures) {
        fprintf(stderr, "test_tensorops_caps: %d test(s) FAILED\n", g_failures);
        return 1;
    }
    fprintf(stderr, "test_tensorops_caps: all tests passed\n");
    return 0;
}
