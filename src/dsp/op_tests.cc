#include <HAP_farf.h>
#include <HAP_perf.h>

// std headers
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "dsp/hmx_mgr.h"
#include "dsp/hvx_math.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"

namespace op_utils {

float compute_rmse(const float *x, const float *y, int n) {
  float squared_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    float err = x[i] - y[i];
    squared_error += err * err;
  }
  float rmse = sqrtf(squared_error / n);
  return rmse;
}

int compare_result(const float *x, const float *y, int n_elems) {
  static int counter = 0;

  int layer = counter++ % 28;
  FARF(ALWAYS, "layer %d attention compare:", layer);

  // hard-coded constants
  constexpr int D = 128;
  constexpr int H = 12;  // 12 query heads
  for (int h = 0; h < n_elems / D; ++h) {
    int q    = h / H;
    int head = h % H;

    float rmse = compute_rmse(&x[h * D], &y[h * D], D);
    FARF(ALWAYS, "query %d head %d RMSE: %g", q, head, rmse);
  }
  return 0;
}

}  // namespace op_utils

namespace internal {

#define OP_TEST_ASSERT(condition)                                                                                     \
  do {                                                                                                                \
    if (!(condition)) {                                                                                               \
      FARF(ALWAYS, "%s:%d: check failed: %s", __FILE__, __LINE__, #condition);                                       \
      abort();                                                                                                        \
    }                                                                                                                 \
  } while (0)

void test_block_sparse_fa_select_plan() {
  constexpr size_t available = 1376256;
  block_sparse_fa_plan_t plan = {};

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 64, 410, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_FULL_KV);
  OP_TEST_ASSERT(plan.blk_sz_c == 448);
  OP_TEST_ASSERT(plan.required_bytes == 183808);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 64, 4250, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_FULL_KV);
  OP_TEST_ASSERT(plan.blk_sz_c == 1984);
  OP_TEST_ASSERT(plan.required_bytes == 1369600);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 128, 4250, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_FULL_K);
  OP_TEST_ASSERT(plan.blk_sz_c == 640);
  OP_TEST_ASSERT(plan.required_bytes == 1368064);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 128, 5281, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_CHUNKED);
  OP_TEST_ASSERT(plan.blk_sz_c == 2048);
  OP_TEST_ASSERT(plan.required_bytes == 1349120);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, false, 1, 64, 410, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_CHUNKED);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(nullptr, true, 1, 64, 410, available) == -1);
  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 0, 64, 410, available) == -1);
  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 0, 410, available) == -1);
  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 48, 410, available) == -1);
  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 64, 0, available) == -1);

  constexpr size_t min_chunked = 43008;
  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, false, 1, 64, 410, min_chunked - 1) == -1);
  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, false, 1, 64, 410, min_chunked) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_CHUNKED);
  OP_TEST_ASSERT(plan.blk_sz_c == 64);
  OP_TEST_ASSERT(plan.required_bytes == min_chunked);
  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, false, 1, 64, 410, min_chunked + 1) == 0);
  OP_TEST_ASSERT(plan.blk_sz_c == 64);
  OP_TEST_ASSERT(plan.required_bytes == min_chunked);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 2147483616, std::numeric_limits<int>::max(),
                                             std::numeric_limits<size_t>::max()) == -1);

  FARF(ALWAYS, "block_sparse_fa_select_plan: PASS");
}

#undef OP_TEST_ASSERT

void test_int16_fp16_conversion() {
#if __HVX_ARCH__ < 73
  FARF(ALWAYS, "HVX native h <-> hf conversion not supported");
  return;
#endif

  static __fp16  input[64];
  static int16_t output[64];

  for (int i = 0; i < 64; ++i) {
    float x  = i * 0.25 - 8;
    input[i] = (__fp16) x;
  }

  vmemu(output) = Q6_Vh_equals_Vhf(vmemu(input));

  for (int i = 0; i < 64; ++i) {
    FARF(ALWAYS, "%s: x=%g y=%d", __func__, (float) input[i], output[i]);
  }
}

void test_fp16_exp2() {
  int    n    = 256;
  size_t size = n * sizeof(__fp16);

  __fp16 *input = nullptr;
  posix_memalign((void **) &input, VLEN, size);

  __fp16 *output = nullptr;
  posix_memalign((void **) &output, VLEN, size);

  __fp16 *output_ref = new __fp16[n];

  for (int i = 0; i < n; ++i) {
    float x       = -0.1 * i;
    input[i]      = (__fp16) x;
    output_ref[i] = (__fp16) exp2f(x);
  }

  auto in_vecs  = (HVX_Vector *) input;
  auto out_vecs = (HVX_Vector *) output;
  for (int i = 0; i < n / 64; ++i) {
    out_vecs[i] = hvx_my_exp2_vhf(in_vecs[i]);
  }

  for (int i = 0; i < n; ++i) {
    float x  = (float) input[i];
    float y0 = (float) output_ref[i];
    float y1 = (float) output[i];
    FARF(ALWAYS, "%s: i=%d, x=%g, my: %g, ref: %g", __func__, i, x, y1, y0);
  }

  delete[] output_ref;
  free(input);
  free(output);
}

void benchmark_hmx_gemm() {
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  __fp16 *a = (__fp16 *) vtcm;
  __fp16 *b = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c = (__fp16 *) (vtcm + 4 * 0x100000);
  __fp16 *s = (__fp16 *) (vtcm + 6 * 0x100000);

  int n_repeat = 1000;
  int sizes[]  = { 32, 64, 128, 256, 512, 1024 };

  hmx_manager_enable_execution();
  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hmx_mat_mul_fp16_core(c, a, b, s, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp16 hmx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }
  hmx_manager_disable_execution();
}

void benchmark_hvx_gemm() {
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  __fp16 *a = (__fp16 *) vtcm;
  __fp16 *b = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c = (__fp16 *) (vtcm + 4 * 0x100000);

  int n_repeat = 10;
  // int sizes[]  = { 32, 64, 128, 256, 512, 1024 };

  /*
  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_fp16_core(c, a, b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp16 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }

  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_fp32_core((float *) c, (float *) a, (float *) b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp32 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }

  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_int16_core((int16_t *) c, (int16_t *) a, (int16_t *) b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core int16 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }

  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_int32_core((int32_t *) c, (int32_t *) a, (int32_t *) b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core int32 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }
  */

  int64_t n = 1024;
  for (int i = 1; i <= 4; i *= 2) {
    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_fp16_core_mt(c, a, b, n, n, n, i);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp16 hvx: %.2lf GFLOPS@%d Threads, %lld us", __func__, gflops, i, elapsed_us);
  }
}

void benchmark_vtcm_bandwidth() {
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  HVX_Vector *a = (HVX_Vector *) vtcm;
  HVX_Vector *b = (HVX_Vector *) (vtcm + 0x400000);

  size_t size = 0x100000;

  int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < size / VLEN; ++i) {
    // HVX_Vector v0 = *a++;
    // HVX_Vector v1 = *a++;
    // HVX_Vector v2 = *a++;
    // *b++ = v0;
    // *b++ = v1;
    // *b++ = v2;

    // 4 packets
    // asm volatile (
    //   "{ v0.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v0 }\n"
    //   "{ v1.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v1 }\n"
    //   "vmem(%1++#1) = v0\n"
    //   "vmem(%1++#1) = v1\n"
    //   :"+r"(a), "+r"(b)::"v0", "v1"
    // );

    // 3 packets
    // asm volatile(
    //   "{ v0.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v0 }\n"
    //   "{ v1.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v1 }\n"
    //   "{ v2.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v2 }\n"
    //   : "+r"(a), "+r"(b)::"v0", "v1", "v2");

    asm volatile(
      "{ v0.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v0 }\n"
      "{ v1.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v1 }\n"
      "{ v2.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v2 }\n"
      "{ v3.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v3 }\n"
      : "+r"(a), "+r"(b)::"v0", "v1", "v2", "v3");
  }
  int64_t t1         = HAP_perf_get_qtimer_count();
  int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

  const int rf       = 4;
  const int tf       = 8;
  double    read_bw  = 1e-3 * rf * size / elapsed_us;
  double    total_bw = 1e-3 * tf * size / elapsed_us;

  FARF(ALWAYS, "%s: %lld us, read bw: %.2lf GB/s, total bw: %.2lf GB/s", __func__, elapsed_us, read_bw, total_bw);
}

}  // namespace internal

extern "C" {

void internal_op_tests();

void internal_op_tests() {
  using namespace internal;

  test_block_sparse_fa_select_plan();

  // test_int16_fp16_conversion();
  // test_fp16_exp2();

  // benchmark_hmx_gemm();
  // benchmark_hvx_gemm();
  // benchmark_vtcm_bandwidth();
}
}
