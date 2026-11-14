#include "dsp/f16_gemm_plan.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>

int main(void) {
  f16_gemm_plan_t p;
  assert(!f16_gemm_make_plan(77, 10240, 4096, 1024, 256, &p));
  assert(!f16_gemm_make_plan(0, 64, 128, 4194304, 256, &p));
  assert(!f16_gemm_make_plan(77, 65, 128, 4194304, 256, &p));
  assert(!f16_gemm_make_plan(77, 64, 127, 4194304, 256, &p));
  assert(!f16_gemm_make_plan(77, 64, 128, 256, 256, &p));
  assert(!f16_gemm_make_plan(77, 64, 128, 4194304, 256, NULL));
  // A single session's half of VTCM must retain the entire 77-row T5 activation.
  assert(f16_gemm_make_plan(77, 10240, 4096, 4194304, 256, &p));
  assert(p.rows == 96 && p.cols >= 64);
  assert(f16_gemm_make_plan(256, 6144, 1536, 4194304, 256, &p));
  assert(p.rows == 256);
  assert(!f16_gemm_make_plan(INT_MAX, INT_MAX - 31, 32, 8388608, 256, &p));

  const int ms[] = {1, 31, 32, 33, 77, 256, 410, 1178, 4096};
  const int ks[] = {32, 64, 768, 1536, 4096, 6144, 10240, 32768};
  const int ns[] = {32, 96, 128, 1536, 4096, 10240};
  const size_t budgets[] = {4096, 65536, 1048576, 4194304, 8388608, 8388631};
  unsigned checks = 0;
  for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); ++i)
    for (size_t j = 0; j < sizeof(ks) / sizeof(ks[0]); ++j)
      for (size_t t = 0; t < sizeof(ns) / sizeof(ns[0]); ++t)
        for (size_t b = 0; b < sizeof(budgets) / sizeof(budgets[0]); ++b)
          for (size_t scratch = 0; scratch <= 16384; scratch += 16384) {
            const size_t reserved = 256 + scratch;
            bool fits = f16_gemm_make_plan(ms[i], ks[j], ns[t], budgets[b], reserved, &p);
            // The smallest legal HMX block is 32x32 with the full K extent.
            assert(fits == (budgets[b] >= reserved + (size_t) ks[j] * 128 + 2048));
            if (!fits) continue;
            assert(p.rows && p.cols && p.rows % 32 == 0 && p.cols % 32 == 0);
            assert(p.rows <= ((size_t) ms[i] + 31) / 32 * 32 && p.cols <= (size_t) ns[t]);
            assert(p.activation_bytes % 2048 == 0 && p.weight_bytes % 2048 == 0 && p.output_bytes % 2048 == 0);
            assert(p.activation_bytes + p.weight_bytes + p.output_bytes + reserved <= budgets[b]);
            ++checks;
          }
  printf("F16 GEMM VTCM bounds passed: %u legal plans\n", checks);
}
