#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  size_t rows, cols;
  size_t activation_bytes, weight_bytes, output_bytes;
} f16_gemm_plan_t;

// Keep K whole so changing the VTCM partition does not change accumulation.
// Minimize full weight scans first, then the number of pack/HMX/store calls.
static inline bool f16_gemm_make_plan(int m, int k, int n, size_t vtcm_bytes,
                                      size_t reserved_bytes, f16_gemm_plan_t *plan) {
  if (!plan || m <= 0 || k <= 0 || n <= 0 || k % 32 || n % 32 || vtcm_bytes <= reserved_bytes) {
    return false;
  }
  const size_t slots = (vtcm_bytes - reserved_bytes) / 2;
  const size_t min_weight = (size_t) k * 32;
  if (slots < min_weight) {
    return false;
  }
  size_t max_rows = (slots - min_weight) / ((size_t) k + 32);
  const size_t padded_m = ((size_t) m + 31) / 32 * 32;
  if (max_rows > padded_m) {
    max_rows = padded_m;
  }
  size_t best_scans = SIZE_MAX, best_calls = SIZE_MAX;
  bool found = false;
  for (size_t rows = 32; rows <= max_rows; rows += 32) {
    size_t cols = (slots - rows * k) / ((size_t) k + rows);
    cols = cols / 32 * 32;
    if (cols > (size_t) n) {
      cols = n;
    }
    if (cols < 32) {
      continue;
    }
    const size_t scans = ((size_t) m + rows - 1) / rows;
    const size_t calls = scans * (((size_t) n + cols - 1) / cols);
    if (scans < best_scans || (scans == best_scans && calls < best_calls)) {
      best_scans = scans;
      best_calls = calls;
      *plan = (f16_gemm_plan_t) {
        .rows = rows, .cols = cols,
        .activation_bytes = rows * k * 2,
        .weight_bytes = cols * k * 2,
        .output_bytes = rows * cols * 2,
      };
      found = true;
    }
  }
  return found;
}
