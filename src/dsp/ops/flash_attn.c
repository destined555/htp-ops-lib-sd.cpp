#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"
#include "dsp/ops.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "op_reg.h"

// for debug
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_memory.h>
#include <qurt/qurt_timer.h>

#define MAX_G_BR        256
#define __vec_aligned__ __attribute__((aligned(VLEN)))
#define FA_LN2          0.69314718055994530942f

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  // int                n_tot_chunks;
  // int                n_chunks_per_task;
  uint8_t           *vtcm_base;
  size_t             vtcm_size_per_thread;
  // params
  __fp16            *O;
  const __fp16      *Q, *K, *V, *mask;
  int                qo_len, kv_len, n_heads, n_kv_heads, head_dim;
  bool               with_lse;
} simple_fa_task_state_t;

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  unsigned int       error_count;
  unsigned int       halo_fallback_count;
  int                n_tasks;
  uint8_t           *vtcm_base;
  size_t             acquired_vtcm_size;
  size_t             vtcm_size_per_thread;
  unsigned int       vtcm_slice_count;
  __fp16            *O;
  const __fp16      *Q, *K, *V;
  const int32_t     *block_mask;
  int                qo_len, kv_len, n_heads, n_kv_heads, head_dim;
  int                block_q, block_k, n_q_blocks, n_k_blocks;
  int                mask_format, max_kept_blocks;
  int                sparse_policy, image_tokens, text_tokens;
  int                image_block_rows, image_block_cols, halo_kv_load_mode;
  bool               halo_trace;
  bool               halo_l2_enabled;
  bool               halo_userdma_enabled;
  bool               halo_force_chunked;
  bool               input_packed;
  int                userdma_scope;
  block_sparse_halo_run_trace_t *halo_run_trace;
  block_sparse_halo_plan_t halo_plan;
  bool               with_lse;
} block_sparse_fa_task_state_t;

static inline void swap_ptr(__fp16 **p0, __fp16 **p1) {
  __fp16 *t = *p0;
  *p0       = *p1;
  *p1       = t;
}

static inline void hvx_fill_uh(void *p, uint16_t v, size_t size) {
  assert(size % VLEN == 0);
  assert(is_aligned(p, VLEN));
  HVX_Vector  v_v    = Q6_Vh_vsplat_R(v);
  HVX_Vector *pv_out = (HVX_Vector *) p;
  for (int i = 0; i < size / VLEN; ++i) {
    *pv_out++ = v_v;
  }
}

static inline void hvx_fill_uw(void *p, uint32_t v, size_t size) {
  assert(size % VLEN == 0);
  assert(is_aligned(p, VLEN));
  HVX_Vector  v_v    = Q6_V_vsplat_R(v);
  HVX_Vector *pv_out = (HVX_Vector *) p;
  for (int i = 0; i < size / VLEN; ++i) {
    *pv_out++ = v_v;
  }
}

static inline void hvx_copy(void *dst, const void *src, size_t size) {
  assert(size % VLEN == 0);
  assert(is_aligned(dst, VLEN) && is_aligned(src, VLEN));
  HVX_Vector *pv_out = (HVX_Vector *) dst;
  const HVX_Vector *pv_in = (const HVX_Vector *) src;
  for (size_t i = 0; i < size / VLEN; ++i) {
    *pv_out++ = *pv_in++;
  }
}

static inline float flash_attn_lse_from_f16_state_scaled(
  const HVX_Vector *mvec_m, const HVX_Vector *mvec_l, int row, float m_scale) {
  enum { N = VLEN / sizeof(__fp16) };
  _Alignas(VLEN) __fp16 m_local[N];
  _Alignas(VLEN) __fp16 l_local[N];

  const int vec_idx = row / N;
  const int vec_off = row % N;
  vmem(m_local) = mvec_m[vec_idx];
  vmem(l_local) = Q6_Vhf_equals_Vqf16(mvec_l[vec_idx]);

  const float m = (float) m_local[vec_off];
  const float l = (float) l_local[vec_off];
  return l > 0.0f ? m * m_scale + logf(l) : -INFINITY;
}

static inline float flash_attn_lse_from_f16_state(const HVX_Vector *mvec_m, const HVX_Vector *mvec_l, int row) {
  return flash_attn_lse_from_f16_state_scaled(mvec_m, mvec_l, row, FA_LN2);
}

static inline float flash_attn_lse_from_f32_state(const HVX_Vector *mvec_m, const HVX_Vector *mvec_l, int row) {
  enum { N = VLEN / sizeof(float) };
  _Alignas(VLEN) float m_local[N];
  _Alignas(VLEN) float l_local[N];

  const int vec_idx = row / N;
  const int vec_off = row % N;
  vmem(m_local) = mvec_m[vec_idx];
  vmem(l_local) = Q6_Vsf_equals_Vqf32(mvec_l[vec_idx]);

  const float m = m_local[vec_off];
  const float l = l_local[vec_off];
  return l > 0.0f ? m * FA_LN2 + logf(l) : -INFINITY;
}

static inline int block_sparse_q_block_for_group_row(int block_q, int n_q_blocks, int n_heads, int q_token_base,
                                                     int q_group_row, int q_head_base, int gqa_factor) {
  const int q_token = q_token_base + q_group_row / gqa_factor;
  const int q_head  = q_head_base + q_group_row % gqa_factor;
  const int q_blocks_per_head = n_q_blocks / n_heads;
  return q_head * q_blocks_per_head + q_token / block_q;
}

static bool checked_size_add(size_t a, size_t b, size_t *result) {
  if (a > SIZE_MAX - b) {
    return false;
  }
  *result = a + b;
  return true;
}

static bool checked_size_mul(size_t a, size_t b, size_t *result) {
  if (a != 0 && b > SIZE_MAX / a) {
    return false;
  }
  *result = a * b;
  return true;
}

static inline bool packed_head_dim_supported(int head_dim) {
  return htp_packed_head_dim_supported(head_dim) != 0;
}

static inline int block_sparse_halo_packed_dma_desc_count(int head_dim) {
  return packed_head_dim_supported(head_dim) ? 1 + head_dim / HMX_FP16_TILE_N_COLS : -1;
}

static bool block_sparse_halo_mask_span_fits(int n_heads, int q_blocks_per_head,
                                             int max_kept_blocks) {
  size_t total_q_blocks;
  size_t mask_elements;
  size_t mask_bytes;
  return n_heads > 0 && q_blocks_per_head > 0 && max_kept_blocks > 0 &&
    checked_size_mul((size_t) n_heads, (size_t) q_blocks_per_head, &total_q_blocks) &&
    checked_size_mul(total_q_blocks, (size_t) max_kept_blocks, &mask_elements) &&
    checked_size_mul(mask_elements, sizeof(int32_t), &mask_bytes);
}

static bool checked_size_align_up(size_t value, size_t alignment, size_t *result) {
  size_t rounded;
  if (alignment == 0 || !checked_size_add(value, alignment - 1, &rounded)) {
    return false;
  }
  *result = rounded / alignment * alignment;
  return true;
}

static bool fused_anchor_fa_vtcm_usage(int head_dim, int n_rows, int n_cols, size_t *result) {
  if (result == NULL || (head_dim != 64 && head_dim != 128) ||
      n_rows <= 0 || n_rows % HMX_FP16_TILE_N_ROWS != 0 ||
      n_cols <= 0 || n_cols % 64 != 0) {
    return false;
  }

  size_t g_br = 0;
  size_t value = 0;
  size_t qo_tile_size = 0;
  size_t kv_tile_size = 0;
  size_t core_tile_size = 0;
  size_t d_tile_size = 0;
  size_t col_vec_size = 0;
  size_t row_vec_size = 0;
  size_t keep_flags_size = 0;
  size_t keep_tile_count = 0;
  size_t total = 0;
  if (!checked_size_align_up((size_t) n_rows, HMX_FP16_TILE_N_ROWS, &g_br) ||
      !checked_size_mul(g_br, (size_t) head_dim, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &qo_tile_size) ||
      !checked_size_mul((size_t) n_cols, (size_t) head_dim, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &kv_tile_size) ||
      !checked_size_mul(g_br, (size_t) n_cols, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &core_tile_size) ||
      !checked_size_mul(g_br, g_br, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &d_tile_size) ||
      !checked_size_mul(g_br, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 256, &col_vec_size) ||
      !checked_size_mul((size_t) n_cols, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 256, &row_vec_size) ||
      !checked_size_add((size_t) n_cols, HMX_FP16_TILE_N_COLS - 1, &keep_tile_count)) {
    return false;
  }
  keep_tile_count /= HMX_FP16_TILE_N_COLS;
  if (!checked_size_align_up(keep_tile_count, 256, &keep_flags_size)) {
    return false;
  }

#define FUSED_ANCHOR_ADD_MULTIPLE(size, count)                                      \
  do {                                                                               \
    if (!checked_size_mul((size), (count), &value) ||                                \
        !checked_size_add(total, value, &total)) {                                   \
      return false;                                                                  \
    }                                                                                \
  } while (0)

  FUSED_ANCHOR_ADD_MULTIPLE(qo_tile_size, 4u);   /* Q + dense/sparse current + shared next. */
  FUSED_ANCHOR_ADD_MULTIPLE(kv_tile_size, 2u);   /* K + V. */
  FUSED_ANCHOR_ADD_MULTIPLE(core_tile_size, 2u); /* S + reusable P. */
  FUSED_ANCHOR_ADD_MULTIPLE(d_tile_size, 1u);
  FUSED_ANCHOR_ADD_MULTIPLE(col_vec_size, 6u);   /* dense/sparse m/l + rowmax/rowsum. */
  FUSED_ANCHOR_ADD_MULTIPLE(row_vec_size, 2u);
  FUSED_ANCHOR_ADD_MULTIPLE(keep_flags_size, 1u);
  if (!checked_size_add(total, 512u, &total)) {
    return false;
  }
#undef FUSED_ANCHOR_ADD_MULTIPLE

  *result = total;
  return true;
}

size_t fused_anchor_fa_compute_vtcm_usage(int head_dim, int n_rows, int n_cols) {
  size_t result = SIZE_MAX;
  return fused_anchor_fa_vtcm_usage(head_dim, n_rows, n_cols, &result) ? result : SIZE_MAX;
}

int fused_anchor_fa_find_chunk_size(fused_anchor_fa_plan_t *plan, int head_dim,
                                    int qo_len, int kv_len, size_t available_bytes) {
  if (plan == NULL || (head_dim != 64 && head_dim != 128) || qo_len <= 0 || kv_len <= 0) {
    return -1;
  }

  size_t max_rows = 0;
  size_t max_cols = 0;
  if (!checked_size_align_up((size_t) qo_len, HMX_FP16_TILE_N_ROWS, &max_rows) ||
      !checked_size_align_up((size_t) kv_len, 64, &max_cols)) {
    return -1;
  }
  fused_anchor_fa_plan_t selected = {0};
  size_t required = 0;
  if (max_rows < HMX_FP16_TILE_N_ROWS ||
      !fused_anchor_fa_vtcm_usage(head_dim, HMX_FP16_TILE_N_ROWS, 64, &required) ||
      required > available_bytes) {
    return -1;
  }
  selected.blk_sz_r = HMX_FP16_TILE_N_ROWS;
  selected.blk_sz_c = 64;
  selected.required_bytes = required;

  for (size_t cols = 128; cols <= max_cols; cols += 64) {
    required = 0;
    if (!fused_anchor_fa_vtcm_usage(head_dim, (int) selected.blk_sz_r, (int) cols, &required)) {
      return -1;
    }
    if (required > available_bytes) {
      break;
    }
    selected.blk_sz_c = cols;
    selected.required_bytes = required;
  }

  *plan = selected;
  return 0;
}

int fused_anchor_block_keep(const int32_t *block_mask, const struct FusedAnchorAttnParams *params,
                            int head, int query_index, int key_block_index) {
  if (block_mask == NULL || params == NULL || head < 0 || head >= params->n_heads ||
      query_index < 0 || query_index >= params->qo_len || key_block_index < 0 ||
      params->block_q != 32 || params->block_k != 32 || params->max_kept_blocks <= 0 ||
      params->n_heads <= 0 || params->n_heads != params->n_kv_heads) {
    return -1;
  }

  const int q_blocks_per_head = params->qo_len / params->block_q +
                                (params->qo_len % params->block_q != 0);
  const int k_blocks_per_head = params->kv_len / params->block_k +
                                (params->kv_len % params->block_k != 0);
  if (key_block_index >= k_blocks_per_head || q_blocks_per_head <= 0 || k_blocks_per_head <= 0 ||
      params->n_q_blocks != params->n_heads * q_blocks_per_head ||
      params->n_k_blocks != params->n_kv_heads * k_blocks_per_head ||
      params->max_kept_blocks < k_blocks_per_head) {
    return -1;
  }

  const int query_block = head * q_blocks_per_head + query_index / params->block_q;
  const int first_key_block = head * k_blocks_per_head;
  const int target = first_key_block + key_block_index;
  const int32_t *row = block_mask + (size_t) query_block * (size_t) params->max_kept_blocks;
  bool found = false;
  bool terminated = false;
  int32_t previous = -1;
  for (int i = 0; i < params->max_kept_blocks; ++i) {
    const int32_t entry = row[i];
    if (entry == -1) {
      terminated = true;
      continue;
    }
    if (terminated || entry < first_key_block || entry >= first_key_block + k_blocks_per_head ||
        entry <= previous) {
      return -1;
    }
    previous = entry;
    found = found || entry == target;
  }
  return found ? 1 : 0;
}

int fused_anchor_plan_selected_segments(fused_anchor_keep_plan_t *plan, const uint8_t *keep_flags,
                                        int tile_count, int first_tile, bool is_text_query) {
  if (plan == NULL || keep_flags == NULL || tile_count < 0 || first_tile < 0) {
    return -1;
  }
  memset(plan, 0, sizeof(*plan));
  plan->first_tile = first_tile;
  plan->tile_count = tile_count;
  plan->is_text_query = is_text_query;
  int tile = 0;
  while (tile < tile_count) {
    if (keep_flags[tile] == 0) {
      ++plan->skipped_tiles;
      ++tile;
      continue;
    }
    if (keep_flags[tile] != 1) {
      return -1;
    }
    const int start = tile;
    while (tile < tile_count && keep_flags[tile] == 1) {
      ++plan->selected_tiles;
      ++tile;
    }
    if (plan->segment_count >= FUSED_ANCHOR_MAX_SELECTED_SEGMENTS) {
      return -1;
    }
    plan->tile_offsets[plan->segment_count] = first_tile + start;
    plan->tile_counts[plan->segment_count] = tile - start;
    ++plan->segment_count;
  }
  return 0;
}

int fused_anchor_prepare_keep_plan(fused_anchor_keep_plan_t *plan, uint8_t *keep_flags,
                                   int keep_flags_capacity, const block_sparse_halo_row_t *row,
                                   int jc, int n_cols, int kv_len, int block_k) {
  if (plan == NULL || keep_flags == NULL || row == NULL || keep_flags_capacity <= 0 ||
      jc < 0 || n_cols <= 0 || kv_len <= 0 || jc >= kv_len || block_k != 32 ||
      n_cols > kv_len - jc ||
      row->k_blocks_per_head <= 0 || row->image_block_total <= 0 ||
      row->image_block_total > row->k_blocks_per_head || row->first_k_block < 0 || row->image_count < 0 ||
      row->image_count > BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS) {
    return -1;
  }
  const int tile_count = (n_cols + block_k - 1) / block_k;
  if (tile_count > keep_flags_capacity) {
    return -1;
  }
  memset(keep_flags, 0, (size_t) tile_count);
  const int image_block_limit = row->image_block_total;
  for (int tile = 0; tile < tile_count; ++tile) {
    const int local_block = (jc + tile * block_k) / block_k;
    const int global_block = row->first_k_block + local_block;
    bool keep = row->is_text_query || local_block >= image_block_limit;
    if (!row->is_text_query && local_block < image_block_limit) {
      keep = false;
      for (int i = 0; i < row->image_count; ++i) {
        if (row->image_blocks[i] == global_block) {
          keep = true;
          break;
        }
      }
    }
    keep_flags[tile] = keep ? 1 : 0;
  }
  return fused_anchor_plan_selected_segments(plan, keep_flags, tile_count, jc / block_k,
                                             row->is_text_query);
}

int block_sparse_packed_qk_tile_offset(size_t *offset, int head, int token_tile, int dim_tile,
                                       int token_tiles, int head_dim) {
  if (offset == NULL || head < 0 || token_tile < 0 || token_tile >= token_tiles ||
      token_tiles <= 0 || head_dim <= 0 || head_dim % HMX_FP16_TILE_N_COLS != 0 ||
      dim_tile < 0 || dim_tile >= head_dim / HMX_FP16_TILE_N_COLS) {
    return -1;
  }
  const size_t dim_tiles = (size_t) head_dim / HMX_FP16_TILE_N_COLS;
  size_t head_tiles;
  size_t tile_index;
  size_t local_tile;
  if (!checked_size_mul((size_t) token_tiles, dim_tiles, &head_tiles) ||
      !checked_size_mul((size_t) head, head_tiles, &tile_index) ||
      !checked_size_mul((size_t) token_tile, dim_tiles, &local_tile) ||
      !checked_size_add(local_tile, (size_t) dim_tile, &local_tile) ||
      !checked_size_add(tile_index, local_tile, &tile_index) ||
      !checked_size_mul(tile_index, HMX_FP16_TILE_N_ELMS, offset)) {
    return -1;
  }
  return 0;
}

int block_sparse_packed_v_tile_offset(size_t *offset, int head, int token_tile, int dim_tile,
                                      int token_tiles, int head_dim) {
  if (offset == NULL || head < 0 || token_tile < 0 || token_tile >= token_tiles ||
      token_tiles <= 0 || head_dim <= 0 || head_dim % HMX_FP16_TILE_N_COLS != 0 ||
      dim_tile < 0 || dim_tile >= head_dim / HMX_FP16_TILE_N_COLS) {
    return -1;
  }
  const size_t dim_tiles = (size_t) head_dim / HMX_FP16_TILE_N_COLS;
  size_t head_tiles;
  size_t tile_index;
  size_t local_tile;
  if (!checked_size_mul((size_t) token_tiles, dim_tiles, &head_tiles) ||
      !checked_size_mul((size_t) head, head_tiles, &tile_index) ||
      !checked_size_mul((size_t) dim_tile, (size_t) token_tiles, &local_tile) ||
      !checked_size_add(local_tile, (size_t) token_tile, &local_tile) ||
      !checked_size_add(tile_index, local_tile, &tile_index) ||
      !checked_size_mul(tile_index, HMX_FP16_TILE_N_ELMS, offset)) {
    return -1;
  }
  return 0;
}

static bool block_sparse_fa_workspace_bytes(int gqa_factor, int head_dim, size_t blk_sz_c, size_t *workspace,
                                            size_t *chunk, size_t *full_head) {
  const size_t br = HMX_FP16_TILE_N_ROWS;
  size_t g_br;
  size_t value;
  size_t qo_tile_size;
  size_t core_tile_size;
  size_t d_tile_size;
  size_t col_vec_size;
  size_t row_vec_size;
  size_t total = 0;

  if (!checked_size_mul((size_t) gqa_factor, br, &g_br) ||
      !checked_size_align_up(g_br, HMX_FP16_TILE_N_ROWS, &g_br) ||
      !checked_size_mul(g_br, (size_t) head_dim, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) || !checked_size_align_up(value, 4096, &qo_tile_size) ||
      !checked_size_mul(g_br, blk_sz_c, &value) || !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &core_tile_size) || !checked_size_mul(g_br, g_br, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) || !checked_size_align_up(value, 4096, &d_tile_size) ||
      !checked_size_mul(g_br, sizeof(__fp16), &value) || !checked_size_align_up(value, 256, &col_vec_size) ||
      !checked_size_mul(blk_sz_c, sizeof(__fp16), &value) || !checked_size_align_up(value, 256, &row_vec_size) ||
      !checked_size_mul(blk_sz_c, (size_t) head_dim, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) || !checked_size_align_up(value, 4096, chunk)) {
    return false;
  }

  if (!checked_size_mul(qo_tile_size, 3, &value) || !checked_size_add(total, value, &total) ||
      !checked_size_mul(core_tile_size, 2, &value) || !checked_size_add(total, value, &total) ||
      !checked_size_add(total, d_tile_size, &total) || !checked_size_mul(col_vec_size, 4, &value) ||
      !checked_size_add(total, value, &total) || !checked_size_mul(row_vec_size, 2, &value) ||
      !checked_size_add(total, value, &total) || !checked_size_add(total, 512, &total)) {
    return false;
  }

  *workspace = total;
  (void) full_head;
  return true;
}

static bool block_sparse_fa_required_bytes(enum block_sparse_fa_mode mode, int gqa_factor, int head_dim, int kv_len,
                                           size_t blk_sz_c, size_t *required) {
  size_t workspace;
  size_t chunk;
  size_t full_head;
  size_t value;
  const size_t k_tiles = ((size_t) kv_len + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
  const size_t d_tiles = (size_t) head_dim / 32;

  if (!block_sparse_fa_workspace_bytes(gqa_factor, head_dim, blk_sz_c, &workspace, &chunk, &full_head) ||
      !checked_size_mul(k_tiles, d_tiles, &full_head) ||
      !checked_size_mul(full_head, HMX_FP16_TILE_SIZE, &full_head)) {
    return false;
  }

  switch (mode) {
    case BLOCK_SPARSE_FA_FULL_KV:
      if (!checked_size_mul(full_head, 2, &value)) {
        return false;
      }
      break;
    case BLOCK_SPARSE_FA_FULL_K:
      if (!checked_size_add(full_head, chunk, &value)) {
        return false;
      }
      break;
    case BLOCK_SPARSE_FA_CHUNKED:
      if (!checked_size_mul(chunk, 2, &value)) {
        return false;
      }
      break;
    default:
      return false;
  }
  return checked_size_add(workspace, value, required);
}

static bool block_sparse_fa_find_mode_plan(block_sparse_fa_plan_t *plan, enum block_sparse_fa_mode mode,
                                           int gqa_factor, int head_dim, int kv_len, size_t available_bytes) {
  size_t max_blk_sz_c;
  if (!checked_size_align_up((size_t) kv_len, 64, &max_blk_sz_c)) {
    return false;
  }

  bool found = false;
  for (size_t blk_sz_c = 64; blk_sz_c <= max_blk_sz_c; blk_sz_c += 64) {
    size_t required;
    if (!block_sparse_fa_required_bytes(mode, gqa_factor, head_dim, kv_len, blk_sz_c, &required)) {
      return false;
    }
    if (required <= available_bytes) {
      plan->mode           = mode;
      plan->blk_sz_c       = blk_sz_c;
      plan->required_bytes = required;
      found                = true;
    }
  }
  return found;
}

int block_sparse_fa_select_plan(block_sparse_fa_plan_t *plan, bool fast_path, int gqa_factor, int head_dim,
                                int kv_len, size_t available_bytes) {
  if (plan == NULL || gqa_factor <= 0 || head_dim <= 0 || head_dim % 32 != 0 || kv_len <= 0) {
    return -1;
  }

  if (fast_path && block_sparse_fa_find_mode_plan(plan, BLOCK_SPARSE_FA_FULL_KV, gqa_factor, head_dim, kv_len,
                                                  available_bytes)) {
    return 0;
  }
  if (fast_path && block_sparse_fa_find_mode_plan(plan, BLOCK_SPARSE_FA_FULL_K, gqa_factor, head_dim, kv_len,
                                                  available_bytes)) {
    return 0;
  }
  if (block_sparse_fa_find_mode_plan(plan, BLOCK_SPARSE_FA_CHUNKED, gqa_factor, head_dim, kv_len, available_bytes)) {
    return 0;
  }
  return -1;
}

int block_sparse_fa_select_halo_fallback_plan(block_sparse_fa_plan_t *plan, bool fast_path, int gqa_factor,
                                              int head_dim, int kv_len, size_t available_bytes) {
  return block_sparse_fa_select_plan(
    plan, fast_path, gqa_factor, head_dim, kv_len, available_bytes);
}

int block_sparse_fa_select_chunked_plan(block_sparse_fa_plan_t *plan, int gqa_factor, int head_dim,
                                        int kv_len, size_t available_bytes) {
  if (plan == NULL) {
    return -1;
  }
  return block_sparse_fa_find_mode_plan(
    plan, BLOCK_SPARSE_FA_CHUNKED, gqa_factor, head_dim, kv_len, available_bytes) ? 0 : -1;
}

int block_sparse_halo_parse_row(block_sparse_halo_row_t *parsed, const int32_t *row, int max_kept_blocks,
                                int q_block, int q_blocks_per_head, int k_blocks_per_head, int image_blocks) {
  if (parsed == NULL || row == NULL || max_kept_blocks <= 0 || q_block < 0 || q_blocks_per_head <= 0 ||
      k_blocks_per_head <= 0 || image_blocks <= 0 || image_blocks > k_blocks_per_head ||
      max_kept_blocks < k_blocks_per_head) {
    return -1;
  }

  block_sparse_halo_row_t result = {0};
  const int kv_head = q_block / q_blocks_per_head;
  if (kv_head > INT_MAX / k_blocks_per_head) {
    return -1;
  }
  const int q_block_in_head = q_block % q_blocks_per_head;
  const int first_k_block = kv_head * k_blocks_per_head;
  const bool is_text_query = q_block_in_head >= image_blocks;
  result.is_text_query = is_text_query;
  result.image_block_total = image_blocks;
  result.first_k_block = first_k_block;
  result.k_blocks_per_head = k_blocks_per_head;

  int entry_count = 0;
  while (entry_count < max_kept_blocks && row[entry_count] != -1) {
    const int32_t block = row[entry_count];
    if (block < first_k_block || block >= first_k_block + k_blocks_per_head ||
        (entry_count > 0 && block <= row[entry_count - 1])) {
      return -1;
    }
    ++entry_count;
  }
  int i = entry_count;
  for (; i <= max_kept_blocks - 8; i += 8) {
    const int32_t all_minus_one = row[i + 0] & row[i + 1] & row[i + 2] & row[i + 3] &
                                  row[i + 4] & row[i + 5] & row[i + 6] & row[i + 7];
    if (all_minus_one != -1) {
      return -1;
    }
  }
  for (; i < max_kept_blocks; ++i) {
    if (row[i] != -1) {
      return -1;
    }
  }

  if (is_text_query) {
    if (entry_count != k_blocks_per_head) {
      return -1;
    }
    for (int i = 0; i < entry_count; ++i) {
      if (row[i] != first_k_block + i) {
        return -1;
      }
    }
  } else {
    int image_count = 0;
    while (image_count < entry_count && row[image_count] < first_k_block + image_blocks) {
      if (image_count >= BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS) {
        return -1;
      }
      result.image_blocks[image_count] = row[image_count];
      ++image_count;
    }
    if (image_count != 4 && image_count != 6 && image_count != 9) {
      return -1;
    }
    const int text_blocks = k_blocks_per_head - image_blocks;
    if (entry_count != image_count + text_blocks) {
      return -1;
    }
    for (int i = 0; i < text_blocks; ++i) {
      if (row[image_count + i] != first_k_block + image_blocks + i) {
        return -1;
      }
    }
    result.image_count = image_count;
  }

  *parsed = result;
  return 0;
}

int block_sparse_halo_schedule_q_row(block_sparse_halo_row_schedule_t *schedule, int q_row, int image_block_rows) {
  if (schedule == NULL || q_row < 0 || image_block_rows <= 0 || q_row >= image_block_rows) {
    return -1;
  }

  const int first_active_row = q_row > 0 ? q_row - 1 : 0;
  const int last_active_row = q_row >= image_block_rows - 1 ? image_block_rows - 1 : q_row + 1;
  schedule->first_active_row = first_active_row;
  schedule->active_row_count = last_active_row - first_active_row + 1;
  schedule->load_row = q_row <= image_block_rows - 3 ? q_row + 2 : -1;
  return 0;
}

int block_sparse_halo_next_q_block(int current_block, int q_block_count) {
  if (current_block < 0 || q_block_count <= 0 || current_block >= q_block_count - 1) {
    return -1;
  }
  return current_block + 1;
}

int block_sparse_halo_schedule_text_row(block_sparse_halo_row_schedule_t *schedule, int image_row,
                                        int image_block_rows) {
  if (schedule == NULL || image_row < 0 || image_block_rows <= 0 || image_row >= image_block_rows) {
    return -1;
  }

  schedule->first_active_row = image_row;
  schedule->active_row_count = 1;
  schedule->load_row = image_row < image_block_rows - 1 ? image_row + 1 : -1;
  return 0;
}

int block_sparse_halo_text_prefetch_target(int prefetched_through, int chunk_last_row, int image_block_rows) {
  if (image_block_rows <= 0 || prefetched_through < 0 || prefetched_through >= image_block_rows ||
      chunk_last_row < 0 || chunk_last_row >= image_block_rows) {
    return -1;
  }

  const int next_row = chunk_last_row < image_block_rows - 1 ? chunk_last_row + 1 : chunk_last_row;
  return smax(prefetched_through, next_row);
}

int block_sparse_halo_find_ready_row_slot(const block_sparse_halo_row_slot_t slots[BLOCK_SPARSE_HALO_ROW_SLOTS],
                                          int row_id) {
  if (slots == NULL || row_id < 0) {
    return -1;
  }
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_ROW_SLOTS; ++slot) {
    if (slots[slot].row_id == row_id && slots[slot].ready && !slots[slot].outstanding) {
      return slot;
    }
  }
  return -1;
}

int block_sparse_halo_select_staging_slot(const block_sparse_halo_row_slot_t slots[BLOCK_SPARSE_HALO_ROW_SLOTS],
                                          int first_active_row, int active_row_count) {
  if (slots == NULL || first_active_row < 0 || active_row_count <= 0 ||
      active_row_count > BLOCK_SPARSE_HALO_ROW_SLOTS) {
    return -1;
  }
  if (first_active_row > INT_MAX - active_row_count) {
    return -1;
  }
  const int active_end = first_active_row + active_row_count;
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_ROW_SLOTS; ++slot) {
    const int row_id = slots[slot].row_id;
    if (!slots[slot].outstanding && (row_id < first_active_row || row_id >= active_end)) {
      return slot;
    }
  }
  return -1;
}

int block_sparse_halo_rotate_row_slots(block_sparse_halo_row_slot_t slots[BLOCK_SPARSE_HALO_ROW_SLOTS],
                                       int evicted_row, int new_row) {
  if (slots == NULL || evicted_row < 0 || new_row < 0 || evicted_row == new_row) {
    return -1;
  }
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_ROW_SLOTS; ++slot) {
    if (slots[slot].row_id == evicted_row) {
      if (slots[slot].outstanding) {
        return -1;
      }
      for (int other = 0; other < BLOCK_SPARSE_HALO_ROW_SLOTS; ++other) {
        if (other != slot && slots[other].row_id == new_row) {
          return -1;
        }
      }
      slots[slot].row_id = new_row;
      slots[slot].ready = false;
      return slot;
    }
  }
  return -1;
}

int block_sparse_halo_find_packed_row(
  const block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS], int row_id) {
  if (slots == NULL || row_id < 0) {
    return -1;
  }
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
    if (slots[slot].row_id == row_id && slots[slot].ready) {
      return slot;
    }
  }
  return -1;
}

int block_sparse_halo_select_packed_row_victim(
  const block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS],
  int first_active_row, int active_row_count) {
  if (slots == NULL || first_active_row < 0 || active_row_count <= 0 ||
      active_row_count > BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS ||
      first_active_row > INT_MAX - active_row_count) {
    return -1;
  }
  const int active_end = first_active_row + active_row_count;
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
    const int row_id = slots[slot].row_id;
    if (row_id < first_active_row || row_id >= active_end) {
      return slot;
    }
  }
  return -1;
}

int block_sparse_halo_bind_packed_row(
  block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS], int slot, int row_id) {
  if (slots == NULL || slot < 0 || slot >= BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS || row_id < 0) {
    return -1;
  }
  for (int other = 0; other < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++other) {
    if (other != slot && slots[other].row_id == row_id) {
      return -1;
    }
  }
  slots[slot].row_id = row_id;
  slots[slot].ready = true;
  return 0;
}

int block_sparse_halo_plan_text_kv_batch(
  block_sparse_halo_text_kv_batch_t *batch,
  const block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS],
  int first_pending_row, int latest_packed_row, int image_block_rows, bool flush_tail) {
  if (batch == NULL || slots == NULL || image_block_rows <= 0 ||
      first_pending_row < 0 || first_pending_row > image_block_rows ||
      latest_packed_row < -1 || latest_packed_row >= image_block_rows ||
      (first_pending_row > 0 && latest_packed_row < first_pending_row - 1) ||
      (first_pending_row == 0 && latest_packed_row < -1) ||
      (flush_tail && latest_packed_row != image_block_rows - 1)) {
    return -1;
  }

  int pending_rows = 0;
  if (latest_packed_row >= first_pending_row) {
    pending_rows = latest_packed_row - first_pending_row + 1;
  }
  if (pending_rows > BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS) {
    return -1;
  }

  int pending_slots[BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS] = {-1, -1, -1};
  for (int row_offset = 0; row_offset < pending_rows; ++row_offset) {
    const int row_id = first_pending_row + row_offset;
    int match_count = 0;
    for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
      if (slots[slot].ready && slots[slot].row_id == row_id) {
        pending_slots[row_offset] = slot;
        ++match_count;
      }
    }
    if (match_count != 1) {
      return -1;
    }
  }

  const int row_count = flush_tail ? pending_rows :
    (pending_rows == BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS ? pending_rows : 0);
  block_sparse_halo_text_kv_batch_t result = {
    .row_count = row_count,
    .next_pending_row = first_pending_row,
  };
  for (int row_offset = 0; row_offset < row_count; ++row_offset) {
    result.row_ids[row_offset] = first_pending_row + row_offset;
    result.packed_slots[row_offset] = pending_slots[row_offset];
  }
  if (row_count > image_block_rows - first_pending_row) {
    return -1;
  }
  result.next_pending_row = first_pending_row + row_count;
  *batch = result;
  return 0;
}

int block_sparse_halo_plan_update_segments(block_sparse_halo_update_segment_plan_t *plan,
                                           const int *real_cols, int count,
                                           size_t scratch_col_capacity) {
  if (plan == NULL || real_cols == NULL || count <= 0 ||
      count > BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS ||
      scratch_col_capacity == 0 || scratch_col_capacity > INT_MAX ||
      scratch_col_capacity % HMX_FP16_TILE_N_COLS != 0) {
    return -1;
  }

  block_sparse_halo_update_segment_plan_t result = {0};
  size_t total_cols = 0;
  for (int segment = 0; segment < count; ++segment) {
    const int cols = real_cols[segment];
    if (cols <= 0 ||
        (segment + 1 < count && cols % HMX_FP16_TILE_N_COLS != 0)) {
      return -1;
    }

    size_t next_total;
    if (!checked_size_add(total_cols, (size_t) cols, &next_total) ||
        next_total > scratch_col_capacity || next_total > INT_MAX) {
      return -1;
    }
    result.tile_offsets[segment] = (int) (total_cols / HMX_FP16_TILE_N_COLS);
    result.tile_counts[segment] = ceil_div(cols, HMX_FP16_TILE_N_COLS);
    total_cols = next_total;
  }
  result.total_cols = (int) total_cols;
  *plan = result;
  return 0;
}

int block_sparse_halo_get_packed_segment(block_sparse_halo_packed_segment_t *segment,
                                         const block_sparse_halo_packed_row_slot_t *slot,
                                         int col_start, int block_count, int image_block_cols, int head_dim) {
  if (segment == NULL || slot == NULL || !slot->ready || slot->packed_k == NULL || slot->packed_v == NULL ||
      col_start < 0 || block_count <= 0 || image_block_cols <= 0 ||
      col_start > image_block_cols - block_count || !packed_head_dim_supported(head_dim)) {
    return -1;
  }

  size_t k_tile_elems;
  size_t k_offset_bytes;
  size_t v_offset_bytes;
  if (!checked_size_mul(HMX_FP16_TILE_N_COLS, (size_t) head_dim, &k_tile_elems) ||
      !checked_size_mul((size_t) col_start, k_tile_elems, &k_tile_elems) ||
      !checked_size_mul(k_tile_elems, sizeof(__fp16), &k_offset_bytes) ||
      !checked_size_mul((size_t) col_start, HMX_FP16_TILE_SIZE, &v_offset_bytes)) {
    return -1;
  }

  const uintptr_t k_base = (uintptr_t) slot->packed_k;
  const uintptr_t v_base = (uintptr_t) slot->packed_v;
  if (k_offset_bytes > UINTPTR_MAX - k_base || v_offset_bytes > UINTPTR_MAX - v_base) {
    return -1;
  }

  block_sparse_halo_packed_segment_t result = {
    .packed_k = (const __fp16 *) (k_base + k_offset_bytes),
    .packed_v = (const __fp16 *) (v_base + v_offset_bytes),
    .v_tile_stride = image_block_cols,
    .block_count = block_count,
  };
  *segment = result;
  return 0;
}

bool block_sparse_halo_can_merge_index_block(bool row_bounded, int start_block, int next_block,
                                             int image_blocks, int image_block_cols,
                                             int current_cols, int max_cols) {
  if (start_block < 0 || next_block < 0 || current_cols <= 0 || max_cols <= 0 ||
      current_cols % HMX_FP16_TILE_N_COLS != 0 || max_cols % HMX_FP16_TILE_N_COLS != 0 ||
      max_cols < HMX_FP16_TILE_N_COLS) {
    return false;
  }
  const int current_blocks = current_cols / HMX_FP16_TILE_N_COLS;
  if (current_blocks <= 0 || start_block > INT_MAX - current_blocks ||
      next_block != start_block + current_blocks ||
      current_cols > max_cols - HMX_FP16_TILE_N_COLS) {
    return false;
  }
  if (row_bounded) {
    if (image_blocks <= 0 || image_block_cols <= 0) {
      return false;
    }
    if (start_block < image_blocks &&
        (next_block >= image_blocks ||
         next_block / image_block_cols != start_block / image_block_cols)) {
      return false;
    }
  }
  return true;
}

int block_sparse_halo_text_q_blocks(int text_tokens) {
  if (text_tokens < 0 || text_tokens > INT_MAX - (HMX_FP16_TILE_N_ROWS - 1)) {
    return -1;
  }
  return ceil_div(text_tokens, HMX_FP16_TILE_N_ROWS);
}

int block_sparse_halo_text_q_rows(int text_tokens, int text_q_block) {
  const int blocks = block_sparse_halo_text_q_blocks(text_tokens);
  if (blocks <= 0 || text_q_block < 0 || text_q_block >= blocks ||
      text_q_block > INT_MAX / HMX_FP16_TILE_N_ROWS) {
    return -1;
  }
  return smin(HMX_FP16_TILE_N_ROWS,
              text_tokens - text_q_block * HMX_FP16_TILE_N_ROWS);
}

int block_sparse_halo_text_o_next_bank(int current_bank, int completed_q_blocks, int total_q_blocks) {
  if ((current_bank != 0 && current_bank != 1) || total_q_blocks <= 0 ||
      completed_q_blocks != total_q_blocks) {
    return -1;
  }
  return current_bank ^ 1;
}

int block_sparse_halo_text_o_bank_after_image_row(int current_bank, int image_block_cols) {
  if ((current_bank != 0 && current_bank != 1) || image_block_cols <= 0) {
    return -1;
  }
  return current_bank ^ 1;
}

int block_sparse_halo_plan_packed_row_event(block_sparse_halo_packed_row_event_t *event,
                                            int row_id, int image_block_rows,
                                            int image_block_cols, int text_q_blocks) {
  if (event == NULL || row_id < 0 || row_id >= image_block_rows ||
      image_block_rows < 2 || image_block_cols <= 0 || text_q_blocks <= 0 ||
      image_block_cols > INT_MAX / text_q_blocks) {
    return -1;
  }
  const block_sparse_halo_packed_row_event_t result = {
    .row_id = row_id,
    .startup_pack = row_id < 2,
    .prefetch_row = row_id >= 2 ? row_id : -1,
    .evicted_row = row_id >= BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS ?
      row_id - BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS : -1,
    .text_q_updates = text_q_blocks,
    .image_block_count = image_block_cols,
    .text_q_block_updates = image_block_cols * text_q_blocks,
  };
  *event = result;
  return 0;
}

static bool block_sparse_halo_row_matches_grid(const block_sparse_halo_row_t *row, int q_block,
                                               int q_blocks_per_head, int k_blocks_per_head,
                                               int image_block_rows, int image_block_cols, int image_blocks) {
  if (row == NULL || row->is_text_query || q_blocks_per_head <= 0 || k_blocks_per_head <= 0 ||
      image_block_rows <= 0 || image_block_cols <= 0 || image_blocks <= 0 || q_block < 0) {
    return false;
  }

  const int q_block_in_head = q_block % q_blocks_per_head;
  if (q_block_in_head >= image_blocks) {
    return false;
  }
  const int qy = q_block_in_head / image_block_cols;
  const int qx = q_block_in_head % image_block_cols;
  const int first_k_block = (q_block / q_blocks_per_head) * k_blocks_per_head;
  int expected = 0;
  for (int ky = qy - 1; ky <= qy + 1; ++ky) {
    if (ky < 0 || ky >= image_block_rows) {
      continue;
    }
    for (int kx = qx - 1; kx <= qx + 1; ++kx) {
      if (kx < 0 || kx >= image_block_cols) {
        continue;
      }
      if (expected >= row->image_count || row->image_blocks[expected] != first_k_block + ky * image_block_cols + kx) {
        return false;
      }
      ++expected;
    }
  }
  return expected == row->image_count;
}

int block_sparse_halo_plan_prefetch(block_sparse_halo_prefetch_plan_t *plan,
                                    const block_sparse_halo_row_t *row, int first_image_block,
                                    int image_blocks, int text_group_start) {
  if (plan == NULL || row == NULL || first_image_block < 0 || image_blocks <= 0 || text_group_start < 0 ||
      text_group_start >= image_blocks) {
    return -1;
  }

  block_sparse_halo_prefetch_plan_t result = {0};
  int32_t blocks[BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS];
  const int count = row->is_text_query ? smin(BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS,
                                               image_blocks - text_group_start) : row->image_count;
  if (count <= 0 || count > BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS) {
    return -1;
  }
  for (int i = 0; i < count; ++i) {
    blocks[i] = row->is_text_query ? first_image_block + text_group_start + i : row->image_blocks[i];
    if (blocks[i] < first_image_block || blocks[i] >= first_image_block + image_blocks ||
        (i > 0 && blocks[i] <= blocks[i - 1])) {
      return -1;
    }
  }

  for (int i = 0; i < count; ++i) {
    if (result.range_count == 0 ||
        blocks[i] != result.ranges[result.range_count - 1].first_block +
                       result.ranges[result.range_count - 1].block_count) {
      result.ranges[result.range_count].first_block = blocks[i];
      result.ranges[result.range_count].block_count = 1;
      ++result.range_count;
    } else {
      ++result.ranges[result.range_count - 1].block_count;
    }
  }
  result.block_count = count;
  *plan = result;
  return 0;
}

int block_sparse_halo_build_dma_descs(dma_desc_2d_t *descs, int desc_capacity, const void *k_base,
                                      const void *v_base, void *raw_bank, int n_kv_heads, int kv_head,
                                      int head_dim, int k_blocks_per_head, const block_sparse_halo_row_t *row) {
  if (descs == NULL || ((uintptr_t) descs & 63u) != 0 || k_base == NULL || v_base == NULL || raw_bank == NULL ||
      n_kv_heads <= 0 || kv_head < 0 || kv_head >= n_kv_heads || head_dim <= 0 || k_blocks_per_head <= 0 ||
      row == NULL || row->is_text_query || row->image_count <= 0 ||
      row->image_count > BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS || desc_capacity < row->image_count * 2) {
    return -1;
  }

  size_t row_bytes;
  size_t src_stride;
  size_t block_bytes;
  size_t pair_bytes;
  if (!checked_size_mul((size_t) head_dim, sizeof(__fp16), &row_bytes) ||
      !checked_size_mul((size_t) n_kv_heads, row_bytes, &src_stride) ||
      !checked_size_mul(HMX_FP16_TILE_N_COLS, row_bytes, &block_bytes) ||
      !checked_size_mul(block_bytes, 2, &pair_bytes)) {
    return -1;
  }

  const uintptr_t k_addr = (uintptr_t) k_base;
  const uintptr_t v_addr = (uintptr_t) v_base;
  const uintptr_t bank_addr = (uintptr_t) raw_bank;
  if (k_addr > UINT32_MAX || v_addr > UINT32_MAX || bank_addr > UINT32_MAX) {
    return -1;
  }

  const int first_head_block = kv_head * k_blocks_per_head;
  for (int slot = 0; slot < row->image_count; ++slot) {
    const int local_block = row->image_blocks[slot] - first_head_block;
    size_t block_token;
    size_t token_offset;
    size_t head_offset;
    size_t src_offset;
    size_t src_last_row_offset;
    size_t src_last_byte;
    size_t bank_offset;
    size_t bank_last_byte;
    if (local_block < 0 || local_block >= k_blocks_per_head ||
        !checked_size_mul((size_t) local_block, HMX_FP16_TILE_N_COLS, &block_token) ||
        !checked_size_mul(block_token, src_stride, &token_offset) ||
        !checked_size_mul((size_t) kv_head, row_bytes, &head_offset) ||
        !checked_size_add(token_offset, head_offset, &src_offset) ||
        !checked_size_mul(HMX_FP16_TILE_N_COLS - 1, src_stride, &src_last_row_offset) ||
        !checked_size_add(src_offset, src_last_row_offset, &src_last_byte) ||
        !checked_size_add(src_last_byte, row_bytes - 1, &src_last_byte) ||
        !checked_size_mul((size_t) slot, pair_bytes, &bank_offset) ||
        !checked_size_add(bank_offset, pair_bytes - 1, &bank_last_byte) ||
        src_last_byte > UINT32_MAX || bank_last_byte > UINT32_MAX ||
        k_addr > UINT32_MAX - src_last_byte || v_addr > UINT32_MAX - src_last_byte ||
        bank_addr > UINT32_MAX - bank_last_byte) {
      return -1;
    }

    const int k_desc = slot * 2;
    const int v_desc = k_desc + 1;
    dma_desc_2d_t *k_next = &descs[v_desc];
    dma_desc_2d_t *v_next = v_desc + 1 < row->image_count * 2 ? &descs[v_desc + 1] : NULL;
    if (dma_desc_2d_init(&descs[k_desc], k_next, (const void *) (k_addr + src_offset),
                         (void *) (bank_addr + bank_offset), row_bytes, HMX_FP16_TILE_N_COLS,
                         src_stride, row_bytes) != DMA_SUBMIT_OK ||
        dma_desc_2d_init(&descs[v_desc], v_next, (const void *) (v_addr + src_offset),
                         (void *) (bank_addr + bank_offset + block_bytes), row_bytes, HMX_FP16_TILE_N_COLS,
                         src_stride, row_bytes) != DMA_SUBMIT_OK) {
      return -1;
    }
  }
  return row->image_count * 2;
}

int block_sparse_halo_build_dma_row_descs(dma_desc_2d_t *descs, int desc_capacity, const void *k_base,
                                          const void *v_base, void *raw_row, int n_kv_heads, int kv_head,
                                          int head_dim, int image_block_cols, int row_id) {
  if (descs == NULL || ((uintptr_t) descs & 63u) != 0 || k_base == NULL || v_base == NULL || raw_row == NULL ||
      desc_capacity < BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW || n_kv_heads <= 0 || kv_head < 0 ||
      kv_head >= n_kv_heads || head_dim <= 0 || head_dim % 64 != 0 || image_block_cols <= 0 || row_id < 0 ||
      image_block_cols > UINT16_MAX / HMX_FP16_TILE_N_COLS) {
    return -1;
  }

  size_t row_bytes;
  size_t src_stride;
  size_t row_height;
  size_t row_half_bytes;
  size_t row_token_start;
  size_t token_offset;
  size_t head_offset;
  size_t src_offset;
  size_t src_last_row_offset;
  size_t src_last_byte;
  size_t v_dst_offset;
  size_t raw_last_byte;
  if (!checked_size_mul((size_t) head_dim, sizeof(__fp16), &row_bytes) ||
      !checked_size_mul((size_t) n_kv_heads, row_bytes, &src_stride) ||
      !checked_size_mul((size_t) image_block_cols, HMX_FP16_TILE_N_COLS, &row_height) ||
      !checked_size_mul(row_height, row_bytes, &row_half_bytes) ||
      !checked_size_mul((size_t) row_id, row_height, &row_token_start) ||
      !checked_size_mul(row_token_start, src_stride, &token_offset) ||
      !checked_size_mul((size_t) kv_head, row_bytes, &head_offset) ||
      !checked_size_mul(row_height - 1, src_stride, &src_last_row_offset) ||
      !checked_size_add(token_offset, head_offset, &src_offset) ||
      !checked_size_add(src_offset, src_last_row_offset, &src_last_byte) ||
      !checked_size_add(src_last_byte, row_bytes - 1, &src_last_byte) ||
      !checked_size_mul(row_half_bytes, 2, &raw_last_byte) || raw_last_byte == 0 ||
      row_bytes > UINT16_MAX || src_stride > UINT16_MAX || row_height > UINT16_MAX ||
      src_last_byte > UINT32_MAX) {
    return -1;
  }
  --raw_last_byte;
  if (raw_last_byte > UINT32_MAX) {
    return -1;
  }

  const uintptr_t k_addr = (uintptr_t) k_base;
  const uintptr_t v_addr = (uintptr_t) v_base;
  const uintptr_t raw_addr = (uintptr_t) raw_row;
  if (k_addr > UINT32_MAX || v_addr > UINT32_MAX || raw_addr > UINT32_MAX ||
      k_addr > UINT32_MAX - src_last_byte || v_addr > UINT32_MAX - src_last_byte ||
      raw_addr > UINT32_MAX - raw_last_byte) {
    return -1;
  }

  if (!checked_size_mul(row_half_bytes, 1, &v_dst_offset)) {
    return -1;
  }
  if (dma_desc_2d_init(&descs[0], &descs[1], (const void *) (k_addr + src_offset),
                       (void *) raw_addr, row_bytes, row_height, src_stride, row_bytes) != DMA_SUBMIT_OK ||
      dma_desc_2d_init(&descs[1], NULL, (const void *) (v_addr + src_offset),
                       (void *) (raw_addr + v_dst_offset), row_bytes, row_height, src_stride, row_bytes) !=
        DMA_SUBMIT_OK) {
    return -1;
  }
  return BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW;
}

static int block_sparse_halo_build_packed_dma_row_descs(
  dma_desc_2d_t *descs, int desc_capacity, const __fp16 *K, const __fp16 *V,
  void *packed_row, int kv_head_idx, int head_dim, int token_tiles,
  int image_block_cols, int row_id) {
  if (descs == NULL || ((uintptr_t) descs & 63u) != 0 || K == NULL || V == NULL ||
      packed_row == NULL || kv_head_idx < 0 || !packed_head_dim_supported(head_dim) || token_tiles <= 0 ||
      image_block_cols <= 0 || row_id < 0 ||
      desc_capacity < block_sparse_halo_packed_dma_desc_count(head_dim)) {
    return -1;
  }
  const int token_tile = row_id * image_block_cols;
  if (token_tile > token_tiles - image_block_cols) {
    return -1;
  }
  const size_t k_tile_bytes = (size_t) head_dim * HMX_FP16_TILE_N_ROWS * sizeof(__fp16);
  const size_t v_tile_bytes = HMX_FP16_TILE_SIZE;
  const int dim_tiles = head_dim / HMX_FP16_TILE_N_COLS;
  const size_t row_half_bytes = (size_t) image_block_cols * k_tile_bytes;
  size_t k_offset;
  size_t v_offset[4] = {};
  if (block_sparse_packed_qk_tile_offset(
        &k_offset, kv_head_idx, token_tile, 0, token_tiles, head_dim) != 0 ||
      dim_tiles > (int) (sizeof(v_offset) / sizeof(v_offset[0]))) {
    return -1;
  }
  for (int d = 0; d < dim_tiles; ++d) {
    if (block_sparse_packed_v_tile_offset(
          &v_offset[d], kv_head_idx, token_tile, d, token_tiles, head_dim) != 0) {
      return -1;
    }
  }
  uint8_t *dst = (uint8_t *) packed_row;
  const int desc_count = block_sparse_halo_packed_dma_desc_count(head_dim);
  if (dma_desc_2d_init(
        &descs[0], &descs[1], K + k_offset, dst,
        k_tile_bytes, image_block_cols, k_tile_bytes, k_tile_bytes) != DMA_SUBMIT_OK) {
    return -1;
  }
  for (int d = 0; d < dim_tiles; ++d) {
    const int desc_idx = 1 + d;
    dma_desc_2d_t *next = desc_idx + 1 < desc_count ? &descs[desc_idx + 1] : NULL;
    if (dma_desc_2d_init(
          &descs[desc_idx], next, V + v_offset[d],
          dst + row_half_bytes + (size_t) d * image_block_cols * v_tile_bytes,
          v_tile_bytes, image_block_cols, v_tile_bytes, v_tile_bytes) != DMA_SUBMIT_OK) {
      return -1;
    }
  }
  return desc_count;
}

#define BLOCK_SPARSE_USERDMA_PROBE_BYTES (64 * 1024)
#define BLOCK_SPARSE_USERDMA_PROBE_GUARD 128
#define BLOCK_SPARSE_USERDMA_PROBE_ITERS 100

typedef struct __attribute__((aligned(64))) {
  dma_desc_2d_t desc[2];
} block_sparse_userdma_probe_desc_t;

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  unsigned int ready_count;
  unsigned int start;
  unsigned int dma_owner;
  int n_jobs;
  bool serialize_dma;
  const uint8_t *src[2];
  uint8_t *vtcm_base;
  size_t vtcm_slice_size;
  block_sparse_userdma_probe_desc_t descs[2];
  int submit_result[2];
  int wait_result[2];
  bool data_ok[2];
  bool guard_ok[2];
  int worker_index[2];
  int64_t begin_ticks[2];
  int64_t end_ticks[2];
} block_sparse_userdma_probe_state_t;

static int block_sparse_halo_dma_wait_bounded(const dma_desc_2d_t *descs, size_t count,
                                              int64_t timeout_us) {
  const int64_t deadline = HAP_perf_get_time_us() + timeout_us;
  int result;
  while ((result = dma_chain_poll(descs, count)) == DMA_SUBMIT_BUSY) {
    if (HAP_perf_get_time_us() >= deadline) {
      // Descriptor storage is worker-local. Quiesce the engine before the
      // caller unwinds, even though the operation is still reported as timed out.
      (void) dma_chain_wait(descs, count);
      return DMA_SUBMIT_HW_ERROR;
    }
    qurt_timer_sleep(1);
  }
  return result;
}

static void block_sparse_userdma_probe_worker(void *data, int worker_index) {
  block_sparse_userdma_probe_state_t *state = (block_sparse_userdma_probe_state_t *) data;
  const unsigned int slot = worker_pool_atomic_inc_return(&state->task_id) - 1;
  if (slot >= (unsigned int) state->n_jobs || worker_index < 0 ||
      (size_t) worker_index > SIZE_MAX / state->vtcm_slice_size) {
    worker_pool_synctoken_jobdone(&state->sync_ctx);
    return;
  }

  uint8_t *slice = state->vtcm_base + (size_t) worker_index * state->vtcm_slice_size;
  uint8_t *dst = slice + BLOCK_SPARSE_USERDMA_PROBE_GUARD;
  memset(slice, 0xa5, BLOCK_SPARSE_USERDMA_PROBE_GUARD);
  memset(dst, 0, BLOCK_SPARSE_USERDMA_PROBE_BYTES);
  memset(dst + BLOCK_SPARSE_USERDMA_PROBE_BYTES, 0x5a, BLOCK_SPARSE_USERDMA_PROBE_GUARD);
  state->worker_index[slot] = worker_index;
  const size_t half_bytes = BLOCK_SPARSE_USERDMA_PROBE_BYTES / 2;
  state->submit_result[slot] = dma_desc_2d_init(&state->descs[slot].desc[0], &state->descs[slot].desc[1],
                                                state->src[slot], dst, 128, half_bytes / 128, 128, 128);
  if (state->submit_result[slot] == DMA_SUBMIT_OK) {
    state->submit_result[slot] = dma_desc_2d_init(&state->descs[slot].desc[1], NULL,
                                                  state->src[slot] + half_bytes, dst + half_bytes,
                                                  128, half_bytes / 128, 128, 128);
  }
  worker_pool_atomic_inc_return(&state->ready_count);
  while (__atomic_load_n(&state->start, __ATOMIC_ACQUIRE) == 0) {
    qurt_timer_sleep(1);
  }

  if (state->serialize_dma) {
    unsigned int expected;
    do {
      expected = 0;
      if (!__atomic_compare_exchange_n(&state->dma_owner, &expected, slot + 1, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        qurt_timer_sleep(1);
      }
    } while (expected != 0);
  }

  state->begin_ticks[slot] = HAP_perf_get_qtimer_count();
  if (state->submit_result[slot] == DMA_SUBMIT_OK) {
    state->submit_result[slot] = dma_chain_submit(&state->descs[slot].desc[0]);
  }
  if (state->submit_result[slot] == DMA_SUBMIT_OK) {
    qurt_timer_sleep(1);
    state->wait_result[slot] = block_sparse_halo_dma_wait_bounded(&state->descs[slot].desc[0], 2, 1000000);
  }
  state->end_ticks[slot] = HAP_perf_get_qtimer_count();
  if (state->serialize_dma) {
    __atomic_store_n(&state->dma_owner, 0, __ATOMIC_RELEASE);
  }

  if (state->submit_result[slot] == DMA_SUBMIT_OK && state->wait_result[slot] == DMA_SUBMIT_OK) {
    state->data_ok[slot] = memcmp(dst, state->src[slot], BLOCK_SPARSE_USERDMA_PROBE_BYTES) == 0;
  }
  state->guard_ok[slot] = true;
  for (int i = 0; i < BLOCK_SPARSE_USERDMA_PROBE_GUARD; ++i) {
    if (slice[i] != 0xa5 || dst[BLOCK_SPARSE_USERDMA_PROBE_BYTES + i] != 0x5a) {
      state->guard_ok[slot] = false;
      break;
    }
  }
  worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static int block_sparse_userdma_probe_batch(block_sparse_userdma_probe_state_t *state, int n_jobs,
                                            bool serialize_dma) {
  state->task_id = 0;
  state->ready_count = 0;
  state->start = 0;
  state->dma_owner = 0;
  state->n_jobs = n_jobs;
  state->serialize_dma = serialize_dma;
  for (int i = 0; i < 2; ++i) {
    state->submit_result[i] = DMA_SUBMIT_BAD_DESC;
    state->wait_result[i] = DMA_SUBMIT_BAD_DESC;
    state->data_ok[i] = false;
    state->guard_ok[i] = false;
    state->worker_index[i] = -1;
  }

  worker_pool_job_t job = { .fptr = block_sparse_userdma_probe_worker, .dptr = state };
  worker_pool_synctoken_init(&state->sync_ctx, n_jobs);
  for (int i = 0; i < n_jobs; ++i) {
    if (worker_pool_submit(NULL, job) != AEE_SUCCESS) {
      __atomic_store_n(&state->start, 1, __ATOMIC_RELEASE);
      for (int unsent = i; unsent < n_jobs; ++unsent) {
        worker_pool_synctoken_jobdone(&state->sync_ctx);
      }
      worker_pool_synctoken_wait(&state->sync_ctx);
      return -1;
    }
  }
  while (__atomic_load_n(&state->ready_count, __ATOMIC_ACQUIRE) < (unsigned int) n_jobs) {
    qurt_timer_sleep(1);
  }
  __atomic_store_n(&state->start, 1, __ATOMIC_RELEASE);
  worker_pool_synctoken_wait(&state->sync_ctx);
  return 0;
}

int block_sparse_halo_classify_userdma_pair(const int submit_result[2], const int wait_result[2],
                                            const bool data_ok[2], const bool guard_ok[2],
                                            const int worker_index[2]) {
  if (submit_result == NULL || wait_result == NULL || data_ok == NULL || guard_ok == NULL ||
      worker_index == NULL || !guard_ok[0] || !guard_ok[1] || worker_index[0] < 0 || worker_index[1] < 0 ||
      worker_index[0] == worker_index[1]) {
    return BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
  }

  int ok_count = 0;
  int busy_count = 0;
  for (int slot = 0; slot < 2; ++slot) {
    if (submit_result[slot] == DMA_SUBMIT_OK && wait_result[slot] == DMA_SUBMIT_OK && data_ok[slot]) {
      ++ok_count;
    } else if (submit_result[slot] == DMA_SUBMIT_BUSY && !data_ok[slot]) {
      ++busy_count;
    } else {
      return BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
    }
  }
  if (ok_count == 2 && busy_count == 0) {
    return BLOCK_SPARSE_HALO_USERDMA_PER_WORKER;
  }
  if (ok_count == 1 && busy_count == 1) {
    return BLOCK_SPARSE_HALO_USERDMA_SHARED;
  }
  return BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
}

int block_sparse_halo_finalize_userdma_scope(int concurrent_scope, bool serialized_ok) {
  if (!serialized_ok) {
    return BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
  }
  return concurrent_scope == BLOCK_SPARSE_HALO_USERDMA_PER_WORKER ?
    BLOCK_SPARSE_HALO_USERDMA_PER_WORKER : BLOCK_SPARSE_HALO_USERDMA_SHARED;
}

int block_sparse_halo_probe_userdma_scope(void) {
  static int cached_scope = BLOCK_SPARSE_HALO_USERDMA_UNKNOWN;
  if (cached_scope != BLOCK_SPARSE_HALO_USERDMA_UNKNOWN) {
    return cached_scope;
  }

  block_sparse_userdma_probe_state_t *state = NULL;
  uint8_t *src0 = NULL;
  uint8_t *src1 = NULL;
  if (posix_memalign((void **) &state, 64, sizeof(*state)) != 0 ||
      posix_memalign((void **) &src0, 128, BLOCK_SPARSE_USERDMA_PROBE_BYTES) != 0 ||
      posix_memalign((void **) &src1, 128, BLOCK_SPARSE_USERDMA_PROBE_BYTES) != 0) {
    free(state);
    free(src0);
    free(src1);
    cached_scope = BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
    return cached_scope;
  }
  memset(state, 0, sizeof(*state));
  for (int i = 0; i < BLOCK_SPARSE_USERDMA_PROBE_BYTES; ++i) {
    src0[i] = (uint8_t) (i * 17 + 3);
    src1[i] = (uint8_t) (i * 29 + 11);
  }
  qurt_mem_cache_clean((qurt_addr_t) src0, BLOCK_SPARSE_USERDMA_PROBE_BYTES,
                       QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t) src1, BLOCK_SPARSE_USERDMA_PROBE_BYTES,
                       QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  state->src[0] = src0;
  state->src[1] = src1;
  state->vtcm_base = (uint8_t *) vtcm_manager_get_vtcm_base();
  const size_t vtcm_size = vtcm_manager_get_vtcm_size();
  state->vtcm_slice_size = vtcm_size / num_workers / (64 * 1024) * (64 * 1024);
  if (num_workers < 2 || state->vtcm_base == NULL ||
      state->vtcm_slice_size < BLOCK_SPARSE_USERDMA_PROBE_BYTES + 2 * BLOCK_SPARSE_USERDMA_PROBE_GUARD) {
    FARF(ALWAYS, "HTP_USERDMA_PROBE_FAIL stage=capacity workers=%u vtcm=%u slice=%u",
         num_workers, (unsigned int) vtcm_size, (unsigned int) state->vtcm_slice_size);
    cached_scope = BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
    goto done;
  }

  for (int iter = 0; iter < BLOCK_SPARSE_USERDMA_PROBE_ITERS; ++iter) {
    if (block_sparse_userdma_probe_batch(state, 1, false) != 0 || state->submit_result[0] != DMA_SUBMIT_OK ||
        state->wait_result[0] != DMA_SUBMIT_OK || !state->data_ok[0] || !state->guard_ok[0]) {
      FARF(ALWAYS,
           "HTP_USERDMA_PROBE_FAIL stage=single iter=%d submit=%d wait=%d data=%d guard=%d worker=%d",
           iter, state->submit_result[0], state->wait_result[0], state->data_ok[0] ? 1 : 0,
           state->guard_ok[0] ? 1 : 0, state->worker_index[0]);
      cached_scope = BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
      goto done;
    }
  }

  int observed_scope = BLOCK_SPARSE_HALO_USERDMA_UNKNOWN;
  for (int iter = 0; iter < BLOCK_SPARSE_USERDMA_PROBE_ITERS; ++iter) {
    if (block_sparse_userdma_probe_batch(state, 2, false) != 0) {
      observed_scope = BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
      break;
    }
    const int iter_scope = block_sparse_halo_classify_userdma_pair(
      state->submit_result, state->wait_result, state->data_ok, state->guard_ok, state->worker_index);
    if (iter_scope == BLOCK_SPARSE_HALO_USERDMA_UNSAFE ||
        (observed_scope != BLOCK_SPARSE_HALO_USERDMA_UNKNOWN && observed_scope != iter_scope)) {
      FARF(ALWAYS,
           "HTP_USERDMA_PROBE_NOTE stage=concurrent iter=%d scope=%d submit=%d,%d wait=%d,%d "
           "data=%d,%d guard=%d,%d worker=%d,%d",
           iter, iter_scope, state->submit_result[0], state->submit_result[1],
           state->wait_result[0], state->wait_result[1], state->data_ok[0] ? 1 : 0,
           state->data_ok[1] ? 1 : 0, state->guard_ok[0] ? 1 : 0, state->guard_ok[1] ? 1 : 0,
           state->worker_index[0], state->worker_index[1]);
      observed_scope = BLOCK_SPARSE_HALO_USERDMA_UNSAFE;
      break;
    }
    observed_scope = iter_scope;
  }

  bool serialized_ok = true;
  for (int iter = 0; iter < BLOCK_SPARSE_USERDMA_PROBE_ITERS; ++iter) {
    if (block_sparse_userdma_probe_batch(state, 2, true) != 0 ||
        block_sparse_halo_classify_userdma_pair(
          state->submit_result, state->wait_result, state->data_ok, state->guard_ok,
          state->worker_index) != BLOCK_SPARSE_HALO_USERDMA_PER_WORKER) {
      FARF(ALWAYS,
           "HTP_USERDMA_PROBE_FAIL stage=serialized iter=%d submit=%d,%d wait=%d,%d "
           "data=%d,%d guard=%d,%d worker=%d,%d",
           iter, state->submit_result[0], state->submit_result[1],
           state->wait_result[0], state->wait_result[1], state->data_ok[0] ? 1 : 0,
           state->data_ok[1] ? 1 : 0, state->guard_ok[0] ? 1 : 0, state->guard_ok[1] ? 1 : 0,
           state->worker_index[0], state->worker_index[1]);
      serialized_ok = false;
      break;
    }
  }
  cached_scope = block_sparse_halo_finalize_userdma_scope(observed_scope, serialized_ok);

done:
  FARF(ALWAYS, "HTP_USERDMA_SCOPE=%s single_iters=%d concurrent_iters=%d serialized_iters=%d",
       cached_scope == BLOCK_SPARSE_HALO_USERDMA_PER_WORKER ? "PER_WORKER" :
       (cached_scope == BLOCK_SPARSE_HALO_USERDMA_SHARED ? "SHARED" : "UNSAFE"),
       BLOCK_SPARSE_USERDMA_PROBE_ITERS, BLOCK_SPARSE_USERDMA_PROBE_ITERS,
       BLOCK_SPARSE_USERDMA_PROBE_ITERS);
  free(src1);
  free(src0);
  free(state);
  return cached_scope;
}

int block_sparse_halo_validate_shape(const int32_t *block_mask, int qo_len, int kv_len,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     int block_q, int block_k, int n_q_blocks, int n_k_blocks,
                                     int mask_format, int max_kept_blocks, int sparse_policy,
                                     int image_tokens, int text_tokens,
                                     int image_block_rows, int image_block_cols) {
  if (block_mask == NULL || sparse_policy != 1 || mask_format != 1 || n_heads <= 0 || n_heads != n_kv_heads ||
      head_dim <= 0 || head_dim % 64 != 0 || block_q != 32 || block_k != 32 || image_tokens <= 0 ||
      image_tokens % 32 != 0 || text_tokens < 0 || image_tokens > INT_MAX - text_tokens || qo_len != kv_len ||
      qo_len != image_tokens + text_tokens || image_block_rows <= 0 || image_block_cols <= 0 ||
      image_block_rows > INT_MAX / image_block_cols) {
    return -1;
  }

  const int grid_image_blocks = image_block_rows * image_block_cols;
  if (grid_image_blocks > INT_MAX / block_k || grid_image_blocks * block_k != image_tokens) {
    return -1;
  }

  const int q_blocks_per_head = qo_len / block_q + (qo_len % block_q != 0);
  const int k_blocks_per_head = kv_len / block_k + (kv_len % block_k != 0);
  const int image_blocks = image_tokens / block_k;
  if (q_blocks_per_head > INT_MAX / n_heads || k_blocks_per_head > INT_MAX / n_kv_heads ||
      n_q_blocks != n_heads * q_blocks_per_head || n_k_blocks != n_kv_heads * k_blocks_per_head ||
      max_kept_blocks < k_blocks_per_head || image_blocks > q_blocks_per_head || image_blocks > k_blocks_per_head ||
      !block_sparse_halo_mask_span_fits(n_heads, q_blocks_per_head, max_kept_blocks)) {
    return -1;
  }
  return 0;
}

int block_sparse_halo_validate_kv_head_rows(const int32_t *block_mask, int kv_head_idx,
                                            int n_kv_heads, int q_blocks_per_head,
                                            int k_blocks_per_head, int max_kept_blocks,
                                            int image_blocks, int image_block_rows,
                                            int image_block_cols) {
  if (block_mask == NULL || kv_head_idx < 0 || kv_head_idx >= n_kv_heads ||
      q_blocks_per_head <= 0 || k_blocks_per_head <= 0 ||
      max_kept_blocks < k_blocks_per_head || image_blocks <= 0 ||
      image_blocks > q_blocks_per_head || image_blocks > k_blocks_per_head ||
      image_block_rows <= 0 || image_block_cols <= 0 ||
      image_block_rows > INT_MAX / image_block_cols ||
      image_block_rows * image_block_cols != image_blocks ||
      q_blocks_per_head > INT_MAX / n_kv_heads ||
      k_blocks_per_head > INT_MAX / n_kv_heads) {
    return -1;
  }

  if (!block_sparse_halo_mask_span_fits(n_kv_heads, q_blocks_per_head, max_kept_blocks)) {
    return -1;
  }

  const int first_q_block = kv_head_idx * q_blocks_per_head;
  for (int qb_in_head = 0; qb_in_head < q_blocks_per_head; ++qb_in_head) {
    const int qb = first_q_block + qb_in_head;
    size_t row_offset;
    if (!checked_size_mul((size_t) qb, (size_t) max_kept_blocks, &row_offset)) {
      return -1;
    }
    block_sparse_halo_row_t parsed;
    if (block_sparse_halo_parse_row(&parsed, block_mask + row_offset,
                                    max_kept_blocks, qb, q_blocks_per_head, k_blocks_per_head, image_blocks) != 0) {
      return -1;
    }
    if (!parsed.is_text_query &&
        !block_sparse_halo_row_matches_grid(&parsed, qb, q_blocks_per_head, k_blocks_per_head,
                                            image_block_rows, image_block_cols, image_blocks)) {
      return -1;
    }
  }
  return 0;
}

int block_sparse_halo_validate(const int32_t *block_mask, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                               int head_dim, int block_q, int block_k, int n_q_blocks, int n_k_blocks,
                               int mask_format, int max_kept_blocks, int sparse_policy, int image_tokens,
                               int text_tokens, int image_block_rows, int image_block_cols) {
  if (block_sparse_halo_validate_shape(
        block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim,
        block_q, block_k, n_q_blocks, n_k_blocks, mask_format, max_kept_blocks,
        sparse_policy, image_tokens, text_tokens, image_block_rows, image_block_cols) != 0) {
    return -1;
  }

  const int q_blocks_per_head = n_q_blocks / n_heads;
  const int k_blocks_per_head = n_k_blocks / n_kv_heads;
  const int image_blocks = image_tokens / block_k;
  for (int kv_head_idx = 0; kv_head_idx < n_kv_heads; ++kv_head_idx) {
    if (block_sparse_halo_validate_kv_head_rows(
          block_mask, kv_head_idx, n_kv_heads, q_blocks_per_head,
          k_blocks_per_head, max_kept_blocks, image_blocks,
          image_block_rows, image_block_cols) != 0) {
      return -1;
    }
  }
  return 0;
}

int block_sparse_halo_select_plan(block_sparse_halo_plan_t *plan, int halo_kv_load_mode, int head_dim,
                                  int text_tokens, int image_block_cols, size_t available_bytes) {
  if (plan == NULL || (halo_kv_load_mode != 1 && halo_kv_load_mode != 2) || head_dim <= 0 ||
      head_dim % 64 != 0 || text_tokens < 0 ||
      text_tokens > INT_MAX - (HMX_FP16_TILE_N_COLS - 1) || image_block_cols <= 0 ||
      image_block_cols > INT_MAX / HMX_FP16_TILE_N_COLS ||
      (halo_kv_load_mode == 2 && image_block_cols > UINT16_MAX / HMX_FP16_TILE_N_COLS)) {
    return -1;
  }

  block_sparse_halo_plan_t result = {0};
  size_t cursor = 0;
  size_t value;
  size_t qo_tile_size;
  size_t kv_tile_size;
  size_t core_tile_size;
  size_t d_tile_size;
  size_t col_vec_size;
  size_t row_vec_size;
  size_t text_blk_sz_c;
  size_t scratch_blk_sz_c;
  size_t three_row_cols;
  size_t image_q_cols;
  result.blk_sz_c = (size_t) smax(3, image_block_cols) * HMX_FP16_TILE_N_COLS;
  if (!checked_size_align_up((size_t) text_tokens, HMX_FP16_TILE_N_COLS, &text_blk_sz_c)) {
    return -1;
  }
  if (!checked_size_mul((size_t) image_block_cols,
                        BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS * HMX_FP16_TILE_N_COLS,
                        &three_row_cols) ||
      three_row_cols > INT_MAX) {
    return -1;
  }
  if (!checked_size_mul((size_t) smin(3, image_block_cols),
                        BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS * HMX_FP16_TILE_N_COLS,
                        &image_q_cols) ||
      !checked_size_add(image_q_cols, text_blk_sz_c, &image_q_cols) ||
      image_q_cols > INT_MAX) {
    return -1;
  }
  scratch_blk_sz_c = smax(smax(result.blk_sz_c, text_blk_sz_c),
                          smax(three_row_cols, image_q_cols));
  result.scratch_col_capacity = scratch_blk_sz_c;
  if (!checked_size_mul(32, (size_t) head_dim, &value) || !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &qo_tile_size) ||
      !checked_size_mul(result.blk_sz_c, (size_t) head_dim, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &value) || !checked_size_align_up(value, 4096, &kv_tile_size) ||
      !checked_size_mul(32, scratch_blk_sz_c, &value) || !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &core_tile_size) ||
      !checked_size_mul(32, 32, &value) || !checked_size_mul(value, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 4096, &d_tile_size) ||
      !checked_size_align_up(32 * sizeof(__fp16), 256, &col_vec_size) ||
      !checked_size_mul(scratch_blk_sz_c, sizeof(__fp16), &value) ||
      !checked_size_align_up(value, 256, &row_vec_size)) {
    return -1;
  }

#define HALO_PLAN_ALLOC(field, bytes)                     \
  do {                                                    \
    result.field = cursor;                                \
    if (!checked_size_add(cursor, (bytes), &cursor)) {    \
      return -1;                                          \
    }                                                     \
  } while (0)

  result.workspace_offset = 0;
  HALO_PLAN_ALLOC(q_offset, qo_tile_size);
  HALO_PLAN_ALLOC(o_offsets[0], smax(qo_tile_size, d_tile_size));
  result.o_offsets[1] = result.o_offsets[0];
  result.diagonal_offset = result.o_offsets[0];
  result.packed_k_bytes = kv_tile_size;
  HALO_PLAN_ALLOC(packed_k_offset, result.packed_k_bytes);
  result.packed_v_bytes = kv_tile_size;
  HALO_PLAN_ALLOC(packed_v_offset, result.packed_v_bytes);
  HALO_PLAN_ALLOC(score_offset, core_tile_size);
  result.probability_offset = result.score_offset;
  result.softmax_state_bytes = 2 * col_vec_size;
  HALO_PLAN_ALLOC(softmax_state_offset, result.softmax_state_bytes);
  result.row_buffer_bytes = 2 * row_vec_size;
  HALO_PLAN_ALLOC(row_buffer_offset, result.row_buffer_bytes);
  result.scales_bytes = 512;
  HALO_PLAN_ALLOC(scales_offset, result.scales_bytes);
  result.workspace_bytes = cursor;

  const size_t text_blocks = ((size_t) text_tokens + 31) / 32;
  if (!checked_size_mul(text_blocks, 2, &value) || !checked_size_mul(value, 32, &value) ||
      !checked_size_mul(value, (size_t) head_dim, &value) || !checked_size_mul(value, sizeof(__fp16), &value)) {
    return -1;
  }
  result.resident_text_bytes = value;

  size_t row_tokens;
  size_t row_half_bytes;
  if (!checked_size_mul((size_t) image_block_cols, HMX_FP16_TILE_N_COLS, &row_tokens) ||
      !checked_size_mul(row_tokens, (size_t) head_dim, &value) ||
      !checked_size_mul(value, sizeof(__fp16), &row_half_bytes) ||
      !checked_size_mul(row_half_bytes, 2, &result.packed_image_row_stride) ||
      !checked_size_mul(result.packed_image_row_stride, BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS,
                        &result.packed_image_row_bytes) ||
      !checked_size_align_up(cursor, 4096, &cursor)) {
    return -1;
  }
  result.packed_image_row_slot_count = BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS;
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
    result.packed_image_row_offsets[slot] = cursor;
    if (!checked_size_add(cursor, result.packed_image_row_stride, &cursor)) {
      return -1;
    }
  }

  if (halo_kv_load_mode == 2) {
    result.image_row_stride = result.packed_image_row_stride;
    if (!checked_size_mul(result.image_row_stride, BLOCK_SPARSE_HALO_RAW_STAGING_SLOTS,
                          &result.image_row_bytes)) {
      return -1;
    }
    for (int slot = 0; slot < BLOCK_SPARSE_HALO_RAW_STAGING_SLOTS; ++slot) {
      result.image_row_offsets[slot] = cursor;
      if (!checked_size_add(cursor, result.image_row_stride, &cursor)) {
        return -1;
      }
    }
    if (!checked_size_mul(BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW, sizeof(dma_desc_2d_t),
                          &result.descriptor_bytes)) {
      return -1;
    }
  }

  if (!checked_size_align_up(cursor, 4096, &cursor)) {
    return -1;
  }
  result.resident_text_offset = cursor;
  if (!checked_size_add(cursor, result.resident_text_bytes, &cursor)) {
    return -1;
  }

  if (text_blocks > 0) {
    if (text_blocks > INT_MAX ||
        !checked_size_mul(text_blocks, qo_tile_size, &result.resident_text_q_bytes)) {
      return -1;
    }
    result.resident_text_q_blocks = (int) text_blocks;
    result.resident_text_q_offset = cursor;
    if (!checked_size_add(cursor, result.resident_text_q_bytes, &cursor)) {
      return -1;
    }

    result.resident_text_o_bytes = result.resident_text_q_bytes;
    for (int bank = 0; bank < 2; ++bank) {
      result.resident_text_o_offsets[bank] = cursor;
      if (!checked_size_add(cursor, result.resident_text_o_bytes, &cursor)) {
        return -1;
      }
    }

    if (!checked_size_mul(text_blocks, 2, &value) ||
        !checked_size_mul(value, col_vec_size, &result.resident_text_state_bytes)) {
      return -1;
    }
    result.resident_text_state_offset = cursor;
    if (!checked_size_add(cursor, result.resident_text_state_bytes, &cursor)) {
      return -1;
    }
  }

#undef HALO_PLAN_ALLOC

  result.required_bytes = cursor;
  if (result.required_bytes > available_bytes) {
    return -1;
  }
  *plan = result;
  return 0;
}

int block_sparse_halo_init_dma_banks(block_sparse_halo_dma_bank_t *banks[BLOCK_SPARSE_HALO_DMA_BANKS],
                                     dma_desc_2d_t *external_descs[BLOCK_SPARSE_HALO_DMA_BANKS],
                                     void *vtcm, size_t vtcm_bytes, const block_sparse_halo_plan_t *plan) {
  if (banks == NULL || external_descs == NULL || vtcm == NULL || plan == NULL ||
      plan->required_bytes > vtcm_bytes ||
      ((uintptr_t) vtcm & 4095u) != 0 || plan->image_bank_stride == 0 ||
      plan->descriptor_bytes != BLOCK_SPARSE_HALO_DMA_BANKS * BLOCK_SPARSE_HALO_DMA_DESCS_PER_BANK *
                                 sizeof(dma_desc_2d_t) ||
      plan->completion_bytes != BLOCK_SPARSE_HALO_DMA_BANKS * sizeof(block_sparse_halo_dma_bank_t)) {
    return -1;
  }

  uint8_t *base = (uint8_t *) vtcm;
  for (int bank = 0; bank < BLOCK_SPARSE_HALO_DMA_BANKS; ++bank) {
    if (external_descs[bank] == NULL || ((uintptr_t) external_descs[bank] & 63u) != 0 ||
        plan->descriptor_offsets[bank] != SIZE_MAX || (plan->completion_offsets[bank] & 63u) != 0 ||
        plan->image_bank_offsets[bank] > plan->required_bytes ||
        plan->image_bank_stride > plan->required_bytes - plan->image_bank_offsets[bank] ||
        plan->completion_offsets[bank] > plan->required_bytes ||
        sizeof(block_sparse_halo_dma_bank_t) > plan->required_bytes - plan->completion_offsets[bank]) {
      return -1;
    }
    block_sparse_halo_dma_bank_t *state =
      (block_sparse_halo_dma_bank_t *) (base + plan->completion_offsets[bank]);
    memset(state, 0, sizeof(*state));
    state->raw = base + plan->image_bank_offsets[bank];
    state->descs = external_descs[bank];
    memset(state->descs, 0, BLOCK_SPARSE_HALO_DMA_DESCS_PER_BANK * sizeof(dma_desc_2d_t));
    for (int slot = 0; slot < BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS; ++slot) {
      state->block_ids[slot] = -1;
    }
    state->submit_result = DMA_SUBMIT_BAD_DESC;
    state->wait_result = DMA_SUBMIT_BAD_DESC;
    banks[bank] = state;
  }
  return 0;
}

int block_sparse_halo_dma_preflight(const void *k_base, const void *v_base, const void *vtcm_base,
                                    size_t vtcm_slice_size, unsigned int vtcm_slice_count,
                                    int n_kv_heads, int head_dim, int image_tokens,
                                    const block_sparse_halo_plan_t *plan) {
  if (k_base == NULL || v_base == NULL || vtcm_base == NULL || plan == NULL ||
      vtcm_slice_size == 0 || vtcm_slice_count == 0 || n_kv_heads <= 0 || head_dim <= 0 ||
      image_tokens <= 0 || image_tokens % HMX_FP16_TILE_N_COLS != 0 ||
      plan->required_bytes > vtcm_slice_size || plan->image_row_stride == 0 || plan->image_row_bytes == 0) {
    return -1;
  }

  size_t row_bytes;
  size_t src_stride;
  size_t row_tokens;
  size_t row_height;
  size_t token_offset;
  size_t head_offset;
  size_t src_last_byte;
  if (!checked_size_mul((size_t) head_dim, sizeof(__fp16), &row_bytes) ||
      !checked_size_mul((size_t) n_kv_heads, row_bytes, &src_stride) ||
      !checked_size_mul(row_bytes, 2, &row_tokens) || row_tokens == 0 ||
      plan->image_row_stride % row_tokens != 0 ||
      (row_height = plan->image_row_stride / row_tokens) == 0 ||
      row_bytes > UINT16_MAX || src_stride > UINT16_MAX || row_height == 0 || row_height > UINT16_MAX ||
      image_tokens % row_height != 0 ||
      !checked_size_mul((size_t) (image_tokens - 1), src_stride, &token_offset) ||
      !checked_size_mul((size_t) (n_kv_heads - 1), row_bytes, &head_offset) ||
      !checked_size_add(token_offset, head_offset, &src_last_byte) ||
      !checked_size_add(src_last_byte, row_bytes - 1, &src_last_byte)) {
    return -1;
  }

  const uintptr_t k_addr = (uintptr_t) k_base;
  const uintptr_t v_addr = (uintptr_t) v_base;
  if (k_addr > UINT32_MAX || v_addr > UINT32_MAX || src_last_byte > UINT32_MAX ||
      k_addr > UINT32_MAX - src_last_byte || v_addr > UINT32_MAX - src_last_byte) {
    return -1;
  }

  size_t last_slice_offset;
  if (!checked_size_mul((size_t) (vtcm_slice_count - 1), vtcm_slice_size, &last_slice_offset)) {
    return -1;
  }
  const uintptr_t vtcm_addr = (uintptr_t) vtcm_base;
  size_t expected_row_bytes;
  if (!checked_size_mul(plan->image_row_stride, BLOCK_SPARSE_HALO_RAW_STAGING_SLOTS, &expected_row_bytes) ||
      plan->image_row_bytes != expected_row_bytes ||
      plan->descriptor_bytes != BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW * sizeof(dma_desc_2d_t)) {
    return -1;
  }
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_RAW_STAGING_SLOTS; ++slot) {
    size_t row_last_byte;
    if (plan->image_row_offsets[slot] > plan->required_bytes ||
        plan->image_row_stride > plan->required_bytes - plan->image_row_offsets[slot] ||
        !checked_size_add(last_slice_offset, plan->image_row_offsets[slot], &row_last_byte) ||
        !checked_size_add(row_last_byte, plan->image_row_stride - 1, &row_last_byte) ||
        vtcm_addr > UINT32_MAX || row_last_byte > UINT32_MAX ||
        vtcm_addr > UINT32_MAX - row_last_byte) {
      return -1;
    }
  }
  return 0;
}

int block_sparse_halo_select_execution(block_sparse_halo_plan_t *plan, int halo_kv_load_mode,
                                       bool geometry_valid, bool dma_preflight_valid, int userdma_scope, int head_dim,
                                       int text_tokens, int image_block_cols, size_t available_bytes) {
  if (plan == NULL || halo_kv_load_mode < 0 || halo_kv_load_mode > 3) {
    return -1;
  }
  memset(plan, 0, sizeof(*plan));
  if (halo_kv_load_mode == 0) {
    return BLOCK_SPARSE_HALO_EXEC_BASELINE;
  }
  if (!geometry_valid) {
    return BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK;
  }
  if (halo_kv_load_mode == 1) {
    // The packed-row worker also supports image-only attention. In that
    // mode the plan omits all resident text buffers and only keeps the
    // packed image-row ring.
    return block_sparse_halo_select_plan(plan, 1, head_dim, text_tokens, image_block_cols, available_bytes) == 0 ?
      BLOCK_SPARSE_HALO_EXEC_L2_PREFETCH : BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK;
  }
  if (halo_kv_load_mode == 3) {
    return BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK;
  }
  if (!dma_preflight_valid || userdma_scope != BLOCK_SPARSE_HALO_USERDMA_PER_WORKER) {
    return BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK;
  }
  return block_sparse_halo_select_plan(plan, 2, head_dim, text_tokens, image_block_cols, available_bytes) == 0 ?
    BLOCK_SPARSE_HALO_EXEC_USERDMA_ROWS : BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK;
}

static unsigned int block_sparse_halo_shared_dma_owner;

static int block_sparse_halo_make_dma_row(block_sparse_halo_row_t *dma_row,
                                          const block_sparse_halo_row_t *query_row,
                                          int first_image_block, int image_blocks, int group_start) {
  if (dma_row == NULL || query_row == NULL || first_image_block < 0 || image_blocks <= 0 ||
      group_start < 0 || group_start >= image_blocks) {
    return -1;
  }
  block_sparse_halo_row_t result = {0};
  if (!query_row->is_text_query) {
    if (group_start != 0 || query_row->image_count <= 0 ||
        query_row->image_count > BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS) {
      return -1;
    }
    result = *query_row;
  } else {
    result.image_count = smin(BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS, image_blocks - group_start);
    for (int i = 0; i < result.image_count; ++i) {
      result.image_blocks[i] = first_image_block + group_start + i;
    }
  }
  result.is_text_query = false;
  *dma_row = result;
  return 0;
}

static int block_sparse_halo_dma_prepare(block_sparse_halo_dma_bank_t *bank,
                                         const __fp16 *K, const __fp16 *V,
                                         int n_kv_heads, int kv_head_idx, int head_dim,
                                         int k_blocks_per_head, const block_sparse_halo_row_t *row) {
  if (bank == NULL || bank->outstanding || bank->ready) {
    return -1;
  }
  const int desc_count = block_sparse_halo_build_dma_descs(
    bank->descs, BLOCK_SPARSE_HALO_DMA_DESCS_PER_BANK, K, V, bank->raw, n_kv_heads, kv_head_idx,
    head_dim, k_blocks_per_head, row);
  if (desc_count <= 0) {
    return -1;
  }
  bank->block_count = row->image_count;
  bank->desc_count = desc_count;
  for (int i = 0; i < BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS; ++i) {
    bank->block_ids[i] = i < row->image_count ? row->image_blocks[i] : -1;
  }
  bank->submit_result = DMA_SUBMIT_BAD_DESC;
  bank->wait_result = DMA_SUBMIT_BAD_DESC;
  return 0;
}

static bool block_sparse_halo_dma_try_acquire(int userdma_scope, int worker_index) {
  if (userdma_scope == BLOCK_SPARSE_HALO_USERDMA_PER_WORKER) {
    return true;
  }
  if (userdma_scope != BLOCK_SPARSE_HALO_USERDMA_SHARED || worker_index < 0) {
    return false;
  }
  unsigned int expected = 0;
  return __atomic_compare_exchange_n(&block_sparse_halo_shared_dma_owner, &expected,
                                     (unsigned int) worker_index + 1, false,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

static void block_sparse_halo_dma_release(int userdma_scope, int worker_index) {
  if (userdma_scope == BLOCK_SPARSE_HALO_USERDMA_SHARED) {
    const unsigned int owner = (unsigned int) worker_index + 1;
    unsigned int expected = owner;
    (void) __atomic_compare_exchange_n(&block_sparse_halo_shared_dma_owner, &expected, 0, false,
                                       __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  }
}

static int block_sparse_halo_dma_try_submit(block_sparse_halo_dma_bank_t *bank,
                                            int userdma_scope, int worker_index) {
  if (bank == NULL || bank->ready || bank->outstanding || bank->desc_count <= 0) {
    return DMA_SUBMIT_BAD_DESC;
  }
  if (!block_sparse_halo_dma_try_acquire(userdma_scope, worker_index)) {
    bank->submit_result = DMA_SUBMIT_BUSY;
    return DMA_SUBMIT_BUSY;
  }
  bank->submit_result = dma_chain_submit(bank->descs);
  if (bank->submit_result != DMA_SUBMIT_OK) {
    block_sparse_halo_dma_release(userdma_scope, worker_index);
    return bank->submit_result;
  }
  bank->outstanding = true;
  return DMA_SUBMIT_OK;
}

static int block_sparse_halo_dma_wait(block_sparse_halo_dma_bank_t *bank,
                                      int userdma_scope, int worker_index) {
  if (bank == NULL) {
    return DMA_SUBMIT_BAD_DESC;
  }
  if (!bank->outstanding) {
    const int submit_result = block_sparse_halo_dma_try_submit(bank, userdma_scope, worker_index);
    if (submit_result != DMA_SUBMIT_OK) {
      return submit_result;
    }
  }
  bank->wait_result = block_sparse_halo_dma_wait_bounded(bank->descs, (size_t) bank->desc_count, 1000000);
  bank->outstanding = false;
  block_sparse_halo_dma_release(userdma_scope, worker_index);
  if (bank->wait_result != DMA_SUBMIT_OK) {
    return bank->wait_result;
  }
  bank->ready = true;
  return DMA_SUBMIT_OK;
}

static __attribute__((unused)) int block_sparse_halo_dma_wait_blocking(block_sparse_halo_dma_bank_t *bank,
                                                                       int userdma_scope, int worker_index) {
  const int64_t deadline = HAP_perf_get_time_us() + 1000000;
  int result;
  while ((result = block_sparse_halo_dma_wait(bank, userdma_scope, worker_index)) == DMA_SUBMIT_BUSY) {
    if (HAP_perf_get_time_us() >= deadline) {
      return DMA_SUBMIT_HW_ERROR;
    }
    qurt_timer_sleep(1);
  }
  return result;
}

static __attribute__((unused)) void block_sparse_halo_dma_consume(block_sparse_halo_dma_bank_t *bank) {
  if (bank != NULL) {
    bank->ready = false;
    bank->block_count = 0;
    bank->desc_count = 0;
  }
}

static __attribute__((unused)) void block_sparse_halo_dma_drain(
  block_sparse_halo_dma_bank_t *banks[BLOCK_SPARSE_HALO_DMA_BANKS], int userdma_scope, int worker_index) {
  if (banks == NULL) {
    return;
  }
  for (int bank = 0; bank < BLOCK_SPARSE_HALO_DMA_BANKS; ++bank) {
    if (banks[bank] != NULL && banks[bank]->outstanding) {
      (void) block_sparse_halo_dma_wait_bounded(banks[bank]->descs, (size_t) banks[bank]->desc_count, 1000000);
      banks[bank]->outstanding = false;
      block_sparse_halo_dma_release(userdma_scope, worker_index);
    }
  }
}

static int block_sparse_halo_row_dma_submit(dma_desc_2d_t *descs, int desc_count, int userdma_scope,
                                            int worker_index, bool *outstanding) {
  if (descs == NULL || desc_count <= 0 || desc_count > BLOCK_SPARSE_HALO_MAX_DMA_DESCS_PER_ROW ||
      outstanding == NULL || *outstanding) {
    return DMA_SUBMIT_BAD_DESC;
  }
  if (!block_sparse_halo_dma_try_acquire(userdma_scope, worker_index)) {
    return DMA_SUBMIT_BUSY;
  }
  const int result = dma_chain_submit(descs);
  if (result != DMA_SUBMIT_OK) {
    block_sparse_halo_dma_release(userdma_scope, worker_index);
    return result;
  }
  *outstanding = true;
  return DMA_SUBMIT_OK;
}

static int block_sparse_halo_row_dma_wait(dma_desc_2d_t *descs, int desc_count, int userdma_scope,
                                          int worker_index, bool *outstanding) {
  if (descs == NULL || desc_count <= 0 || desc_count > BLOCK_SPARSE_HALO_MAX_DMA_DESCS_PER_ROW ||
      outstanding == NULL) {
    return DMA_SUBMIT_BAD_DESC;
  }
  if (!*outstanding) {
    return DMA_SUBMIT_OK;
  }
  const int result = block_sparse_halo_dma_wait_bounded(descs, (size_t) desc_count, 1000000);
  *outstanding = false;
  block_sparse_halo_dma_release(userdma_scope, worker_index);
  return result;
}

static void block_sparse_halo_row_dma_drain(dma_desc_2d_t *descs, int desc_count, int userdma_scope,
                                            int worker_index, bool *outstanding) {
  if (descs == NULL || outstanding == NULL || !*outstanding) {
    return;
  }
  (void) block_sparse_halo_dma_wait_bounded(descs, (size_t) desc_count, 1000000);
  *outstanding = false;
  block_sparse_halo_dma_release(userdma_scope, worker_index);
}

static int block_sparse_halo_row_dma_load_async(block_sparse_halo_row_slot_t *slot, dma_desc_2d_t *descs,
                                                const __fp16 *K, const __fp16 *V, int n_kv_heads,
                                                int kv_head_idx, int head_dim, int image_block_cols, int row_id,
                                                bool input_packed, int token_tiles,
                                                int userdma_scope, int worker_index) {
  const int expected_descs = input_packed ? block_sparse_halo_packed_dma_desc_count(head_dim) :
                                            BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW;
  if (slot == NULL || descs == NULL || slot->outstanding || slot->raw == NULL || row_id < 0 ||
      (input_packed ?
         block_sparse_halo_build_packed_dma_row_descs(
           descs, BLOCK_SPARSE_HALO_MAX_DMA_DESCS_PER_ROW, K, V, slot->raw,
           kv_head_idx, head_dim, token_tiles, image_block_cols, row_id) :
         block_sparse_halo_build_dma_row_descs(
           descs, BLOCK_SPARSE_HALO_MAX_DMA_DESCS_PER_ROW, K, V, slot->raw,
           n_kv_heads, kv_head_idx, head_dim, image_block_cols, row_id)) != expected_descs) {
    return -1;
  }
  slot->row_id = row_id;
  slot->ready = false;
  if (block_sparse_halo_row_dma_submit(descs, expected_descs,
                                       userdma_scope, worker_index, &slot->outstanding) != DMA_SUBMIT_OK) {
    slot->outstanding = false;
    return -1;
  }
  return 0;
}

static int block_sparse_halo_dma_row_for_q(block_sparse_halo_row_t *dma_row,
                                           block_sparse_halo_row_t *query_row,
                                           const int32_t *block_mask, int max_kept_blocks,
                                           int q_head_base, int q_blocks_per_head,
                                           int k_blocks_per_head, int image_blocks,
                                           int block_q, int q_token, int group_start) {
  if (dma_row == NULL || query_row == NULL || block_mask == NULL || q_token < 0) {
    return -1;
  }
  const int q_block = q_head_base * q_blocks_per_head + q_token / block_q;
  const int32_t *row_data = block_mask + (size_t) q_block * (size_t) max_kept_blocks;
  if (block_sparse_halo_parse_row(query_row, row_data, max_kept_blocks, q_block,
                                  q_blocks_per_head, k_blocks_per_head, image_blocks) != 0) {
    return -1;
  }
  return block_sparse_halo_make_dma_row(dma_row, query_row,
                                        (q_head_base * k_blocks_per_head), image_blocks, group_start);
}

static __attribute__((unused)) int block_sparse_halo_dma_prepare_for_q(
  block_sparse_halo_dma_bank_t *bank, const __fp16 *K, const __fp16 *V, const int32_t *block_mask,
  int max_kept_blocks, int n_kv_heads, int kv_head_idx, int head_dim, int q_head_base,
  int q_blocks_per_head, int k_blocks_per_head, int image_blocks, int block_q, int q_token, int group_start) {
  block_sparse_halo_row_t query_row;
  block_sparse_halo_row_t dma_row;
  if (block_sparse_halo_dma_row_for_q(&dma_row, &query_row, block_mask, max_kept_blocks,
                                      q_head_base, q_blocks_per_head, k_blocks_per_head,
                                      image_blocks, block_q, q_token, group_start) != 0 ||
      block_sparse_halo_dma_prepare(bank, K, V, n_kv_heads, kv_head_idx, head_dim,
                                    k_blocks_per_head, &dma_row) != 0) {
    return -1;
  }
  return dma_row.image_count;
}

static void block_sparse_halo_l2_prefetch_image_row(const __fp16 *K, const __fp16 *V,
                                                    int n_kv_heads, int kv_head_idx, int head_dim,
                                                    int block_k, int image_block_cols, int row_id) {
  if (K == NULL || V == NULL || n_kv_heads <= 0 || kv_head_idx < 0 || kv_head_idx >= n_kv_heads ||
      head_dim <= 0 || block_k <= 0 || image_block_cols <= 0 || row_id < 0) {
    return;
  }
  const int row_tokens = image_block_cols * block_k;
  const size_t kv_ld_blk_sz = (size_t) head_dim;
  const size_t kv_ld_stride = (size_t) n_kv_heads * kv_ld_blk_sz;
  const size_t first_token = (size_t) row_id * (size_t) row_tokens;
  const __fp16 *row_k = K + first_token * kv_ld_stride + (size_t) kv_head_idx * kv_ld_blk_sz;
  const __fp16 *row_v = V + first_token * kv_ld_stride + (size_t) kv_head_idx * kv_ld_blk_sz;
  l2fetch(row_k, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), row_tokens, 0);
  l2fetch(row_v, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), row_tokens, 0);
}

static void block_sparse_pack_k_f16(__fp16 *packed, const __fp16 *src, int src_stride, int n_cols, int head_dim) {
  static int32_t transpose_vscatter_indices_base[32] __vec_aligned__;
  for (int i = 0; i < 32; ++i) {
    transpose_vscatter_indices_base[i] = i * 128;
  }

  const int n_k_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
  const HVX_Vector v_step = Q6_V_vsplat_R(4);
  const HVX_Vector v_offsets_base = vmem(transpose_vscatter_indices_base);
  for (int k_tile_idx = 0; k_tile_idx < n_k_tiles; ++k_tile_idx) {
    __fp16 *out_base = packed + k_tile_idx * HMX_FP16_TILE_N_COLS * head_dim;
    HVX_Vector v_offsets = v_offsets_base;
    for (int r1 = 0; r1 < HMX_FP16_TILE_N_COLS; ++r1) {
      const int r = k_tile_idx * HMX_FP16_TILE_N_COLS + r1;
      if (r >= n_cols) {
        break;
      }
      const HVX_Vector *pv_in = (const HVX_Vector *) (src + r * src_stride);
#pragma unroll
      for (int d = 0; d < head_dim / 64; ++d) {
        __fp16 *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
        Q6_vscatter_RMVwV((size_t) out_dual_tile, HMX_FP16_TILE_SIZE * 2 - 1, v_offsets, *pv_in++);
      }
      v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
    }
  }
}

static void block_sparse_pack_v_f16(__fp16 *packed, const __fp16 *src, int src_stride, int n_cols,
                                    int padded_cols, int head_dim) {
  assert(padded_cols >= n_cols && padded_cols % HMX_FP16_TILE_N_COLS == 0);
  const int n_k_tiles = padded_cols / HMX_FP16_TILE_N_COLS;
  for (int r = 0; r < n_cols; r += 2) {
    const bool next_row_valid = (r + 1) < n_cols;
    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * src_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * src_stride);
#pragma unroll
    for (int c = 0; c < head_dim; c += 64) {
      const HVX_Vector v0 = *pv_in0++;
      const HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
      const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
      const int k_tile_idx = r / HMX_FP16_TILE_N_ROWS;
      const int row_in_tile = r % HMX_FP16_TILE_N_ROWS;
      const int d_tile_idx = c / HMX_FP16_TILE_N_COLS;
      __fp16 *tile_base0 = packed + ((d_tile_idx + 0) * n_k_tiles + k_tile_idx) * HMX_FP16_TILE_N_ELMS;
      __fp16 *tile_base1 = packed + ((d_tile_idx + 1) * n_k_tiles + k_tile_idx) * HMX_FP16_TILE_N_ELMS;
      HVX_Vector *pv_out0 = ((HVX_Vector *) tile_base0) + row_in_tile / 2;
      HVX_Vector *pv_out1 = ((HVX_Vector *) tile_base1) + row_in_tile / 2;
      *pv_out0 = Q6_V_lo_W(vp);
      *pv_out1 = Q6_V_hi_W(vp);
    }
  }
}

static void block_sparse_pack_v_block_f16(__fp16 *packed, const __fp16 *src, int src_stride,
                                          int dst_col_tile, int total_col_tiles, int head_dim) {
  assert(dst_col_tile >= 0 && dst_col_tile < total_col_tiles);
  for (int r = 0; r < HMX_FP16_TILE_N_COLS; r += 2) {
    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * src_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * src_stride);
#pragma unroll
    for (int c = 0; c < head_dim; c += 64) {
      const HVX_VectorPair vp = Q6_W_vshuff_VVR(*pv_in1++, *pv_in0++, -2);
      const int d_tile_idx = c / HMX_FP16_TILE_N_COLS;
      __fp16 *tile_base0 = packed + ((d_tile_idx + 0) * total_col_tiles + dst_col_tile) * HMX_FP16_TILE_N_ELMS;
      __fp16 *tile_base1 = packed + ((d_tile_idx + 1) * total_col_tiles + dst_col_tile) * HMX_FP16_TILE_N_ELMS;
      HVX_Vector *pv_out0 = ((HVX_Vector *) tile_base0) + r / 2;
      HVX_Vector *pv_out1 = ((HVX_Vector *) tile_base1) + r / 2;
      *pv_out0 = Q6_V_lo_W(vp);
      *pv_out1 = Q6_V_hi_W(vp);
    }
  }
}

static __attribute__((unused)) int block_sparse_pack_raw_bank_segment(
  __fp16 *packed_k, __fp16 *packed_v, size_t packed_bytes, const block_sparse_halo_dma_bank_t *bank,
  int first_slot, int block_count, int head_dim) {
  if (packed_k == NULL || packed_v == NULL || bank == NULL || !bank->ready || first_slot < 0 ||
      block_count <= 0 || block_count > 2 || first_slot > bank->block_count - block_count) {
    return -1;
  }
  hvx_fill_uh(packed_k, 0, packed_bytes);
  hvx_fill_uh(packed_v, 0, packed_bytes);
  const size_t block_elems = (size_t) HMX_FP16_TILE_N_COLS * (size_t) head_dim;
  const size_t pair_bytes = 2 * block_elems * sizeof(__fp16);
  for (int i = 0; i < block_count; ++i) {
    const __fp16 *raw_k = (const __fp16 *) (bank->raw + (size_t) (first_slot + i) * pair_bytes);
    const __fp16 *raw_v = raw_k + block_elems;
    block_sparse_pack_k_f16(packed_k + (size_t) i * block_elems, raw_k, head_dim,
                            HMX_FP16_TILE_N_COLS, head_dim);
    block_sparse_pack_v_block_f16(packed_v, raw_v, head_dim, i, 2, head_dim);
  }
  return 0;
}

int block_sparse_halo_pack_raw_row_segment(__fp16 *packed_k, __fp16 *packed_v, size_t packed_bytes,
                                           const block_sparse_halo_row_slot_t *slot, int col_start,
                                           int block_count, int image_block_cols, int head_dim) {
  if (packed_k == NULL || packed_v == NULL || slot == NULL || !slot->ready || slot->outstanding ||
      slot->raw == NULL ||
      col_start < 0 || block_count <= 0 || block_count > 2 || image_block_cols <= 0 ||
      col_start > image_block_cols - block_count || head_dim <= 0) {
    return -1;
  }
  const size_t block_elems = (size_t) HMX_FP16_TILE_N_COLS * (size_t) head_dim;
  const size_t row_half_bytes = (size_t) image_block_cols * HMX_FP16_TILE_N_COLS * (size_t) head_dim *
                                sizeof(__fp16);
  const size_t token_offset = (size_t) col_start * HMX_FP16_TILE_N_COLS * (size_t) head_dim * sizeof(__fp16);
  const size_t segment_bytes = (size_t) block_count * block_elems * sizeof(__fp16);
  if (token_offset > row_half_bytes || segment_bytes > row_half_bytes - token_offset ||
      packed_bytes < block_count * block_elems * sizeof(__fp16)) {
    return -1;
  }
  hvx_fill_uh(packed_k, 0, packed_bytes);
  hvx_fill_uh(packed_v, 0, packed_bytes);
  const __fp16 *raw_k = (const __fp16 *) (slot->raw + token_offset);
  const __fp16 *raw_v = (const __fp16 *) (slot->raw + row_half_bytes + token_offset);
  block_sparse_pack_k_f16(packed_k, raw_k, head_dim, block_count * HMX_FP16_TILE_N_COLS, head_dim);
  block_sparse_pack_v_f16(packed_v, raw_v, head_dim, block_count * HMX_FP16_TILE_N_COLS,
                          2 * HMX_FP16_TILE_N_COLS, head_dim);
  return 0;
}

size_t fa_f16_compute_vtcm_usage(int gqa_factor, int head_dim, int n_rows, int n_cols) {
  const size_t g_br = align_up(gqa_factor * n_rows, HMX_FP16_TILE_N_ROWS);

  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);    // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(n_cols * head_dim * sizeof(__fp16), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * n_cols * sizeof(__fp16), 4096);      // S, P: [Br', Bc]
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);        // D: [Br', Br']
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);                // m, l, rowmax, rowsum: [Br']
  const size_t row_vec_size   = align_up(n_cols * sizeof(__fp16), 256);

  size_t total = qo_tile_size * 3 /* Q, O0, O1 */ + kv_tile_size * 2 /* K, V */ + core_tile_size * 2 /* S, P */ +
                 d_tile_size /* D */ + col_vec_size * 4 + row_vec_size * 2 /* 2x row buffer */ +
                 512 /* HMX column scales */;
  return total;
}

void find_chunk_size_common(size_t *blk_r, size_t *blk_c, int gqa_factor, int head_dim, int qo_len, int kv_len,
                            size_t limit, int nr_unit, int nc_unit, size_t (*compute_vtcm_usage)(int, int, int, int)) {
  size_t nr = nr_unit, nc = nc_unit;
  size_t nr_ok = nr, nc_ok = nc;
  assert(compute_vtcm_usage(gqa_factor, head_dim, nr, nc) <= limit);

  const size_t max_g_nr = MAX_G_BR;
  const size_t max_nr   = align_up(qo_len, nr_unit);
  const size_t max_nc   = align_up(kv_len, nc_unit);

  // increase Br first
  for (; nr <= max_nr && gqa_factor * nr <= max_g_nr; nr += nr_unit) {
    if (compute_vtcm_usage(gqa_factor, head_dim, nr, nc) > limit) {
      break;
    }

    nr_ok = nr;
  }

  // then increase Bc
  for (; nc <= max_nc; nc += nc_unit) {
    if (compute_vtcm_usage(gqa_factor, head_dim, nr_ok, nc) > limit) {
      break;
    }

    nc_ok = nc;
  }

  *blk_r = nr_ok, *blk_c = nc_ok;
}

void fa_f16_find_chunk_size(size_t *blk_r, size_t *blk_c, int gqa_factor, int head_dim, int qo_len, int kv_len,
                            size_t limit) {
  const int nr_unit = ceil_div(HMX_FP16_TILE_N_ROWS, gqa_factor);
  const int nc_unit = 64;

  find_chunk_size_common(blk_r, blk_c, gqa_factor, head_dim, qo_len, kv_len, limit, nr_unit, nc_unit,
                         fa_f16_compute_vtcm_usage);
}

// #define ENABLE_PROFILE_TIMERS

#if defined(ENABLE_PROFILE_TIMERS)

#  define TIMER_DEFINE(name) int64_t name##_ticks = 0
#  define TIMER_START(name)  int64_t name##_t0 = HAP_perf_get_qtimer_count()
#  define TIMER_STOP(name)   name##_ticks += HAP_perf_get_qtimer_count() - name##_t0
#  define TIMER_US(name)     HAP_perf_qtimer_count_to_us(name##_ticks)

#else

#  define TIMER_DEFINE(name)
#  define TIMER_START(name)
#  define TIMER_STOP(name)
#  define TIMER_US(name)

#endif

// pre-assert: D is multiple of 64
void simple_flash_attn_f16_core(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O,
                                const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                                const __fp16 *restrict qk_mask, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                                int head_dim, bool with_lse, int worker_index) {
  (void) worker_index;

  // "compile-time" configs
  // TODO: make them real compile-time constants (constexpr or template parameters)
  const int G = n_heads / n_kv_heads;  // GQA factor
  const int D = head_dim;

  const bool   qo_fp32_element = true;  // whether Q/O has fp32 elements
  const size_t qo_element_size = qo_fp32_element ? sizeof(float) : sizeof(__fp16);

  // NOTE(hzx): confirmed that non-constant `has_qk_mask` affects softmax performance, disable it for now
  assert(qk_mask != NULL);
  // const bool   has_qk_mask = (qk_mask != NULL);
  const bool   has_qk_mask = true;
  const size_t kv_pad_len  = align_up(kv_len, 64);

  const bool use_fp32_exp = false;  // compute FP32 exp

  // determine block sizes
  size_t blk_sz_r, blk_sz_c;  // Br, Bc
  fa_f16_find_chunk_size(&blk_sz_r, &blk_sz_c, G, head_dim, qo_len, kv_len, vtcm_limit - vtcm);
  assert(blk_sz_c % 64 == 0);

  const size_t g_br = align_up(G * blk_sz_r, HMX_FP16_TILE_N_ROWS);  // Br'

  // FARF(ALWAYS, "%s: Br=%d Bc=%d Br'=%d", __func__, blk_sz_r, blk_sz_c, g_br);

  const size_t n_tiles_per_blk_r = g_br / HMX_FP16_TILE_N_ROWS;
  const size_t n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;

  // compute tile/vector sizes
  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);      // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(blk_sz_c * head_dim * sizeof(__fp16), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * blk_sz_c * sizeof(__fp16), 4096);      // S, P: [Br', Bc]
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);          // D: [Br', Br']
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);                  // m, l, rowmax, rowsum: [Br']
  const size_t row_vec_size   = align_up(blk_sz_c * sizeof(__fp16), 256);

  const size_t kv_ld_blk_sz   = head_dim;               // no * element_size
  const size_t kv_ld_stride   = n_kv_heads * head_dim;  // no * element_size
  const size_t q_ld_blk_sz    = G * head_dim;           // no * element_size
  const size_t q_ld_stride    = n_heads * head_dim;     // no * element_size
  const size_t o_head_dim     = head_dim + (with_lse ? 1 : 0);
  const size_t o_st_blk_sz    = G * o_head_dim;         // no * element_size
  const size_t o_st_stride    = n_heads * o_head_dim;   // no * element_size

  // begin VTCM allocation
  uint8_t *vtcm_cur = vtcm;
  __fp16  *q_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile0  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile1  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);

  __fp16 *k_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  __fp16 *v_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);

  __fp16 *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *p_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  __fp16 *d_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_size);

  HVX_Vector *mvec_m        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_s_rowmax = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  HVX_Vector *row_buffer0 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  HVX_Vector *row_buffer1 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);

  uint8_t *hmx_output_scales_id = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  uint8_t *hmx_output_scales_qk = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);

  // end VTCM allocation
  assert(vtcm_cur <= vtcm_limit);

  float  qk_scale    = 1.0f / sqrtf(head_dim) * 1.44269504f;  // log2(e) = 1.44269504
  __fp16 qk_scale_hf = (__fp16) qk_scale;                     // NOTE: this conversion can be very slow

  // NOTE: there are 32 effective elements in scales, use 4 bytes splat (not Q6_Vh_vsplat_R)
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));

  // prepare constants
  static int32_t transpose_vscatter_indices_base[32] __vec_aligned__;
  for (int i = 0; i < 32; ++i) {
    transpose_vscatter_indices_base[i] = i * 128;  // range [0, 4096), two HMX tiles
  }

  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    // offsets within the first tile
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }
  // profile timers
  TIMER_DEFINE(q_load);
  TIMER_DEFINE(k_load);
  TIMER_DEFINE(v_load);
  TIMER_DEFINE(qk_dot);
  TIMER_DEFINE(safe_sm);   // safe softmax
  TIMER_DEFINE(core_acc);  // core accumulation
  TIMER_DEFINE(o_scale);
  TIMER_DEFINE(o_store);

  /////////////// CORE LOGIC BEGIN

  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows        = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g      = n_rows * G;
    const size_t n_row_tiles   = ceil_div(n_rows_g, HMX_FP16_TILE_N_ROWS);
    const size_t n_row_vec_cnt = ceil_div(n_rows_g, 64);

    // load [n_rows*G, D] tile of Q into VTCM
    TIMER_START(q_load);
    {
      // load block size: G*D elements
      const size_t q_ld_blk_sz_bytes = q_ld_blk_sz * qo_element_size;
      const size_t q_ld_stride_bytes = q_ld_stride * qo_element_size;  // a.k.a. hidden_size

      const uint8_t *q_ld_base = ((uint8_t *) Q) + ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;

      // FIXME(hzx): This L2 fetch may not be very useful
      // NOTE(hzx): what about prefetching in reverse order?
      l2fetch(q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, n_rows, 1);

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        // input positions
        int q_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int q_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        const HVX_Vector *pv_in0 =
          (const HVX_Vector *) (q_ld_base + q_idx0 * q_ld_stride_bytes + h_idx0 * head_dim * qo_element_size);
        const HVX_Vector *pv_in1 =
          (const HVX_Vector *) (q_ld_base + q_idx1 * q_ld_stride_bytes + h_idx1 * head_dim * qo_element_size);

        // output positions
        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector v0 = *pv_in0++;
            const HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

            const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]     = v_out;
          }
        } else {
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            // locate target dual-tile
            __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
            HVX_Vector *pv_out1       = pv_out0 + 16;  // 16 * 128B = 2048B (1 tile)

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
          }
        }
        // clang-format on
      }
    }
    TIMER_STOP(q_load);

    hvx_fill_uh(mvec_m, 0xfbff, col_vec_size);  // init to -65504 (-inf)
    hvx_fill_uh(mvec_l, 0, col_vec_size);       // init: 0

    __fp16 *o_tile_prev = o_tile0;
    __fp16 *o_tile_curr = o_tile1;
    hvx_fill_uh(o_tile_prev, 0, qo_tile_size);
    hvx_fill_uh(d_tile, 0, d_tile_size);

    // inner loop over kv_len
    for (int jc = 0; jc < kv_len; jc += blk_sz_c) {
      const size_t n_cols      = smin(kv_len - jc, blk_sz_c);
      const size_t n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

      // load [Bc, D] tile of K^T into VTCM
      // TODO(hzx): use DMA? if DMA used, we should read from VTCM
      TIMER_START(k_load);
      {
        const __fp16 *k_ld_base = K + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        // FIXME: Is this necessary?
        l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 1);

        const HVX_Vector v_step         = Q6_V_vsplat_R(4);
        const HVX_Vector v_offsets_base = vmem(transpose_vscatter_indices_base);

        // continuous fetch loop: [Bc/32, 32, D]
        for (int r0 = 0; r0 < n_col_tiles; ++r0) {
          __fp16 *out_base = k_tile + r0 * HMX_FP16_TILE_N_COLS * head_dim;  // transposed [D, 32] column chunk

          HVX_Vector v_offsets = v_offsets_base;                             // reset to base offsets

          for (int r1 = 0; r1 < HMX_FP16_TILE_N_COLS; ++r1) {
            int r = r0 * HMX_FP16_TILE_N_COLS + r1;
            if (r >= n_cols) {
              break;
            }

            const HVX_Vector *pv_in = (const HVX_Vector *) (k_ld_base + r * kv_ld_stride);

            // clang-format off
            #pragma unroll
            for (int d = 0; d < D / 64; ++d) {
              __fp16 *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
              Q6_vscatter_RMVwV((size_t) out_dual_tile, HMX_FP16_TILE_SIZE * 2 - 1, v_offsets, *pv_in++);
            }
            // clang-format on

            v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
          }
        }
      }
      TIMER_STOP(k_load);

      // issue L2 prefetch of V tile
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 0);
      }

      // compute dot product of tiles: dot(Q[Br', D], K[Bc, D]) ==> [Br', Bc]
      TIMER_START(qk_dot);
      {
        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_qk);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < n_col_tiles; ++c) {
              const __fp16 *row_tiles = q_tile + r * HMX_FP16_TILE_N_ROWS * head_dim;
              const __fp16 *col_tiles = k_tile + c * HMX_FP16_TILE_N_COLS * head_dim;

              // NOTE: we use `n_tiles_per_blk_c` instead of `n_col_tiles` here
              __fp16 *out_tile = s_tile + (r * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS;
              hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
            }
          }
        }
        hmx_unit_release();
      }
      TIMER_STOP(qk_dot);

      // core softmax computation
      TIMER_START(safe_sm);
      {
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);  // fp16: -65504

        // read from S tile, process 2 rows at a time, generate P tile
        for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
          // vector registers, empty when initialized, fill in 2 rows at a time
          HVX_Vector v_s_rowmax_local = v_neg_inf;
          HVX_Vector v_p_rowsum_local = Q6_V_vzero();

          for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            int r = r_vec_idx * 64 + r_vec_off;
            if (r >= align_up(n_rows_g, 2)) {
              break;
            }

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            // NOTE: make sure this match with S tile generation logic
            __fp16 *s_ld_base = s_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;
            __fp16 *p_st_base = p_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;

            // decode 2 rows into row buffers
            HVX_Vector *pv_row_buf0 = row_buffer0;
            HVX_Vector *pv_row_buf1 = row_buffer1;
            for (int c = 0; c < n_cols; c += 64) {
              const __fp16     *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              const HVX_Vector *pv_s_in0     = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
              const HVX_Vector *pv_s_in1     = pv_s_in0 + 16;  // 16 * 128B = 2048B (1 tile)

              HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
              *pv_row_buf0++               = Q6_V_lo_W(vp_s_dual_row);
              *pv_row_buf1++               = Q6_V_hi_W(vp_s_dual_row);
            }

            // apply mask & compute rowmax(S)
            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            // reduction phase 1: inter-vector
            for (int c = 0; c < n_cols; c += 64) {
              int q_idx0 = ir + (r + 0) / G;
              int q_idx1 = ir + (r + 1) / G;
              int k_idx  = jc + c;

              HVX_VectorPred q_mask_keep0, q_mask_keep1;
              if (has_qk_mask) {
                HVX_Vector v_mask0 = vmemu(qk_mask + q_idx0 * kv_pad_len + k_idx);
                HVX_Vector v_mask1 = vmemu(qk_mask + q_idx1 * kv_pad_len + k_idx);

                const HVX_Vector v_fp16_mask_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16: -16.0

                q_mask_keep0 = Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_fp16_mask_threshold);
                q_mask_keep1 = Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_fp16_mask_threshold);
              } else {
                const size_t ne = smin(n_cols - c, 64);

                q_mask_keep0 = q_mask_keep1 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              }

              HVX_Vector v_s_row0 = Q6_V_vmux_QVV(q_mask_keep0, row_buffer0[c / 64], v_neg_inf);
              HVX_Vector v_s_row1 = Q6_V_vmux_QVV(q_mask_keep1, row_buffer1[c / 64], v_neg_inf);

              row_buffer0[c / 64] = v_s_row0;
              row_buffer1[c / 64] = v_s_row1;

              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
            }

            // clang-format off
            // reduction phase 2: intra-vector
            #pragma unroll
            for (int s = 64; s >= 2; s >>= 1) {
              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
            }
            // clang-format on
            // now, v_s_rowmax0[63] = rowmax(S)_0, v_s_rowmax1[63] = rowmax(S)_1

            // shift rowmax(S_i^j) into v_s_rowmax_local
            HVX_Vector v_s_rowmax_pack2 =
              Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));    // highest 4 bytes are valid
            HVX_Vector v_s_rowmax_pack2_rot =
              Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));   // lowest 4 bytes valid
            HVX_Vector v_s_rowmax_local_rot =
              Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));  // highest r*2 bytes valid
            v_s_rowmax_local = Q6_V_vlalign_VVR(v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r_vec_off * sizeof(__fp16));

            // compute m_i^j = max(m_i^{j-1}, rowmax(S_i^j))
            HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(mvec_m[r_vec_idx], v_s_rowmax_local);

            // broadcast new m_0^j and m_1^j to whole vectors using LUT
            HVX_Vector v_m_lut  = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));  // lowest 4 bytes are valid
            HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
            HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));

            // compute rows of P_i^j = exp(S_i^j - m_i^j)
            // write permuted rows of P tile into VTCM
            // compute rowsum(P)
            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;  // qfloat
            HVX_Vector       v_p_rowsum1 = v_zero;  // qfloat

            for (int c = 0; c < n_cols; c += 64) {
              HVX_Vector v_p_row0_hf, v_p_row1_hf;

              HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0);  // qf16
              HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1);  // qf16

              if (use_fp32_exp) {
                HVX_VectorPair vp_s_minus_m0_sf = hvx_my_vqf16_to_wsf(v_s_minus_m0);
                HVX_VectorPair vp_s_minus_m1_sf = hvx_my_vqf16_to_wsf(v_s_minus_m1);

                HVX_Vector v_p_row00_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row01_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row10_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m1_sf));
                HVX_Vector v_p_row11_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m1_sf));

                v_p_row0_hf = hvx_my_wsf_to_vhf(v_p_row01_sf, v_p_row00_sf);
                v_p_row1_hf = hvx_my_wsf_to_vhf(v_p_row11_sf, v_p_row10_sf);
              } else {
                v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
                v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);
              }

              // compute P tile output positions
              __fp16     *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              HVX_Vector *pv_p_out0     = ((HVX_Vector *) out_dual_tile) + r1 / 2;
              HVX_Vector *pv_p_out1     = pv_p_out0 + 16;  // 16 * 128B = 2048B (1 tile)

              // write to P tile
              HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
              *pv_p_out0                   = Q6_V_lo_W(vp_p_dual_row);
              *pv_p_out1                   = Q6_V_hi_W(vp_p_dual_row);

              // rowsum(P) phase 1 reduction
              // v_p_rowsum0 = Q6_Vqf16_vadd_Vqf16Vhf(v_p_rowsum0, v_p_row0_hf);
              // v_p_rowsum1 = Q6_Vqf16_vadd_Vqf16Vhf(v_p_rowsum1, v_p_row1_hf);

              // reduce sum using qf32 precision
              HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0_hf);
              HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1_hf);

              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
            }

            // clang-format off
            // rowsum(P) phase 2 reduction
            // #pragma unroll
            // for (int s = 64; s >= 2; s >>= 1) {
            //   v_p_rowsum0 = Q6_Vqf16_vadd_Vqf16Vqf16(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
            //   v_p_rowsum1 = Q6_Vqf16_vadd_Vqf16Vqf16(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            // }
            // clang-format on
            // now, v_p_rowsum0[63] = rowsum(P)_0, v_p_rowsum1[63] = rowsum(P)_1

#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            }
            HVX_Vector v_p_rowsum_pack2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));

            // shift rowsum(P) into v_p_rowsum_local
            // HVX_Vector v_p_rowsum_pack2     = Q6_V_hi_W(Q6_W_vshuff_VVR(v_p_rowsum1, v_p_rowsum0, -2));
            HVX_Vector v_p_rowsum_pack2_rot = Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_p_rowsum_local_rot = Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
            v_p_rowsum_local = Q6_V_vlalign_VVR(v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r_vec_off * sizeof(__fp16));
          }

          // write local vector registers back to VTCM
          mvec_s_rowmax[r_vec_idx] = v_s_rowmax_local;
          mvec_p_rowsum[r_vec_idx] = v_p_rowsum_local;
        }
      }
      TIMER_STOP(safe_sm);

      // load [Bc, D] tile of V into VTCM
      TIMER_START(v_load);
      {
        // NOTE: at tile granularity, tile V's layout is column-major rather than row-major
        // because V tile is an RHS of matmul and HMX's dot RHS operands are column-major tiles
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        for (int r = 0; r < n_cols; r += 2) {
          const bool next_row_valid = (r + 1) < n_cols;

          const HVX_Vector *pv_in0 = (const HVX_Vector *) (v_ld_base + (r + 0) * kv_ld_stride);
          const HVX_Vector *pv_in1 = (const HVX_Vector *) (v_ld_base + (r + 1) * kv_ld_stride);

          // clang-format off
          #pragma unroll
          for (int c = 0; c < D; c += 64) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;
            int c0 = c / HMX_FP16_TILE_N_COLS;

            // transposed tile index: (c0, r0) => c0 * Bc/32 + r0
            int     tile_idx0  = (c0 + 0) * n_tiles_per_blk_c + r0;
            int     tile_idx1  = (c0 + 1) * n_tiles_per_blk_c + r0;
            __fp16 *tile_base0 = v_tile + tile_idx0 * HMX_FP16_TILE_N_ELMS;
            __fp16 *tile_base1 = v_tile + tile_idx1 * HMX_FP16_TILE_N_ELMS;

            HVX_Vector *pv_out0 = ((HVX_Vector *) tile_base0) + r1 / 2;
            HVX_Vector *pv_out1 = ((HVX_Vector *) tile_base1) + r1 / 2;
            *pv_out0            = Q6_V_lo_W(vp);
            *pv_out1            = Q6_V_hi_W(vp);
          }
          // clang-format on
        }
      }
      TIMER_STOP(v_load);

      // issue L2 prefetch of the next K tile
      {
        int jc_next = jc + blk_sz_c;
        if (jc_next < kv_len) {
          const size_t n_cols_next = smin(kv_len - jc_next, blk_sz_c);

          const __fp16 *k_ld_base = K + jc_next * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
          l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols_next, 0);
        }
      }

      // NOTE: after the use of rowmax(S), store exp(m_i^{j-1} - m_i^j) in the very same VTCM buffer
      HVX_Vector *mvec_exp_m_diff = mvec_s_rowmax;

      TIMER_START(core_acc);
      // update rowmax vector m_i and vector l_i
      {
        for (int i = 0; i < n_row_vec_cnt; ++i) {  // i => r_vec_idx?
          HVX_Vector v_m_prev = mvec_m[i];
          HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, mvec_s_rowmax[i]);
          HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);  // qf16

          HVX_Vector v_exp_m_diff_hf = hvx_my_exp2_vhf_vqf16(v_m_diff);    // fp16

          // l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
          HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(mvec_l[i], v_exp_m_diff_hf);  // qf16
          v_l_curr            = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, mvec_p_rowsum[i]);

          mvec_m[i] = v_m_curr;
          mvec_l[i] = v_l_curr;

          mvec_exp_m_diff[i] = v_exp_m_diff_hf;  // fp16
        }
      }

      // compute O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + P_i^j V_j
      {
        // generate D tile = diag(exp(m_i^{j-1} - m_i^j))
        const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
        const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        for (int i = 0; i < n_row_tiles; ++i) {
          const HVX_Vector v_content = Q6_V_vror_VR(mvec_exp_m_diff[i / 2], (i % 2) * 64);

          __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
          Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_id);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < head_dim / 32; ++c) {
              __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;  // D: [Br', Br']
              __fp16 *o_tile_in =
                o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;  // O: [Br', D] --T-> [D, Br']
              hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

              __fp16 *p_tile_in = p_tile + (r * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;  // P: [Br', Bc]
              __fp16 *v_tile_in = v_tile + (c * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;  // V: [Bc, D] --T-> [D, Bc]
              // NOTE: `n_col_tiles` may exceed 32, we need to explicitly split and accumulate
              for (int k = 0; k < n_col_tiles; k += 32) {
                int    offset  = k * HMX_FP16_TILE_N_ELMS;
                size_t n_tiles = smin(n_col_tiles - k, 32);
                hmx_load_tiles_fp16(p_tile_in + offset, v_tile_in + offset, n_tiles);
              }

              // NOTE: O's layout is also column-major as O is always on the RHS
              __fp16 *o_tile_out = o_tile_curr + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
              hmx_consume_accumulator_fp16(o_tile_out);
            }
          }
        }
        hmx_unit_release();

        swap_ptr(&o_tile_curr, &o_tile_prev);
      }
      TIMER_STOP(core_acc);
    }

    // generate final output: scale O_i = diag(l_i^{-1}) O_i
    TIMER_START(o_scale);
    {
      const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
      const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

      HVX_Vector v_content;
      for (int i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
          v_content = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(mvec_l[i / 2]));
        } else {
          v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
      }

      hmx_unit_acquire();
      {
        hmx_set_output_scales(hmx_output_scales_id);
        for (int r = 0; r < n_row_tiles; ++r) {
          for (int c = 0; c < head_dim / 32; ++c) {
            __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;

            // NOTE: to simplify final output procedure, we turn final O into row-major layout
            __fp16 *o_tile_out = o_tile_curr + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;

            hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
          }
        }
      }
      hmx_unit_release();
    }
    TIMER_STOP(o_scale);

    // store [n_rows*G, D] tile of O back to memory
    TIMER_START(o_store);
    {
      const size_t o_st_blk_sz_bytes = o_st_blk_sz * qo_element_size;
      const size_t o_st_stride_bytes = o_st_stride * qo_element_size;

      uint8_t *o_st_base = ((uint8_t *) O) + ir * o_st_stride_bytes + kv_head_idx * o_st_blk_sz_bytes;

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int o_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int o_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        uint8_t *row_out0 = o_st_base + o_idx0 * o_st_stride_bytes + h_idx0 * o_head_dim * qo_element_size;
        uint8_t *row_out1 = o_st_base + o_idx1 * o_st_stride_bytes + h_idx1 * o_head_dim * qo_element_size;

        HVX_Vector *pv_out0 = (HVX_Vector *) row_out0;
        HVX_Vector *pv_out1 = (HVX_Vector *) row_out1;

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *in_base = o_tile_curr + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);
        
            const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);
            const HVX_Vector v_out0 = Q6_V_lo_W(vp);
            const HVX_Vector v_out1 = Q6_V_hi_W(vp);

            if (with_lse) {
              vstu_variable(row_out0 + d * 32 * sizeof(float), sizeof(v_out0), v_out0);
              if (next_row_valid) {
                vstu_variable(row_out1 + d * 32 * sizeof(float), sizeof(v_out1), v_out1);
              }
            } else {
              *pv_out0++ = v_out0;
              if (next_row_valid) {
                *pv_out1++ = v_out1;
              }
            }
          }

          if (with_lse) {
            const float lse0 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 0);
            memcpy(row_out0 + D * sizeof(float), &lse0, sizeof(lse0));
            if (next_row_valid) {
              const float lse1 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 1);
              memcpy(row_out1 + D * sizeof(float), &lse1, sizeof(lse1));
            }
          }
        } else {
          assert(!with_lse);
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const __fp16     *in_dual_tile = in_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector *pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
            const HVX_Vector *pv_in1       = pv_in0 + 16;

            const HVX_VectorPair vp = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        }
        // clang-format on
      }
    }
    TIMER_STOP(o_store);
  }

#if defined(ENABLE_PROFILE_TIMERS)
  {
    FARF(ALWAYS, "q_load: %lld us, k_load: %lld us, v_load: %lld us, qk_dot: %lld us", TIMER_US(q_load),
         TIMER_US(k_load), TIMER_US(v_load), TIMER_US(qk_dot));
    FARF(ALWAYS, "safe_sm: %lld us, core_acc: %lld us, o_scale: %lld us, o_store: %lld us", TIMER_US(safe_sm),
         TIMER_US(core_acc), TIMER_US(o_scale), TIMER_US(o_store));
  }
#endif
}

size_t fa_f32_compute_vtcm_usage(int group_size, int head_dim, int n_rows, int n_cols) {
  const size_t g_br = align_up(group_size * n_rows, 32);

  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(float), 4096);    // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(n_cols * head_dim * sizeof(float), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * n_cols * sizeof(float), 4096);      // S, P: [Br', Bc]
  const size_t col_vec_size   = align_up(g_br * sizeof(float), 256);                // m_prev, m_curr, l, rowsum: [Br']

  size_t total =
    qo_tile_size * 2 /* Q, O */ + kv_tile_size * 2 /* K, V */ + core_tile_size * 1 /* S/P */ + col_vec_size * 4;
  return total;
}

void fa_f32_find_chunk_size(size_t *blk_r, size_t *blk_c, int group_size, int head_dim, int qo_len, int kv_len,
                            size_t limit) {
  const int nr_unit = 32 / group_size;
  const int nc_unit = 32;

  find_chunk_size_common(blk_r, blk_c, group_size, head_dim, qo_len, kv_len, limit, nr_unit, nc_unit,
                         fa_f32_compute_vtcm_usage);
}

void simple_flash_attn_f32_core(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O,
                                const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                                const __fp16 *restrict qk_mask, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                                int head_dim, bool with_lse) {
  const int G = n_heads / n_kv_heads;  // group size
  const int D = head_dim;

  const bool   has_qk_mask = (qk_mask != NULL);
  const size_t kv_pad_len  = align_up(kv_len, 64);

  const float qk_scale = 1.0f / sqrtf(head_dim) * 1.44269504f;

  const size_t q_ld_stride    = n_heads * head_dim;
  const size_t q_ld_blk_sz    = G * head_dim;
  const size_t o_head_dim     = head_dim + (with_lse ? 1 : 0);
  const size_t o_st_stride    = n_heads * o_head_dim;
  const size_t o_st_blk_sz    = G * o_head_dim;
  const size_t kv_ld_stride   = n_kv_heads * head_dim;
  const size_t kv_ld_blk_sz   = head_dim;

  size_t blk_sz_r, blk_sz_c;  // Br, Bc
  fa_f32_find_chunk_size(&blk_sz_r, &blk_sz_c, G, head_dim, qo_len, kv_len, vtcm_limit - vtcm);

  const size_t g_br = align_up(G * blk_sz_r, 32);  // Br'
  FARF(ALWAYS, "%s: Br=%d Bc=%d Br'=%d", __func__, blk_sz_r, blk_sz_c, g_br);

  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(float), 4096);      // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(blk_sz_c * head_dim * sizeof(float), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * blk_sz_c * sizeof(float), 4096);      // S, P: [Br', Bc]
  const size_t col_vec_size   = align_up(g_br * sizeof(float), 256);  // m_prev, m_curr, l, rowsum: [Br']

  uint8_t *vtcm_cur = vtcm;

  float *q_tile = (float *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  float *o_tile = (float *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  float *k_tile = (float *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  float *v_tile = (float *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  float *s_tile = (float *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  HVX_Vector *mvec_m_prev   = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_m_curr   = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  assert(vtcm_cur <= vtcm_limit);

  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows   = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g = n_rows * G;

    // load [n_rows*G, D] tile of Q into VTCM
    {
      const float *q_ld_base = ((const float *) Q) + ir * q_ld_stride + kv_head_idx * q_ld_blk_sz;

      // l2fetch(q_ld_base, qo_ldst_stride * sizeof(float), qo_ldst_blk_sz * sizeof(float), n_rows, 1);

      // HVX_Vector *pv_out = (HVX_Vector *) q_tile;
      // for (int r = 0; r < n_rows; ++r) {
      //   const HVX_Vector *pv_in = (const HVX_Vector *) (q_ld_base + r * qo_ldst_stride);
      //   for (int gd = 0; gd < G * D; gd += 32) {
      //     *pv_out++ = *pv_in++;
      //   }
      // }

      _Alignas(64) dma_desc_2d_t desc = { 0 };

      desc.next       = 0;  // no next descriptor
      desc.length     = 0;  // not used
      desc.type       = DMA_DESC_TYPE_2D;
      desc.dst_bypass = 0;
      desc.src_bypass = 1;                        // bypass L2 cache
      desc.ordered    = 1;                        // ordered, doesn't matter
      desc.dstate     = DMA_DESC_DSTATE_PENDING;  // to be processed

      desc.src        = (uint32_t) q_ld_base;
      desc.dst        = (uint32_t) q_tile;
      desc.roi_width  = q_ld_blk_sz * sizeof(float);
      desc.roi_height = n_rows;
      desc.src_stride = q_ld_stride * sizeof(float);
      desc.dst_stride = q_ld_blk_sz * sizeof(float);

      desc.src_width_offset = 0;
      desc.dst_width_offset = 0;

      dma_wait_for_idle();
      dma_submit_one((dma_desc_1d_t *) &desc);
      dma_wait_for_idle();
    }

    hvx_fill_uw(mvec_m_prev, 0xff7fffff, col_vec_size);
    hvx_fill_uw(mvec_l, 0, col_vec_size);

    hvx_fill_uw(o_tile, 0, core_tile_size);

    for (int jc = 0; jc < kv_len; jc += blk_sz_c) {
      const size_t n_cols = smin(kv_len - jc, blk_sz_c);

      // load and convert [Bc, D] tile of K into VTCM
      {
        const __fp16 *k_ld_base = K + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 1);

        /*
        HVX_Vector *pv_out = (HVX_Vector *) k_tile;
        for (int c = 0; c < n_cols; ++c) {
          const HVX_Vector *pv_in = (const HVX_Vector *) (k_ld_base + c * kv_ld_stride);

#pragma unroll
          for (int d = 0; d < D; d += 64) {
            HVX_VectorPair vp = hvx_my_vhf_to_wsf(*pv_in++);
            vp                = Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -4);
            *pv_out++         = Q6_V_lo_W(vp);
            *pv_out++         = Q6_V_hi_W(vp);
          }
        }
        */

        // NOTE: This scalar version is slow but more accurate
        for (int c = 0; c < n_cols; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            k_tile[c * head_dim + d] = (float) k_ld_base[c * kv_ld_stride + d];
          }
        }
      }

      // issue L2 prefetch of V tile
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 0);
      }

      // compute QK^T
      {
        const HVX_Vector v_zero          = Q6_V_vzero();
        const HVX_Vector v_qk_scale_sf   = Q6_V_vsplat_R(*(uint32_t *) (&qk_scale));  // fp32_to_bits
        const HVX_Vector v_qk_scale_qf32 = Q6_Vqf32_vadd_VsfVsf(v_qk_scale_sf, v_zero);

        for (int r = 0; r < n_rows_g; ++r) {
          for (int c0 = 0; c0 < n_cols; c0 += 32) {
            _Alignas(VLEN) float qk_dot_local[32], tmp[32];

            for (int c1 = 0; c1 < 32; ++c1) {
              int c = c0 + c1;
              if (c >= n_cols) {
                break;
              }

              const HVX_Vector *pv_row = (const HVX_Vector *) (q_tile + r * head_dim);
              const HVX_Vector *pv_col = (const HVX_Vector *) (k_tile + c * head_dim);

              HVX_Vector v_sum = v_zero;
              for (int d = 0; d < D; d += 32) {
                HVX_Vector v_prod;
                v_prod = Q6_Vqf32_vmpy_VsfVsf(*pv_row++, *pv_col++);
                v_sum  = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, v_prod);
              }
#pragma unroll
              for (int s = 64; s >= 4; s >>= 1) {
                v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_V_vlalign_VVR(v_sum, v_zero, s));
              }
              vmem(tmp)        = v_sum;
              qk_dot_local[c1] = tmp[31];
            }

            HVX_Vector v_scaled_qk           = Q6_Vqf32_vmpy_Vqf32Vqf32(vmem(qk_dot_local), v_qk_scale_qf32);
            vmem(&s_tile[r * blk_sz_c + c0]) = Q6_Vsf_equals_Vqf32(v_scaled_qk);
          }
        }

        // for (int r = 0; r < n_rows_g; ++r) {
        //   for (int c = 0; c < n_cols; ++c) {
        //     float s = 0.0f;
        //     for (int d = 0; d < head_dim; ++d) {
        //       s += q_tile[r * head_dim + d] * k_tile[c * head_dim + d];
        //     }
        //     s_tile[r * blk_sz_c + c] = s * qk_scale;
        //   }
        // }
      }

      // core softmax computation
      {
        const HVX_Vector v_neg_inf = Q6_V_vsplat_R(0xff7fffff);  // 1 11111110 11...1

        for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
          _Alignas(VLEN) float m_prev_local[32], m_curr_local[32], p_rowsum_local[32], tmp[32];

          vmem(m_prev_local) = mvec_m_prev[r0 / 32];

          for (int r1 = 0; r1 < 32; ++r1) {
            int r = r0 + r1;
            if (r >= n_rows_g) {
              break;
            }

            HVX_Vector *const pv_s_row = (HVX_Vector *) (s_tile + r * blk_sz_c);

            // apply mask & compute rowmax(S)
            HVX_Vector v_s_rowmax = v_neg_inf;
            for (int c = 0; c < n_cols; c += 32) {
              int q_idx = ir + r / G;
              int k_idx = jc + c;

              // TODO(hzx): handle leftover mask when qk_mask == null
              HVX_Vector     v_mask_hf  = has_qk_mask ? vmemu(qk_mask + q_idx * kv_pad_len + k_idx) : Q6_V_vzero();
              HVX_VectorPair vp_mask_sf = hvx_my_vhf_to_wsf(v_mask_hf);
              HVX_Vector     v_mask_sf  = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp_mask_sf), Q6_V_lo_W(vp_mask_sf), -4));

              const HVX_Vector v_fp32_mask_threshold = Q6_V_vsplat_R(0xc3000000);  // fp32: -128.0
              HVX_VectorPred   q_mask_out            = Q6_Q_vcmp_gt_VsfVsf(v_fp32_mask_threshold, v_mask_sf);

              HVX_Vector v_s_row = Q6_V_vmux_QVV(q_mask_out, v_neg_inf, pv_s_row[c / 32]);
              pv_s_row[c / 32]   = v_s_row;

              v_s_rowmax = Q6_Vsf_vmax_VsfVsf(v_s_rowmax, v_s_row);
            }
#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_s_rowmax = Q6_Vsf_vmax_VsfVsf(v_s_rowmax, Q6_V_vlalign_VVR(v_s_rowmax, v_neg_inf, s));
            }
            vmem(tmp)      = v_s_rowmax;
            float s_rowmax = tmp[31];

            float m_cur      = s_rowmax > m_prev_local[r1] ? s_rowmax : m_prev_local[r1];  // fmaxf
            m_curr_local[r1] = m_cur;

            HVX_Vector v_dup_m = Q6_V_vsplat_R(*(uint32_t *) (&m_cur));  // fp32_to_bits

            // compute rows of P = exp(S - m)
            HVX_Vector *const pv_p_row   = pv_s_row;  // inplace replacement of S tile
            const HVX_Vector  v_zero     = Q6_V_vzero();
            HVX_Vector        v_p_rowsum = v_zero;    // qf32
            for (int c = 0; c < n_cols; c += 32) {
              HVX_Vector v_s_minus_m = Q6_Vqf32_vsub_VsfVsf(pv_s_row[c / 32], v_dup_m);
              HVX_Vector v_p_row_sf  = hvx_my_exp2_vsf_vqf32(v_s_minus_m);

              pv_p_row[c / 32] = v_p_row_sf;

              v_p_rowsum = Q6_Vqf32_vadd_Vqf32Vsf(v_p_rowsum, v_p_row_sf);
            }
#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_p_rowsum = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum, Q6_V_vlalign_VVR(v_p_rowsum, v_zero, s));
            }
            vmem(tmp)      = v_p_rowsum;
            float p_rowsum = tmp[31];

            p_rowsum_local[r1] = p_rowsum;
          }

          mvec_m_curr[r0 / 32]   = vmem(m_curr_local);
          mvec_p_rowsum[r0 / 32] = vmem(p_rowsum_local);
        }
      }

      // load and convert [Bc, D] tile of V into VTCM
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        /*
        HVX_Vector *pv_out = (HVX_Vector *) v_tile;
        for (int c = 0; c < n_cols; ++c) {
          const HVX_Vector *pv_in = (const HVX_Vector *) (v_ld_base + c * kv_ld_stride);

#pragma unroll
          for (int d = 0; d < D; d += 64) {
            HVX_VectorPair vp = hvx_my_vhf_to_wsf(*pv_in++);
            vp                = Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -4);
            *pv_out++         = Q6_V_lo_W(vp);
            *pv_out++         = Q6_V_hi_W(vp);
          }
        }
        */

        // NOTE: This scalar version is slow but more accurate
        for (int c = 0; c < n_cols; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            v_tile[c * head_dim + d] = (float) v_ld_base[c * kv_ld_stride + d];
          }
        }
      }

      // issue L2 prefetch of the next K tile
      {
        int jc_next = jc + blk_sz_c;
        if (jc_next < kv_len) {
          const size_t n_cols_next = smin(kv_len - jc_next, blk_sz_c);

          const __fp16 *k_ld_base = K + jc_next * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
          l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols_next, 0);
        }
      }

      // core accumulation (update)
      {
        HVX_Vector *mvec_exp_m_diff = mvec_m_curr;  // inplace replacement, qf32
        // update vector m_i, l_i
        for (int r = 0; r < n_rows_g; r += 32) {
          int i = r / 32;

          HVX_Vector v_m_prev = mvec_m_prev[i];
          HVX_Vector v_m_curr = mvec_m_curr[i];
          HVX_Vector v_m_diff = Q6_Vqf32_vsub_VsfVsf(v_m_prev, v_m_curr);

          HVX_Vector v_exp_m_diff = hvx_my_exp2_vqf32(v_m_diff);

          // l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
          HVX_Vector v_l_curr = Q6_Vqf32_vmpy_Vqf32Vqf32(mvec_l[i], v_exp_m_diff);
          v_l_curr            = Q6_Vqf32_vadd_Vqf32Vqf32(v_l_curr, mvec_p_rowsum[i]);

          mvec_m_prev[i] = v_m_curr;
          mvec_l[i]      = v_l_curr;

          mvec_exp_m_diff[i] = v_exp_m_diff;  // qf32
        }

        // compute O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + P_i^j V_j

        /*
        // scalar impl ref: assume o_tile fp32
        _Alignas(VLEN) float o_row_scale_local[32];
        for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
          vmem(o_row_scale_local) = Q6_Vsf_equals_Vqf32(mvec_exp_m_diff[r0 / 32]);

          for (int r1 = 0; r1 < 32; ++r1) {
            int r = r0 + r1;
            if (r >= n_rows_g) {
              break;
            }

            float *p_row = s_tile + r * blk_sz_c;
            float *o_row = o_tile + r * head_dim;

            for (int d = 0; d < head_dim; ++d) {
              o_row[d] *= o_row_scale_local[r1];

              float sum = 0.0f;
              for (int c = 0; c < n_cols; ++c) {
                sum += p_row[c] * v_tile[c * head_dim + d];
              }
              o_row[d] += sum;
            }
          }
        }
        */

        _Alignas(VLEN) int32_t o_row_scale_qf32_local[32], p_row_local[32];
        for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
          vmem(o_row_scale_qf32_local) = mvec_exp_m_diff[r0 / 32];

          for (int r1 = 0; r1 < 32; ++r1) {
            int r = r0 + r1;
            if (r >= n_rows_g) {
              break;
            }

            int32_t          o_row_scale_qf32   = o_row_scale_qf32_local[r1];
            const HVX_Vector v_o_row_scale_qf32 = Q6_V_vsplat_R(o_row_scale_qf32);

            const HVX_Vector *const pv_p_row = (const HVX_Vector *) (s_tile + r * blk_sz_c);
            HVX_Vector *const       pv_o_row = (HVX_Vector *) (o_tile + r * head_dim);

            for (int d = 0; d < head_dim; d += 32) {
              HVX_Vector v_sum = Q6_V_vzero();

              // reduction axis: column (along `kv_len` axis)
              for (int c0 = 0; c0 < n_cols; c0 += 32) {
                // NOTE(hzx): consider alternative solution
                // sol 1# outer loop: VTCM -> vec reg -> L2; inner loop: L2 -> L1 -> scalar reg -> vec reg
                // sol 2# outer loop: VTCM -> vec reg; inner loop: vec reg ---> vec reg
                // NOTE: broadcasting a 4B value to the whole vector register may have long critical path
                //     example instruction sequence: vror + 2x vlut16 + vmux
                vmem(p_row_local) = pv_p_row[c0 / 32];

                for (int c1 = 0; c1 < 32; ++c1) {
                  int c = c0 + c1;
                  if (c >= n_cols) {
                    break;
                  }

                  HVX_Vector v_p_elem = Q6_V_vsplat_R(p_row_local[c1]);
                  HVX_Vector v_v_vec  = vmem(&v_tile[c * head_dim + d]);

                  v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vmpy_VsfVsf(v_p_elem, v_v_vec));
                }
              }

              // now `v_sum` contains the dot product of P[1, Bc] & V[Bc, 32]
              HVX_Vector v_scaled_o = Q6_Vqf32_vmpy_Vqf32Vqf32(v_o_row_scale_qf32, pv_o_row[d / 32]);
              pv_o_row[d / 32]      = Q6_Vqf32_vadd_Vqf32Vqf32(v_scaled_o, v_sum);
            }
          }
        }
      }
    }

    // scale O tile: O_i = diag(l_i^{-1}) O_i
    {
      _Alignas(VLEN) int32_t inv_l_local[32];
      for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
        HVX_Vector v_inv_l_qf32 = hvx_my_inv_vqf32_vsf(Q6_Vsf_equals_Vqf32(mvec_l[r0 / 32]));
        vmem(inv_l_local)       = v_inv_l_qf32;

        for (int r1 = 0; r1 < 32; ++r1) {
          int r = r0 + r1;
          if (r >= n_rows_g) {
            break;
          }

          const HVX_Vector v_o_row_scale_qf32 = Q6_V_vsplat_R(inv_l_local[r1]);

          HVX_Vector *pv_o_row = (HVX_Vector *) (o_tile + r * head_dim);
          for (int d = 0; d < head_dim; d += 32) {
            HVX_Vector v_scaled_o = Q6_Vqf32_vmpy_Vqf32Vqf32(v_o_row_scale_qf32, pv_o_row[d / 32]);
            pv_o_row[d / 32]      = Q6_Vsf_equals_Vqf32(v_scaled_o);
          }
        }
      }

      /*
      // scalar impl ref: assume o_tile fp32
      _Alignas(VLEN) float inv_l_local[32];
      for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
        // vmem(inv_l_local) = Q6_Vsf_equals_Vqf32(hvx_my_inv_vqf32_vsf(Q6_Vsf_equals_Vqf32(mvec_l[r0 / 32])));
        vmem(inv_l_local) = Q6_Vsf_equals_Vqf32(mvec_l[r0 / 32]);

        for (int r1 = 0; r1 < 32; ++r1) {
          int r = r0 + r1;
          if (r >= n_rows_g) {
            break;
          }

          inv_l_local[r1] = 1.0f / inv_l_local[r1];

          float *o_row = o_tile + r * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            o_row[d] *= inv_l_local[r1];
          }
        }
      }
      */
    }

    // store [n_rows*G, D] tile of O back to DDR memory
    {
      float *o_st_base = ((float *) O) + ir * o_st_stride + kv_head_idx * o_st_blk_sz;

      const HVX_Vector *pv_in = (HVX_Vector *) o_tile;
      for (int r = 0; r < n_rows; ++r) {
        if (with_lse) {
          for (int g = 0; g < G; ++g) {
            float *row_out = o_st_base + r * o_st_stride + g * o_head_dim;
            memcpy(row_out, o_tile + (r * G + g) * D, D * sizeof(float));
            row_out[D] = flash_attn_lse_from_f32_state(mvec_m_prev, mvec_l, r * G + g);
          }
          pv_in += (G * D) / 32;
        } else {
          HVX_Vector *pv_out = (HVX_Vector *) (o_st_base + r * o_st_stride);
          for (int gd = 0; gd < G * D; gd += 32) {
            *pv_out++ = *pv_in++;
          }
        }
      }

      // TODO(hzx): investigate why DMA is not working here
      /*
      _Alignas(64) dma_desc_2d_t desc = { 0 };

      desc.next       = 0;
      desc.length     = 0;
      desc.type       = DMA_DESC_TYPE_2D;
      desc.dst_bypass = 1;
      desc.src_bypass = 0;
      desc.order      = 1;
      desc.dstate     = DMA_DESC_DSTATE_PENDING;

      desc.src        = (uint32_t) o_tile;
      desc.dst        = (uint32_t) o_st_base;
      desc.roi_width  = o_st_blk_sz * sizeof(float);
      desc.roi_height = n_rows;
      desc.src_stride = q_ld_blk_sz * sizeof(float);
      desc.dst_stride = o_st_stride * sizeof(float);

      desc.src_width_offset = 0;
      desc.dst_width_offset = 0;

      dma_wait_for_idle();
      dma_submit_one((dma_desc_1d_t *) &desc);
      dma_wait_for_idle();
      */
    }
  }
}

void simple_flash_attn_worker(void *data, int worker_index) {
  simple_fa_task_state_t *s = (simple_fa_task_state_t *) data;

  uint8_t *vtcm       = s->vtcm_base + worker_index * s->vtcm_size_per_thread;
  uint8_t *vtcm_limit = vtcm + s->vtcm_size_per_thread;

  hmx_manager_enable_execution();

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(s->task_id)) - 1;
    if (task_id >= s->n_tasks) {
      break;
    }

    int kv_head_idx = task_id;
    simple_flash_attn_f16_core(kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->mask, s->qo_len, s->kv_len,
                               s->n_heads, s->n_kv_heads, s->head_dim, s->with_lse, worker_index);
  }

  hmx_manager_disable_execution();

  worker_pool_synctoken_jobdone(&(s->sync_ctx));
}

int simple_flash_attn_sp_hdim(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                              const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                              int n_heads, int n_kv_heads, int head_dim);

/**
 * Simple llama.cpp-style FlashAttention implementation
 *
 * batch_size dimension is omitted
 * 
 * Q: [qo_len, n_heads, head_dim], K/V: [kv_len, n_kv_heads, head_dim]
 * mask: [qo_len*, kv_len] broadcast to each head (first dimension maybe larger than qo_len)
 */
int simple_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                      const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                      bool with_lse) {
  if (head_dim % 64 != 0) {
    if (with_lse) {
      FARF(ALWAYS, "FA with LSE not supported for non-64-aligned head_dim=%d", head_dim);
      return -1;
    }
    return simple_flash_attn_sp_hdim(O, Q, K, V, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim);
  }
  if (n_heads % n_kv_heads != 0) {
    FARF(ALWAYS, "FA not supported: head_dim=%d n_heads=%d n_kv_heads=%d", head_dim, n_heads, n_kv_heads);
    return -1;
  }

  const size_t vtcm_size_per_thread = 1024 * 1024;
  const int    n_jobs               = num_hvx128_contexts;
  const unsigned int vtcm_slice_count = num_workers;
  if (n_jobs <= 0 || vtcm_slice_count == 0 || (size_t) vtcm_slice_count > SIZE_MAX / vtcm_size_per_thread) {
    FARF(ERROR, "%s: invalid worker counts: jobs=%d pool_workers=%u", __func__, n_jobs, vtcm_slice_count);
    return -1;
  }

  const size_t required_vtcm_size = (size_t) vtcm_slice_count * vtcm_size_per_thread;
  uint8_t *const vtcm_base        = (uint8_t *) vtcm_manager_get_vtcm_base();
  const size_t acquired_vtcm_size = vtcm_manager_get_vtcm_size();
  if (!vtcm_base || acquired_vtcm_size < required_vtcm_size) {
    FARF(ERROR, "%s: insufficient VTCM: acquired=%u required=%u pool_workers=%u", __func__,
         (unsigned int) acquired_vtcm_size, (unsigned int) required_vtcm_size, vtcm_slice_count);
    return -1;
  }

  simple_fa_task_state_t state;
  state.O          = O;
  state.Q          = Q;
  state.K          = K;
  state.V          = V;
  state.mask       = mask;
  state.qo_len     = qo_len;
  state.kv_len     = kv_len;
  state.n_heads    = n_heads;
  state.n_kv_heads = n_kv_heads;
  state.head_dim   = head_dim;
  state.with_lse   = with_lse;

  // TODO(hzx): parallelize along query_len x n_kv_heads dimension
  // size_t n_tot_chunks      = qo_len * n_kv_heads;
  // size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_jobs);

  state.task_id              = 0;
  state.n_tasks              = n_kv_heads;
  state.vtcm_base            = vtcm_base;
  state.vtcm_size_per_thread = vtcm_size_per_thread;

  worker_pool_job_t job;
  job.fptr = simple_flash_attn_worker;
  job.dptr = &state;

  int64_t t0 = HAP_perf_get_time_us();

  worker_pool_synctoken_init(&(state.sync_ctx), n_jobs);
  for (int i = 0; i < n_jobs; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  int64_t elapsed_us = HAP_perf_get_time_us() - t0;
  FARF(ALWAYS, "%s: %lld us, qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d", __func__, elapsed_us, qo_len,
       kv_len, n_heads, n_kv_heads, head_dim);

  return 0;
}

typedef struct {
  __fp16    *q;
  __fp16    *o_prev;
  __fp16    *o_next;
  HVX_Vector *m;
  HVX_Vector *l;
  int         n_rows;
} block_sparse_halo_q_state_t;

typedef struct {
  __fp16    *score;
  __fp16    *probability;
  __fp16    *diagonal;
  HVX_Vector *rowmax;
  HVX_Vector *rowsum;
  HVX_Vector *row_buffer0;
  HVX_Vector *row_buffer1;
} block_sparse_halo_update_scratch_t;

typedef struct {
  const __fp16 *packed_k;
  const __fp16 *packed_v;
  int           v_tile_stride;
  int           n_real_cols;
} block_sparse_halo_update_segment_t;

static int block_sparse_halo_pack_q_f32(__fp16 *packed_q, const float *raw_q,
                                        size_t raw_q_stride, int n_rows, int head_dim) {
  if (packed_q == NULL || raw_q == NULL || raw_q_stride < (size_t) head_dim ||
      n_rows <= 0 || n_rows > HMX_FP16_TILE_N_ROWS || head_dim <= 0 || head_dim % 32 != 0) {
    return -1;
  }

  for (int r = 0; r < n_rows; r += 2) {
    const bool next_row_valid = r + 1 < n_rows;
    const HVX_Vector *pv_in0 = (const HVX_Vector *) (raw_q + (size_t) r * raw_q_stride);
    const HVX_Vector *pv_in1 = next_row_valid ?
      (const HVX_Vector *) (raw_q + (size_t) (r + 1) * raw_q_stride) : pv_in0;
    for (int d = 0; d < head_dim / 32; ++d) {
      const HVX_Vector v0 = *pv_in0++;
      const HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
      HVX_Vector *out_tile =
        (HVX_Vector *) (packed_q + (size_t) d * HMX_FP16_TILE_N_ELMS);
      out_tile[r / 2] = hvx_my_wsf_to_vhf(v1, v0);
    }
  }
  return 0;
}

static int block_sparse_halo_update_packed_segments(
  block_sparse_halo_q_state_t *state, block_sparse_halo_update_scratch_t *scratch,
  const block_sparse_halo_update_segment_t *segments, int segment_count,
  int scratch_col_capacity, int head_dim, const uint8_t *hmx_output_scales_qk,
  const uint8_t *hmx_output_scales_id, const int16_t *d_tile_vscatter_offsets) {
  if (state == NULL || scratch == NULL || segments == NULL ||
      state->q == NULL || state->o_prev == NULL || state->o_next == NULL ||
      state->m == NULL || state->l == NULL || scratch->score == NULL ||
      scratch->probability == NULL || scratch->diagonal == NULL || scratch->rowmax == NULL ||
      scratch->rowsum == NULL || scratch->row_buffer0 == NULL || scratch->row_buffer1 == NULL ||
      hmx_output_scales_qk == NULL || hmx_output_scales_id == NULL || d_tile_vscatter_offsets == NULL ||
      state->n_rows <= 0 || segment_count <= 0 ||
      segment_count > BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS ||
      scratch_col_capacity <= 0 || scratch_col_capacity % HMX_FP16_TILE_N_COLS != 0 ||
      head_dim <= 0 || head_dim % 32 != 0) {
    return -1;
  }

  int segment_cols[BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS] = {0};
  for (int segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
    const block_sparse_halo_update_segment_t *segment = &segments[segment_idx];
    if (segment->packed_k == NULL || segment->packed_v == NULL || segment->n_real_cols <= 0 ||
        segment->v_tile_stride < ceil_div(segment->n_real_cols, HMX_FP16_TILE_N_COLS)) {
      return -1;
    }
    segment_cols[segment_idx] = segment->n_real_cols;
  }
  block_sparse_halo_update_segment_plan_t segment_plan;
  if (block_sparse_halo_plan_update_segments(
        &segment_plan, segment_cols, segment_count, (size_t) scratch_col_capacity) != 0) {
    return -1;
  }
  const int n_cols = segment_plan.total_cols;

  const int n_rows = state->n_rows;
  const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
  const int n_row_vec_cnt = ceil_div(n_rows, 64);
  const int n_tiles_per_blk_r = align_up(n_rows, HMX_FP16_TILE_N_ROWS) / HMX_FP16_TILE_N_ROWS;
  const int n_tiles_per_scratch_row = scratch_col_capacity / HMX_FP16_TILE_N_COLS;

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_qk);
  for (int segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
    const block_sparse_halo_update_segment_t *segment = &segments[segment_idx];
    for (int r = 0; r < n_row_tiles; ++r) {
      for (int c = 0; c < segment_plan.tile_counts[segment_idx]; ++c) {
        const __fp16 *row_tiles = state->q + r * HMX_FP16_TILE_N_ROWS * head_dim;
        const __fp16 *col_tiles = segment->packed_k + c * HMX_FP16_TILE_N_COLS * head_dim;
        __fp16 *out_tile = scratch->score +
          (r * n_tiles_per_scratch_row + segment_plan.tile_offsets[segment_idx] + c) *
          HMX_FP16_TILE_N_ELMS;
        hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
      }
    }
  }
  hmx_unit_release();

  const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
  for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
    HVX_Vector v_s_rowmax_local = v_neg_inf;
    HVX_Vector v_p_rowsum_local = Q6_V_vzero();

    for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
      const int r = r_vec_idx * 64 + r_vec_off;
      if (r >= align_up(n_rows, 2)) {
        break;
      }

      const int r0 = r / HMX_FP16_TILE_N_ROWS;
      const int r1 = r % HMX_FP16_TILE_N_ROWS;
      __fp16 *s_ld_base = scratch->score + r0 * HMX_FP16_TILE_N_ROWS * scratch_col_capacity;
      __fp16 *p_st_base = scratch->probability + r0 * HMX_FP16_TILE_N_ROWS * scratch_col_capacity;

      HVX_Vector *pv_row_buf0 = scratch->row_buffer0;
      HVX_Vector *pv_row_buf1 = scratch->row_buffer1;
      for (int c = 0; c < n_cols; c += 64) {
        const __fp16 *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
        const HVX_Vector *pv_s_in0 = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
        const bool second_tile_valid = c + HMX_FP16_TILE_N_COLS < n_cols;
        const HVX_Vector v_s_in1 = second_tile_valid ? pv_s_in0[16] : Q6_V_vzero();
        const HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(v_s_in1, *pv_s_in0, -2);
        *pv_row_buf0++ = Q6_V_lo_W(vp_s_dual_row);
        *pv_row_buf1++ = Q6_V_hi_W(vp_s_dual_row);
      }

      HVX_Vector v_s_rowmax0 = v_neg_inf;
      HVX_Vector v_s_rowmax1 = v_neg_inf;
      for (int c = 0; c < n_cols; c += 64) {
        const size_t ne = smin(n_cols - c, 64);
        const HVX_VectorPred q_mask_keep0 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
        const HVX_VectorPred q_mask_keep1 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
        const HVX_Vector v_s_row0 =
          Q6_V_vmux_QVV(q_mask_keep0, scratch->row_buffer0[c / 64], v_neg_inf);
        const HVX_Vector v_s_row1 =
          Q6_V_vmux_QVV(q_mask_keep1, scratch->row_buffer1[c / 64], v_neg_inf);
        scratch->row_buffer0[c / 64] = v_s_row0;
        scratch->row_buffer1[c / 64] = v_s_row1;
        v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
        v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
      }

#pragma unroll
      for (int s = 64; s >= 2; s >>= 1) {
        v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
        v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
      }

      const HVX_Vector v_s_rowmax_pack2 = Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));
      const HVX_Vector v_s_rowmax_pack2_rot =
        Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));
      const HVX_Vector v_s_rowmax_local_rot =
        Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));
      v_s_rowmax_local =
        Q6_V_vlalign_VVR(v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r_vec_off * sizeof(__fp16));

      const HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(state->m[r_vec_idx], v_s_rowmax_local);
      const HVX_Vector v_m_lut = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));
      const HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
      const HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));
      const HVX_Vector v_zero = Q6_V_vzero();
      HVX_Vector v_p_rowsum0 = v_zero;
      HVX_Vector v_p_rowsum1 = v_zero;

      for (int c = 0; c < n_cols; c += 64) {
        const int ne = smax(0, smin(n_cols - c, 64));
        const HVX_VectorPred q_mask_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
        const HVX_Vector v_s_minus_m0 =
          Q6_Vqf16_vsub_VhfVhf(scratch->row_buffer0[c / 64], v_dup_m0);
        const HVX_Vector v_s_minus_m1 =
          Q6_Vqf16_vsub_VhfVhf(scratch->row_buffer1[c / 64], v_dup_m1);
        HVX_Vector v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
        HVX_Vector v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);
        v_p_row0_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row0_hf, v_zero);
        v_p_row1_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row1_hf, v_zero);

        __fp16 *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
        HVX_Vector *pv_p_out0 = ((HVX_Vector *) out_dual_tile) + r1 / 2;
        const HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
        *pv_p_out0 = Q6_V_lo_W(vp_p_dual_row);
        if (c + HMX_FP16_TILE_N_COLS < n_cols) {
          pv_p_out0[16] = Q6_V_hi_W(vp_p_dual_row);
        }

        const HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0_hf);
        const HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1_hf);
        v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
          v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
        v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
          v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
      }

#pragma unroll
      for (int s = 64; s >= 4; s >>= 1) {
        v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
          v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
        v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
          v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
      }
      const HVX_Vector v_p_rowsum_pack2 =
        Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));
      const HVX_Vector v_p_rowsum_pack2_rot =
        Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
      const HVX_Vector v_p_rowsum_local_rot =
        Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
      v_p_rowsum_local =
        Q6_V_vlalign_VVR(v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r_vec_off * sizeof(__fp16));
    }

    scratch->rowmax[r_vec_idx] = v_s_rowmax_local;
    scratch->rowsum[r_vec_idx] = v_p_rowsum_local;
  }

  HVX_Vector *mvec_exp_m_diff = scratch->rowmax;
  for (int i = 0; i < n_row_vec_cnt; ++i) {
    const HVX_Vector v_m_prev = state->m[i];
    const HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, scratch->rowmax[i]);
    const HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);
    const HVX_Vector v_exp_m_diff_hf = hvx_my_exp2_vhf_vqf16(v_m_diff);
    HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(state->l[i], v_exp_m_diff_hf);
    v_l_curr = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, scratch->rowsum[i]);
    state->m[i] = v_m_curr;
    state->l[i] = v_l_curr;
    mvec_exp_m_diff[i] = v_exp_m_diff_hf;
  }

  const HVX_Vector v_offsets = vmem(d_tile_vscatter_offsets);
  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  for (int i = 0; i < n_row_tiles; ++i) {
    const HVX_Vector v_content = Q6_V_vror_VR(mvec_exp_m_diff[i / 2], (i % 2) * 64);
    __fp16 *out_base = scratch->diagonal + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
    Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1,
                       v_offsets, v_content);
  }

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_id);
  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < head_dim / 32; ++c) {
      __fp16 *d_tile_in = scratch->diagonal + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
      __fp16 *o_tile_in = state->o_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
      hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

      __fp16 *p_tile_in = scratch->probability +
        (r * n_tiles_per_scratch_row) * HMX_FP16_TILE_N_ELMS;
      for (int segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
        const block_sparse_halo_update_segment_t *segment = &segments[segment_idx];
        const __fp16 *v_tile_in = segment->packed_v +
          (size_t) c * (size_t) segment->v_tile_stride * HMX_FP16_TILE_N_ELMS;
        for (int k = 0; k < segment_plan.tile_counts[segment_idx]; ++k) {
          const int p_offset = (segment_plan.tile_offsets[segment_idx] + k) * HMX_FP16_TILE_N_ELMS;
          const int v_offset = k * HMX_FP16_TILE_N_ELMS;
          hmx_load_tiles_fp16(p_tile_in + p_offset, v_tile_in + v_offset, 1);
        }
      }

      __fp16 *o_tile_out = state->o_next + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(o_tile_out);
    }
  }
  hmx_unit_release();

  swap_ptr(&state->o_next, &state->o_prev);
  return 0;
}

static int block_sparse_halo_stable_normal_packed_segments(
  __fp16 *q, __fp16 *o, block_sparse_halo_update_scratch_t *scratch,
  const block_sparse_halo_update_segment_t *segments, int segment_count,
  int n_rows, int scratch_col_capacity, int head_dim,
  const uint8_t *hmx_output_scales_qk, const uint8_t *hmx_output_scales_id,
  const int16_t *d_tile_vscatter_offsets) {
  if (q == NULL || o == NULL || scratch == NULL || segments == NULL ||
      scratch->score == NULL || scratch->probability == NULL ||
      scratch->rowmax == NULL || scratch->rowsum == NULL ||
      scratch->row_buffer0 == NULL || scratch->row_buffer1 == NULL ||
      hmx_output_scales_qk == NULL || hmx_output_scales_id == NULL ||
      d_tile_vscatter_offsets == NULL ||
      n_rows <= 0 || n_rows > HMX_FP16_TILE_N_ROWS || segment_count <= 0 ||
      segment_count > BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS ||
      scratch_col_capacity <= 0 || scratch_col_capacity % HMX_FP16_TILE_N_COLS != 0 ||
      head_dim <= 0 || head_dim % 32 != 0) {
    return -1;
  }

  int segment_cols[BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS] = {0};
  for (int segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
    const block_sparse_halo_update_segment_t *segment = &segments[segment_idx];
    if (segment->packed_k == NULL || segment->packed_v == NULL || segment->n_real_cols <= 0 ||
        segment->v_tile_stride < ceil_div(segment->n_real_cols, HMX_FP16_TILE_N_COLS)) {
      return -1;
    }
    segment_cols[segment_idx] = segment->n_real_cols;
  }

  block_sparse_halo_update_segment_plan_t segment_plan;
  if (block_sparse_halo_plan_update_segments(
        &segment_plan, segment_cols, segment_count, (size_t) scratch_col_capacity) != 0) {
    return -1;
  }
  const int n_cols = segment_plan.total_cols;
  const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
  const int n_tiles_per_scratch_row = scratch_col_capacity / HMX_FP16_TILE_N_COLS;

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_qk);
  for (int segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
    const block_sparse_halo_update_segment_t *segment = &segments[segment_idx];
    for (int r = 0; r < n_row_tiles; ++r) {
      for (int c = 0; c < segment_plan.tile_counts[segment_idx]; ++c) {
        const __fp16 *row_tiles = q + r * HMX_FP16_TILE_N_ROWS * head_dim;
        const __fp16 *col_tiles = segment->packed_k + c * HMX_FP16_TILE_N_COLS * head_dim;
        __fp16 *out_tile = scratch->score +
          (r * n_tiles_per_scratch_row + segment_plan.tile_offsets[segment_idx] + c) *
          HMX_FP16_TILE_N_ELMS;
        hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
      }
    }
  }
  hmx_unit_release();

  const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
  const HVX_Vector v_zero = Q6_V_vzero();
  HVX_Vector v_s_rowmax_local = v_neg_inf;
  HVX_Vector v_p_rowsum_local = v_zero;
  for (int r = 0; r < align_up(n_rows, 2); r += 2) {
    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;
    __fp16 *s_ld_base = scratch->score + r0 * HMX_FP16_TILE_N_ROWS * scratch_col_capacity;
    __fp16 *p_st_base = scratch->probability + r0 * HMX_FP16_TILE_N_ROWS * scratch_col_capacity;

    HVX_Vector v_s_rowmax0 = v_neg_inf;
    HVX_Vector v_s_rowmax1 = v_neg_inf;
    for (int c = 0; c < n_cols; c += 64) {
      const __fp16 *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
      const HVX_Vector *pv_s_in0 = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
      const bool second_tile_valid = c + HMX_FP16_TILE_N_COLS < n_cols;
      const HVX_Vector v_s_in1 = second_tile_valid ? pv_s_in0[16] : Q6_V_vzero();
      const HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(v_s_in1, *pv_s_in0, -2);
      const int ne = smax(0, smin(n_cols - c, 64));
      const HVX_VectorPred q_mask_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
      const HVX_Vector v_s_row0 =
        Q6_V_vmux_QVV(q_mask_keep, Q6_V_lo_W(vp_s_dual_row), v_neg_inf);
      const HVX_Vector v_s_row1 =
        Q6_V_vmux_QVV(q_mask_keep, Q6_V_hi_W(vp_s_dual_row), v_neg_inf);
      scratch->row_buffer0[c / 64] = v_s_row0;
      scratch->row_buffer1[c / 64] = v_s_row1;
      v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
      v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
    }

#pragma unroll
    for (int s = 64; s >= 2; s >>= 1) {
      v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
      v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
    }

    // Keep rowmax in the same packed HVX form used by online Flash Attention.
    // The last lane of each reduction is packed into its row slot without a
    // half-to-float scalar helper call.
    const HVX_Vector v_s_rowmax_pack2 =
      Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));
    const HVX_Vector v_s_rowmax_pack2_rot =
      Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));
    const HVX_Vector v_s_rowmax_local_rot =
      Q6_V_vror_VR(v_s_rowmax_local, r * sizeof(__fp16));
    v_s_rowmax_local =
      Q6_V_vlalign_VVR(v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r * sizeof(__fp16));

    const HVX_Vector v_m_lut = Q6_V_vror_VR(v_s_rowmax_local, r * sizeof(__fp16));
    const HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
    const HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));
    HVX_Vector v_p_rowsum0 = v_zero;
    HVX_Vector v_p_rowsum1 = v_zero;
    for (int c = 0; c < n_cols; c += 64) {
      const int ne = smax(0, smin(n_cols - c, 64));
      const HVX_VectorPred q_mask_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
      const HVX_Vector v_s_minus_m0 =
        Q6_Vqf16_vsub_VhfVhf(scratch->row_buffer0[c / 64], v_dup_m0);
      const HVX_Vector v_s_minus_m1 =
        Q6_Vqf16_vsub_VhfVhf(scratch->row_buffer1[c / 64], v_dup_m1);
      HVX_Vector v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
      HVX_Vector v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);
      v_p_row0_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row0_hf, v_zero);
      v_p_row1_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row1_hf, v_zero);

      __fp16 *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
      HVX_Vector *pv_p_out0 = ((HVX_Vector *) out_dual_tile) + r1 / 2;
      const HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
      *pv_p_out0 = Q6_V_lo_W(vp_p_dual_row);
      if (c + HMX_FP16_TILE_N_COLS < n_cols) {
        pv_p_out0[16] = Q6_V_hi_W(vp_p_dual_row);
      }

      const HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0_hf);
      const HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1_hf);
      v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
        v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
      v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
        v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
    }

#pragma unroll
    for (int s = 64; s >= 4; s >>= 1) {
      v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
        v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
      v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
        v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
    }
    const HVX_Vector v_p_rowsum_pack2 =
      Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));
    const HVX_Vector v_p_rowsum_pack2_rot =
      Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
    const HVX_Vector v_p_rowsum_local_rot =
      Q6_V_vror_VR(v_p_rowsum_local, r * sizeof(__fp16));
    v_p_rowsum_local =
      Q6_V_vlalign_VVR(v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r * sizeof(__fp16));
  }

  scratch->rowmax[0] = v_s_rowmax_local;
  scratch->rowsum[0] = Q6_Vqf16_vadd_Vqf16Vhf(v_zero, v_p_rowsum_local);

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_id);
  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < head_dim / 32; ++c) {
      __fp16 *p_tile_in = scratch->probability +
        (r * n_tiles_per_scratch_row) * HMX_FP16_TILE_N_ELMS;
      for (int segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
        const block_sparse_halo_update_segment_t *segment = &segments[segment_idx];
        const __fp16 *v_tile_in = segment->packed_v +
          (size_t) c * (size_t) segment->v_tile_stride * HMX_FP16_TILE_N_ELMS;
        for (int k = 0; k < segment_plan.tile_counts[segment_idx]; ++k) {
          const int p_offset = (segment_plan.tile_offsets[segment_idx] + k) * HMX_FP16_TILE_N_ELMS;
          const int v_offset = k * HMX_FP16_TILE_N_ELMS;
          hmx_load_tiles_fp16(p_tile_in + p_offset, v_tile_in + v_offset, 1);
        }
      }
      __fp16 *o_tile_out = o + (c * n_row_tiles + r) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(o_tile_out);
    }
  }
  hmx_unit_release();

  // Q is dead after QK and score/probability is dead after PV. Reuse those
  // compact-plan aliases for the same FP16 diagonal normalization as Flash.
  __fp16 *normalization_diagonal = scratch->score;
  hvx_fill_uh(normalization_diagonal, 0, HMX_FP16_TILE_SIZE);
  const HVX_Vector v_inv_rowsum = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(scratch->rowsum[0]));
  const HVX_Vector v_offsets = vmem(d_tile_vscatter_offsets);
  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) normalization_diagonal,
                     HMX_FP16_TILE_SIZE - 1, v_offsets, v_inv_rowsum);

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_id);
  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < head_dim / 32; ++c) {
      const __fp16 *d_tile_in =
        normalization_diagonal + (r * n_row_tiles) * HMX_FP16_TILE_N_ELMS;
      const __fp16 *o_tile_in = o + (c * n_row_tiles) * HMX_FP16_TILE_N_ELMS;
      __fp16 *o_tile_out = q + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;
      hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
    }
  }
  hmx_unit_release();
  return 0;
}

static int block_sparse_halo_update_packed_segment(
  block_sparse_halo_q_state_t *state, block_sparse_halo_update_scratch_t *scratch,
  const __fp16 *packed_k, const __fp16 *packed_v, int v_tile_stride,
  int n_real_cols, int scratch_col_capacity, int head_dim,
  const uint8_t *hmx_output_scales_qk, const uint8_t *hmx_output_scales_id,
  const int16_t *d_tile_vscatter_offsets) {
  const block_sparse_halo_update_segment_t segment = {
    .packed_k = packed_k,
    .packed_v = packed_v,
    .v_tile_stride = v_tile_stride,
    .n_real_cols = n_real_cols,
  };
  return block_sparse_halo_update_packed_segments(
    state, scratch, &segment, 1, scratch_col_capacity, head_dim,
    hmx_output_scales_qk, hmx_output_scales_id, d_tile_vscatter_offsets);
}

int block_sparse_halo_pack_image_row(
  block_sparse_halo_packed_row_slot_t *slot, const __fp16 *K, const __fp16 *V,
  int n_kv_heads, int kv_head_idx, int head_dim, int image_block_cols, int row_id) {
  if (slot == NULL || slot->packed_k == NULL || slot->packed_v == NULL || K == NULL || V == NULL ||
      n_kv_heads <= 0 || kv_head_idx < 0 || kv_head_idx >= n_kv_heads || head_dim <= 0 ||
      image_block_cols <= 0 || row_id < 0 || image_block_cols > INT_MAX / HMX_FP16_TILE_N_COLS) {
    return -1;
  }
  const int row_tokens = image_block_cols * HMX_FP16_TILE_N_COLS;
  const int kv_stride = n_kv_heads * head_dim;
  const size_t first_token = (size_t) row_id * (size_t) row_tokens;
  const __fp16 *row_k = K + first_token * (size_t) kv_stride + (size_t) kv_head_idx * head_dim;
  const __fp16 *row_v = V + first_token * (size_t) kv_stride + (size_t) kv_head_idx * head_dim;

  slot->ready = false;
  block_sparse_pack_k_f16(slot->packed_k, row_k, kv_stride, row_tokens, head_dim);
  block_sparse_pack_v_f16(slot->packed_v, row_v, kv_stride, row_tokens, row_tokens, head_dim);
  slot->row_id = row_id;
  slot->ready = true;
  return 0;
}

static int block_sparse_halo_copy_packed_image_row(
  block_sparse_halo_packed_row_slot_t *slot, const __fp16 *K, const __fp16 *V,
  int kv_head_idx, int head_dim, int token_tiles, int image_block_cols, int row_id) {
  if (slot == NULL || slot->packed_k == NULL || slot->packed_v == NULL || K == NULL || V == NULL ||
      kv_head_idx < 0 || head_dim <= 0 || head_dim % HMX_FP16_TILE_N_COLS != 0 || token_tiles <= 0 ||
      image_block_cols <= 0 || row_id < 0 || row_id > (token_tiles - 1) / image_block_cols) {
    return -1;
  }
  const int token_tile = row_id * image_block_cols;
  if (token_tile > token_tiles - image_block_cols) {
    return -1;
  }
  size_t src_offset;
  if (block_sparse_packed_qk_tile_offset(
        &src_offset, kv_head_idx, token_tile, 0, token_tiles, head_dim) != 0) {
    return -1;
  }
  const size_t k_elems = (size_t) image_block_cols * (head_dim / HMX_FP16_TILE_N_COLS) *
                         HMX_FP16_TILE_N_ELMS;
  hvx_copy(slot->packed_k, K + src_offset, k_elems * sizeof(__fp16));
  for (int d_tile = 0; d_tile < head_dim / HMX_FP16_TILE_N_COLS; ++d_tile) {
    if (block_sparse_packed_v_tile_offset(
          &src_offset, kv_head_idx, token_tile, d_tile, token_tiles, head_dim) != 0) {
      return -1;
    }
    hvx_copy(slot->packed_v + (size_t) d_tile * image_block_cols * HMX_FP16_TILE_N_ELMS,
             V + src_offset,
             (size_t) image_block_cols * HMX_FP16_TILE_N_ELMS * sizeof(__fp16));
  }
  slot->row_id = row_id;
  slot->ready = true;
  return 0;
}

static int block_sparse_halo_prefetch_packed_image_row(
  const __fp16 *K, const __fp16 *V, int kv_head_idx, int head_dim,
  int token_tiles, int image_block_cols, int row_id) {
  if (K == NULL || V == NULL || kv_head_idx < 0 || head_dim <= 0 || token_tiles <= 0 ||
      image_block_cols <= 0 || row_id < 0) {
    return -1;
  }
  const int token_tile = row_id * image_block_cols;
  if (token_tile > token_tiles - image_block_cols) {
    return -1;
  }
  size_t src_offset;
  if (block_sparse_packed_qk_tile_offset(
        &src_offset, kv_head_idx, token_tile, 0, token_tiles, head_dim) != 0) {
    return -1;
  }
  const uint32_t k_tile_bytes = (uint32_t) head_dim * HMX_FP16_TILE_N_ROWS * sizeof(__fp16);
  l2fetch(K + src_offset, k_tile_bytes, k_tile_bytes, image_block_cols, 0);
  for (int d_tile = 0; d_tile < head_dim / HMX_FP16_TILE_N_COLS; ++d_tile) {
    if (block_sparse_packed_v_tile_offset(
          &src_offset, kv_head_idx, token_tile, d_tile, token_tiles, head_dim) != 0) {
      return -1;
    }
    l2fetch(V + src_offset, HMX_FP16_TILE_SIZE, HMX_FP16_TILE_SIZE, image_block_cols, 0);
  }
  return 0;
}

static int block_sparse_halo_copy_packed_text_kv(
  __fp16 *resident_text_k, __fp16 *resident_text_v,
  const __fp16 *K, const __fp16 *V, int kv_head_idx, int head_dim,
  int token_tiles, int image_tokens, int text_tokens) {
  if (resident_text_k == NULL || resident_text_v == NULL || K == NULL || V == NULL ||
      image_tokens < 0 || image_tokens % HMX_FP16_TILE_N_COLS != 0 || text_tokens < 0) {
    return -1;
  }
  const int start_tile = image_tokens / HMX_FP16_TILE_N_COLS;
  const int text_tiles = ceil_div(text_tokens, HMX_FP16_TILE_N_COLS);
  if (text_tiles == 0) {
    return 0;
  }
  if (start_tile < 0 || start_tile > token_tiles - text_tiles) {
    return -1;
  }
  size_t src_offset;
  if (block_sparse_packed_qk_tile_offset(
        &src_offset, kv_head_idx, start_tile, 0, token_tiles, head_dim) != 0) {
    return -1;
  }
  hvx_copy(resident_text_k, K + src_offset,
           (size_t) text_tiles * (head_dim / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_SIZE);
  for (int d_tile = 0; d_tile < head_dim / HMX_FP16_TILE_N_COLS; ++d_tile) {
    if (block_sparse_packed_v_tile_offset(
          &src_offset, kv_head_idx, start_tile, d_tile, token_tiles, head_dim) != 0) {
      return -1;
    }
    hvx_copy(resident_text_v + (size_t) d_tile * text_tiles * HMX_FP16_TILE_N_ELMS,
             V + src_offset, (size_t) text_tiles * HMX_FP16_TILE_SIZE);
  }
  return 0;
}

static int block_sparse_halo_pack_staged_row(
  block_sparse_halo_packed_row_slot_t *packed_slot,
  const block_sparse_halo_row_slot_t *staging_slot,
  int image_block_cols, int head_dim) {
  if (packed_slot == NULL || staging_slot == NULL || staging_slot->raw == NULL ||
      !staging_slot->ready || staging_slot->outstanding || staging_slot->row_id < 0 ||
      image_block_cols <= 0 || image_block_cols > INT_MAX / HMX_FP16_TILE_N_COLS ||
      head_dim <= 0) {
    return -1;
  }
  const size_t row_half_elems =
    (size_t) image_block_cols * HMX_FP16_TILE_N_COLS * (size_t) head_dim;
  const __fp16 *raw_k = (const __fp16 *) staging_slot->raw;
  const __fp16 *raw_v = raw_k + row_half_elems;
  const int row_id = staging_slot->row_id;
  if (block_sparse_halo_pack_image_row(
        packed_slot, raw_k, raw_v, 1, 0, head_dim, image_block_cols, 0) != 0) {
    return -1;
  }
  packed_slot->row_id = row_id;
  return 0;
}

static int block_sparse_halo_adopt_packed_staged_row(
  block_sparse_halo_packed_row_slot_t *packed_slot,
  block_sparse_halo_row_slot_t *staging_slot,
  int image_block_cols, int head_dim) {
  if (packed_slot == NULL || packed_slot->packed_k == NULL || packed_slot->packed_v == NULL ||
      staging_slot == NULL || staging_slot->raw == NULL || !staging_slot->ready ||
      staging_slot->outstanding || staging_slot->row_id < 0 || image_block_cols <= 0 ||
      head_dim <= 0) {
    return -1;
  }
  const size_t row_half_elems =
    (size_t) image_block_cols * HMX_FP16_TILE_N_COLS * (size_t) head_dim;
  if (packed_slot->packed_v != packed_slot->packed_k + row_half_elems) {
    return -1;
  }
  __fp16 *retired = packed_slot->packed_k;
  packed_slot->packed_k = (__fp16 *) staging_slot->raw;
  packed_slot->packed_v = packed_slot->packed_k + row_half_elems;
  packed_slot->row_id = staging_slot->row_id;
  packed_slot->ready = true;
  staging_slot->raw = (uint8_t *) retired;
  staging_slot->ready = false;
  return 0;
}

static int block_sparse_halo_update_resident_text_q_segments(
  __fp16 *resident_text_q, __fp16 *resident_text_o[2], uint8_t *resident_text_state,
  int resident_text_q_blocks, int text_tokens, size_t qo_tile_size, size_t col_vec_size,
  int *current_o_bank, block_sparse_halo_update_scratch_t *scratch,
  const block_sparse_halo_update_segment_t *segments, int segment_count,
  int scratch_col_capacity, int head_dim, const uint8_t *hmx_output_scales_qk,
  const uint8_t *hmx_output_scales_id, const int16_t *d_tile_vscatter_offsets) {
  if (resident_text_q == NULL || resident_text_o == NULL || resident_text_o[0] == NULL ||
      resident_text_o[1] == NULL || resident_text_state == NULL || current_o_bank == NULL ||
      scratch == NULL || scratch->diagonal == NULL ||
      (*current_o_bank != 0 && *current_o_bank != 1)) {
    return -1;
  }
  // Image-Q output aliases this tile. Reestablish a zero off-diagonal before
  // the text-Q online update scatters its new diagonal entries.
  hvx_fill_uh(scratch->diagonal, 0, HMX_FP16_TILE_SIZE);
  const int next_o_bank = *current_o_bank ^ 1;
  for (int text_q_block = 0; text_q_block < resident_text_q_blocks; ++text_q_block) {
    uint8_t *state_block = resident_text_state + (size_t) text_q_block * 2 * col_vec_size;
    block_sparse_halo_q_state_t q_state = {
      .q = (__fp16 *) ((uint8_t *) resident_text_q + (size_t) text_q_block * qo_tile_size),
      .o_prev = (__fp16 *) ((uint8_t *) resident_text_o[*current_o_bank] +
                            (size_t) text_q_block * qo_tile_size),
      .o_next = (__fp16 *) ((uint8_t *) resident_text_o[next_o_bank] +
                            (size_t) text_q_block * qo_tile_size),
      .m = (HVX_Vector *) state_block,
      .l = (HVX_Vector *) (state_block + col_vec_size),
      .n_rows = block_sparse_halo_text_q_rows(text_tokens, text_q_block),
    };
    if (q_state.n_rows <= 0 || block_sparse_halo_update_packed_segments(
          &q_state, scratch, segments, segment_count, scratch_col_capacity,
          head_dim, hmx_output_scales_qk, hmx_output_scales_id,
          d_tile_vscatter_offsets) != 0 ||
        q_state.o_prev != (__fp16 *) ((uint8_t *) resident_text_o[next_o_bank] +
                                      (size_t) text_q_block * qo_tile_size)) {
      return -1;
    }
  }
  const int completed_bank = block_sparse_halo_text_o_next_bank(
    *current_o_bank, resident_text_q_blocks, resident_text_q_blocks);
  if (completed_bank < 0) {
    return -1;
  }
  *current_o_bank = completed_bank;
  return 0;
}

static int block_sparse_halo_update_resident_text_q(
  __fp16 *resident_text_q, __fp16 *resident_text_o[2], uint8_t *resident_text_state,
  int resident_text_q_blocks, int text_tokens, size_t qo_tile_size, size_t col_vec_size,
  int *current_o_bank, block_sparse_halo_update_scratch_t *scratch,
  const __fp16 *packed_k, const __fp16 *packed_v, int v_tile_stride, int n_real_cols,
  int scratch_col_capacity, int head_dim, const uint8_t *hmx_output_scales_qk,
  const uint8_t *hmx_output_scales_id, const int16_t *d_tile_vscatter_offsets) {
  const block_sparse_halo_update_segment_t segment = {
    .packed_k = packed_k,
    .packed_v = packed_v,
    .v_tile_stride = v_tile_stride,
    .n_real_cols = n_real_cols,
  };
  return block_sparse_halo_update_resident_text_q_segments(
    resident_text_q, resident_text_o, resident_text_state,
    resident_text_q_blocks, text_tokens, qo_tile_size, col_vec_size,
    current_o_bank, scratch, &segment, 1, scratch_col_capacity, head_dim,
    hmx_output_scales_qk, hmx_output_scales_id, d_tile_vscatter_offsets);
}

static int block_sparse_halo_update_resident_text_q_from_packed_batch(
  const block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS],
  const block_sparse_halo_text_kv_batch_t *batch, int image_block_cols,
  __fp16 *resident_text_q, __fp16 *resident_text_o[2], uint8_t *resident_text_state,
  int resident_text_q_blocks, int text_tokens, size_t qo_tile_size, size_t col_vec_size,
  int *current_o_bank, block_sparse_halo_update_scratch_t *scratch,
  int scratch_col_capacity, int head_dim, const uint8_t *hmx_output_scales_qk,
  const uint8_t *hmx_output_scales_id, const int16_t *d_tile_vscatter_offsets) {
  if (slots == NULL || batch == NULL || batch->row_count <= 0 ||
      batch->row_count > BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS ||
      image_block_cols <= 0 || image_block_cols > INT_MAX / HMX_FP16_TILE_N_COLS ||
      current_o_bank == NULL || (*current_o_bank != 0 && *current_o_bank != 1)) {
    return -1;
  }
  const int row_cols = image_block_cols * HMX_FP16_TILE_N_COLS;
  if (row_cols > scratch_col_capacity / batch->row_count) {
    return -1;
  }

  block_sparse_halo_update_segment_t segments[BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS] = {0};
  bool used_slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS] = {false};
  for (int batch_row = 0; batch_row < batch->row_count; ++batch_row) {
    const int packed_slot = batch->packed_slots[batch_row];
    const int row_id = batch->row_ids[batch_row];
    if (packed_slot < 0 || packed_slot >= BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS ||
        used_slots[packed_slot] || row_id < 0 ||
        (batch_row > 0 && row_id != batch->row_ids[batch_row - 1] + 1) ||
        !slots[packed_slot].ready || slots[packed_slot].row_id != row_id) {
      return -1;
    }
    used_slots[packed_slot] = true;

    block_sparse_halo_packed_segment_t packed_segment;
    if (block_sparse_halo_get_packed_segment(
          &packed_segment, &slots[packed_slot], 0, image_block_cols,
          image_block_cols, head_dim) != 0) {
      return -1;
    }
    segments[batch_row] = (block_sparse_halo_update_segment_t) {
      .packed_k = packed_segment.packed_k,
      .packed_v = packed_segment.packed_v,
      .v_tile_stride = packed_segment.v_tile_stride,
      .n_real_cols = row_cols,
    };
  }
  if (batch->row_ids[batch->row_count - 1] >= INT_MAX ||
      batch->next_pending_row != batch->row_ids[batch->row_count - 1] + 1) {
    return -1;
  }

  const int expected_o_bank = *current_o_bank ^ 1;
  if (block_sparse_halo_update_resident_text_q_segments(
        resident_text_q, resident_text_o, resident_text_state,
        resident_text_q_blocks, text_tokens, qo_tile_size, col_vec_size,
        current_o_bank, scratch, segments, batch->row_count, scratch_col_capacity,
        head_dim, hmx_output_scales_qk, hmx_output_scales_id,
        d_tile_vscatter_offsets) != 0) {
    return -1;
  }
  return *current_o_bank == expected_o_bank ? 0 : -1;
}

static int block_sparse_halo_store_resident_text_q(
  block_sparse_halo_q_state_t *state, __fp16 *diagonal, float *raw_o, size_t raw_o_stride,
  int head_dim, bool with_lse, const uint8_t *hmx_output_scales_id,
  const int16_t *d_tile_vscatter_offsets) {
  if (state == NULL || diagonal == NULL || raw_o == NULL || state->n_rows <= 0 ||
      state->n_rows > HMX_FP16_TILE_N_ROWS || head_dim <= 0 || head_dim % 32 != 0) {
    return -1;
  }
  const HVX_Vector v_offsets = vmem(d_tile_vscatter_offsets);
  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  const HVX_Vector v_inverse_l = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(state->l[0]));
  Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) diagonal, HMX_FP16_TILE_SIZE - 1,
                     v_offsets, v_inverse_l);

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_id);
  for (int c = 0; c < head_dim / 32; ++c) {
    __fp16 *o_tile_in = state->o_prev + (size_t) c * HMX_FP16_TILE_N_ELMS;
    __fp16 *o_tile_out = state->o_next + (size_t) c * HMX_FP16_TILE_N_ELMS;
    hmx_dot_fp16(o_tile_out, diagonal, o_tile_in, 1);
  }
  hmx_unit_release();

  for (int r = 0; r < state->n_rows; r += 2) {
    const bool next_row_valid = r + 1 < state->n_rows;
    float *row_out0 = raw_o + (size_t) r * raw_o_stride;
    float *row_out1 = next_row_valid ? raw_o + (size_t) (r + 1) * raw_o_stride : row_out0;
    for (int d = 0; d < head_dim / 32; ++d) {
      const HVX_Vector *in_tile =
        (const HVX_Vector *) (state->o_next + (size_t) d * HMX_FP16_TILE_N_ELMS);
      const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r / 2]);
      const HVX_Vector v_out0 = Q6_V_lo_W(vp);
      const HVX_Vector v_out1 = Q6_V_hi_W(vp);
      vstu_variable(row_out0 + d * 32, sizeof(v_out0), v_out0);
      if (next_row_valid) {
        vstu_variable(row_out1 + d * 32, sizeof(v_out1), v_out1);
      }
    }
    if (with_lse) {
      row_out0[head_dim] = flash_attn_lse_from_f16_state(state->m, state->l, r);
      if (next_row_valid) {
        row_out1[head_dim] = flash_attn_lse_from_f16_state(state->m, state->l, r + 1);
      }
    }
  }
  return 0;
}

static int fused_anchor_build_probability(
  __fp16 *s_tile, __fp16 *p_tile, HVX_Vector *mvec_m,
  HVX_Vector *mvec_s_rowmax, HVX_Vector *mvec_p_rowsum,
  HVX_Vector *row_buffer0, HVX_Vector *row_buffer1,
  const __fp16 *dense_mask, size_t dense_mask_stride,
  const int32_t *block_mask, const struct FusedAnchorAttnParams *params,
  const uint8_t *keep_flags, int keep_flag_count,
  fused_anchor_fa_stats_t *stats,
  int head, int ir, int jc, int n_rows, int n_cols, int blk_sz_c, bool sparse) {
  const int n_row_vec_cnt = ceil_div(n_rows, 64);
  const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
  const HVX_Vector v_zero = Q6_V_vzero();
  const HVX_Vector v_mask_threshold = Q6_Vh_vsplat_R(0xcc00);
  const HVX_Vector v_keep = Q6_Vh_vsplat_R(0x3c00);
  const HVX_VectorPred q_first_half = Q6_Q_vsetq2_R(32 * sizeof(__fp16));

  for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
    HVX_Vector v_s_rowmax_local = v_neg_inf;
    HVX_Vector v_p_rowsum_local = v_zero;

    for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
      const int r = r_vec_idx * 64 + r_vec_off;
      if (r >= align_up(n_rows, 2)) {
        break;
      }
      const bool row1_valid = r + 1 < n_rows;
      const int r0 = r / HMX_FP16_TILE_N_ROWS;
      const int r1 = r % HMX_FP16_TILE_N_ROWS;
      __fp16 *s_ld_base = s_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;
      __fp16 *p_st_base = p_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;

      for (int c = 0; c < n_cols; c += 64) {
        const __fp16 *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
        const HVX_Vector *pv_s_in0 = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
        const HVX_Vector *pv_s_in1 = pv_s_in0 + 16;
        const HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
        row_buffer0[c / 64] = Q6_V_lo_W(vp_s_dual_row);
        row_buffer1[c / 64] = Q6_V_hi_W(vp_s_dual_row);
      }

      HVX_Vector v_s_rowmax0 = v_neg_inf;
      HVX_Vector v_s_rowmax1 = v_neg_inf;
      const int query0 = ir + r;
      const int query1 = ir + r + 1;
      for (int c = 0; c < n_cols; c += 64) {
        const int key = jc + c;
        const int ne = smin(n_cols - c, 64);
        const HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
        const HVX_Vector v_mask0 = vmemu(dense_mask + (size_t) query0 * dense_mask_stride + key);
        const HVX_Vector v_mask1 = row1_valid ?
          vmemu(dense_mask + (size_t) query1 * dense_mask_stride + key) : v_neg_inf;
        HVX_VectorPred q_keep0 = Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_mask_threshold);
        HVX_VectorPred q_keep1 = Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_mask_threshold);

        if (sparse) {
          const int key_block0 = key / params->block_k;
          const int key_block1 = key_block0 + 1;
          const bool second_block_valid = key + HMX_FP16_TILE_N_COLS < params->kv_len;
          int keep00;
          int keep01;
          int keep10;
          int keep11;
          if (keep_flags != NULL) {
            const int flag0 = (key - jc) / params->block_k;
            const int flag1 = flag0 + 1;
            if (flag0 < 0 || flag0 >= keep_flag_count ||
                (second_block_valid && (flag1 < 0 || flag1 >= keep_flag_count))) {
              return -1;
            }
            keep00 = keep_flags[flag0];
            keep01 = second_block_valid ? keep_flags[flag1] : 0;
            keep10 = row1_valid ? keep00 : 0;
            keep11 = row1_valid && second_block_valid ? keep01 : 0;
          } else {
            if (stats != NULL) {
              const uint64_t scans = 1u + (uint64_t) second_block_valid +
                                     (uint64_t) row1_valid +
                                     (uint64_t) (row1_valid && second_block_valid);
              stats->mask_row_scans += scans;
              stats->mask_entry_scans += scans * (uint64_t) params->max_kept_blocks;
            }
            keep00 = fused_anchor_block_keep(block_mask, params, head, query0, key_block0);
            keep01 = second_block_valid ?
              fused_anchor_block_keep(block_mask, params, head, query0, key_block1) : 0;
            keep10 = row1_valid ?
              fused_anchor_block_keep(block_mask, params, head, query1, key_block0) : 0;
            keep11 = row1_valid && second_block_valid ?
              fused_anchor_block_keep(block_mask, params, head, query1, key_block1) : 0;
          }
          if (keep00 < 0 || keep01 < 0 || keep10 < 0 || keep11 < 0) {
            return -1;
          }
          const HVX_Vector v_block0 = Q6_V_vmux_QVV(
            q_first_half, keep00 ? v_keep : v_neg_inf, keep01 ? v_keep : v_neg_inf);
          const HVX_Vector v_block1 = Q6_V_vmux_QVV(
            q_first_half, keep10 ? v_keep : v_neg_inf, keep11 ? v_keep : v_neg_inf);
          q_keep0 = Q6_Q_and_QQ(q_keep0, Q6_Q_vcmp_gt_VhfVhf(v_block0, v_mask_threshold));
          q_keep1 = Q6_Q_and_QQ(q_keep1, Q6_Q_vcmp_gt_VhfVhf(v_block1, v_mask_threshold));
        }

        q_keep0 = Q6_Q_and_QQ(q_keep0, q_tail_keep);
        q_keep1 = Q6_Q_and_QQ(q_keep1, q_tail_keep);
        const HVX_Vector v_s_row0 = Q6_V_vmux_QVV(q_keep0, row_buffer0[c / 64], v_neg_inf);
        const HVX_Vector v_s_row1 = Q6_V_vmux_QVV(q_keep1, row_buffer1[c / 64], v_neg_inf);
        row_buffer0[c / 64] = v_s_row0;
        row_buffer1[c / 64] = v_s_row1;
        v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
        v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
      }

#pragma unroll
      for (int shift = 64; shift >= 2; shift >>= 1) {
        v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, shift));
        v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, shift));
      }
      const HVX_Vector v_s_rowmax_pack2 = Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));
      const HVX_Vector v_s_rowmax_pack2_rot = Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));
      const HVX_Vector v_s_rowmax_local_rot = Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));
      v_s_rowmax_local = Q6_V_vlalign_VVR(
        v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r_vec_off * sizeof(__fp16));

      const HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(mvec_m[r_vec_idx], v_s_rowmax_local);
      const HVX_Vector v_m_lut = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));
      const HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(v_zero, v_m_lut, 0));
      const HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(v_zero, v_m_lut, 2));
      HVX_Vector v_p_rowsum0 = v_zero;
      HVX_Vector v_p_rowsum1 = v_zero;

      for (int c = 0; c < n_cols; c += 64) {
        if (sparse && keep_flags != NULL && keep_flags[c / params->block_k] == 0 &&
            (c / params->block_k + 1 >= keep_flag_count ||
             keep_flags[c / params->block_k + 1] == 0)) {
          continue;
        }
        const HVX_Vector v_score0 = row_buffer0[c / 64];
        const HVX_Vector v_score1 = row_buffer1[c / 64];
        const HVX_VectorPred q_score_keep0 = Q6_Q_vcmp_gt_VhfVhf(v_score0, v_neg_inf);
        const HVX_VectorPred q_score_keep1 = Q6_Q_vcmp_gt_VhfVhf(v_score1, v_neg_inf);
        HVX_Vector v_p_row0 = hvx_my_exp2_vhf_vqf16(Q6_Vqf16_vsub_VhfVhf(v_score0, v_dup_m0));
        HVX_Vector v_p_row1 = hvx_my_exp2_vhf_vqf16(Q6_Vqf16_vsub_VhfVhf(v_score1, v_dup_m1));
        v_p_row0 = Q6_V_vmux_QVV(q_score_keep0, v_p_row0, v_zero);
        v_p_row1 = Q6_V_vmux_QVV(q_score_keep1, v_p_row1, v_zero);

        __fp16 *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
        HVX_Vector *pv_p_out0 = ((HVX_Vector *) out_dual_tile) + r1 / 2;
        HVX_Vector *pv_p_out1 = pv_p_out0 + 16;
        const HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1, v_p_row0, -2);
        *pv_p_out0 = Q6_V_lo_W(vp_p_dual_row);
        *pv_p_out1 = Q6_V_hi_W(vp_p_dual_row);

        const HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0);
        const HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1);
        v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
          v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
        v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
          v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
      }

#pragma unroll
      for (int shift = 64; shift >= 4; shift >>= 1) {
        v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, shift));
        v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, shift));
      }
      const HVX_Vector v_p_rowsum_pack2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));
      const HVX_Vector v_p_rowsum_pack2_rot = Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
      const HVX_Vector v_p_rowsum_local_rot = Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
      v_p_rowsum_local = Q6_V_vlalign_VVR(
        v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r_vec_off * sizeof(__fp16));
    }

    mvec_s_rowmax[r_vec_idx] = v_s_rowmax_local;
    mvec_p_rowsum[r_vec_idx] = v_p_rowsum_local;
  }
  return 0;
}

static void fused_anchor_accumulate_probability(
  __fp16 *o_tile_prev, __fp16 *o_tile_curr, const __fp16 *p_tile, const __fp16 *v_tile,
  __fp16 *d_tile, HVX_Vector *mvec_m, HVX_Vector *mvec_l,
  HVX_Vector *mvec_s_rowmax, const HVX_Vector *mvec_p_rowsum,
  int n_row_tiles, int n_row_vec_cnt, int n_col_tiles,
  int n_tiles_per_blk_r, int n_tiles_per_blk_c, int head_dim,
  const fused_anchor_keep_plan_t *keep_plan,
  const uint8_t *hmx_output_scales_id, const int16_t *d_tile_vscatter_offsets) {
  if (keep_plan != NULL && keep_plan->selected_tiles == 0) {
    return;
  }
  for (int i = 0; i < n_row_vec_cnt; ++i) {
    const HVX_Vector v_m_prev = mvec_m[i];
    const HVX_VectorPred q_has_work = Q6_Q_vcmp_gt_VhfVhf(mvec_p_rowsum[i], Q6_V_vzero());
    const HVX_Vector v_m_candidate = Q6_Vhf_vmax_VhfVhf(v_m_prev, mvec_s_rowmax[i]);
    const HVX_Vector v_m_curr = Q6_V_vmux_QVV(q_has_work, v_m_candidate, v_m_prev);
    const HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);
    HVX_Vector v_exp_m_diff = hvx_my_exp2_vhf_vqf16(v_m_diff);
    v_exp_m_diff = Q6_V_vmux_QVV(q_has_work, v_exp_m_diff, Q6_Vh_vsplat_R(0x3c00));
    HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(mvec_l[i], v_exp_m_diff);
    v_l_curr = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, mvec_p_rowsum[i]);
    mvec_m[i] = v_m_curr;
    mvec_l[i] = v_l_curr;
    mvec_s_rowmax[i] = v_exp_m_diff;
  }

  const HVX_Vector v_offsets = vmem(d_tile_vscatter_offsets);
  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  for (int i = 0; i < n_row_tiles; ++i) {
    const HVX_Vector v_content = Q6_V_vror_VR(mvec_s_rowmax[i / 2], (i % 2) * 64);
    __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
    Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base,
                       HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
  }

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_id);
  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < head_dim / 32; ++c) {
      __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
      __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
      hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

      const __fp16 *p_tile_in = p_tile + (r * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;
      const __fp16 *v_tile_in = v_tile + (c * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;
      if (keep_plan == NULL) {
        for (int k = 0; k < n_col_tiles; k += 32) {
          const int offset = k * HMX_FP16_TILE_N_ELMS;
          const int n_tiles = smin(n_col_tiles - k, 32);
          hmx_load_tiles_fp16(p_tile_in + offset, v_tile_in + offset, n_tiles);
        }
      } else {
        for (int s = 0; s < keep_plan->segment_count; ++s) {
          const int local_tile = keep_plan->tile_offsets[s] - keep_plan->first_tile;
          const int offset = local_tile * HMX_FP16_TILE_N_ELMS;
          const int n_tiles = keep_plan->tile_counts[s];
          if (local_tile < 0 || local_tile + n_tiles > n_col_tiles || n_tiles <= 0) {
            continue;
          }
          hmx_load_tiles_fp16(p_tile_in + offset, v_tile_in + offset, n_tiles);
        }
      }

      __fp16 *o_tile_out = o_tile_curr + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(o_tile_out);
    }
  }
  hmx_unit_release();
}

static void fused_anchor_normalize_and_store(
  float *output, const __fp16 *o_tile_prev, __fp16 *o_tile_scaled,
  __fp16 *d_tile, const HVX_Vector *mvec_m, const HVX_Vector *mvec_l,
  int head, int ir, int n_rows, int n_row_tiles, int n_tiles_per_blk_r,
  int head_dim, int n_heads, const uint8_t *hmx_output_scales_id,
  const int16_t *d_tile_vscatter_offsets) {
  const HVX_Vector v_offsets = vmem(d_tile_vscatter_offsets);
  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  HVX_Vector v_content;
  for (int i = 0; i < n_row_tiles; ++i) {
    HVX_Vector v_l_content = Q6_Vhf_equals_Vqf16(mvec_l[i / 2]);
    if ((i % 2) != 0) {
      v_l_content = Q6_V_vror_VR(v_l_content, 64);
    }
    const HVX_VectorPred q_l_valid = Q6_Q_vcmp_gt_VhfVhf(v_l_content, Q6_V_vzero());
    const HVX_Vector v_inv_l = Q6_V_vmux_QVV(
      q_l_valid, hvx_my_inv_vhf(v_l_content), Q6_V_vzero());
    if ((i % 2) == 0) {
      v_content = v_inv_l;
    } else {
      v_content = Q6_V_vror_VR(v_content, 64);
    }
    __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
    Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base,
                       HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
  }

  hmx_unit_acquire();
  hmx_set_output_scales(hmx_output_scales_id);
  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < head_dim / 32; ++c) {
      __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
      const __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
      __fp16 *o_tile_out = o_tile_scaled + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;
      hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
    }
  }
  hmx_unit_release();

  const size_t output_stride = (size_t) n_heads * (head_dim + 1);
  float *output_base = output + (size_t) ir * output_stride + (size_t) head * (head_dim + 1);
  for (int r = 0; r < n_rows; r += 2) {
    const bool row1_valid = r + 1 < n_rows;
    float *row_out0 = output_base + (size_t) r * output_stride;
    float *row_out1 = output_base + (size_t) (r + 1) * output_stride;
    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;
    const __fp16 *in_base = o_tile_scaled + r0 * HMX_FP16_TILE_N_ROWS * head_dim;

#pragma unroll
    for (int d = 0; d < head_dim / 32; ++d) {
      const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);
      const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);
      vstu_variable(row_out0 + d * 32, sizeof(HVX_Vector), Q6_V_lo_W(vp));
      if (row1_valid) {
        vstu_variable(row_out1 + d * 32, sizeof(HVX_Vector), Q6_V_hi_W(vp));
      }
    }

    row_out0[head_dim] = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r);
    if (row1_valid) {
      row_out1[head_dim] = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 1);
    }
  }
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  unsigned int error_count;
  int n_tasks;
  uint8_t *vtcm_base;
  size_t vtcm_size_per_thread;
  float *output;
  const float *q;
  const __fp16 *k;
  const __fp16 *v;
  const __fp16 *dense_mask;
  const int32_t *block_mask;
  const struct FusedAnchorAttnParams *params;
  fused_anchor_fa_stats_t *stats;
} fused_anchor_fa_task_state_t;

static int fused_anchor_flash_attn_head(
  int head, uint8_t *vtcm, uint8_t *vtcm_limit, float *output, const float *q,
  const __fp16 *k, const __fp16 *v, const __fp16 *dense_mask,
  const int32_t *block_mask, const struct FusedAnchorAttnParams *params,
  fused_anchor_fa_stats_t *stats) {
  fused_anchor_fa_plan_t plan;
  if (fused_anchor_fa_find_chunk_size(
        &plan, params->head_dim, params->qo_len, params->kv_len,
        (size_t) (vtcm_limit - vtcm)) != 0 || plan.blk_sz_r != HMX_FP16_TILE_N_ROWS) {
    return -1;
  }

  const int D = params->head_dim;
  const int blk_sz_r = (int) plan.blk_sz_r;
  const int blk_sz_c = (int) plan.blk_sz_c;
  const int n_tiles_per_blk_r = blk_sz_r / HMX_FP16_TILE_N_ROWS;
  const int n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;
  const size_t qo_tile_size = align_up((size_t) blk_sz_r * D * sizeof(__fp16), 4096);
  const size_t kv_tile_size = align_up((size_t) blk_sz_c * D * sizeof(__fp16), 4096);
  const size_t core_tile_size = align_up((size_t) blk_sz_r * blk_sz_c * sizeof(__fp16), 4096);
  const size_t d_tile_size = align_up((size_t) blk_sz_r * blk_sz_r * sizeof(__fp16), 4096);
  const size_t col_vec_size = align_up((size_t) blk_sz_r * sizeof(__fp16), 256);
  const size_t row_vec_size = align_up((size_t) blk_sz_c * sizeof(__fp16), 256);
  const size_t keep_flags_size = align_up((size_t) ceil_div(blk_sz_c, HMX_FP16_TILE_N_COLS), 256);

  uint8_t *vtcm_cur = vtcm;
  __fp16 *q_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16 *dense_o = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16 *sparse_o = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16 *shared_o = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16 *k_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  __fp16 *v_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  __fp16 *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *p_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *d_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_size);
  HVX_Vector *dense_m = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *dense_l = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *sparse_m = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *sparse_l = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *rowmax = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *row_buffer0 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  HVX_Vector *row_buffer1 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  uint8_t *keep_flags = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, keep_flags_size);
  uint8_t *hmx_output_scales_id = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  uint8_t *hmx_output_scales_qk = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  (void) keep_flags;
  if (vtcm_cur > vtcm_limit) {
    return -1;
  }

  const float qk_scale = 1.0f / sqrtf((float) D) * 1.44269504f;
  __fp16 qk_scale_hf = (__fp16) qk_scale;
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));
  static int32_t transpose_vscatter_indices_base[32] __vec_aligned__;
  for (int i = 0; i < 32; ++i) {
    transpose_vscatter_indices_base[i] = i * 128;
  }
  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }

  const size_t q_stride = (size_t) params->n_heads * D;
  const size_t kv_stride = (size_t) params->n_kv_heads * D;
  const size_t output_stride = (size_t) params->n_heads * (D + 1);
  const size_t slab_elements = (size_t) params->qo_len * output_stride;
  const size_t dense_mask_stride = align_up(params->kv_len, 64);

  for (int ir = 0; ir < params->qo_len; ir += blk_sz_r) {
    const int n_rows = smin(params->qo_len - ir, blk_sz_r);
    const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
    const int n_row_vec_cnt = ceil_div(n_rows, 64);
    if (stats != NULL) {
      ++stats->q_blocks_decoded;
    }

    block_sparse_halo_row_t keep_row = {};
    const int q_blocks_per_head = ceil_div(params->qo_len, params->block_q);
    const int image_blocks = params->image_block_rows * params->image_block_cols;
    const int q_block = head * q_blocks_per_head + ir / params->block_q;
    const int32_t *row_data = block_mask + (size_t) q_block * (size_t) params->max_kept_blocks;
    if (block_sparse_halo_parse_row(&keep_row, row_data, params->max_kept_blocks,
                                    q_block, q_blocks_per_head,
                                    ceil_div(params->kv_len, params->block_k), image_blocks) != 0) {
      return -1;
    }
    if (stats != NULL) {
      ++stats->mask_row_scans;
      stats->mask_entry_scans += (uint64_t) params->max_kept_blocks;
      if (keep_row.is_text_query) {
        ++stats->text_full_keep_blocks;
      }
    }

    const size_t q_stride_bytes = q_stride * sizeof(float);
    const uint8_t *q_base = (const uint8_t *) q +
                            (size_t) ir * q_stride_bytes + (size_t) head * D * sizeof(float);
    l2fetch(q_base, q_stride_bytes, (size_t) D * sizeof(float), n_rows, 1);
    for (int r = 0; r < n_rows; r += 2) {
      const bool row1_valid = r + 1 < n_rows;
      const HVX_Vector *pv_in0 = (const HVX_Vector *) (q_base + (size_t) r * q_stride_bytes);
      const HVX_Vector *pv_in1 = (const HVX_Vector *) (q_base + (size_t) (r + 1) * q_stride_bytes);
      const int r0 = r / HMX_FP16_TILE_N_ROWS;
      const int r1 = r % HMX_FP16_TILE_N_ROWS;
      __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * D;
#pragma unroll
      for (int d = 0; d < D / 32; ++d) {
        const HVX_Vector v0 = *pv_in0++;
        const HVX_Vector v1 = row1_valid ? *pv_in1++ : Q6_V_vzero();
        HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
        out_tile[r1 / 2] = hvx_my_wsf_to_vhf(v1, v0);
      }
    }

    hvx_fill_uh(dense_m, 0xfbff, col_vec_size);
    hvx_fill_uh(dense_l, 0, col_vec_size);
    hvx_fill_uh(sparse_m, 0xfbff, col_vec_size);
    hvx_fill_uh(sparse_l, 0, col_vec_size);
    hvx_fill_uh(dense_o, 0, qo_tile_size);
    hvx_fill_uh(sparse_o, 0, qo_tile_size);
    hvx_fill_uh(shared_o, 0, qo_tile_size);
    hvx_fill_uh(d_tile, 0, d_tile_size);
    __fp16 *dense_o_prev = dense_o;
    __fp16 *sparse_o_prev = sparse_o;
    __fp16 *o_scratch = shared_o;

    for (int jc = 0; jc < params->kv_len; jc += blk_sz_c) {
      const int n_cols = smin(params->kv_len - jc, blk_sz_c);
      const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
      if (stats != NULL) {
        ++stats->k_chunks;
        stats->qk_tiles += (uint64_t) n_row_tiles * (uint64_t) n_col_tiles;
      }
      fused_anchor_keep_plan_t keep_plan = {};
      if (fused_anchor_prepare_keep_plan(&keep_plan, keep_flags,
                                         (int) keep_flags_size, &keep_row,
                                         jc, n_cols, params->kv_len, params->block_k) != 0) {
        return -1;
      }
      const fused_anchor_keep_plan_t *sparse_keep_plan = &keep_plan;

      const __fp16 *k_base = k + (size_t) jc * kv_stride + (size_t) head * D;
      l2fetch(k_base, kv_stride * sizeof(__fp16), (size_t) D * sizeof(__fp16), n_cols, 1);
      const HVX_Vector v_step = Q6_V_vsplat_R(4);
      const HVX_Vector v_offsets_base = vmem(transpose_vscatter_indices_base);
      for (int r0 = 0; r0 < n_col_tiles; ++r0) {
        __fp16 *out_base = k_tile + r0 * HMX_FP16_TILE_N_COLS * D;
        HVX_Vector v_offsets = v_offsets_base;
        for (int r1 = 0; r1 < HMX_FP16_TILE_N_COLS; ++r1) {
          const int r = r0 * HMX_FP16_TILE_N_COLS + r1;
          if (r >= n_cols) {
            break;
          }
          const HVX_Vector *pv_in = (const HVX_Vector *) (k_base + (size_t) r * kv_stride);
#pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            __fp16 *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            Q6_vscatter_RMVwV((size_t) out_dual_tile,
                              HMX_FP16_TILE_SIZE * 2 - 1, v_offsets, *pv_in++);
          }
          v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
        }
      }

      const __fp16 *v_base = v + (size_t) jc * kv_stride + (size_t) head * D;
      l2fetch(v_base, kv_stride * sizeof(__fp16), (size_t) D * sizeof(__fp16), n_cols, 0);

      hmx_unit_acquire();
      hmx_set_output_scales(hmx_output_scales_qk);
      for (int r = 0; r < n_row_tiles; ++r) {
        for (int c = 0; c < n_col_tiles; ++c) {
          const __fp16 *row_tiles = q_tile + r * HMX_FP16_TILE_N_ROWS * D;
          const __fp16 *col_tiles = k_tile + c * HMX_FP16_TILE_N_COLS * D;
          __fp16 *out_tile = s_tile + (r * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS;
          hmx_dot_fp16(out_tile, row_tiles, col_tiles, D / 32);
        }
      }
      hmx_unit_release();

      if (fused_anchor_build_probability(
            s_tile, p_tile, dense_m, rowmax, rowsum, row_buffer0, row_buffer1,
            dense_mask, dense_mask_stride, block_mask, params,
            NULL, 0, stats, head, ir, jc, n_rows, n_cols, blk_sz_c, false) != 0) {
        return -1;
      }

      for (int r = 0; r < n_cols; r += 2) {
        const bool row1_valid = r + 1 < n_cols;
        const HVX_Vector *pv_in0 = (const HVX_Vector *) (v_base + (size_t) r * kv_stride);
        const HVX_Vector *pv_in1 = (const HVX_Vector *) (v_base + (size_t) (r + 1) * kv_stride);
#pragma unroll
        for (int c = 0; c < D; c += 64) {
          const HVX_Vector v0 = *pv_in0++;
          const HVX_Vector v1 = row1_valid ? *pv_in1++ : Q6_V_vzero();
          const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
          const int r0 = r / HMX_FP16_TILE_N_ROWS;
          const int r1 = r % HMX_FP16_TILE_N_ROWS;
          const int c0 = c / HMX_FP16_TILE_N_COLS;
          const int tile_idx0 = (c0 + 0) * n_tiles_per_blk_c + r0;
          const int tile_idx1 = (c0 + 1) * n_tiles_per_blk_c + r0;
          HVX_Vector *pv_out0 = ((HVX_Vector *) (v_tile + tile_idx0 * HMX_FP16_TILE_N_ELMS)) + r1 / 2;
          HVX_Vector *pv_out1 = ((HVX_Vector *) (v_tile + tile_idx1 * HMX_FP16_TILE_N_ELMS)) + r1 / 2;
          *pv_out0 = Q6_V_lo_W(vp);
          *pv_out1 = Q6_V_hi_W(vp);
        }
      }

      fused_anchor_accumulate_probability(
        dense_o_prev, o_scratch, p_tile, v_tile, d_tile,
        dense_m, dense_l, rowmax, rowsum, n_row_tiles, n_row_vec_cnt,
        n_col_tiles, n_tiles_per_blk_r, n_tiles_per_blk_c, D,
        NULL,
        hmx_output_scales_id, d_tile_vscatter_offsets);
      if (stats != NULL) {
        stats->dense_pv_tiles += (uint64_t) n_row_tiles *
                                 (uint64_t) (D / HMX_FP16_TILE_N_COLS) *
                                 (uint64_t) n_col_tiles;
      }
      __fp16 *old_dense = dense_o_prev;
      dense_o_prev = o_scratch;
      o_scratch = old_dense;

      if (keep_row.is_text_query) {
        sparse_o_prev = dense_o_prev;
        hvx_copy(sparse_m, dense_m, col_vec_size);
        hvx_copy(sparse_l, dense_l, col_vec_size);
        continue;
      }

      if (fused_anchor_build_probability(
            s_tile, p_tile, sparse_m, rowmax, rowsum, row_buffer0, row_buffer1,
            dense_mask, dense_mask_stride, block_mask, params,
            keep_flags, keep_plan.tile_count,
            stats, head, ir, jc, n_rows, n_cols, blk_sz_c, true) != 0) {
        return -1;
      }

      if (stats != NULL && sparse_keep_plan != NULL) {
        stats->sparse_selected_tiles += (uint64_t) n_row_tiles *
                                        (uint64_t) sparse_keep_plan->selected_tiles;
        stats->sparse_skipped_tiles += (uint64_t) n_row_tiles *
                                       (uint64_t) sparse_keep_plan->skipped_tiles;
        if (sparse_keep_plan->selected_tiles == 0) {
          ++stats->sparse_empty_chunks;
        }
      }

      fused_anchor_accumulate_probability(
        sparse_o_prev, o_scratch, p_tile, v_tile, d_tile,
        sparse_m, sparse_l, rowmax, rowsum, n_row_tiles, n_row_vec_cnt,
        n_col_tiles, n_tiles_per_blk_r, n_tiles_per_blk_c, D,
        sparse_keep_plan,
        hmx_output_scales_id, d_tile_vscatter_offsets);
      __fp16 *old_sparse = sparse_o_prev;
      sparse_o_prev = o_scratch;
      o_scratch = old_sparse;
    }

    fused_anchor_normalize_and_store(
      output, dense_o_prev, o_scratch, d_tile, dense_m, dense_l,
      head, ir, n_rows, n_row_tiles, n_tiles_per_blk_r, D, params->n_heads,
      hmx_output_scales_id, d_tile_vscatter_offsets);
    fused_anchor_normalize_and_store(
      output + slab_elements, sparse_o_prev, o_scratch, d_tile, sparse_m, sparse_l,
      head, ir, n_rows, n_row_tiles, n_tiles_per_blk_r, D, params->n_heads,
      hmx_output_scales_id, d_tile_vscatter_offsets);
  }
  return 0;
}

static void fused_anchor_flash_attn_worker(void *data, int worker_index) {
  fused_anchor_fa_task_state_t *state = (fused_anchor_fa_task_state_t *) data;
  uint8_t *vtcm = state->vtcm_base + (size_t) worker_index * state->vtcm_size_per_thread;
  uint8_t *vtcm_limit = vtcm + state->vtcm_size_per_thread;
  hmx_manager_enable_execution();
  while (1) {
    const unsigned int task_id = worker_pool_atomic_inc_return(&state->task_id) - 1;
    if (task_id >= (unsigned int) state->n_tasks) {
      break;
    }
    if (fused_anchor_flash_attn_head(
          (int) task_id, vtcm, vtcm_limit, state->output, state->q, state->k, state->v,
          state->dense_mask, state->block_mask, state->params,
          state->stats != NULL ? &state->stats[worker_index] : NULL) != 0) {
      worker_pool_atomic_inc_return(&state->error_count);
    }
  }
  hmx_manager_disable_execution();
  worker_pool_synctoken_jobdone(&state->sync_ctx);
}

int fused_anchor_flash_attn(float *restrict output, const float *restrict q,
                            const __fp16 *restrict k, const __fp16 *restrict v,
                            const __fp16 *restrict dense_mask,
                            const int32_t *restrict block_mask,
                            const struct FusedAnchorAttnParams *params) {
  if (output == NULL || q == NULL || k == NULL || v == NULL || dense_mask == NULL ||
      block_mask == NULL || params == NULL ||
      block_sparse_halo_validate(
        block_mask, params->qo_len, params->kv_len, params->n_heads, params->n_kv_heads,
        params->head_dim, params->block_q, params->block_k, params->n_q_blocks,
        params->n_k_blocks, params->mask_format, params->max_kept_blocks,
        params->sparse_policy, params->image_tokens, params->text_tokens,
        params->image_block_rows, params->image_block_cols) != 0) {
    return -1;
  }

  const int n_jobs = num_hvx128_contexts;
  const unsigned int vtcm_slice_count = num_workers;
  fused_anchor_fa_stats_t worker_stats[64] = {};
  uint8_t *vtcm_base = (uint8_t *) vtcm_manager_get_vtcm_base();
  const size_t acquired_vtcm_size = vtcm_manager_get_vtcm_size();
  if (n_jobs <= 0 || vtcm_slice_count == 0 || vtcm_slice_count > 64 || vtcm_base == NULL ||
      acquired_vtcm_size < (size_t) vtcm_slice_count * 1024u * 1024u) {
    return -1;
  }

  fused_anchor_fa_task_state_t state = {
    .task_id = 0,
    .error_count = 0,
    .n_tasks = params->n_heads,
    .vtcm_base = vtcm_base,
    .vtcm_size_per_thread = 1024u * 1024u,
    .output = output,
    .q = q,
    .k = k,
    .v = v,
    .dense_mask = dense_mask,
    .block_mask = block_mask,
    .params = params,
    .stats = worker_stats,
  };
  worker_pool_job_t job = { .fptr = fused_anchor_flash_attn_worker, .dptr = &state };
  worker_pool_synctoken_init(&state.sync_ctx, n_jobs);
  for (int i = 0; i < n_jobs; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&state.sync_ctx);
  fused_anchor_fa_stats_t total_stats = {};
  for (unsigned int worker = 0; worker < vtcm_slice_count; ++worker) {
    const uint64_t *src = (const uint64_t *) &worker_stats[worker];
    uint64_t *dst = (uint64_t *) &total_stats;
    for (size_t i = 0; i < sizeof(total_stats) / sizeof(uint64_t); ++i) {
      dst[i] += src[i];
    }
  }
#ifdef HTP_FUSED_ANCHOR_PROFILE
  FARF(ALWAYS,
       "HTP_FUSED_ANCHOR_STATS q_blocks_decoded=%llu mask_row_scans=%llu mask_entry_scans=%llu "
       "k_chunks=%llu qk_tiles=%llu dense_pv_tiles=%llu sparse_selected_tiles=%llu "
       "sparse_skipped_tiles=%llu sparse_empty_chunks=%llu text_full_keep_blocks=%llu",
       (unsigned long long) total_stats.q_blocks_decoded,
       (unsigned long long) total_stats.mask_row_scans,
       (unsigned long long) total_stats.mask_entry_scans,
       (unsigned long long) total_stats.k_chunks,
       (unsigned long long) total_stats.qk_tiles,
       (unsigned long long) total_stats.dense_pv_tiles,
       (unsigned long long) total_stats.sparse_selected_tiles,
       (unsigned long long) total_stats.sparse_skipped_tiles,
       (unsigned long long) total_stats.sparse_empty_chunks,
       (unsigned long long) total_stats.text_full_keep_blocks);
#endif
  return state.error_count == 0 ? 0 : -1;
}

int block_sparse_flash_attn_f16_core_chunked(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit,
                                             __fp16 *restrict O, const __fp16 *restrict Q,
                                             const __fp16 *restrict K, const __fp16 *restrict V,
                                             const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
                                             int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                                             int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
                                             int worker_index, bool input_packed) {
  (void) worker_index;

  // 一个 KV head 对应 G 个 Q heads
  // 当前函数只处理 kv_head_idx 对应的 K/V head
  // 以及 Q/O 中 [kv_head_idx * G, (kv_head_idx + 1) * G) 这 G 个 heads
  const int G = n_heads / n_kv_heads;
  const int D = head_dim;
  const int q_head_base = kv_head_idx * G;
  const int k_blocks_per_head = n_k_blocks / n_kv_heads;
  const bool use_host_index =
    mask_format == 1 && max_kept_blocks > 0 && G == 1 && block_q == HMX_FP16_TILE_N_ROWS &&
    block_k == HMX_FP16_TILE_N_COLS;
  if (input_packed && (G != 1 || !htp_packed_shape_supported(n_heads, n_kv_heads, head_dim, 1) ||
                       block_q != HMX_FP16_TILE_N_ROWS || block_k != HMX_FP16_TILE_N_COLS ||
                       qo_len != kv_len)) {
    return -1;
  }

  // 当前实现认为 Q/O 外部存储是 fp32
  // 虽然函数参数类型是 __fp16*，
  // 但实际通过 uint8_t* + qo_element_size 访问。
  // qo_fp32_element = true 时：
  //   Q/O 外部按 float 存储；
  //   读入 VTCM 后转成 fp16 HMX tile；
  //   写回 O 时再从 fp16 转回 float
  const bool   qo_fp32_element = true;
  const size_t qo_element_size = qo_fp32_element ? sizeof(float) : sizeof(__fp16);

  // 是否用 fp32 路径算 exp
  // 当前 false，表示使用 fp16/qf16 exp2 路径
  const bool use_fp32_exp = false;

  block_sparse_fa_plan_t plan;
  if (block_sparse_fa_select_plan(&plan, false, G, head_dim, kv_len, vtcm_limit - vtcm) != 0) {
    if (input_packed) {
      FARF(ERROR, "packed chunk planner failed: kv_head=%d kv_len=%d available=%u", kv_head_idx, kv_len,
           (unsigned int) (vtcm_limit - vtcm));
    }
    return -1;
  }
  const size_t blk_sz_r = HMX_FP16_TILE_N_ROWS;
  const size_t blk_sz_c = plan.blk_sz_c;
  assert(blk_sz_c % 64 == 0);

  // 当前 Q block 展开 G 个 Q heads 后的行数是 G * Br
  // 为适配 HMX FP16 tile 的行数要求，按 HMX_FP16_TILE_N_ROWS 对齐
  // g_br 表示对齐后的 Br'
  const size_t g_br = align_up(G * blk_sz_r, HMX_FP16_TILE_N_ROWS);

  // 当前 Q block 在行方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_r = g_br / HMX_FP16_TILE_N_ROWS;

  // 当前 K/V block 在列方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;

  // q_tile / o_tile:
  //   逻辑形状 [Br', D]
  //
  // k_tile / v_tile:
  //   逻辑形状 [Bc, D]
  //
  // s_tile / p_tile:
  //   逻辑形状 [Br', Bc]
  //   s_tile 保存 QK^T score；
  //   p_tile 保存 softmax numerator P
  //
  // d_tile:
  //   逻辑形状 [Br', Br']
  //   保存对角矩阵，用于：
  //     1. alpha * O_old
  //     2. final O / l
  //
  // col_vec_size:
  //   保存每一行的 m/l/rowmax/rowsum
  //
  // row_vec_size:
  //   softmax 时把 HMX tile layout 中的 S/P 解码成连续 row buffer
  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);
  const size_t kv_tile_size   = align_up(blk_sz_c * head_dim * sizeof(__fp16), 4096);
  const size_t core_tile_size = align_up(g_br * blk_sz_c * sizeof(__fp16), 4096);
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);
  const size_t row_vec_size   = align_up(blk_sz_c * sizeof(__fp16), 256);

  // 原始 K/V 布局：
  //   K/V: [kv_len, n_kv_heads, head_dim]
  //
  // kv_ld_blk_sz:
  //   当前 kv_head_idx 对应的一个 head 的大小
  //
  // kv_ld_stride:
  //   K/V 中一个 token 的完整 stride，即所有 KV heads 的大小
  const size_t kv_ld_blk_sz   = head_dim;
  const size_t kv_ld_stride   = n_kv_heads * head_dim;

  // 原始 Q/O 布局：
  //   Q: [qo_len, n_heads, head_dim]
  //   O: [qo_len, n_heads, head_dim + (with_lse ? 1 : 0)]
  //
  // q_ld_blk_sz/o_st_blk_sz:
  //   当前 kv_head_idx 对应的 G 个 Q heads 的总大小
  //
  // q_ld_stride/o_st_stride:
  //   Q/O 中一个 token 的完整 stride，即所有 Q heads 的大小
  const size_t q_ld_blk_sz = G * head_dim;
  const size_t q_ld_stride = n_heads * head_dim;
  const size_t o_head_dim  = head_dim + (with_lse ? 1 : 0);
  const size_t o_st_blk_sz = G * o_head_dim;
  const size_t o_st_stride = n_heads * o_head_dim;

  // 从 vtcm 开始顺序分配所有临时 buffer
  uint8_t *vtcm_cur = vtcm;
  __fp16  *q_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile0  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile1  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);

  __fp16 *k_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  __fp16 *v_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);

  __fp16 *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *p_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  // D tile 保存对角矩阵
  // 用于 HMX 计算 diag(scale) @ O
  __fp16 *d_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_size);

  // online softmax 状态：
  //
  // mvec_m:
  //   m_i，当前已经处理过的 K/V blocks 的 row max
  //
  // mvec_l:
  //   l_i，softmax denominator
  //
  // mvec_s_rowmax:
  //   当前 K/V block 中每一行 S 的 rowmax
  //   后面会复用为 exp(m_old - m_new)
  //
  // mvec_p_rowsum:
  //   当前 K/V block 中每一行 P 的 rowsum
  HVX_Vector *mvec_m        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_s_rowmax = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  // softmax 时一次处理两行 S
  // 从 HMX tile layout 解码出来后，放到这两个 row buffer 中
  HVX_Vector *row_buffer0 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  HVX_Vector *row_buffer1 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);

  // HMX output scale buffer
  //
  // hmx_output_scales_id:
  //   scale = 1，用于普通 HMX matmul 输出
  //
  // hmx_output_scales_qk:
  //   scale = log2(e) / sqrt(D)，用于 QK^T
  uint8_t *hmx_output_scales_id = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  uint8_t *hmx_output_scales_qk = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  assert(vtcm_cur <= vtcm_limit);

  // qk_scale 用于 attention score：
  //
  //   QK^T / sqrt(D)
  //
  // 但后面 softmax 用的是 exp2，而不是 exp
  // 因此这里提前乘上 log2(e)
  float  qk_scale    = 1.0f / sqrtf(head_dim) * 1.44269504f;
  __fp16 qk_scale_hf = (__fp16) qk_scale;

  // 初始化 HMX 输出 scale
  // 0x3c00 是 fp16 1.0
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));

  // QK^T 输出时直接乘 qk_scale_hf
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));

  // transpose_vscatter_indices_base 用于 K tile 转置/重排
  // vscatter 会把连续读入的 K row 写到 HMX RHS operand 需要的位置
  static int32_t transpose_vscatter_indices_base[32] __vec_aligned__;
  for (int i = 0; i < 32; ++i) {
    transpose_vscatter_indices_base[i] = i * 128;
  }

  // d_tile_vscatter_offsets 用于生成对角矩阵 D tile
  // offsets 指向 HMX tile layout 中对角线元素的位置
  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }

  // 外层循环：按 Q/O token 方向分块
  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows        = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g      = n_rows * G;
    const size_t n_row_tiles   = ceil_div(n_rows_g, HMX_FP16_TILE_N_ROWS);
    const size_t n_row_vec_cnt = ceil_div(n_rows_g, 64);

    // load [n_rows * G, D] tile of Q into VTCM. Packed Q is already HMX tiled.
    {
      if (input_packed) {
        const int token_tiles = ceil_div(qo_len, HMX_FP16_TILE_N_ROWS);
        const int token_tile = ir / HMX_FP16_TILE_N_ROWS;
        for (int d_tile = 0; d_tile < D / HMX_FP16_TILE_N_COLS; ++d_tile) {
          size_t src_offset = 0;
          if (block_sparse_packed_qk_tile_offset(&src_offset, kv_head_idx, token_tile, d_tile,
                                                 token_tiles, D) != 0) {
            FARF(ERROR, "packed chunk Q offset failed: head=%d tile=%d d=%d tiles=%d", kv_head_idx,
                 token_tile, d_tile, token_tiles);
            return -1;
          }
          hvx_copy(q_tile + (size_t) d_tile * HMX_FP16_TILE_N_ELMS,
                   Q + src_offset, HMX_FP16_TILE_SIZE);
        }
      } else {
      const size_t q_ld_blk_sz_bytes = q_ld_blk_sz * qo_element_size;
      const size_t q_ld_stride_bytes = q_ld_stride * qo_element_size;

      const uint8_t *q_ld_base = ((uint8_t *) Q) + ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;

      l2fetch(q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, n_rows, 1);

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int q_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int q_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        const HVX_Vector *pv_in0 =
          (const HVX_Vector *) (q_ld_base + q_idx0 * q_ld_stride_bytes + h_idx0 * head_dim * qo_element_size);
        const HVX_Vector *pv_in1 =
          (const HVX_Vector *) (q_ld_base + q_idx1 * q_ld_stride_bytes + h_idx1 * head_dim * qo_element_size);

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * head_dim;

        if (qo_fp32_element) {
#pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector v0    = *pv_in0++;
            const HVX_Vector v1    = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]     = v_out;
          }
        } else {
#pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
            HVX_Vector *pv_out1       = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
          }
        }
      }
      }
    }

    __fp16 *o_tile_prev = o_tile0;
    __fp16 *o_tile_curr = o_tile1;

    int dense_jc = 0;
    int index_pos = 0;
    const int index_qb = block_sparse_q_block_for_group_row(block_q, n_q_blocks, n_heads, ir, 0, q_head_base, G);
    const int32_t *index_row =
      use_host_index && index_qb >= 0 && index_qb < n_q_blocks ? block_mask + (size_t) index_qb * max_kept_blocks :
                                                                  NULL;
    hvx_fill_uh(mvec_m, 0xfbff, col_vec_size);
    hvx_fill_uh(mvec_l, 0, col_vec_size);
    hvx_fill_uh(o_tile_prev, 0, qo_tile_size);
    hvx_fill_uh(d_tile, 0, d_tile_size);
    while (1) {
      int jc = -1;
      int index_cols = 0;
      if (use_host_index) {
        bool found_index_chunk = false;
        while (index_row != NULL && index_pos < max_kept_blocks) {
          const int kb_abs = index_row[index_pos++];
          if (kb_abs < 0) {
            break;
          }
          if (kb_abs < 0 || kb_abs >= n_k_blocks || kb_abs / k_blocks_per_head != kv_head_idx) {
            continue;
          }
          const int start_block_in_head = kb_abs % k_blocks_per_head;
          int end_block_in_head = start_block_in_head + 1;
          while (index_pos < max_kept_blocks) {
            const int next_kb_abs = index_row[index_pos];
            if (next_kb_abs < 0 || next_kb_abs >= n_k_blocks || next_kb_abs / k_blocks_per_head != kv_head_idx) {
              break;
            }
            const int next_block_in_head = next_kb_abs % k_blocks_per_head;
            if (next_block_in_head != end_block_in_head) {
              break;
            }
            const int next_cols = (next_block_in_head + 1 - start_block_in_head) * block_k;
            if (next_cols > (int) blk_sz_c) {
              break;
            }
            ++index_pos;
            ++end_block_in_head;
          }
          jc = start_block_in_head * block_k;
          index_cols = (end_block_in_head - start_block_in_head) * block_k;
          if (jc + index_cols > kv_len) {
            index_cols = kv_len - jc;
          }
          if (index_cols <= 0) {
            continue;
          }
          found_index_chunk = true;
          break;
        }
        if (!found_index_chunk) {
          break;
        }
      } else {
        if (dense_jc >= kv_len) {
          break;
        }
        jc = dense_jc;
        dense_jc += blk_sz_c;
      }
      const size_t n_real_cols = use_host_index ? (size_t) index_cols : smin(kv_len - jc, blk_sz_c);
      const size_t n_cols      = n_real_cols;
      const size_t n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

      // load [Bc, D] tile of K^T into VTCM. Packed K is copied without transpose.
      {
        if (input_packed) {
          const int token_tiles = ceil_div(kv_len, HMX_FP16_TILE_N_ROWS);
          const int token_tile = jc / HMX_FP16_TILE_N_ROWS;
          for (int c = 0; c < n_col_tiles; ++c) {
            for (int d_tile = 0; d_tile < D / HMX_FP16_TILE_N_COLS; ++d_tile) {
              size_t src_offset = 0;
              if (block_sparse_packed_qk_tile_offset(&src_offset, kv_head_idx, token_tile + c,
                                                     d_tile, token_tiles, D) != 0) {
                FARF(ERROR, "packed chunk K offset failed: head=%d tile=%d d=%d tiles=%d c=%d", kv_head_idx,
                     token_tile + c, d_tile, token_tiles, c);
                return -1;
              }
              hvx_copy(k_tile + ((size_t) c * (D / HMX_FP16_TILE_N_COLS) + d_tile) * HMX_FP16_TILE_N_ELMS,
                       K + src_offset, HMX_FP16_TILE_SIZE);
            }
          }
        } else {
        const __fp16 *k_ld_base = K + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_real_cols, 1);

        const HVX_Vector v_step         = Q6_V_vsplat_R(4);
        const HVX_Vector v_offsets_base = vmem(transpose_vscatter_indices_base);

        for (int r0 = 0; r0 < n_col_tiles; ++r0) {
          __fp16 *out_base = k_tile + r0 * HMX_FP16_TILE_N_COLS * head_dim;

          HVX_Vector v_offsets = v_offsets_base;

          for (int r1 = 0; r1 < HMX_FP16_TILE_N_COLS; ++r1) {
            int r = r0 * HMX_FP16_TILE_N_COLS + r1;
            if (r >= n_real_cols) {
              break;
            }

            const HVX_Vector *pv_in = (const HVX_Vector *) (k_ld_base + r * kv_ld_stride);

#pragma unroll
            for (int d = 0; d < D / 64; ++d) {
              __fp16 *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
              Q6_vscatter_RMVwV((size_t) out_dual_tile, HMX_FP16_TILE_SIZE * 2 - 1, v_offsets, *pv_in++);
            }

            v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
          }
        }
        }
      }

      // 预取当前 block 对应的 V. Packed V is already in dim-tile-major layout.
      {
        if (input_packed) {
          const int token_tiles = ceil_div(kv_len, HMX_FP16_TILE_N_ROWS);
          const int token_tile = jc / HMX_FP16_TILE_N_ROWS;
          for (int d_tile = 0; d_tile < D / HMX_FP16_TILE_N_COLS; ++d_tile) {
            for (int c = 0; c < n_col_tiles; ++c) {
              size_t src_offset = 0;
              if (block_sparse_packed_v_tile_offset(&src_offset, kv_head_idx, token_tile + c, d_tile,
                                                    token_tiles, D) != 0) {
                FARF(ERROR, "packed chunk V offset failed: head=%d tile=%d d=%d tiles=%d c=%d", kv_head_idx,
                     token_tile + c, d_tile, token_tiles, c);
                return -1;
              }
              // PV consumes V with a fixed stride based on the selected chunk
              // capacity, not the tail's actual tile count.
              hvx_copy(v_tile + ((size_t) d_tile * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS,
                       V + src_offset, HMX_FP16_TILE_SIZE);
            }
          }
        } else {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_real_cols, 0);
      }

      // 计算块级 score：
      //
      //   dot(Q[Br', D], K[Bc, D]) ==> S[Br', Bc]
      {
        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_qk);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < n_col_tiles; ++c) {
              const __fp16 *row_tiles = q_tile + r * HMX_FP16_TILE_N_ROWS * head_dim;
              const __fp16 *col_tiles = k_tile + c * HMX_FP16_TILE_N_COLS * head_dim;
              __fp16 *out_tile = s_tile + (r * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS;

              hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
            }
          }
        }
        hmx_unit_release();
      }

      // safe softmax：
      //   1. 从 S tile 解码出连续行
      //   2. 对 HMX 对齐产生的 padding 列做 -inf mask
      //   3. 计算当前 block 的 rowmax / rowsum
      //   4. 生成 P tile
      //
      // host-index mode only visits kept K ranges. Dense-mask fallback computes full dense chunks.
      {
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

        for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
          HVX_Vector v_s_rowmax_local = v_neg_inf;
          HVX_Vector v_p_rowsum_local = Q6_V_vzero();

          for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            int r = r_vec_idx * 64 + r_vec_off;
            if (r >= align_up(n_rows_g, 2)) {
              break;
            }

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            __fp16 *s_ld_base = s_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;
            __fp16 *p_st_base = p_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;

            HVX_Vector *pv_row_buf0 = row_buffer0;
            HVX_Vector *pv_row_buf1 = row_buffer1;
            for (int c = 0; c < n_cols; c += 64) {
              const __fp16     *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              const HVX_Vector *pv_s_in0     = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
              const HVX_Vector *pv_s_in1     = pv_s_in0 + 16;

              HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
              *pv_row_buf0++               = Q6_V_lo_W(vp_s_dual_row);
              *pv_row_buf1++               = Q6_V_hi_W(vp_s_dual_row);
            }

            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            for (int c = 0; c < n_cols; c += 64) {
              const size_t ne = smin((int) n_real_cols - c, 64);
              HVX_VectorPred q_mask_keep0 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              HVX_VectorPred q_mask_keep1 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

              HVX_Vector v_s_row0 = Q6_V_vmux_QVV(q_mask_keep0, row_buffer0[c / 64], v_neg_inf);
              HVX_Vector v_s_row1 = Q6_V_vmux_QVV(q_mask_keep1, row_buffer1[c / 64], v_neg_inf);

              // 将被 mask 掉的位置直接改写成 -inf
              // 后续 rowmax / exp / rowsum 都基于这个结果继续
              row_buffer0[c / 64] = v_s_row0;
              row_buffer1[c / 64] = v_s_row1;

              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
            }

#pragma unroll
            for (int s = 64; s >= 2; s >>= 1) {
              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
            }

            HVX_Vector v_s_rowmax_pack2     = Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));
            HVX_Vector v_s_rowmax_pack2_rot = Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_s_rowmax_local_rot =
              Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));
            v_s_rowmax_local =
              Q6_V_vlalign_VVR(v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r_vec_off * sizeof(__fp16));

            HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(mvec_m[r_vec_idx], v_s_rowmax_local);

            HVX_Vector v_m_lut  = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));
            HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
            HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));

            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;
            HVX_Vector       v_p_rowsum1 = v_zero;

            for (int c = 0; c < n_cols; c += 64) {
              const int ne = smax(0, smin((int) n_real_cols - c, 64));
              HVX_VectorPred q_mask_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              HVX_Vector v_p_row0_hf, v_p_row1_hf;

              HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0);
              HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1);

              if (use_fp32_exp) {
                HVX_VectorPair vp_s_minus_m0_sf = hvx_my_vqf16_to_wsf(v_s_minus_m0);
                HVX_VectorPair vp_s_minus_m1_sf = hvx_my_vqf16_to_wsf(v_s_minus_m1);

                HVX_Vector v_p_row00_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row01_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row10_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m1_sf));
                HVX_Vector v_p_row11_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m1_sf));

                v_p_row0_hf = hvx_my_wsf_to_vhf(v_p_row01_sf, v_p_row00_sf);
                v_p_row1_hf = hvx_my_wsf_to_vhf(v_p_row11_sf, v_p_row10_sf);
              } else {
                v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
                v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);
              }

              v_p_row0_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row0_hf, v_zero);
              v_p_row1_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row1_hf, v_zero);

              __fp16     *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              HVX_Vector *pv_p_out0     = ((HVX_Vector *) out_dual_tile) + r1 / 2;
              HVX_Vector *pv_p_out1     = pv_p_out0 + 16;

              HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
              *pv_p_out0                   = Q6_V_lo_W(vp_p_dual_row);
              *pv_p_out1                   = Q6_V_hi_W(vp_p_dual_row);

              HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0_hf);
              HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1_hf);

              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
            }

#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            }
            HVX_Vector v_p_rowsum_pack2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));

            HVX_Vector v_p_rowsum_pack2_rot = Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_p_rowsum_local_rot = Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
            v_p_rowsum_local =
              Q6_V_vlalign_VVR(v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r_vec_off * sizeof(__fp16));
          }

          mvec_s_rowmax[r_vec_idx] = v_s_rowmax_local;
          mvec_p_rowsum[r_vec_idx] = v_p_rowsum_local;
        }
        }
      }

      // load [Bc, D] tile of V into VTCM for raw input. Packed input was
      // loaded above in the final HMX RHS layout and must not be overwritten.
      if (!input_packed) {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        for (int r = 0; r < n_real_cols; r += 2) {
          const bool next_row_valid = (r + 1) < n_real_cols;

          const HVX_Vector *pv_in0 = (const HVX_Vector *) (v_ld_base + (r + 0) * kv_ld_stride);
          const HVX_Vector *pv_in1 = (const HVX_Vector *) (v_ld_base + (r + 1) * kv_ld_stride);

#pragma unroll
          for (int c = 0; c < D; c += 64) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;
            int c0 = c / HMX_FP16_TILE_N_COLS;

            int     tile_idx0  = (c0 + 0) * n_tiles_per_blk_c + r0;
            int     tile_idx1  = (c0 + 1) * n_tiles_per_blk_c + r0;
            __fp16 *tile_base0 = v_tile + tile_idx0 * HMX_FP16_TILE_N_ELMS;
            __fp16 *tile_base1 = v_tile + tile_idx1 * HMX_FP16_TILE_N_ELMS;

            HVX_Vector *pv_out0 = ((HVX_Vector *) tile_base0) + r1 / 2;
            HVX_Vector *pv_out1 = ((HVX_Vector *) tile_base1) + r1 / 2;
            *pv_out0            = Q6_V_lo_W(vp);
            *pv_out1            = Q6_V_hi_W(vp);
          }
        }
      }

      // 复用 mvec_s_rowmax 这块 buffer，
      // 后续存 exp(m_i^{j-1} - m_i^j)
      HVX_Vector *mvec_exp_m_diff = mvec_s_rowmax;

      // 更新 online softmax 状态：
      //
      //   m_i^j = max(m_i^{j-1}, rowmax(S_i^j))
      //   l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
      {
        for (int i = 0; i < n_row_vec_cnt; ++i) {
          HVX_Vector v_m_prev = mvec_m[i];
          HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, mvec_s_rowmax[i]);
          HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);

          HVX_Vector v_exp_m_diff_hf = hvx_my_exp2_vhf_vqf16(v_m_diff);

          HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(mvec_l[i], v_exp_m_diff_hf);
          v_l_curr            = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, mvec_p_rowsum[i]);

          mvec_m[i] = v_m_curr;
          mvec_l[i] = v_l_curr;

          mvec_exp_m_diff[i] = v_exp_m_diff_hf;
        }
      }

      // 更新当前输出：
      //
      //   O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + P_i^j V_j
      {
        const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
        const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        for (int i = 0; i < n_row_tiles; ++i) {
          const HVX_Vector v_content = Q6_V_vror_VR(mvec_exp_m_diff[i / 2], (i % 2) * 64);

          __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
          Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_id);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < head_dim / 32; ++c) {
              __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
              __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
              hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

              __fp16 *p_tile_in = p_tile + (r * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;
              __fp16 *v_tile_in = v_tile + (c * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;
              for (int k = 0; k < n_col_tiles; ++k) {
                int    offset  = k * HMX_FP16_TILE_N_ELMS;
                size_t n_tiles = 1;
                hmx_load_tiles_fp16(p_tile_in + offset, v_tile_in + offset, n_tiles);
              }

              __fp16 *o_tile_out = o_tile_curr + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
              hmx_consume_accumulator_fp16(o_tile_out);
            }
          }
        }
        hmx_unit_release();

        swap_ptr(&o_tile_curr, &o_tile_prev);
      }
    }

    // 最终归一化：
    //
    //   O_i = diag(l_i^{-1}) O_i
    {
      const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
      const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

      HVX_Vector v_content;
      for (int i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
          v_content = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(mvec_l[i / 2]));
        } else {
          v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
      }

      hmx_unit_acquire();
      {
        hmx_set_output_scales(hmx_output_scales_id);
        for (int r = 0; r < n_row_tiles; ++r) {
          for (int c = 0; c < head_dim / 32; ++c) {
            __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_out = o_tile_curr + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;

            hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
          }
        }
      }
      hmx_unit_release();
    }

    // 将当前 Q block 对应的结果从 VTCM 写回外部 O
    {
      const size_t o_st_blk_sz_bytes = o_st_blk_sz * qo_element_size;
      const size_t o_st_stride_bytes = o_st_stride * qo_element_size;

      uint8_t *o_st_base = ((uint8_t *) O) + ir * o_st_stride_bytes + kv_head_idx * o_st_blk_sz_bytes;

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int o_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int o_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        uint8_t *row_out0 = o_st_base + o_idx0 * o_st_stride_bytes + h_idx0 * o_head_dim * qo_element_size;
        uint8_t *row_out1 = o_st_base + o_idx1 * o_st_stride_bytes + h_idx1 * o_head_dim * qo_element_size;

        HVX_Vector *pv_out0 = (HVX_Vector *) row_out0;
        HVX_Vector *pv_out1 = (HVX_Vector *) row_out1;

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *in_base = o_tile_curr + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);

            const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);
            const HVX_Vector v_out0 = Q6_V_lo_W(vp);
            const HVX_Vector v_out1 = Q6_V_hi_W(vp);

            if (with_lse) {
              vstu_variable(row_out0 + d * 32 * sizeof(float), sizeof(v_out0), v_out0);
              if (next_row_valid) {
                vstu_variable(row_out1 + d * 32 * sizeof(float), sizeof(v_out1), v_out1);
              }
            } else {
              *pv_out0++ = v_out0;
              if (next_row_valid) {
                *pv_out1++ = v_out1;
              }
            }
          }

          if (with_lse) {
            const float lse0 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 0);
            memcpy(row_out0 + D * sizeof(float), &lse0, sizeof(lse0));
            if (next_row_valid) {
              const float lse1 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 1);
              memcpy(row_out1 + D * sizeof(float), &lse1, sizeof(lse1));
            }
          }
        } else {
          assert(!with_lse);
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const __fp16     *in_dual_tile = in_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector *pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
            const HVX_Vector *pv_in1       = pv_in0 + 16;

            const HVX_VectorPair vp = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        }
        // clang-format on
      }
    }
  }
  return 0;
}

/*
 * Halo-row FlashAttention 的统一执行核心。
 *
 * 这个函数由两种入口共用：
 *   - halo_l2：通过 L2 prefetch 提前准备 image K/V row；
 *   - halo_userdma：通过 per-worker UserDMA 把 image K/V row 搬到 raw staging，
 *     再转换为 HMX 可直接消费的 packed row。
 *
 * 每次调用只处理一个 KV head，以及 G = n_heads / n_kv_heads 个对应的
 * Q heads。逻辑上的输入/输出布局为：
 *
 *   Q: [qo_len, n_heads,    head_dim]
 *   K/V: [kv_len, n_kv_heads, head_dim]
 *   O: [qo_len, n_heads,    head_dim + (with_lse ? 1 : 0)]
 *
 * 在 halo 场景中，KV 序列由 image tokens 和 text tokens 组成。对每个
 * image query block，block_mask 给出需要访问的 image block；相邻 image
 * block row 被当作 halo 一起复用，text K/V 则可常驻 VTCM。当前 Q block
 * 的每个 K/V segment 会按以下优先级选择数据源：
 *
 *   1. resident_text_k/v：text K/V 已经预打包并常驻 VTCM；
 *   2. packed_row_slots：image row 已经被 L2 或 UserDMA 准备好；
 *   3. 临时 k_tile/v_tile：从外部 K/V 读取并现场转换为 HMX layout。
 *
 * 当前有效的 halo 执行走 image-Q 融合路径：先收集当前 image query 所需的
 * halo/text segments，再一次性执行稳定的 packed-segment 更新；image query
 * 的输出直接复用 q_tile，text query 则使用 resident state 在循环外更新。
 * 函数内保留的逐 segment online softmax 更新代码是旧路径/防御性代码，
 * 当前入口会在进入该路径前拒绝不满足融合条件的配置。
 *
 * halo_plan 描述了 VTCM workspace、packed row slot、resident text state 和
 * DMA staging 的布局；因此本函数只接受已经完成预分配/校验的 plan，不在
 * 这里动态申请 workspace。Q/O 的外部存储当前按 fp32 处理，VTCM 内部使用
 * fp16 HMX tile；with_lse 为 true 时，最后额外写回每行的 LSE。
 */
static int block_sparse_flash_attn_f16_core_halo_rows(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit,
                                             __fp16 *restrict O, const __fp16 *restrict Q,
                                             const __fp16 *restrict K, const __fp16 *restrict V,
                                             const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
                                             int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                                             int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
                                             int worker_index, const block_sparse_halo_plan_t *halo_plan,
                                             int image_tokens, int text_tokens, int image_block_rows,
                                             int image_block_cols, bool resident_text_enabled,
                                             bool userdma_enabled, int userdma_scope, bool input_packed,
                                             bool halo_trace,
                                             block_sparse_halo_run_trace_t *halo_run_trace) {
  if (userdma_enabled && userdma_scope != BLOCK_SPARSE_HALO_USERDMA_PER_WORKER) {
    return -1;
  }

  // 一个 KV head 对应 G 个 Q heads
  // 当前函数只处理 kv_head_idx 对应的 K/V head
  // 以及 Q/O 中 [kv_head_idx * G, (kv_head_idx + 1) * G) 这 G 个 heads
  const int G = n_heads / n_kv_heads;
  const int D = head_dim;
  const int q_head_base = kv_head_idx * G;
  const int k_blocks_per_head = n_k_blocks / n_kv_heads;
  const bool use_host_index =
    mask_format == 1 && max_kept_blocks > 0 && G == 1 && block_q == HMX_FP16_TILE_N_ROWS &&
    block_k == HMX_FP16_TILE_N_COLS;

  // L2 模式提前把 image K/V 按 row 打包到 packed_row_slots；
  // UserDMA 模式则先 DMA 到 raw staging，再使用同一组 packed row slots。
  const bool row_l2_prefetch_enabled =
    resident_text_enabled && !userdma_enabled && image_block_rows > 0 && image_block_cols > 0;

  // 两种 halo loader 都复用 packed image row。启用后，image Q 可以直接消费
  // packed segment，避免在每个 Q block 中重复读取/重排同一 image row。
  const bool packed_row_reuse_enabled =
    resident_text_enabled && image_block_rows > 0 && image_block_cols > 0 &&
    (row_l2_prefetch_enabled || userdma_enabled);

  // text Q 的状态也常驻 VTCM：先处理 image halo 对 text Q 的增量，再处理
  // resident text K/V，最后统一写回 text Q 的 O/LSE。
  const bool row_text_reuse_enabled =
    resident_text_enabled && text_tokens > 0 && image_block_rows > 0 && image_block_cols > 0 &&
    packed_row_reuse_enabled;

  // 当前实现认为 Q/O 外部存储是 fp32
  // 虽然函数参数类型是 __fp16*，
  // 但实际通过 uint8_t* + qo_element_size 访问。
  // qo_fp32_element = true 时：
  //   Q/O 外部按 float 存储；
  //   读入 VTCM 后转成 fp16 HMX tile；
  //   写回 O 时再从 fp16 转回 float
  const bool   qo_fp32_element = true;
  const size_t qo_element_size = qo_fp32_element ? sizeof(float) : sizeof(__fp16);

  block_sparse_fa_plan_t plan;
  if (halo_plan != NULL) {
    plan.mode = BLOCK_SPARSE_FA_CHUNKED;
    plan.blk_sz_c = halo_plan->blk_sz_c;
    plan.required_bytes = halo_plan->required_bytes;
  } else if (block_sparse_fa_select_plan(&plan, false, G, head_dim, kv_len, vtcm_limit - vtcm) != 0) {
    return -1;
  }
  const size_t blk_sz_r = HMX_FP16_TILE_N_ROWS;
  const size_t blk_sz_c = plan.blk_sz_c;
  if (text_tokens > INT_MAX - (HMX_FP16_TILE_N_COLS - 1)) {
    return -1;
  }
  const size_t text_blk_sz_c = align_up((size_t) text_tokens, HMX_FP16_TILE_N_COLS);
  size_t three_row_cols;
  size_t image_q_cols;
  if (image_block_cols <= 0 ||
      !checked_size_mul((size_t) image_block_cols,
                        BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS * HMX_FP16_TILE_N_COLS,
                        &three_row_cols) ||
      three_row_cols > INT_MAX ||
      !checked_size_mul((size_t) smin(3, image_block_cols),
                        BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS * HMX_FP16_TILE_N_COLS,
                        &image_q_cols) ||
      !checked_size_add(image_q_cols, text_blk_sz_c, &image_q_cols) ||
      image_q_cols > INT_MAX) {
    return -1;
  }
  const size_t minimum_scratch_cols =
    smax(smax(blk_sz_c, text_blk_sz_c), smax(three_row_cols, image_q_cols));
  const size_t scratch_blk_sz_c = halo_plan != NULL ?
    halo_plan->scratch_col_capacity : minimum_scratch_cols;
  if (scratch_blk_sz_c != minimum_scratch_cols || scratch_blk_sz_c > INT_MAX ||
      scratch_blk_sz_c % HMX_FP16_TILE_N_COLS != 0) {
    return -1;
  }
  assert(blk_sz_c % HMX_FP16_TILE_N_COLS == 0);

  // 当前 Q block 展开 G 个 Q heads 后的行数是 G * Br
  // 为适配 HMX FP16 tile 的行数要求，按 HMX_FP16_TILE_N_ROWS 对齐
  // g_br 表示对齐后的 Br'
  const size_t g_br = align_up(G * blk_sz_r, HMX_FP16_TILE_N_ROWS);

  // 当前 Q block 在行方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_r = g_br / HMX_FP16_TILE_N_ROWS;

  // 当前 K/V block 在列方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;

  // q_tile / o_tile:
  //   逻辑形状 [Br', D]
  //
  // k_tile / v_tile:
  //   逻辑形状 [Bc, D]
  //
  // s_tile / p_tile:
  //   逻辑形状 [Br', Bc]
  //   s_tile 保存 QK^T score；
  //   p_tile 保存 softmax numerator P
  //
  // d_tile:
  //   逻辑形状 [Br', Br']
  //   保存对角矩阵，用于：
  //     1. alpha * O_old
  //     2. final O / l
  //
  // col_vec_size:
  //   保存每一行的 m/l/rowmax/rowsum
  //
  // row_vec_size:
  //   softmax 时把 HMX tile layout 中的 S/P 解码成连续 row buffer
  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);
  const size_t kv_tile_size   = align_up(blk_sz_c * head_dim * sizeof(__fp16), 4096);
  const size_t core_tile_size = align_up(g_br * scratch_blk_sz_c * sizeof(__fp16), 4096);
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);
  const size_t row_vec_size   = align_up(scratch_blk_sz_c * sizeof(__fp16), 256);

  // 原始 K/V 布局：
  //   K/V: [kv_len, n_kv_heads, head_dim]
  //
  // kv_ld_blk_sz:
  //   当前 kv_head_idx 对应的一个 head 的大小
  //
  // kv_ld_stride:
  //   K/V 中一个 token 的完整 stride，即所有 KV heads 的大小
  const size_t kv_ld_blk_sz   = head_dim;
  const size_t kv_ld_stride   = n_kv_heads * head_dim;

  // 原始 Q/O 布局：
  //   Q: [qo_len, n_heads, head_dim]
  //   O: [qo_len, n_heads, head_dim + (with_lse ? 1 : 0)]
  //
  // q_ld_blk_sz/o_st_blk_sz:
  //   当前 kv_head_idx 对应的 G 个 Q heads 的总大小
  //
  // q_ld_stride/o_st_stride:
  //   Q/O 中一个 token 的完整 stride，即所有 Q heads 的大小
  const size_t q_ld_blk_sz = G * head_dim;
  const size_t q_ld_stride = n_heads * head_dim;
  const size_t o_head_dim  = head_dim + (with_lse ? 1 : 0);
  const size_t o_st_blk_sz = G * o_head_dim;
  const size_t o_st_stride = n_heads * o_head_dim;
  const int packed_token_tiles = ceil_div(kv_len, HMX_FP16_TILE_N_COLS);

  const size_t halo_available_bytes = (size_t) (vtcm_limit - vtcm);
  if (halo_plan == NULL || halo_plan->workspace_offset != 0 ||
      halo_plan->workspace_bytes > halo_available_bytes ||
      halo_plan->q_offset > halo_plan->workspace_bytes ||
      qo_tile_size > halo_plan->workspace_bytes - halo_plan->q_offset ||
      halo_plan->o_offsets[0] > halo_plan->workspace_bytes ||
      smax(qo_tile_size, d_tile_size) > halo_plan->workspace_bytes - halo_plan->o_offsets[0] ||
      halo_plan->o_offsets[1] != halo_plan->o_offsets[0] ||
      halo_plan->diagonal_offset != halo_plan->o_offsets[0] ||
      halo_plan->packed_k_bytes != kv_tile_size ||
      halo_plan->packed_k_offset > halo_plan->workspace_bytes ||
      kv_tile_size > halo_plan->workspace_bytes - halo_plan->packed_k_offset ||
      halo_plan->packed_v_bytes != kv_tile_size ||
      halo_plan->packed_v_offset > halo_plan->workspace_bytes ||
      kv_tile_size > halo_plan->workspace_bytes - halo_plan->packed_v_offset ||
      halo_plan->score_offset > halo_plan->workspace_bytes ||
      core_tile_size > halo_plan->workspace_bytes - halo_plan->score_offset ||
      halo_plan->probability_offset != halo_plan->score_offset ||
      halo_plan->softmax_state_bytes != 2 * col_vec_size ||
      halo_plan->softmax_state_offset > halo_plan->workspace_bytes ||
      halo_plan->softmax_state_bytes > halo_plan->workspace_bytes - halo_plan->softmax_state_offset ||
      halo_plan->row_buffer_bytes != 2 * row_vec_size ||
      halo_plan->row_buffer_offset > halo_plan->workspace_bytes ||
      halo_plan->row_buffer_bytes > halo_plan->workspace_bytes - halo_plan->row_buffer_offset ||
      halo_plan->scales_bytes != 512 || halo_plan->scales_offset > halo_plan->workspace_bytes ||
      halo_plan->scales_bytes > halo_plan->workspace_bytes - halo_plan->scales_offset) {
    return -1;
  }

  __fp16 *q_tile = (__fp16 *) (vtcm + halo_plan->q_offset);
  __fp16 *o_tile0 = (__fp16 *) (vtcm + halo_plan->o_offsets[0]);
  __fp16 *o_tile1 = (__fp16 *) (vtcm + halo_plan->o_offsets[1]);
  __fp16 *k_tile = (__fp16 *) (vtcm + halo_plan->packed_k_offset);
  __fp16 *v_tile = (__fp16 *) (vtcm + halo_plan->packed_v_offset);
  __fp16 *s_tile = (__fp16 *) (vtcm + halo_plan->score_offset);
  __fp16 *p_tile = (__fp16 *) (vtcm + halo_plan->probability_offset);
  __fp16 *d_tile = (__fp16 *) (vtcm + halo_plan->diagonal_offset);

  HVX_Vector *mvec_s_rowmax = (HVX_Vector *) (vtcm + halo_plan->softmax_state_offset);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *)
    (vtcm + halo_plan->softmax_state_offset + col_vec_size);
  // 当前 halo 执行要求走 fused image-Q 路径，因此不再单独分配 m/l：
  // mvec_s_rowmax 和 mvec_p_rowsum 在融合更新中分别复用为 rowmax/rowsum；
  // 下面的别名只保证被拒绝的 non-fused 分支保持类型/布局完整。
  HVX_Vector *mvec_m = mvec_s_rowmax;
  HVX_Vector *mvec_l = mvec_p_rowsum;

  HVX_Vector *row_buffer0 = (HVX_Vector *) (vtcm + halo_plan->row_buffer_offset);
  HVX_Vector *row_buffer1 = (HVX_Vector *)
    (vtcm + halo_plan->row_buffer_offset + row_vec_size);
  uint8_t *hmx_output_scales_id = vtcm + halo_plan->scales_offset;
  uint8_t *hmx_output_scales_qk = hmx_output_scales_id + 256;
  uint8_t *vtcm_cur = vtcm + halo_plan->workspace_bytes;
  __fp16 *resident_text_k = NULL;
  __fp16 *resident_text_v = NULL;
  int resident_text_tiles = 0;
  size_t resident_text_half_bytes = 0;
  __fp16 *resident_text_q = NULL;
  __fp16 *resident_text_o[2] = {NULL, NULL};
  uint8_t *resident_text_state = NULL;
  int resident_text_q_blocks = 0;
  block_sparse_halo_packed_row_slot_t packed_row_slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS] = {};
  if (resident_text_enabled) {
    if (halo_plan == NULL || halo_plan->resident_text_offset < halo_plan->workspace_bytes ||
        halo_plan->resident_text_offset > (size_t) (vtcm_limit - vtcm) ||
        halo_plan->resident_text_bytes > (size_t) (vtcm_limit - vtcm) - halo_plan->resident_text_offset) {
      return -1;
    }
    vtcm_cur = vtcm + halo_plan->resident_text_offset;
    resident_text_tiles = ceil_div(text_tokens, HMX_FP16_TILE_N_COLS);
    resident_text_half_bytes = halo_plan->resident_text_bytes / 2;
    resident_text_k = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, resident_text_half_bytes);
    resident_text_v = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, resident_text_half_bytes);
  }
  if (row_text_reuse_enabled) {
    resident_text_q_blocks = block_sparse_halo_text_q_blocks(text_tokens);
    if (resident_text_q_blocks <= 0 || halo_plan->resident_text_q_blocks != resident_text_q_blocks ||
        image_tokens < 0 || text_tokens > qo_len - image_tokens ||
        halo_plan->resident_text_q_bytes != (size_t) resident_text_q_blocks * qo_tile_size ||
        halo_plan->resident_text_o_bytes != (size_t) resident_text_q_blocks * qo_tile_size ||
        halo_plan->resident_text_state_bytes != (size_t) resident_text_q_blocks * 2 * col_vec_size ||
        halo_plan->resident_text_q_offset > halo_available_bytes ||
        halo_plan->resident_text_q_bytes > halo_available_bytes - halo_plan->resident_text_q_offset ||
        halo_plan->resident_text_o_offsets[0] > halo_available_bytes ||
        halo_plan->resident_text_o_bytes > halo_available_bytes - halo_plan->resident_text_o_offsets[0] ||
        halo_plan->resident_text_o_offsets[1] > halo_available_bytes ||
        halo_plan->resident_text_o_bytes > halo_available_bytes - halo_plan->resident_text_o_offsets[1] ||
        halo_plan->resident_text_state_offset > halo_available_bytes ||
        halo_plan->resident_text_state_bytes > halo_available_bytes - halo_plan->resident_text_state_offset) {
      return -1;
    }
    resident_text_q = (__fp16 *) (vtcm + halo_plan->resident_text_q_offset);
    resident_text_o[0] = (__fp16 *) (vtcm + halo_plan->resident_text_o_offsets[0]);
    resident_text_o[1] = (__fp16 *) (vtcm + halo_plan->resident_text_o_offsets[1]);
    resident_text_state = vtcm + halo_plan->resident_text_state_offset;
  }
  if (packed_row_reuse_enabled) {
    if (halo_plan == NULL ||
        halo_plan->packed_image_row_slot_count != BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS ||
        halo_plan->packed_image_row_stride == 0 ||
        (halo_plan->packed_image_row_stride & 1u) != 0 ||
        halo_plan->packed_image_row_bytes !=
          BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS * halo_plan->packed_image_row_stride) {
      return -1;
    }
    const size_t packed_row_half_bytes = halo_plan->packed_image_row_stride / 2;
    for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
      const size_t row_offset = halo_plan->packed_image_row_offsets[slot];
      if (row_offset > halo_available_bytes ||
          halo_plan->packed_image_row_stride > halo_available_bytes - row_offset) {
        return -1;
      }
      packed_row_slots[slot].packed_k = (__fp16 *) (vtcm + row_offset);
      packed_row_slots[slot].packed_v = (__fp16 *) (vtcm + row_offset + packed_row_half_bytes);
      packed_row_slots[slot].row_id = -1;
    }
  }
  assert(vtcm_cur <= vtcm_limit);

  // qk_scale 用于 attention score：
  //
  //   QK^T / sqrt(D)
  //
  // 但后面 softmax 用的是 exp2，而不是 exp
  // 因此这里提前乘上 log2(e)
  float  qk_scale    = 1.0f / sqrtf(head_dim) * 1.44269504f;
  __fp16 qk_scale_hf = (__fp16) qk_scale;

  // 初始化 HMX 输出 scale
  // 0x3c00 是 fp16 1.0
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));

  // QK^T 输出时直接乘 qk_scale_hf
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));

  // d_tile_vscatter_offsets 用于生成对角矩阵 D tile
  // offsets 指向 HMX tile layout 中对角线元素的位置
  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }
  const float stable_lse_scale = 1.0f / ((float) qk_scale_hf * sqrtf((float) D));
  block_sparse_halo_update_scratch_t halo_update_scratch = {
    .score = s_tile,
    .probability = p_tile,
    .diagonal = d_tile,
    .rowmax = mvec_s_rowmax,
    .rowsum = mvec_p_rowsum,
    .row_buffer0 = row_buffer0,
    .row_buffer1 = row_buffer1,
  };

  int64_t text_pack_us = 0;
  uint64_t requested_k_bytes = 0;
  uint64_t requested_v_bytes = 0;
  int halo_prefetch_rows = 0;
  int packed_image_rows = 0;
  int packed_image_blocks = 0;
  int image_q_direct_segment_uses = 0;
  int text_q_image_row_updates = 0;
  int text_q_image_block_updates = 0;
  int text_q_image_batches = 0;
  int image_q_batch_updates = 0;
  int image_q_segment_updates = 0;
  int resident_text_segments = 0;
  int64_t startup_row_pack_us = 0;
  int64_t boundary_row_pack_us = 0;
  int64_t text_q_image_update_us = 0;
  int64_t text_q_resident_update_us = 0;
  int64_t text_q_init_us = 0;
  int64_t text_q_store_us = 0;
  int64_t image_q_normal_us = 0;
  int64_t image_q_output_us = 0;
  if (resident_text_enabled) {
    const int64_t text_pack_begin = halo_trace ? HAP_perf_get_time_us() : 0;
    hvx_fill_uh(resident_text_k, 0, resident_text_half_bytes);
    hvx_fill_uh(resident_text_v, 0, resident_text_half_bytes);
    if (text_tokens > 0) {
      if (input_packed) {
        if (block_sparse_halo_copy_packed_text_kv(
              resident_text_k, resident_text_v, K, V, kv_head_idx, D,
              packed_token_tiles, image_tokens, text_tokens) != 0) {
          return -1;
        }
      } else {
        const __fp16 *text_k = K + (size_t) image_tokens * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        const __fp16 *text_v = V + (size_t) image_tokens * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(text_k, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), text_tokens, 1);
        block_sparse_pack_k_f16(resident_text_k, text_k, (int) kv_ld_stride, text_tokens, D);
        l2fetch(text_v, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), text_tokens, 1);
        block_sparse_pack_v_f16(resident_text_v, text_v, (int) kv_ld_stride, text_tokens,
                                resident_text_tiles * HMX_FP16_TILE_N_COLS, D);
      }
    }
    if (halo_trace) {
      text_pack_us = HAP_perf_get_time_us() - text_pack_begin;
    }
  }
  if (row_text_reuse_enabled) {
    const int64_t text_q_init_begin = halo_trace ? HAP_perf_get_time_us() : 0;
    hvx_fill_uh(resident_text_q, 0, halo_plan->resident_text_q_bytes);
    hvx_fill_uh(resident_text_o[0], 0, halo_plan->resident_text_o_bytes);
    for (int text_q_block = 0; text_q_block < resident_text_q_blocks; ++text_q_block) {
      const int n_text_rows = block_sparse_halo_text_q_rows(text_tokens, text_q_block);
      const int q_token = image_tokens + text_q_block * HMX_FP16_TILE_N_ROWS;
      __fp16 *packed_text_q = (__fp16 *) ((uint8_t *) resident_text_q +
        (size_t) text_q_block * qo_tile_size);
      if (n_text_rows <= 0) {
        return -1;
      }
      if (input_packed) {
        size_t q_offset;
        if (q_token % HMX_FP16_TILE_N_ROWS != 0 ||
            block_sparse_packed_qk_tile_offset(
              &q_offset, kv_head_idx, q_token / HMX_FP16_TILE_N_ROWS, 0,
              packed_token_tiles, D) != 0) {
          return -1;
        }
        hvx_copy(packed_text_q, Q + q_offset,
                 (size_t) (D / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_SIZE);
      } else {
        const float *raw_text_q = (const float *) Q +
          (size_t) q_token * q_ld_stride + kv_head_idx * q_ld_blk_sz;
        if (block_sparse_halo_pack_q_f32(
              packed_text_q, raw_text_q, q_ld_stride, n_text_rows, D) != 0) {
          return -1;
        }
      }

      uint8_t *text_state_block = resident_text_state +
        (size_t) text_q_block * 2 * col_vec_size;
      hvx_fill_uh(text_state_block, 0xfbff, col_vec_size);
      hvx_fill_uh(text_state_block + col_vec_size, 0, col_vec_size);
    }
    if (halo_trace) {
      text_q_init_us = HAP_perf_get_time_us() - text_q_init_begin;
    }
  }

  int resident_text_o_bank = 0;
  int text_q_next_pending_row = 0;
  int last_image_prefetched_row = -1;
  int packed_last_q_row = -1;
  if (row_l2_prefetch_enabled) {
    if (image_block_rows < 2) {
      return -1;
    }
    const int row_tokens = image_block_cols * block_k;
    const uint64_t row_bytes = (uint64_t) row_tokens * (uint64_t) head_dim * sizeof(__fp16);
    for (int row = 0; row < 2; ++row) {
      const int64_t pack_begin = halo_trace ? HAP_perf_get_time_us() : 0;
      if (input_packed &&
          block_sparse_halo_prefetch_packed_image_row(
            K, V, kv_head_idx, head_dim, packed_token_tiles,
            image_block_cols, row) != 0) {
        return -1;
      }
      const int load_result = input_packed ?
        block_sparse_halo_copy_packed_image_row(
          &packed_row_slots[row], K, V, kv_head_idx, D,
          packed_token_tiles, image_block_cols, row) :
        block_sparse_halo_pack_image_row(
          &packed_row_slots[row], K, V, n_kv_heads,
          kv_head_idx, D, image_block_cols, row);
      if (load_result != 0) {
        return -1;
      }
      if (halo_trace) {
        startup_row_pack_us += HAP_perf_get_time_us() - pack_begin;
      }
      ++packed_image_rows;
      packed_image_blocks += image_block_cols;
    }
    if (image_block_rows > 2) {
      if (input_packed) {
        if (block_sparse_halo_prefetch_packed_image_row(
              K, V, kv_head_idx, head_dim, packed_token_tiles,
              image_block_cols, 2) != 0) {
          return -1;
        }
      } else {
        block_sparse_halo_l2_prefetch_image_row(K, V, n_kv_heads, kv_head_idx, head_dim,
                                                block_k, image_block_cols, 2);
      }
      ++halo_prefetch_rows;
      requested_k_bytes += row_bytes;
      requested_v_bytes += row_bytes;
      last_image_prefetched_row = 2;
    }
  }

  block_sparse_halo_row_slot_t dma_row_slots[BLOCK_SPARSE_HALO_RAW_STAGING_SLOTS] = {};
  _Alignas(64) dma_desc_2d_t dma_desc_storage[BLOCK_SPARSE_HALO_MAX_DMA_DESCS_PER_ROW];
  const int dma_desc_count = input_packed ? block_sparse_halo_packed_dma_desc_count(head_dim) :
                                            BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW;
  int dma_pending_slot = -1;
  int dma_pending_row = -1;
  int dma_last_q_row = -1;
  uint64_t dma_requested_bytes = 0;
  int dma_image_query_rows_loaded = 0;
  int dma_async_rows_loaded = 0;
  int dma_descriptors_submitted = 0;
  int64_t dma_submit_us = 0;
  int64_t dma_overlap_window_us = 0;
  int64_t dma_exposed_wait_us = 0;
  int64_t dma_pack_us = 0;
  int64_t dma_pending_submit_done_us = 0;
  const int dma_image_blocks = image_tokens / block_k;
  const uint64_t dma_row_transfer_bytes = userdma_enabled ?
    (uint64_t) image_block_cols * (uint64_t) block_k * (uint64_t) head_dim * sizeof(__fp16) * 2u : 0;
  if (userdma_enabled) {
    if (halo_plan == NULL || halo_plan->image_row_stride == 0 || image_block_rows <= 0 || image_block_cols <= 0 ||
        halo_plan->image_row_bytes == 0 ||
        halo_plan->image_row_bytes > (size_t) (vtcm_limit - vtcm)) {
      return -1;
    }
    for (int slot = 0; slot < BLOCK_SPARSE_HALO_RAW_STAGING_SLOTS; ++slot) {
      dma_row_slots[slot].raw = vtcm + halo_plan->image_row_offsets[slot];
      dma_row_slots[slot].row_id = -1;
    }
    const int startup_rows = smin(image_block_rows, 2);
    for (int row = 0; row < startup_rows; ++row) {
      const int64_t pack_begin = halo_trace ? HAP_perf_get_time_us() : 0;
      const int load_result = input_packed ?
        block_sparse_halo_copy_packed_image_row(
          &packed_row_slots[row], K, V, kv_head_idx, D,
          packed_token_tiles, image_block_cols, row) :
        block_sparse_halo_pack_image_row(
          &packed_row_slots[row], K, V, n_kv_heads,
          kv_head_idx, D, image_block_cols, row);
      if (load_result != 0) {
        return -1;
      }
      if (halo_trace) {
        startup_row_pack_us += HAP_perf_get_time_us() - pack_begin;
      }
      ++packed_image_rows;
      packed_image_blocks += image_block_cols;
    }
  }

#define BLOCK_SPARSE_HALO_CORE_FAIL()                                                              \
  do {                                                                                             \
    if (userdma_enabled && dma_pending_slot >= 0) {                                                \
      block_sparse_halo_row_dma_drain(dma_desc_storage, dma_desc_count,                           \
                                      userdma_scope, worker_index,                                 \
                                      &dma_row_slots[dma_pending_slot].outstanding);                \
    }                                                                                              \
    return -1;                                                                                     \
  } while (0)

  // 外层循环按 Q/O token 方向分块。
  // row-reuse 模式只在这里处理 image Q；text Q 使用常驻的 Q/O/softmax state，
  // 在 image row 处理完后统一更新和写回。
  const int q_loop_tokens = row_text_reuse_enabled ? image_tokens : qo_len;
  const int q_block_count = (int) ceil_div(q_loop_tokens, blk_sz_r);
  for (int ir = 0; ir < q_loop_tokens; ir += blk_sz_r) {
    const size_t n_rows      = smin(q_loop_tokens - ir, blk_sz_r);
    const size_t n_rows_g    = n_rows * G;
    const size_t n_row_tiles = ceil_div(n_rows_g, HMX_FP16_TILE_N_ROWS);

    if (row_l2_prefetch_enabled) {
      const int q_image_block = ir / block_q;
      const int q_image_row = q_image_block / image_block_cols;
      if (q_image_row != packed_last_q_row) {
        const int incoming_row = q_image_row + 1;
        if (q_image_row > 0 && incoming_row < image_block_rows &&
            block_sparse_halo_find_packed_row(packed_row_slots, incoming_row) < 0) {
          const int first_protected_row = q_image_row - 1;
          const int protected_rows = smin(2, image_block_rows - first_protected_row);
          const int victim = block_sparse_halo_select_packed_row_victim(
            packed_row_slots, first_protected_row, protected_rows);
          const int64_t pack_begin = halo_trace ? HAP_perf_get_time_us() : 0;
          if (victim < 0 ||
              (row_text_reuse_enabled &&
               packed_row_slots[victim].row_id >= text_q_next_pending_row) ||
              (input_packed ?
                 block_sparse_halo_copy_packed_image_row(
                   &packed_row_slots[victim], K, V, kv_head_idx, D,
                   packed_token_tiles, image_block_cols, incoming_row) :
                 block_sparse_halo_pack_image_row(
                   &packed_row_slots[victim], K, V, n_kv_heads, kv_head_idx, D,
                   image_block_cols, incoming_row)) != 0) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          if (halo_trace) {
            boundary_row_pack_us += HAP_perf_get_time_us() - pack_begin;
          }
          ++packed_image_rows;
          packed_image_blocks += image_block_cols;

          const int next_prefetch_row = incoming_row + 1;
          if (next_prefetch_row < image_block_rows &&
              next_prefetch_row > last_image_prefetched_row) {
            if (input_packed) {
              if (block_sparse_halo_prefetch_packed_image_row(
                    K, V, kv_head_idx, head_dim, packed_token_tiles,
                    image_block_cols, next_prefetch_row) != 0) {
                BLOCK_SPARSE_HALO_CORE_FAIL();
              }
            } else {
              block_sparse_halo_l2_prefetch_image_row(K, V, n_kv_heads, kv_head_idx, head_dim,
                                                      block_k, image_block_cols, next_prefetch_row);
            }
            last_image_prefetched_row = next_prefetch_row;
            ++halo_prefetch_rows;
            const uint64_t row_bytes = (uint64_t) image_block_cols * (uint64_t) block_k *
                                       (uint64_t) head_dim * sizeof(__fp16);
            requested_k_bytes += row_bytes;
            requested_v_bytes += row_bytes;
          }

          if (row_text_reuse_enabled) {
            block_sparse_halo_text_kv_batch_t batch;
            const int64_t text_update_begin = halo_trace ? HAP_perf_get_time_us() : 0;
            if (block_sparse_halo_plan_text_kv_batch(
                  &batch, packed_row_slots, text_q_next_pending_row, incoming_row,
                  image_block_rows, false) != 0 ||
                (batch.row_count > 0 &&
                 block_sparse_halo_update_resident_text_q_from_packed_batch(
                   packed_row_slots, &batch, image_block_cols,
                   resident_text_q, resident_text_o, resident_text_state,
                   resident_text_q_blocks, text_tokens, qo_tile_size, col_vec_size,
                   &resident_text_o_bank, &halo_update_scratch,
                   (int) scratch_blk_sz_c, D, hmx_output_scales_qk,
                   hmx_output_scales_id, d_tile_vscatter_offsets) != 0)) {
              BLOCK_SPARSE_HALO_CORE_FAIL();
            }
            if (batch.row_count > 0) {
              text_q_next_pending_row = batch.next_pending_row;
              text_q_image_row_updates += batch.row_count * resident_text_q_blocks;
              text_q_image_block_updates +=
                batch.row_count * resident_text_q_blocks * image_block_cols;
              ++text_q_image_batches;
              if (halo_trace) {
                text_q_image_update_us += HAP_perf_get_time_us() - text_update_begin;
              }
            }
          }
        }
        const int first_active_row = q_image_row > 0 ? q_image_row - 1 : 0;
        const int last_active_row = smin(image_block_rows - 1, q_image_row + 1);
        for (int row = first_active_row; row <= last_active_row; ++row) {
          if (block_sparse_halo_find_packed_row(packed_row_slots, row) < 0) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
        }
        packed_last_q_row = q_image_row;
      }
    }

    if (userdma_enabled) {
      const int q_image_block = ir / block_q;
      const int q_image_row = q_image_block / image_block_cols;
      if (q_image_row != dma_last_q_row) {
        block_sparse_halo_row_schedule_t row_schedule;
        if (block_sparse_halo_schedule_q_row(&row_schedule, q_image_row, image_block_rows) != 0) {
          BLOCK_SPARSE_HALO_CORE_FAIL();
        }

        int completed_row = -1;
        int completed_packed_slot = -1;
        if (dma_pending_row >= 0) {
          if (dma_pending_row != q_image_row + 1 || dma_pending_slot != 0 ||
              dma_row_slots[0].row_id != dma_pending_row || !dma_row_slots[0].outstanding) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          completed_row = dma_pending_row;
          const int64_t wait_begin = halo_trace ? HAP_perf_get_time_us() : 0;
          if (halo_trace && dma_pending_submit_done_us > 0 && wait_begin >= dma_pending_submit_done_us) {
            dma_overlap_window_us += wait_begin - dma_pending_submit_done_us;
          }
          if (block_sparse_halo_row_dma_wait(
                dma_desc_storage, dma_desc_count,
                userdma_scope, worker_index, &dma_row_slots[0].outstanding) != DMA_SUBMIT_OK) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          if (halo_trace) {
            dma_exposed_wait_us += HAP_perf_get_time_us() - wait_begin;
          }
          dma_row_slots[0].ready = true;

          completed_packed_slot = block_sparse_halo_select_packed_row_victim(
            packed_row_slots, q_image_row - 1, 2);
          const int64_t pack_begin = halo_trace ? HAP_perf_get_time_us() : 0;
          if (completed_packed_slot < 0 ||
              (row_text_reuse_enabled &&
               packed_row_slots[completed_packed_slot].row_id >= text_q_next_pending_row) ||
              (input_packed ?
                 block_sparse_halo_adopt_packed_staged_row(
                   &packed_row_slots[completed_packed_slot], &dma_row_slots[0],
                   image_block_cols, D) :
                 block_sparse_halo_pack_staged_row(
                   &packed_row_slots[completed_packed_slot], &dma_row_slots[0],
                   image_block_cols, D)) != 0) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          if (halo_trace) {
            dma_pack_us += HAP_perf_get_time_us() - pack_begin;
          }
          ++packed_image_rows;
          packed_image_blocks += image_block_cols;
          dma_pending_slot = -1;
          dma_pending_row = -1;
          dma_pending_submit_done_us = 0;
        }

        const int first_active_row = row_schedule.first_active_row;
        const int last_active_row = first_active_row + row_schedule.active_row_count - 1;
        for (int row = first_active_row; row <= last_active_row; ++row) {
          if (block_sparse_halo_find_packed_row(packed_row_slots, row) < 0) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
        }

        const int incoming_row = row_schedule.load_row;
        if (incoming_row >= 0) {
          const int64_t submit_begin = halo_trace ? HAP_perf_get_time_us() : 0;
          if (dma_pending_row >= 0 || dma_row_slots[0].outstanding ||
              block_sparse_halo_row_dma_load_async(
                &dma_row_slots[0], dma_desc_storage, K, V, n_kv_heads, kv_head_idx,
                head_dim, image_block_cols, incoming_row, input_packed, packed_token_tiles,
                userdma_scope, worker_index) != 0) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          dma_pending_submit_done_us = halo_trace ? HAP_perf_get_time_us() : 0;
          if (halo_trace) {
            dma_submit_us += dma_pending_submit_done_us - submit_begin;
          }
          ++dma_image_query_rows_loaded;
          ++dma_async_rows_loaded;
          dma_descriptors_submitted += dma_desc_count;
          dma_requested_bytes += dma_row_transfer_bytes;
          dma_pending_slot = 0;
          dma_pending_row = incoming_row;
        }

        if (completed_row >= 0 && row_text_reuse_enabled) {
          block_sparse_halo_text_kv_batch_t batch;
          const int64_t text_update_begin = halo_trace ? HAP_perf_get_time_us() : 0;
          if (block_sparse_halo_plan_text_kv_batch(
                &batch, packed_row_slots, text_q_next_pending_row, completed_row,
                image_block_rows, false) != 0 ||
              (batch.row_count > 0 &&
               block_sparse_halo_update_resident_text_q_from_packed_batch(
                 packed_row_slots, &batch, image_block_cols,
                 resident_text_q, resident_text_o, resident_text_state,
                 resident_text_q_blocks, text_tokens, qo_tile_size, col_vec_size,
                 &resident_text_o_bank, &halo_update_scratch,
                 (int) scratch_blk_sz_c, D, hmx_output_scales_qk,
                 hmx_output_scales_id, d_tile_vscatter_offsets) != 0)) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          if (batch.row_count > 0) {
            text_q_next_pending_row = batch.next_pending_row;
            text_q_image_row_updates += batch.row_count * resident_text_q_blocks;
            text_q_image_block_updates +=
              batch.row_count * resident_text_q_blocks * image_block_cols;
            ++text_q_image_batches;
            if (halo_trace) {
              text_q_image_update_us += HAP_perf_get_time_us() - text_update_begin;
            }
          }
        }
        dma_last_q_row = q_image_row;
      }
    }

    // load [n_rows * G, D] tile of Q into VTCM
    {
      if (input_packed) {
        size_t q_offset;
        if (G != 1 || ir % HMX_FP16_TILE_N_ROWS != 0 || n_rows_g > HMX_FP16_TILE_N_ROWS ||
            block_sparse_packed_qk_tile_offset(
              &q_offset, kv_head_idx, ir / HMX_FP16_TILE_N_ROWS, 0,
              packed_token_tiles, D) != 0) {
          BLOCK_SPARSE_HALO_CORE_FAIL();
        }
        hvx_copy(q_tile, Q + q_offset,
                 (size_t) (D / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_SIZE);
      } else {
      const size_t q_ld_blk_sz_bytes = q_ld_blk_sz * qo_element_size;
      const size_t q_ld_stride_bytes = q_ld_stride * qo_element_size;

      const uint8_t *q_ld_base = ((uint8_t *) Q) + ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;

      l2fetch(q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, n_rows, 1);

      // Keep only the immediately following Q block hot. The next block is
      // consumed in the next iteration, while larger row-wide prefetches
      // would unnecessarily increase L2 pressure on the resident K/V data.
      const int current_q_block = ir / (int) blk_sz_r;
      const int next_q_block = block_sparse_halo_next_q_block(current_q_block, q_block_count);
      if (next_q_block >= 0) {
        const int next_ir = ir + (int) blk_sz_r;
        const size_t next_n_rows = smin(q_loop_tokens - next_ir, (int) blk_sz_r);
        const uint8_t *next_q_ld_base =
          ((uint8_t *) Q) + next_ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;
        l2fetch(next_q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, next_n_rows, 1);
      }

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int q_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int q_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        const HVX_Vector *pv_in0 =
          (const HVX_Vector *) (q_ld_base + q_idx0 * q_ld_stride_bytes + h_idx0 * head_dim * qo_element_size);
        const HVX_Vector *pv_in1 =
          (const HVX_Vector *) (q_ld_base + q_idx1 * q_ld_stride_bytes + h_idx1 * head_dim * qo_element_size);

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * head_dim;

        if (qo_fp32_element) {
#pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector v0    = *pv_in0++;
            const HVX_Vector v1    = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]     = v_out;
          }
        } else {
#pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
            HVX_Vector *pv_out1       = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
          }
        }
      }
      }
    }

    __fp16 *o_tile_prev = o_tile0;
    __fp16 *o_tile_curr = o_tile1;

    int dense_jc = 0;
    int index_pos = 0;
    const int index_qb = block_sparse_q_block_for_group_row(block_q, n_q_blocks, n_heads, ir, 0, q_head_base, G);
    const int32_t *index_row =
      use_host_index && index_qb >= 0 && index_qb < n_q_blocks ? block_mask + (size_t) index_qb * max_kept_blocks :
                                                                  NULL;
    block_sparse_halo_row_t halo_row = {0};
    const int halo_image_blocks = image_tokens / HMX_FP16_TILE_N_COLS;
    if (resident_text_enabled) {
      const int q_blocks_per_head = n_q_blocks / n_heads;
      if (block_sparse_halo_parse_row(&halo_row, index_row, max_kept_blocks, index_qb, q_blocks_per_head,
                                      k_blocks_per_head, halo_image_blocks) != 0) {
        BLOCK_SPARSE_HALO_CORE_FAIL();
      }
    }
    // packed-row + host-index 是融合 image-Q 更新的必要条件：
    // 当前 Q block 必须能从 block_mask 解析出完整的 halo image row 集合，
    // 并把这些 segment 连同 text segment 一起提交给稳定更新例程。
    const bool fuse_image_q =
      packed_row_reuse_enabled && use_host_index && !halo_row.is_text_query;
    if (!fuse_image_q) {
      BLOCK_SPARSE_HALO_CORE_FAIL();
    }
    block_sparse_halo_update_segment_t image_q_segments[BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS] = {0};
    int image_q_segment_count = 0;
    int image_q_image_segment_count = 0;
    int image_q_gathered_image_blocks = 0;
    int image_q_last_image_row = -1;
    bool image_q_text_seen = false;
    while (1) {
      int jc = -1;
      int index_cols = 0;
      int raw_row_id = -1;
      int raw_col_start = -1;
      int raw_block_count = 0;
      if (use_host_index) {
        bool found_index_chunk = false;
        while (index_row != NULL && index_pos < max_kept_blocks) {
          const int kb_abs = index_row[index_pos++];
          if (kb_abs < 0) {
            break;
          }
          if (kb_abs < 0 || kb_abs >= n_k_blocks || kb_abs / k_blocks_per_head != kv_head_idx) {
            continue;
          }
          const int start_block_in_head = kb_abs % k_blocks_per_head;
          int end_block_in_head = start_block_in_head + 1;
          while (index_pos < max_kept_blocks) {
            const int next_kb_abs = index_row[index_pos];
            if (next_kb_abs < 0 || next_kb_abs >= n_k_blocks || next_kb_abs / k_blocks_per_head != kv_head_idx) {
              break;
            }
            const int next_block_in_head = next_kb_abs % k_blocks_per_head;
            const int current_cols = (end_block_in_head - start_block_in_head) * block_k;
            const int merge_max_cols =
              packed_row_reuse_enabled && start_block_in_head >= dma_image_blocks && text_blk_sz_c > 0 ?
                (int) text_blk_sz_c : (int) blk_sz_c;
            if (!block_sparse_halo_can_merge_index_block(
                  packed_row_reuse_enabled,
                  start_block_in_head, next_block_in_head, dma_image_blocks,
                  image_block_cols, current_cols, merge_max_cols)) {
              break;
            }
            ++index_pos;
            ++end_block_in_head;
          }
          jc = start_block_in_head * block_k;
          index_cols = (end_block_in_head - start_block_in_head) * block_k;
          if (packed_row_reuse_enabled && start_block_in_head < dma_image_blocks) {
            raw_row_id = start_block_in_head / image_block_cols;
            raw_col_start = start_block_in_head % image_block_cols;
            raw_block_count = index_cols / block_k;
          }
          if (jc + index_cols > kv_len) {
            index_cols = kv_len - jc;
          }
          if (index_cols <= 0) {
            continue;
          }
          found_index_chunk = true;
          break;
        }
        if (!found_index_chunk) {
          break;
        }
      } else {
        if (dense_jc >= kv_len) {
          break;
        }
        jc = dense_jc;
        dense_jc += blk_sz_c;
      }
      const size_t n_real_cols = use_host_index ? (size_t) index_cols : smin(kv_len - jc, blk_sz_c);
      const bool use_resident_text = resident_text_enabled && jc >= image_tokens;
      const bool mixed_image_text = resident_text_enabled && jc < image_tokens &&
                                    jc + (int) n_real_cols > image_tokens;
      const bool use_packed_row = packed_row_reuse_enabled && raw_row_id >= 0;
      block_sparse_halo_packed_segment_t packed_segment = {};
      if (use_packed_row) {
        const int packed_slot_index = block_sparse_halo_find_packed_row(packed_row_slots, raw_row_id);
        if (packed_slot_index < 0 || raw_block_count <= 0 || raw_block_count > image_block_cols ||
            block_sparse_halo_get_packed_segment(
              &packed_segment, &packed_row_slots[packed_slot_index], raw_col_start,
              raw_block_count, image_block_cols, D) != 0) {
          BLOCK_SPARSE_HALO_CORE_FAIL();
        }
        ++image_q_direct_segment_uses;
      }
      const int resident_text_tile_start = use_resident_text ?
        (jc - image_tokens) / HMX_FP16_TILE_N_COLS : 0;
      const __fp16 *active_k_tile = k_tile;
      const __fp16 *active_v_tile = v_tile;
      int active_v_tile_stride = (int) n_tiles_per_blk_c;

      // load [Bc, D] tile of K^T into VTCM
      // K tile 会被转置/重排成 HMX RHS 需要的布局
      if (use_resident_text) {
        active_k_tile = resident_text_k +
          (size_t) resident_text_tile_start * HMX_FP16_TILE_N_COLS * head_dim;
      } else if (use_packed_row) {
        active_k_tile = packed_segment.packed_k;
      } else {
        const __fp16 *k_ld_base = K + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        const int k_ddr_cols = mixed_image_text ? image_tokens - jc : (int) n_real_cols;

        l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), k_ddr_cols, 1);
        block_sparse_pack_k_f16(k_tile, k_ld_base, (int) kv_ld_stride, k_ddr_cols, D);
        if (mixed_image_text) {
          assert(k_ddr_cols == HMX_FP16_TILE_N_COLS);
          memcpy(k_tile + (size_t) k_ddr_cols * head_dim,
                 resident_text_k,
                 (size_t) HMX_FP16_TILE_N_COLS * head_dim * sizeof(__fp16));
        }
      }

      // 预取当前 block 对应的 V
      if (!use_resident_text && !use_packed_row) {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        const int v_ddr_cols = mixed_image_text ? image_tokens - jc : (int) n_real_cols;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), v_ddr_cols, 0);
      }

      // load [Bc, D] tile of V into VTCM
      // 与 K 类似，V 也要重排成 HMX RHS 可直接消费的布局
      if (use_resident_text) {
        active_v_tile = resident_text_v + (size_t) resident_text_tile_start * HMX_FP16_TILE_N_ELMS;
        active_v_tile_stride = resident_text_tiles;
      } else if (use_packed_row) {
        active_v_tile = packed_segment.packed_v;
        active_v_tile_stride = packed_segment.v_tile_stride;
      } else {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        const int v_ddr_cols = mixed_image_text ? image_tokens - jc : (int) n_real_cols;
        block_sparse_pack_v_f16(v_tile, v_ld_base, (int) kv_ld_stride, v_ddr_cols, (int) blk_sz_c, D);
        if (mixed_image_text) {
          const int dst_tile = v_ddr_cols / HMX_FP16_TILE_N_COLS;
          const int copy_tiles = ceil_div((int) n_real_cols - v_ddr_cols, HMX_FP16_TILE_N_COLS);
          for (int d = 0; d < D / HMX_FP16_TILE_N_COLS; ++d) {
            memcpy(v_tile + ((size_t) d * n_tiles_per_blk_c + dst_tile) * HMX_FP16_TILE_N_ELMS,
                   resident_text_v + (size_t) d * resident_text_tiles * HMX_FP16_TILE_N_ELMS,
                   (size_t) copy_tiles * HMX_FP16_TILE_SIZE);
          }
        }
      }

      if (fuse_image_q) {
        if (mixed_image_text || image_q_segment_count >= BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS) {
          BLOCK_SPARSE_HALO_CORE_FAIL();
        }
        if (use_resident_text) {
          if (image_q_text_seen || jc != image_tokens || resident_text_tile_start != 0 ||
              n_real_cols != (size_t) text_tokens) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          image_q_text_seen = true;
        } else {
          if (!use_packed_row || image_q_text_seen || raw_row_id <= image_q_last_image_row ||
              raw_block_count <= 0 || block_k <= 0 || raw_block_count > INT_MAX / block_k ||
              n_real_cols != (size_t) (raw_block_count * block_k)) {
            BLOCK_SPARSE_HALO_CORE_FAIL();
          }
          image_q_last_image_row = raw_row_id;
          image_q_gathered_image_blocks += raw_block_count;
          ++image_q_image_segment_count;
        }
        image_q_segments[image_q_segment_count++] = (block_sparse_halo_update_segment_t) {
          .packed_k = active_k_tile,
          .packed_v = active_v_tile,
          .v_tile_stride = active_v_tile_stride,
          .n_real_cols = (int) n_real_cols,
        };
        continue;
      }

      block_sparse_halo_q_state_t q_state = {
        .q = q_tile,
        .o_prev = o_tile_prev,
        .o_next = o_tile_curr,
        .m = mvec_m,
        .l = mvec_l,
        .n_rows = (int) n_rows_g,
      };
      if (block_sparse_halo_update_packed_segment(
            &q_state, &halo_update_scratch, active_k_tile, active_v_tile, active_v_tile_stride,
            (int) n_real_cols, (int) scratch_blk_sz_c, D, hmx_output_scales_qk,
            hmx_output_scales_id, d_tile_vscatter_offsets) != 0) {
        BLOCK_SPARSE_HALO_CORE_FAIL();
      }
      o_tile_prev = q_state.o_prev;
      o_tile_curr = q_state.o_next;

    }

    if (fuse_image_q) {
      const int image_q_row = (ir / block_q) / image_block_cols;
      const int expected_image_segments =
        1 + (image_q_row > 0 ? 1 : 0) + (image_q_row + 1 < image_block_rows ? 1 : 0);
      const int expected_text_segments = text_tokens > 0 ? 1 : 0;
      if (image_q_row < 0 || image_q_row >= image_block_rows ||
          image_q_image_segment_count != expected_image_segments ||
          image_q_gathered_image_blocks != halo_row.image_count ||
          image_q_text_seen != (text_tokens > 0) ||
          image_q_segment_count != image_q_image_segment_count + expected_text_segments) {
        BLOCK_SPARSE_HALO_CORE_FAIL();
      }

      const int64_t image_q_normal_begin = halo_trace ? HAP_perf_get_time_us() : 0;
      if (block_sparse_halo_stable_normal_packed_segments(
            q_tile, o_tile0, &halo_update_scratch, image_q_segments, image_q_segment_count,
            (int) n_rows_g, (int) scratch_blk_sz_c, D, hmx_output_scales_qk,
            hmx_output_scales_id, d_tile_vscatter_offsets) != 0) {
        BLOCK_SPARSE_HALO_CORE_FAIL();
      }
      if (halo_trace) {
        image_q_normal_us += HAP_perf_get_time_us() - image_q_normal_begin;
      }
      ++image_q_batch_updates;
      image_q_segment_updates += image_q_segment_count;
    }

    // 最终归一化：
    //
    //   O_i = diag(l_i^{-1}) O_i
    if (!fuse_image_q) {
      const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
      const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

      HVX_Vector v_content;
      for (int i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
          v_content = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(mvec_l[i / 2]));
        } else {
          v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
      }

      hmx_unit_acquire();
      {
        hmx_set_output_scales(hmx_output_scales_id);
        for (int r = 0; r < n_row_tiles; ++r) {
          for (int c = 0; c < head_dim / 32; ++c) {
            __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_out = o_tile_curr + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;

            hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
          }
        }
      }
      hmx_unit_release();
    }

    // 将当前 Q block 对应的结果从 VTCM 写回外部 O
    const int64_t image_q_output_begin = fuse_image_q && halo_trace ? HAP_perf_get_time_us() : 0;
    {
      const size_t o_st_blk_sz_bytes = o_st_blk_sz * qo_element_size;
      const size_t o_st_stride_bytes = o_st_stride * qo_element_size;

      uint8_t *o_st_base = ((uint8_t *) O) + ir * o_st_stride_bytes + kv_head_idx * o_st_blk_sz_bytes;

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int o_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int o_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        uint8_t *row_out0 = o_st_base + o_idx0 * o_st_stride_bytes + h_idx0 * o_head_dim * qo_element_size;
        uint8_t *row_out1 = o_st_base + o_idx1 * o_st_stride_bytes + h_idx1 * o_head_dim * qo_element_size;

        HVX_Vector *pv_out0 = (HVX_Vector *) row_out0;
        HVX_Vector *pv_out1 = (HVX_Vector *) row_out1;

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *in_base = (fuse_image_q ? q_tile : o_tile_curr) +
          r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);

            const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);
            HVX_Vector v_out0 = Q6_V_lo_W(vp);
            HVX_Vector v_out1 = Q6_V_hi_W(vp);

            if (with_lse) {
              vstu_variable(row_out0 + d * 32 * sizeof(float), sizeof(v_out0), v_out0);
              if (next_row_valid) {
                vstu_variable(row_out1 + d * 32 * sizeof(float), sizeof(v_out1), v_out1);
              }
            } else {
              *pv_out0++ = v_out0;
              if (next_row_valid) {
                *pv_out1++ = v_out1;
              }
            }
          }

          if (with_lse) {
            const float lse0 = fuse_image_q ?
              flash_attn_lse_from_f16_state_scaled(mvec_m, mvec_l, r + 0, stable_lse_scale) :
              flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 0);
            memcpy(row_out0 + D * sizeof(float), &lse0, sizeof(lse0));
            if (next_row_valid) {
              const float lse1 = fuse_image_q ?
                flash_attn_lse_from_f16_state_scaled(mvec_m, mvec_l, r + 1, stable_lse_scale) :
                flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 1);
              memcpy(row_out1 + D * sizeof(float), &lse1, sizeof(lse1));
            }
          }
        } else {
          assert(!with_lse);
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const __fp16     *in_dual_tile = in_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector *pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
            const HVX_Vector *pv_in1       = pv_in0 + 16;

            const HVX_VectorPair vp = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        }
        // clang-format on
      }
    }
    if (fuse_image_q && halo_trace) {
      image_q_output_us += HAP_perf_get_time_us() - image_q_output_begin;
    }
  }
  if (row_text_reuse_enabled) {
    block_sparse_halo_text_kv_batch_t tail_batch;
    const int64_t tail_update_begin = halo_trace ? HAP_perf_get_time_us() : 0;
    if (block_sparse_halo_plan_text_kv_batch(
          &tail_batch, packed_row_slots, text_q_next_pending_row,
          image_block_rows - 1, image_block_rows, true) != 0 ||
        (tail_batch.row_count > 0 &&
         block_sparse_halo_update_resident_text_q_from_packed_batch(
           packed_row_slots, &tail_batch, image_block_cols,
           resident_text_q, resident_text_o, resident_text_state,
           resident_text_q_blocks, text_tokens, qo_tile_size, col_vec_size,
           &resident_text_o_bank, &halo_update_scratch,
           (int) scratch_blk_sz_c, D, hmx_output_scales_qk,
           hmx_output_scales_id, d_tile_vscatter_offsets) != 0)) {
      BLOCK_SPARSE_HALO_CORE_FAIL();
    }
    if (tail_batch.row_count > 0) {
      text_q_next_pending_row = tail_batch.next_pending_row;
      text_q_image_row_updates += tail_batch.row_count * resident_text_q_blocks;
      text_q_image_block_updates +=
        tail_batch.row_count * resident_text_q_blocks * image_block_cols;
      ++text_q_image_batches;
      if (halo_trace) {
        text_q_image_update_us += HAP_perf_get_time_us() - tail_update_begin;
      }
    }
    if (text_q_next_pending_row != image_block_rows) {
      BLOCK_SPARSE_HALO_CORE_FAIL();
    }

    const int64_t resident_update_begin = halo_trace ? HAP_perf_get_time_us() : 0;
    if (block_sparse_halo_update_resident_text_q(
          resident_text_q, resident_text_o, resident_text_state, resident_text_q_blocks,
          text_tokens, qo_tile_size, col_vec_size, &resident_text_o_bank,
          &halo_update_scratch, resident_text_k, resident_text_v, resident_text_tiles,
          text_tokens, (int) scratch_blk_sz_c, D, hmx_output_scales_qk,
          hmx_output_scales_id, d_tile_vscatter_offsets) != 0) {
      BLOCK_SPARSE_HALO_CORE_FAIL();
    }
    resident_text_segments = text_tokens > 0 ? 1 : 0;
    if (halo_trace) {
      text_q_resident_update_us += HAP_perf_get_time_us() - resident_update_begin;
    }

    const int64_t text_q_store_begin = halo_trace ? HAP_perf_get_time_us() : 0;
    for (int text_q_block = 0; text_q_block < resident_text_q_blocks; ++text_q_block) {
      uint8_t *state_block = resident_text_state +
        (size_t) text_q_block * 2 * col_vec_size;
      block_sparse_halo_q_state_t text_q_state = {
        .q = (__fp16 *) ((uint8_t *) resident_text_q + (size_t) text_q_block * qo_tile_size),
        .o_prev = (__fp16 *) ((uint8_t *) resident_text_o[resident_text_o_bank] +
                              (size_t) text_q_block * qo_tile_size),
        .o_next = (__fp16 *) ((uint8_t *) resident_text_o[resident_text_o_bank ^ 1] +
                              (size_t) text_q_block * qo_tile_size),
        .m = (HVX_Vector *) state_block,
        .l = (HVX_Vector *) (state_block + col_vec_size),
        .n_rows = block_sparse_halo_text_q_rows(text_tokens, text_q_block),
      };
      const int q_token = image_tokens + text_q_block * HMX_FP16_TILE_N_ROWS;
      float *text_o = (float *) O + (size_t) q_token * o_st_stride +
        (size_t) kv_head_idx * o_st_blk_sz;
      if (block_sparse_halo_store_resident_text_q(
            &text_q_state, d_tile, text_o, o_st_stride, D, with_lse,
            hmx_output_scales_id, d_tile_vscatter_offsets) != 0) {
        BLOCK_SPARSE_HALO_CORE_FAIL();
      }
    }
    if (halo_trace) {
      text_q_store_us = HAP_perf_get_time_us() - text_q_store_begin;
    }
  }
  if (userdma_enabled && dma_pending_slot >= 0) {
    const int64_t wait_begin = halo_trace ? HAP_perf_get_time_us() : 0;
    if (halo_trace && dma_pending_submit_done_us > 0 && wait_begin >= dma_pending_submit_done_us) {
      dma_overlap_window_us += wait_begin - dma_pending_submit_done_us;
    }
    if (block_sparse_halo_row_dma_wait(dma_desc_storage, dma_desc_count,
                                       userdma_scope, worker_index,
                                       &dma_row_slots[dma_pending_slot].outstanding) != DMA_SUBMIT_OK) {
      BLOCK_SPARSE_HALO_CORE_FAIL();
    }
    if (halo_trace) {
      dma_exposed_wait_us += HAP_perf_get_time_us() - wait_begin;
    }
    dma_row_slots[dma_pending_slot].ready = true;
    dma_pending_slot = -1;
    dma_pending_row = -1;
    dma_pending_submit_done_us = 0;
  }
  if (halo_run_trace != NULL && kv_head_idx == 0 && packed_row_reuse_enabled) {
    halo_run_trace->image_rows_updated = row_text_reuse_enabled ? text_q_next_pending_row : 0;
    halo_run_trace->text_q_image_row_updates = text_q_image_row_updates;
    halo_run_trace->text_q_image_block_updates = text_q_image_block_updates;
    halo_run_trace->image_q_batch_updates = image_q_batch_updates;
    halo_run_trace->image_q_segment_updates = image_q_segment_updates;
  }
  if (userdma_enabled && halo_trace) {
    const int dma_image_query_blocks_loaded = dma_image_query_rows_loaded * image_block_cols;
    const char *trace_mode = text_tokens == 0 ? "IMAGE_ONLY_USERDMA" : "HALO_USERDMA_ROWS";
    FARF(ALWAYS,
         "mode=%s worker=%d kv_head=%d packed_rows=%llu raw_staging=%llu "
         "row_stride=%llu resident_text=%llu shared_kv_scratch=%llu descriptors_ddr=%llu available=%llu",
         trace_mode, worker_index, kv_head_idx, (unsigned long long) halo_plan->packed_image_row_bytes,
         (unsigned long long) halo_plan->image_row_bytes,
         (unsigned long long) halo_plan->image_row_stride,
         (unsigned long long) halo_plan->resident_text_bytes,
         (unsigned long long) (halo_plan->packed_k_bytes + halo_plan->packed_v_bytes),
         (unsigned long long) halo_plan->descriptor_bytes,
         (unsigned long long) (vtcm_limit - vtcm));
    FARF(ALWAYS,
         "mode=%s worker=%d kv_head=%d image_query_rows=%d image_query_blocks=%d "
         "descs=%d bytes=%llu async_rows=%d packed_image_rows=%d packed_image_blocks=%d "
         "image_q_batches=%d image_q_segments=%d",
         trace_mode, worker_index, kv_head_idx, dma_image_query_rows_loaded, dma_image_query_blocks_loaded,
         dma_descriptors_submitted, (unsigned long long) dma_requested_bytes,
         dma_async_rows_loaded, packed_image_rows, packed_image_blocks,
         image_q_batch_updates, image_q_segment_updates);
    FARF(ALWAYS,
         "mode=%s worker=%d kv_head=%d resident_text_q_blocks=%d "
         "text_q_image_row_updates=%d text_q_image_block_updates=%d text_q_image_batches=%d "
         "resident_text_segments=%d "
         "resident_text_qo_state_bytes=%llu",
         trace_mode, worker_index, kv_head_idx, resident_text_q_blocks, text_q_image_row_updates,
         text_q_image_block_updates, text_q_image_batches, resident_text_segments,
         (unsigned long long) (halo_plan->resident_text_q_bytes +
                               2 * halo_plan->resident_text_o_bytes +
                               halo_plan->resident_text_state_bytes));
    FARF(ALWAYS,
         "mode=%s worker=%d kv_head=%d submit_us=%lld "
         "overlap_window_us=%lld wait_us=%lld pack_us=%lld image_q_normal_us=%lld image_q_output_us=%lld "
         "text_q_image_update_us=%lld text_q_resident_update_us=%lld",
         trace_mode, worker_index, kv_head_idx, dma_submit_us, dma_overlap_window_us,
         dma_exposed_wait_us, dma_pack_us, image_q_normal_us, image_q_output_us,
         text_q_image_update_us, text_q_resident_update_us);
  } else if (row_l2_prefetch_enabled && halo_trace) {
    const char *trace_mode = text_tokens == 0 ? "IMAGE_ONLY_L2" : "HALO_L2_PACKED_ROWS";
    FARF(ALWAYS,
         "mode=%s worker=%d kv_head=%d packed_image_rows=%d packed_image_blocks=%d "
         "packed_row_slots=%d packed_row_slot_bytes=%llu image_q_direct_segments=%d "
         "image_q_batches=%d image_q_segments=%d raw_prefetch_rows=%d "
         "requested_k_bytes=%llu requested_v_bytes=%llu",
         trace_mode, worker_index, kv_head_idx, packed_image_rows, packed_image_blocks,
         halo_plan->packed_image_row_slot_count, (unsigned long long) halo_plan->packed_image_row_stride,
         image_q_direct_segment_uses, image_q_batch_updates, image_q_segment_updates, halo_prefetch_rows,
         requested_k_bytes, requested_v_bytes);
    FARF(ALWAYS,
         "mode=%s worker=%d kv_head=%d text_q_blocks=%d "
         "text_q_image_row_updates=%d text_q_image_block_updates=%d text_q_image_batches=%d "
         "resident_text_segments=%d "
         "resident_text_kv_bytes=%llu resident_text_qo_state_bytes=%llu",
         trace_mode, worker_index, kv_head_idx, resident_text_q_blocks, text_q_image_row_updates,
         text_q_image_block_updates, text_q_image_batches, resident_text_segments,
         (unsigned long long) halo_plan->resident_text_bytes,
         (unsigned long long) (halo_plan->resident_text_q_bytes +
                               2 * halo_plan->resident_text_o_bytes +
                               halo_plan->resident_text_state_bytes));
    FARF(ALWAYS,
         "mode=%s worker=%d kv_head=%d text_kv_pack_us=%lld "
         "startup_row_pack_us=%lld boundary_row_pack_us=%lld image_q_normal_us=%lld image_q_output_us=%lld "
         "text_q_image_update_us=%lld text_q_resident_update_us=%lld "
         "text_q_init_us=%lld text_q_store_us=%lld",
         trace_mode, worker_index, kv_head_idx, text_pack_us, startup_row_pack_us, boundary_row_pack_us,
         image_q_normal_us, image_q_output_us, text_q_image_update_us, text_q_resident_update_us,
         text_q_init_us, text_q_store_us);
  }
#undef BLOCK_SPARSE_HALO_CORE_FAIL
  return 0;
}

int block_sparse_flash_attn_f16_core_halo_l2(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit,
                                             __fp16 *restrict O, const __fp16 *restrict Q,
                                             const __fp16 *restrict K, const __fp16 *restrict V,
                                             const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
                                             int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                                             int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
                                             int worker_index, const block_sparse_halo_plan_t *halo_plan,
                                             int image_tokens, int text_tokens, int image_block_rows,
                                             int image_block_cols, bool resident_text_enabled,
                                             bool halo_trace,
                                             block_sparse_halo_run_trace_t *halo_run_trace) {
  if (halo_plan == NULL || !resident_text_enabled ||
      halo_plan->packed_image_row_slot_count != BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS ||
      halo_plan->packed_image_row_stride == 0 || halo_plan->image_row_stride != 0) {
    return -1;
  }
  return block_sparse_flash_attn_f16_core_halo_rows(
    kv_head_idx, vtcm, vtcm_limit, O, Q, K, V, block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim,
    block_q, block_k, n_q_blocks, n_k_blocks, mask_format, max_kept_blocks, with_lse, worker_index, halo_plan,
    image_tokens, text_tokens, image_block_rows, image_block_cols, resident_text_enabled, false,
    BLOCK_SPARSE_HALO_USERDMA_UNKNOWN, false, halo_trace, halo_run_trace);
}

static int block_sparse_flash_attn_f16_core_halo_l2_packed(
    int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O,
    const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
    const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
    int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
    int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
    int worker_index, const block_sparse_halo_plan_t *halo_plan, int image_tokens,
    int text_tokens, int image_block_rows, int image_block_cols, bool halo_trace,
    block_sparse_halo_run_trace_t *halo_run_trace) {
  if (halo_plan == NULL || halo_plan->packed_image_row_slot_count != BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS ||
      halo_plan->packed_image_row_stride == 0 || halo_plan->image_row_stride != 0) {
    return -1;
  }
  return block_sparse_flash_attn_f16_core_halo_rows(
    kv_head_idx, vtcm, vtcm_limit, O, Q, K, V, block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim,
    block_q, block_k, n_q_blocks, n_k_blocks, mask_format, max_kept_blocks, with_lse, worker_index, halo_plan,
    image_tokens, text_tokens, image_block_rows, image_block_cols, true, false,
    BLOCK_SPARSE_HALO_USERDMA_UNKNOWN, true, halo_trace, halo_run_trace);
}

int block_sparse_flash_attn_f16_core_halo_userdma(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit,
                                             __fp16 *restrict O, const __fp16 *restrict Q,
                                             const __fp16 *restrict K, const __fp16 *restrict V,
                                             const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
                                             int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                                             int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
                                             int worker_index, const block_sparse_halo_plan_t *halo_plan,
                                             int image_tokens, int text_tokens, int image_block_rows,
                                             int image_block_cols, int userdma_scope, bool halo_trace,
                                             block_sparse_halo_run_trace_t *halo_run_trace) {
  if (halo_plan == NULL || userdma_scope != BLOCK_SPARSE_HALO_USERDMA_PER_WORKER ||
      halo_plan->image_row_stride == 0 || halo_plan->image_row_bytes == 0) {
    return -1;
  }
  return block_sparse_flash_attn_f16_core_halo_rows(
    kv_head_idx, vtcm, vtcm_limit, O, Q, K, V, block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim,
    block_q, block_k, n_q_blocks, n_k_blocks, mask_format, max_kept_blocks, with_lse, worker_index, halo_plan,
    image_tokens, text_tokens, image_block_rows, image_block_cols, true, true, userdma_scope, false,
    halo_trace, halo_run_trace);
}

static int block_sparse_flash_attn_f16_core_halo_userdma_packed(
    int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O,
    const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
    const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
    int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
    int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
    int worker_index, const block_sparse_halo_plan_t *halo_plan, int image_tokens,
    int text_tokens, int image_block_rows, int image_block_cols, int userdma_scope,
    bool halo_trace, block_sparse_halo_run_trace_t *halo_run_trace) {
  if (halo_plan == NULL || userdma_scope != BLOCK_SPARSE_HALO_USERDMA_PER_WORKER ||
      halo_plan->image_row_stride == 0 || halo_plan->image_row_bytes == 0) {
    return -1;
  }
  return block_sparse_flash_attn_f16_core_halo_rows(
    kv_head_idx, vtcm, vtcm_limit, O, Q, K, V, block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim,
    block_q, block_k, n_q_blocks, n_k_blocks, mask_format, max_kept_blocks, with_lse, worker_index, halo_plan,
    image_tokens, text_tokens, image_block_rows, image_block_cols, true, true, userdma_scope, true,
    halo_trace, halo_run_trace);
}

int block_sparse_flash_attn_f16_core_full_k(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit,
                                             __fp16 *restrict O, const __fp16 *restrict Q,
                                             const __fp16 *restrict K, const __fp16 *restrict V,
                                             const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
                                             int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                                             int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
                                             int worker_index) {
  (void) worker_index;

  // 一个 KV head 对应 G 个 Q heads
  // 当前函数只处理 kv_head_idx 对应的 K/V head
  // 以及 Q/O 中 [kv_head_idx * G, (kv_head_idx + 1) * G) 这 G 个 heads
  const int G = n_heads / n_kv_heads;
  const int D = head_dim;
  const int q_head_base = kv_head_idx * G;
  const int k_blocks_per_head = n_k_blocks / n_kv_heads;
  const bool use_host_index =
    mask_format == 1 && max_kept_blocks > 0 && G == 1 && block_q == HMX_FP16_TILE_N_ROWS &&
    block_k == HMX_FP16_TILE_N_COLS;

  // 当前实现认为 Q/O 外部存储是 fp32
  // 虽然函数参数类型是 __fp16*，
  // 但实际通过 uint8_t* + qo_element_size 访问。
  // qo_fp32_element = true 时：
  //   Q/O 外部按 float 存储；
  //   读入 VTCM 后转成 fp16 HMX tile；
  //   写回 O 时再从 fp16 转回 float
  const bool   qo_fp32_element = true;
  const size_t qo_element_size = qo_fp32_element ? sizeof(float) : sizeof(__fp16);

  // 是否用 fp32 路径算 exp
  // 当前 false，表示使用 fp16/qf16 exp2 路径
  const bool use_fp32_exp = false;

  block_sparse_fa_plan_t plan;
  if (!use_host_index ||
      !block_sparse_fa_find_mode_plan(&plan, BLOCK_SPARSE_FA_FULL_K, G, head_dim, kv_len, vtcm_limit - vtcm)) {
    return -1;
  }
  const size_t blk_sz_r = HMX_FP16_TILE_N_ROWS;
  const size_t blk_sz_c = plan.blk_sz_c;
  assert(blk_sz_c % 64 == 0);

  // 当前 Q block 展开 G 个 Q heads 后的行数是 G * Br
  // 为适配 HMX FP16 tile 的行数要求，按 HMX_FP16_TILE_N_ROWS 对齐
  // g_br 表示对齐后的 Br'
  const size_t g_br = align_up(G * blk_sz_r, HMX_FP16_TILE_N_ROWS);

  // 当前 Q block 在行方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_r = g_br / HMX_FP16_TILE_N_ROWS;

  // 当前 K/V block 在列方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;

  // q_tile / o_tile:
  //   逻辑形状 [Br', D]
  //
  // k_tile / v_tile:
  //   逻辑形状 [Bc, D]
  //
  // s_tile / p_tile:
  //   逻辑形状 [Br', Bc]
  //   s_tile 保存 QK^T score；
  //   p_tile 保存 softmax numerator P
  //
  // d_tile:
  //   逻辑形状 [Br', Br']
  //   保存对角矩阵，用于：
  //     1. alpha * O_old
  //     2. final O / l
  //
  // col_vec_size:
  //   保存每一行的 m/l/rowmax/rowsum
  //
  // row_vec_size:
  //   softmax 时把 HMX tile layout 中的 S/P 解码成连续 row buffer
  const size_t qo_tile_size     = align_up(g_br * head_dim * sizeof(__fp16), 4096);
  const size_t full_k_tile_size = (size_t) ceil_div(kv_len, HMX_FP16_TILE_N_COLS) * (head_dim / 32) *
                                  HMX_FP16_TILE_SIZE;
  const size_t v_tile_size      = align_up(blk_sz_c * head_dim * sizeof(__fp16), 4096);
  const size_t core_tile_size   = align_up(g_br * blk_sz_c * sizeof(__fp16), 4096);
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);
  const size_t row_vec_size   = align_up(blk_sz_c * sizeof(__fp16), 256);

  // 原始 K/V 布局：
  //   K/V: [kv_len, n_kv_heads, head_dim]
  //
  // kv_ld_blk_sz:
  //   当前 kv_head_idx 对应的一个 head 的大小
  //
  // kv_ld_stride:
  //   K/V 中一个 token 的完整 stride，即所有 KV heads 的大小
  const size_t kv_ld_blk_sz   = head_dim;
  const size_t kv_ld_stride   = n_kv_heads * head_dim;

  // 原始 Q/O 布局：
  //   Q: [qo_len, n_heads, head_dim]
  //   O: [qo_len, n_heads, head_dim + (with_lse ? 1 : 0)]
  //
  // q_ld_blk_sz/o_st_blk_sz:
  //   当前 kv_head_idx 对应的 G 个 Q heads 的总大小
  //
  // q_ld_stride/o_st_stride:
  //   Q/O 中一个 token 的完整 stride，即所有 Q heads 的大小
  const size_t q_ld_blk_sz = G * head_dim;
  const size_t q_ld_stride = n_heads * head_dim;
  const size_t o_head_dim  = head_dim + (with_lse ? 1 : 0);
  const size_t o_st_blk_sz = G * o_head_dim;
  const size_t o_st_stride = n_heads * o_head_dim;

  // 从 vtcm 开始顺序分配所有临时 buffer
  uint8_t *vtcm_cur = vtcm;
  __fp16  *q_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile0  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile1  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);

  __fp16 *packed_k = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, full_k_tile_size);
  __fp16 *v_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, v_tile_size);

  __fp16 *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *p_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  // D tile 保存对角矩阵
  // 用于 HMX 计算 diag(scale) @ O
  __fp16 *d_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_size);

  // online softmax 状态：
  //
  // mvec_m:
  //   m_i，当前已经处理过的 K/V blocks 的 row max
  //
  // mvec_l:
  //   l_i，softmax denominator
  //
  // mvec_s_rowmax:
  //   当前 K/V block 中每一行 S 的 rowmax
  //   后面会复用为 exp(m_old - m_new)
  //
  // mvec_p_rowsum:
  //   当前 K/V block 中每一行 P 的 rowsum
  HVX_Vector *mvec_m        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_s_rowmax = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  // softmax 时一次处理两行 S
  // 从 HMX tile layout 解码出来后，放到这两个 row buffer 中
  HVX_Vector *row_buffer0 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  HVX_Vector *row_buffer1 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);

  // HMX output scale buffer
  //
  // hmx_output_scales_id:
  //   scale = 1，用于普通 HMX matmul 输出
  //
  // hmx_output_scales_qk:
  //   scale = log2(e) / sqrt(D)，用于 QK^T
  uint8_t *hmx_output_scales_id = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  uint8_t *hmx_output_scales_qk = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  assert(vtcm_cur <= vtcm_limit);

  // qk_scale 用于 attention score：
  //
  //   QK^T / sqrt(D)
  //
  // 但后面 softmax 用的是 exp2，而不是 exp
  // 因此这里提前乘上 log2(e)
  float  qk_scale    = 1.0f / sqrtf(head_dim) * 1.44269504f;
  __fp16 qk_scale_hf = (__fp16) qk_scale;

  // 初始化 HMX 输出 scale
  // 0x3c00 是 fp16 1.0
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));

  // QK^T 输出时直接乘 qk_scale_hf
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));

  // d_tile_vscatter_offsets 用于生成对角矩阵 D tile
  // offsets 指向 HMX tile layout 中对角线元素的位置
  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }

  hvx_fill_uh(packed_k, 0, full_k_tile_size);
  {
    const __fp16 *k_ld_base = K + kv_head_idx * kv_ld_blk_sz;

    l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), kv_len, 1);
    block_sparse_pack_k_f16(packed_k, k_ld_base, (int) kv_ld_stride, kv_len, D);
  }

  // 外层循环：按 Q/O token 方向分块
  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows        = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g      = n_rows * G;
    const size_t n_row_tiles   = ceil_div(n_rows_g, HMX_FP16_TILE_N_ROWS);
    const size_t n_row_vec_cnt = ceil_div(n_rows_g, 64);

    // load [n_rows * G, D] tile of Q into VTCM
    {
      const size_t q_ld_blk_sz_bytes = q_ld_blk_sz * qo_element_size;
      const size_t q_ld_stride_bytes = q_ld_stride * qo_element_size;

      const uint8_t *q_ld_base = ((uint8_t *) Q) + ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;

      l2fetch(q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, n_rows, 1);

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int q_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int q_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        const HVX_Vector *pv_in0 =
          (const HVX_Vector *) (q_ld_base + q_idx0 * q_ld_stride_bytes + h_idx0 * head_dim * qo_element_size);
        const HVX_Vector *pv_in1 =
          (const HVX_Vector *) (q_ld_base + q_idx1 * q_ld_stride_bytes + h_idx1 * head_dim * qo_element_size);

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * head_dim;

        if (qo_fp32_element) {
#pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector v0    = *pv_in0++;
            const HVX_Vector v1    = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]     = v_out;
          }
        } else {
#pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
            HVX_Vector *pv_out1       = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
          }
        }
      }
    }

    // 初始化 online softmax 状态
    hvx_fill_uh(mvec_m, 0xfbff, col_vec_size);
    hvx_fill_uh(mvec_l, 0, col_vec_size);

    __fp16 *o_tile_prev = o_tile0;
    __fp16 *o_tile_curr = o_tile1;

    hvx_fill_uh(o_tile_prev, 0, qo_tile_size);
    hvx_fill_uh(d_tile, 0, d_tile_size);

    int dense_jc = 0;
    int index_pos = 0;
    const int index_qb = block_sparse_q_block_for_group_row(block_q, n_q_blocks, n_heads, ir, 0, q_head_base, G);
    const int32_t *index_row =
      use_host_index && index_qb >= 0 && index_qb < n_q_blocks ? block_mask + (size_t) index_qb * max_kept_blocks :
                                                                  NULL;
    while (1) {
      int jc = -1;
      int index_cols = 0;
      if (use_host_index) {
        bool found_index_chunk = false;
        while (index_row != NULL && index_pos < max_kept_blocks) {
          const int kb_abs = index_row[index_pos++];
          if (kb_abs < 0) {
            break;
          }
          if (kb_abs < 0 || kb_abs >= n_k_blocks || kb_abs / k_blocks_per_head != kv_head_idx) {
            continue;
          }
          const int start_block_in_head = kb_abs % k_blocks_per_head;
          int end_block_in_head = start_block_in_head + 1;
          while (index_pos < max_kept_blocks) {
            const int next_kb_abs = index_row[index_pos];
            if (next_kb_abs < 0 || next_kb_abs >= n_k_blocks || next_kb_abs / k_blocks_per_head != kv_head_idx) {
              break;
            }
            const int next_block_in_head = next_kb_abs % k_blocks_per_head;
            if (next_block_in_head != end_block_in_head) {
              break;
            }
            const int next_cols = (next_block_in_head + 1 - start_block_in_head) * block_k;
            if (next_cols > (int) blk_sz_c) {
              break;
            }
            ++index_pos;
            ++end_block_in_head;
          }
          jc = start_block_in_head * block_k;
          index_cols = (end_block_in_head - start_block_in_head) * block_k;
          if (jc + index_cols > kv_len) {
            index_cols = kv_len - jc;
          }
          if (index_cols <= 0) {
            continue;
          }
          found_index_chunk = true;
          break;
        }
        if (!found_index_chunk) {
          break;
        }
      } else {
        if (dense_jc >= kv_len) {
          break;
        }
        jc = dense_jc;
        dense_jc += blk_sz_c;
      }
      const size_t n_real_cols = use_host_index ? (size_t) index_cols : smin(kv_len - jc, blk_sz_c);
      const size_t n_cols      = n_real_cols;
      const size_t n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

      // 预取当前 block 对应的 V
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_real_cols, 0);
      }

      // 计算块级 score：
      //
      //   dot(Q[Br', D], K[Bc, D]) ==> S[Br', Bc]
      {
        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_qk);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < n_col_tiles; ++c) {
              const __fp16 *row_tiles = q_tile + r * HMX_FP16_TILE_N_ROWS * head_dim;
              const __fp16 *col_tiles = packed_k + (jc / HMX_FP16_TILE_N_COLS + c) * HMX_FP16_TILE_N_COLS * head_dim;
              __fp16 *out_tile = s_tile + (r * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS;

              hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
            }
          }
        }
        hmx_unit_release();
      }

      // safe softmax：
      //   1. 从 S tile 解码出连续行
      //   2. 对 HMX 对齐产生的 padding 列做 -inf mask
      //   3. 计算当前 block 的 rowmax / rowsum
      //   4. 生成 P tile
      //
      // host-index mode only visits kept K ranges. Dense-mask fallback computes full dense chunks.
      {
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

        for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
          HVX_Vector v_s_rowmax_local = v_neg_inf;
          HVX_Vector v_p_rowsum_local = Q6_V_vzero();

          for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            int r = r_vec_idx * 64 + r_vec_off;
            if (r >= align_up(n_rows_g, 2)) {
              break;
            }

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            __fp16 *s_ld_base = s_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;
            __fp16 *p_st_base = p_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;

            HVX_Vector *pv_row_buf0 = row_buffer0;
            HVX_Vector *pv_row_buf1 = row_buffer1;
            for (int c = 0; c < n_cols; c += 64) {
              const __fp16     *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              const HVX_Vector *pv_s_in0     = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
              const HVX_Vector *pv_s_in1     = pv_s_in0 + 16;

              HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
              *pv_row_buf0++               = Q6_V_lo_W(vp_s_dual_row);
              *pv_row_buf1++               = Q6_V_hi_W(vp_s_dual_row);
            }

            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            for (int c = 0; c < n_cols; c += 64) {
              const size_t ne = smin((int) n_real_cols - c, 64);
              HVX_VectorPred q_mask_keep0 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              HVX_VectorPred q_mask_keep1 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

              HVX_Vector v_s_row0 = Q6_V_vmux_QVV(q_mask_keep0, row_buffer0[c / 64], v_neg_inf);
              HVX_Vector v_s_row1 = Q6_V_vmux_QVV(q_mask_keep1, row_buffer1[c / 64], v_neg_inf);

              // 将被 mask 掉的位置直接改写成 -inf
              // 后续 rowmax / exp / rowsum 都基于这个结果继续
              row_buffer0[c / 64] = v_s_row0;
              row_buffer1[c / 64] = v_s_row1;

              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
            }

#pragma unroll
            for (int s = 64; s >= 2; s >>= 1) {
              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
            }

            HVX_Vector v_s_rowmax_pack2     = Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));
            HVX_Vector v_s_rowmax_pack2_rot = Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_s_rowmax_local_rot =
              Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));
            v_s_rowmax_local =
              Q6_V_vlalign_VVR(v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r_vec_off * sizeof(__fp16));

            HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(mvec_m[r_vec_idx], v_s_rowmax_local);

            HVX_Vector v_m_lut  = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));
            HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
            HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));

            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;
            HVX_Vector       v_p_rowsum1 = v_zero;

            for (int c = 0; c < n_cols; c += 64) {
              const int ne = smax(0, smin((int) n_real_cols - c, 64));
              HVX_VectorPred q_mask_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              HVX_Vector v_p_row0_hf, v_p_row1_hf;

              HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0);
              HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1);

              if (use_fp32_exp) {
                HVX_VectorPair vp_s_minus_m0_sf = hvx_my_vqf16_to_wsf(v_s_minus_m0);
                HVX_VectorPair vp_s_minus_m1_sf = hvx_my_vqf16_to_wsf(v_s_minus_m1);

                HVX_Vector v_p_row00_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row01_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row10_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m1_sf));
                HVX_Vector v_p_row11_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m1_sf));

                v_p_row0_hf = hvx_my_wsf_to_vhf(v_p_row01_sf, v_p_row00_sf);
                v_p_row1_hf = hvx_my_wsf_to_vhf(v_p_row11_sf, v_p_row10_sf);
              } else {
                v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
                v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);
              }

              v_p_row0_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row0_hf, v_zero);
              v_p_row1_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row1_hf, v_zero);

              __fp16     *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              HVX_Vector *pv_p_out0     = ((HVX_Vector *) out_dual_tile) + r1 / 2;
              HVX_Vector *pv_p_out1     = pv_p_out0 + 16;

              HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
              *pv_p_out0                   = Q6_V_lo_W(vp_p_dual_row);
              *pv_p_out1                   = Q6_V_hi_W(vp_p_dual_row);

              HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0_hf);
              HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1_hf);

              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
            }

#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            }
            HVX_Vector v_p_rowsum_pack2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));

            HVX_Vector v_p_rowsum_pack2_rot = Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_p_rowsum_local_rot = Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
            v_p_rowsum_local =
              Q6_V_vlalign_VVR(v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r_vec_off * sizeof(__fp16));
          }

          mvec_s_rowmax[r_vec_idx] = v_s_rowmax_local;
          mvec_p_rowsum[r_vec_idx] = v_p_rowsum_local;
        }
      }

      // load [Bc, D] tile of V into VTCM
      // 与 K 类似，V 也要重排成 HMX RHS 可直接消费的布局
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        block_sparse_pack_v_f16(v_tile, v_ld_base, (int) kv_ld_stride, (int) n_real_cols, (int) blk_sz_c, D);
      }

      // 复用 mvec_s_rowmax 这块 buffer，
      // 后续存 exp(m_i^{j-1} - m_i^j)
      HVX_Vector *mvec_exp_m_diff = mvec_s_rowmax;

      // 更新 online softmax 状态：
      //
      //   m_i^j = max(m_i^{j-1}, rowmax(S_i^j))
      //   l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
      {
        for (int i = 0; i < n_row_vec_cnt; ++i) {
          HVX_Vector v_m_prev = mvec_m[i];
          HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, mvec_s_rowmax[i]);
          HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);

          HVX_Vector v_exp_m_diff_hf = hvx_my_exp2_vhf_vqf16(v_m_diff);

          HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(mvec_l[i], v_exp_m_diff_hf);
          v_l_curr            = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, mvec_p_rowsum[i]);

          mvec_m[i] = v_m_curr;
          mvec_l[i] = v_l_curr;

          mvec_exp_m_diff[i] = v_exp_m_diff_hf;
        }
      }

      // 更新当前输出：
      //
      //   O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + P_i^j V_j
      {
        const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
        const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        for (int i = 0; i < n_row_tiles; ++i) {
          const HVX_Vector v_content = Q6_V_vror_VR(mvec_exp_m_diff[i / 2], (i % 2) * 64);

          __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
          Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_id);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < head_dim / 32; ++c) {
              __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
              __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
              hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

              __fp16 *p_tile_in = p_tile + (r * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;
              __fp16 *v_tile_in = v_tile + (c * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;
              for (int k = 0; k < n_col_tiles; ++k) {
                int    offset  = k * HMX_FP16_TILE_N_ELMS;
                size_t n_tiles = 1;
                hmx_load_tiles_fp16(p_tile_in + offset, v_tile_in + offset, n_tiles);
              }

              __fp16 *o_tile_out = o_tile_curr + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
              hmx_consume_accumulator_fp16(o_tile_out);
            }
          }
        }
        hmx_unit_release();

        swap_ptr(&o_tile_curr, &o_tile_prev);
      }
    }

    // 最终归一化：
    //
    //   O_i = diag(l_i^{-1}) O_i
    {
      const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
      const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

      HVX_Vector v_content;
      for (int i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
          v_content = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(mvec_l[i / 2]));
        } else {
          v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
      }

      hmx_unit_acquire();
      {
        hmx_set_output_scales(hmx_output_scales_id);
        for (int r = 0; r < n_row_tiles; ++r) {
          for (int c = 0; c < head_dim / 32; ++c) {
            __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_out = o_tile_curr + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;

            hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
          }
        }
      }
      hmx_unit_release();
    }

    // 将当前 Q block 对应的结果从 VTCM 写回外部 O
    {
      const size_t o_st_blk_sz_bytes = o_st_blk_sz * qo_element_size;
      const size_t o_st_stride_bytes = o_st_stride * qo_element_size;

      uint8_t *o_st_base = ((uint8_t *) O) + ir * o_st_stride_bytes + kv_head_idx * o_st_blk_sz_bytes;

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int o_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int o_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        uint8_t *row_out0 = o_st_base + o_idx0 * o_st_stride_bytes + h_idx0 * o_head_dim * qo_element_size;
        uint8_t *row_out1 = o_st_base + o_idx1 * o_st_stride_bytes + h_idx1 * o_head_dim * qo_element_size;

        HVX_Vector *pv_out0 = (HVX_Vector *) row_out0;
        HVX_Vector *pv_out1 = (HVX_Vector *) row_out1;

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *in_base = o_tile_curr + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);

            const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);
            const HVX_Vector v_out0 = Q6_V_lo_W(vp);
            const HVX_Vector v_out1 = Q6_V_hi_W(vp);

            if (with_lse) {
              vstu_variable(row_out0 + d * 32 * sizeof(float), sizeof(v_out0), v_out0);
              if (next_row_valid) {
                vstu_variable(row_out1 + d * 32 * sizeof(float), sizeof(v_out1), v_out1);
              }
            } else {
              *pv_out0++ = v_out0;
              if (next_row_valid) {
                *pv_out1++ = v_out1;
              }
            }
          }

          if (with_lse) {
            const float lse0 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 0);
            memcpy(row_out0 + D * sizeof(float), &lse0, sizeof(lse0));
            if (next_row_valid) {
              const float lse1 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 1);
              memcpy(row_out1 + D * sizeof(float), &lse1, sizeof(lse1));
            }
          }
        } else {
          assert(!with_lse);
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const __fp16     *in_dual_tile = in_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector *pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
            const HVX_Vector *pv_in1       = pv_in0 + 16;

            const HVX_VectorPair vp = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        }
        // clang-format on
      }
    }
  }
  return 0;
}

int block_sparse_flash_attn_f16_core_full_kv(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit,
                                             __fp16 *restrict O, const __fp16 *restrict Q,
                                             const __fp16 *restrict K, const __fp16 *restrict V,
                                             const int32_t *restrict block_mask, int qo_len, int kv_len, int n_heads,
                                             int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                                             int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse,
                                             int worker_index) {
  (void) worker_index;

  // 一个 KV head 对应 G 个 Q heads
  // 当前函数只处理 kv_head_idx 对应的 K/V head
  // 以及 Q/O 中 [kv_head_idx * G, (kv_head_idx + 1) * G) 这 G 个 heads
  const int G = n_heads / n_kv_heads;
  const int D = head_dim;
  const int q_head_base = kv_head_idx * G;
  const int k_blocks_per_head = n_k_blocks / n_kv_heads;
  const bool use_host_index =
    mask_format == 1 && max_kept_blocks > 0 && G == 1 && block_q == HMX_FP16_TILE_N_ROWS &&
    block_k == HMX_FP16_TILE_N_COLS;

  // 当前实现认为 Q/O 外部存储是 fp32
  // 虽然函数参数类型是 __fp16*，
  // 但实际通过 uint8_t* + qo_element_size 访问。
  // qo_fp32_element = true 时：
  //   Q/O 外部按 float 存储；
  //   读入 VTCM 后转成 fp16 HMX tile；
  //   写回 O 时再从 fp16 转回 float
  const bool   qo_fp32_element = true;
  const size_t qo_element_size = qo_fp32_element ? sizeof(float) : sizeof(__fp16);

  // 是否用 fp32 路径算 exp
  // 当前 false，表示使用 fp16/qf16 exp2 路径
  const bool use_fp32_exp = false;

  block_sparse_fa_plan_t plan;
  if (!use_host_index ||
      !block_sparse_fa_find_mode_plan(&plan, BLOCK_SPARSE_FA_FULL_KV, G, head_dim, kv_len, vtcm_limit - vtcm)) {
    return -1;
  }
  const size_t blk_sz_r = HMX_FP16_TILE_N_ROWS;
  const size_t blk_sz_c = plan.blk_sz_c;
  assert(blk_sz_c % 64 == 0);

  // 当前 Q block 展开 G 个 Q heads 后的行数是 G * Br
  // 为适配 HMX FP16 tile 的行数要求，按 HMX_FP16_TILE_N_ROWS 对齐
  // g_br 表示对齐后的 Br'
  const size_t g_br = align_up(G * blk_sz_r, HMX_FP16_TILE_N_ROWS);

  // 当前 Q block 在行方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_r = g_br / HMX_FP16_TILE_N_ROWS;

  // 当前 K/V block 在列方向需要多少个 HMX tile
  const size_t n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;

  // q_tile / o_tile:
  //   逻辑形状 [Br', D]
  //
  // k_tile / v_tile:
  //   逻辑形状 [Bc, D]
  //
  // s_tile / p_tile:
  //   逻辑形状 [Br', Bc]
  //   s_tile 保存 QK^T score；
  //   p_tile 保存 softmax numerator P
  //
  // d_tile:
  //   逻辑形状 [Br', Br']
  //   保存对角矩阵，用于：
  //     1. alpha * O_old
  //     2. final O / l
  //
  // col_vec_size:
  //   保存每一行的 m/l/rowmax/rowsum
  //
  // row_vec_size:
  //   softmax 时把 HMX tile layout 中的 S/P 解码成连续 row buffer
  const size_t qo_tile_size      = align_up(g_br * head_dim * sizeof(__fp16), 4096);
  const size_t full_kv_tile_size = (size_t) ceil_div(kv_len, HMX_FP16_TILE_N_COLS) * (head_dim / 32) *
                                   HMX_FP16_TILE_SIZE;
  const size_t core_tile_size    = align_up(g_br * blk_sz_c * sizeof(__fp16), 4096);
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);
  const size_t row_vec_size   = align_up(blk_sz_c * sizeof(__fp16), 256);

  // 原始 K/V 布局：
  //   K/V: [kv_len, n_kv_heads, head_dim]
  //
  // kv_ld_blk_sz:
  //   当前 kv_head_idx 对应的一个 head 的大小
  //
  // kv_ld_stride:
  //   K/V 中一个 token 的完整 stride，即所有 KV heads 的大小
  const size_t kv_ld_blk_sz   = head_dim;
  const size_t kv_ld_stride   = n_kv_heads * head_dim;

  // 原始 Q/O 布局：
  //   Q: [qo_len, n_heads, head_dim]
  //   O: [qo_len, n_heads, head_dim + (with_lse ? 1 : 0)]
  //
  // q_ld_blk_sz/o_st_blk_sz:
  //   当前 kv_head_idx 对应的 G 个 Q heads 的总大小
  //
  // q_ld_stride/o_st_stride:
  //   Q/O 中一个 token 的完整 stride，即所有 Q heads 的大小
  const size_t q_ld_blk_sz = G * head_dim;
  const size_t q_ld_stride = n_heads * head_dim;
  const size_t o_head_dim  = head_dim + (with_lse ? 1 : 0);
  const size_t o_st_blk_sz = G * o_head_dim;
  const size_t o_st_stride = n_heads * o_head_dim;

  // 从 vtcm 开始顺序分配所有临时 buffer
  uint8_t *vtcm_cur = vtcm;
  __fp16  *q_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile0  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile1  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);

  __fp16 *packed_k = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, full_kv_tile_size);
  __fp16 *packed_v = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, full_kv_tile_size);

  __fp16 *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *p_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  // D tile 保存对角矩阵
  // 用于 HMX 计算 diag(scale) @ O
  __fp16 *d_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_size);

  // online softmax 状态：
  //
  // mvec_m:
  //   m_i，当前已经处理过的 K/V blocks 的 row max
  //
  // mvec_l:
  //   l_i，softmax denominator
  //
  // mvec_s_rowmax:
  //   当前 K/V block 中每一行 S 的 rowmax
  //   后面会复用为 exp(m_old - m_new)
  //
  // mvec_p_rowsum:
  //   当前 K/V block 中每一行 P 的 rowsum
  HVX_Vector *mvec_m        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_s_rowmax = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  // softmax 时一次处理两行 S
  // 从 HMX tile layout 解码出来后，放到这两个 row buffer 中
  HVX_Vector *row_buffer0 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  HVX_Vector *row_buffer1 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);

  // HMX output scale buffer
  //
  // hmx_output_scales_id:
  //   scale = 1，用于普通 HMX matmul 输出
  //
  // hmx_output_scales_qk:
  //   scale = log2(e) / sqrt(D)，用于 QK^T
  uint8_t *hmx_output_scales_id = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  uint8_t *hmx_output_scales_qk = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  assert(vtcm_cur <= vtcm_limit);

  // qk_scale 用于 attention score：
  //
  //   QK^T / sqrt(D)
  //
  // 但后面 softmax 用的是 exp2，而不是 exp
  // 因此这里提前乘上 log2(e)
  float  qk_scale    = 1.0f / sqrtf(head_dim) * 1.44269504f;
  __fp16 qk_scale_hf = (__fp16) qk_scale;

  // 初始化 HMX 输出 scale
  // 0x3c00 是 fp16 1.0
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));

  // QK^T 输出时直接乘 qk_scale_hf
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));

  // d_tile_vscatter_offsets 用于生成对角矩阵 D tile
  // offsets 指向 HMX tile layout 中对角线元素的位置
  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }

  hvx_fill_uh(packed_k, 0, full_kv_tile_size);
  hvx_fill_uh(packed_v, 0, full_kv_tile_size);
  {
    const __fp16 *k_ld_base = K + kv_head_idx * kv_ld_blk_sz;
    const __fp16 *v_ld_base = V + kv_head_idx * kv_ld_blk_sz;
    const int padded_kv_len = ceil_div(kv_len, HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_COLS;

    l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), kv_len, 1);
    block_sparse_pack_k_f16(packed_k, k_ld_base, (int) kv_ld_stride, kv_len, D);

    l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), kv_len, 1);
    block_sparse_pack_v_f16(packed_v, v_ld_base, (int) kv_ld_stride, kv_len, padded_kv_len, D);
  }
  // 外层循环：按 Q/O token 方向分块
  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows        = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g      = n_rows * G;
    const size_t n_row_tiles   = ceil_div(n_rows_g, HMX_FP16_TILE_N_ROWS);
    const size_t n_row_vec_cnt = ceil_div(n_rows_g, 64);

    // load [n_rows * G, D] tile of Q into VTCM
    {
      const size_t q_ld_blk_sz_bytes = q_ld_blk_sz * qo_element_size;
      const size_t q_ld_stride_bytes = q_ld_stride * qo_element_size;

      const uint8_t *q_ld_base = ((uint8_t *) Q) + ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;

      l2fetch(q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, n_rows, 1);

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int q_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int q_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        const HVX_Vector *pv_in0 =
          (const HVX_Vector *) (q_ld_base + q_idx0 * q_ld_stride_bytes + h_idx0 * head_dim * qo_element_size);
        const HVX_Vector *pv_in1 =
          (const HVX_Vector *) (q_ld_base + q_idx1 * q_ld_stride_bytes + h_idx1 * head_dim * qo_element_size);

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * head_dim;

        if (qo_fp32_element) {
#pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector v0    = *pv_in0++;
            const HVX_Vector v1    = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]     = v_out;
          }
        } else {
#pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
            HVX_Vector *pv_out1       = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
          }
        }
      }
    }
    // 初始化 online softmax 状态
    hvx_fill_uh(mvec_m, 0xfbff, col_vec_size);
    hvx_fill_uh(mvec_l, 0, col_vec_size);

    __fp16 *o_tile_prev = o_tile0;
    __fp16 *o_tile_curr = o_tile1;

    hvx_fill_uh(o_tile_prev, 0, qo_tile_size);
    hvx_fill_uh(d_tile, 0, d_tile_size);

    int dense_jc = 0;
    int index_pos = 0;
    const int index_qb = block_sparse_q_block_for_group_row(block_q, n_q_blocks, n_heads, ir, 0, q_head_base, G);
    const int32_t *index_row =
      use_host_index && index_qb >= 0 && index_qb < n_q_blocks ? block_mask + (size_t) index_qb * max_kept_blocks :
                                                                  NULL;
    while (1) {
      int jc = -1;
      int index_cols = 0;
      if (use_host_index) {
        bool found_index_chunk = false;
        while (index_row != NULL && index_pos < max_kept_blocks) {
          const int kb_abs = index_row[index_pos++];
          if (kb_abs < 0) {
            break;
          }
          if (kb_abs < 0 || kb_abs >= n_k_blocks || kb_abs / k_blocks_per_head != kv_head_idx) {
            continue;
          }
          const int start_block_in_head = kb_abs % k_blocks_per_head;
          int end_block_in_head = start_block_in_head + 1;
          while (index_pos < max_kept_blocks) {
            const int next_kb_abs = index_row[index_pos];
            if (next_kb_abs < 0 || next_kb_abs >= n_k_blocks || next_kb_abs / k_blocks_per_head != kv_head_idx) {
              break;
            }
            const int next_block_in_head = next_kb_abs % k_blocks_per_head;
            if (next_block_in_head != end_block_in_head) {
              break;
            }
            const int next_cols = (next_block_in_head + 1 - start_block_in_head) * block_k;
            if (next_cols > (int) blk_sz_c) {
              break;
            }
            ++index_pos;
            ++end_block_in_head;
          }
          jc = start_block_in_head * block_k;
          index_cols = (end_block_in_head - start_block_in_head) * block_k;
          if (jc + index_cols > kv_len) {
            index_cols = kv_len - jc;
          }
          if (index_cols <= 0) {
            continue;
          }
          found_index_chunk = true;
          break;
        }
        if (!found_index_chunk) {
          break;
        }
      } else {
        if (dense_jc >= kv_len) {
          break;
        }
        jc = dense_jc;
        dense_jc += blk_sz_c;
      }
      const size_t n_real_cols = use_host_index ? (size_t) index_cols : smin(kv_len - jc, blk_sz_c);
      const size_t n_cols      = n_real_cols;
      const size_t n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

      // 计算块级 score：
      //
      //   dot(Q[Br', D], K[Bc, D]) ==> S[Br', Bc]
      {
        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_qk);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < n_col_tiles; ++c) {
              const __fp16 *row_tiles = q_tile + r * HMX_FP16_TILE_N_ROWS * head_dim;
              const __fp16 *col_tiles = packed_k + (jc / HMX_FP16_TILE_N_COLS + c) * HMX_FP16_TILE_N_COLS * head_dim;
              __fp16 *out_tile = s_tile + (r * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS;

              hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
            }
          }
        }
        hmx_unit_release();
      }

      // safe softmax：
      //   1. 从 S tile 解码出连续行
      //   2. 对 HMX 对齐产生的 padding 列做 -inf mask
      //   3. 计算当前 block 的 rowmax / rowsum
      //   4. 生成 P tile
      //
      // host-index mode only visits kept K ranges. Dense-mask fallback computes full dense chunks.
      {
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

        for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
          HVX_Vector v_s_rowmax_local = v_neg_inf;
          HVX_Vector v_p_rowsum_local = Q6_V_vzero();

          for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            int r = r_vec_idx * 64 + r_vec_off;
            if (r >= align_up(n_rows_g, 2)) {
              break;
            }

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            __fp16 *s_ld_base = s_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;
            __fp16 *p_st_base = p_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;

            HVX_Vector *pv_row_buf0 = row_buffer0;
            HVX_Vector *pv_row_buf1 = row_buffer1;
            for (int c = 0; c < n_cols; c += 64) {
              const __fp16     *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              const HVX_Vector *pv_s_in0     = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
              const HVX_Vector *pv_s_in1     = pv_s_in0 + 16;

              HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
              *pv_row_buf0++               = Q6_V_lo_W(vp_s_dual_row);
              *pv_row_buf1++               = Q6_V_hi_W(vp_s_dual_row);
            }

            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            for (int c = 0; c < n_cols; c += 64) {
              const size_t ne = smin((int) n_real_cols - c, 64);
              HVX_VectorPred q_mask_keep0 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              HVX_VectorPred q_mask_keep1 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

              HVX_Vector v_s_row0 = Q6_V_vmux_QVV(q_mask_keep0, row_buffer0[c / 64], v_neg_inf);
              HVX_Vector v_s_row1 = Q6_V_vmux_QVV(q_mask_keep1, row_buffer1[c / 64], v_neg_inf);

              // 将被 mask 掉的位置直接改写成 -inf
              // 后续 rowmax / exp / rowsum 都基于这个结果继续
              row_buffer0[c / 64] = v_s_row0;
              row_buffer1[c / 64] = v_s_row1;

              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
            }

#pragma unroll
            for (int s = 64; s >= 2; s >>= 1) {
              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
            }

            HVX_Vector v_s_rowmax_pack2     = Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));
            HVX_Vector v_s_rowmax_pack2_rot = Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_s_rowmax_local_rot =
              Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));
            v_s_rowmax_local =
              Q6_V_vlalign_VVR(v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r_vec_off * sizeof(__fp16));

            HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(mvec_m[r_vec_idx], v_s_rowmax_local);

            HVX_Vector v_m_lut  = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));
            HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
            HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));

            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;
            HVX_Vector       v_p_rowsum1 = v_zero;

            for (int c = 0; c < n_cols; c += 64) {
              const int ne = smax(0, smin((int) n_real_cols - c, 64));
              HVX_VectorPred q_mask_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              HVX_Vector v_p_row0_hf, v_p_row1_hf;

              HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0);
              HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1);

              if (use_fp32_exp) {
                HVX_VectorPair vp_s_minus_m0_sf = hvx_my_vqf16_to_wsf(v_s_minus_m0);
                HVX_VectorPair vp_s_minus_m1_sf = hvx_my_vqf16_to_wsf(v_s_minus_m1);

                HVX_Vector v_p_row00_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row01_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m0_sf));
                HVX_Vector v_p_row10_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m1_sf));
                HVX_Vector v_p_row11_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m1_sf));

                v_p_row0_hf = hvx_my_wsf_to_vhf(v_p_row01_sf, v_p_row00_sf);
                v_p_row1_hf = hvx_my_wsf_to_vhf(v_p_row11_sf, v_p_row10_sf);
              } else {
                v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
                v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);
              }

              v_p_row0_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row0_hf, v_zero);
              v_p_row1_hf = Q6_V_vmux_QVV(q_mask_keep, v_p_row1_hf, v_zero);

              __fp16     *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              HVX_Vector *pv_p_out0     = ((HVX_Vector *) out_dual_tile) + r1 / 2;
              HVX_Vector *pv_p_out1     = pv_p_out0 + 16;

              HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
              *pv_p_out0                   = Q6_V_lo_W(vp_p_dual_row);
              *pv_p_out1                   = Q6_V_hi_W(vp_p_dual_row);

              HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0_hf);
              HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1_hf);

              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
            }

#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            }
            HVX_Vector v_p_rowsum_pack2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));

            HVX_Vector v_p_rowsum_pack2_rot = Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_p_rowsum_local_rot = Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
            v_p_rowsum_local =
              Q6_V_vlalign_VVR(v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r_vec_off * sizeof(__fp16));
          }

          mvec_s_rowmax[r_vec_idx] = v_s_rowmax_local;
          mvec_p_rowsum[r_vec_idx] = v_p_rowsum_local;
        }
      }

      // 复用 mvec_s_rowmax 这块 buffer，
      // 后续存 exp(m_i^{j-1} - m_i^j)
      HVX_Vector *mvec_exp_m_diff = mvec_s_rowmax;

      // 更新 online softmax 状态：
      //
      //   m_i^j = max(m_i^{j-1}, rowmax(S_i^j))
      //   l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
      {
        for (int i = 0; i < n_row_vec_cnt; ++i) {
          HVX_Vector v_m_prev = mvec_m[i];
          HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, mvec_s_rowmax[i]);
          HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);

          HVX_Vector v_exp_m_diff_hf = hvx_my_exp2_vhf_vqf16(v_m_diff);

          HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(mvec_l[i], v_exp_m_diff_hf);
          v_l_curr            = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, mvec_p_rowsum[i]);

          mvec_m[i] = v_m_curr;
          mvec_l[i] = v_l_curr;

          mvec_exp_m_diff[i] = v_exp_m_diff_hf;
        }
      }

      // 更新当前输出：
      //
      //   O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + P_i^j V_j
      {
        const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
        const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        for (int i = 0; i < n_row_tiles; ++i) {
          const HVX_Vector v_content = Q6_V_vror_VR(mvec_exp_m_diff[i / 2], (i % 2) * 64);

          __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
          Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_id);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < head_dim / 32; ++c) {
              __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
              __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
              hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

              __fp16 *p_tile_in = p_tile + (r * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;
              __fp16 *v_tile_in = packed_v + (c * ceil_div(kv_len, HMX_FP16_TILE_N_COLS) +
                                                jc / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_ELMS;
              for (int k = 0; k < n_col_tiles; ++k) {
                int    offset  = k * HMX_FP16_TILE_N_ELMS;
                size_t n_tiles = 1;
                hmx_load_tiles_fp16(p_tile_in + offset, v_tile_in + offset, n_tiles);
              }

              __fp16 *o_tile_out = o_tile_curr + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
              hmx_consume_accumulator_fp16(o_tile_out);
            }
          }
        }
        hmx_unit_release();

        swap_ptr(&o_tile_curr, &o_tile_prev);
      }
    }
    // 最终归一化：
    //
    //   O_i = diag(l_i^{-1}) O_i
    {
      const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
      const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

      HVX_Vector v_content;
      for (int i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
          v_content = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(mvec_l[i / 2]));
        } else {
          v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
      }

      hmx_unit_acquire();
      {
        hmx_set_output_scales(hmx_output_scales_id);
        for (int r = 0; r < n_row_tiles; ++r) {
          for (int c = 0; c < head_dim / 32; ++c) {
            __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_out = o_tile_curr + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;

            hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
          }
        }
      }
      hmx_unit_release();
    }

    // 将当前 Q block 对应的结果从 VTCM 写回外部 O
    {
      const size_t o_st_blk_sz_bytes = o_st_blk_sz * qo_element_size;
      const size_t o_st_stride_bytes = o_st_stride * qo_element_size;

      uint8_t *o_st_base = ((uint8_t *) O) + ir * o_st_stride_bytes + kv_head_idx * o_st_blk_sz_bytes;

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int o_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int o_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        uint8_t *row_out0 = o_st_base + o_idx0 * o_st_stride_bytes + h_idx0 * o_head_dim * qo_element_size;
        uint8_t *row_out1 = o_st_base + o_idx1 * o_st_stride_bytes + h_idx1 * o_head_dim * qo_element_size;

        HVX_Vector *pv_out0 = (HVX_Vector *) row_out0;
        HVX_Vector *pv_out1 = (HVX_Vector *) row_out1;

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *in_base = o_tile_curr + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);

            const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);
            const HVX_Vector v_out0 = Q6_V_lo_W(vp);
            const HVX_Vector v_out1 = Q6_V_hi_W(vp);

            if (with_lse) {
              vstu_variable(row_out0 + d * 32 * sizeof(float), sizeof(v_out0), v_out0);
              if (next_row_valid) {
                vstu_variable(row_out1 + d * 32 * sizeof(float), sizeof(v_out1), v_out1);
              }
            } else {
              *pv_out0++ = v_out0;
              if (next_row_valid) {
                *pv_out1++ = v_out1;
              }
            }
          }

          if (with_lse) {
            const float lse0 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 0);
            memcpy(row_out0 + D * sizeof(float), &lse0, sizeof(lse0));
            if (next_row_valid) {
              const float lse1 = flash_attn_lse_from_f16_state(mvec_m, mvec_l, r + 1);
              memcpy(row_out1 + D * sizeof(float), &lse1, sizeof(lse1));
            }
          }
        } else {
          assert(!with_lse);
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const __fp16     *in_dual_tile = in_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector *pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
            const HVX_Vector *pv_in1       = pv_in0 + 16;

            const HVX_VectorPair vp = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        }
        // clang-format on
      }
    }
  }
  return 0;
}

static void block_sparse_flash_attn_worker(void *data, int worker_index) {
  block_sparse_fa_task_state_t *s = (block_sparse_fa_task_state_t *) data;

  if (worker_index < 0 || (unsigned int) worker_index >= s->vtcm_slice_count ||
      (size_t) worker_index > SIZE_MAX / s->vtcm_size_per_thread) {
    worker_pool_atomic_inc_return(&(s->error_count));
    worker_pool_synctoken_jobdone(&(s->sync_ctx));
    return;
  }

  const size_t vtcm_offset = (size_t) worker_index * s->vtcm_size_per_thread;
  if (vtcm_offset > s->acquired_vtcm_size || s->vtcm_size_per_thread > s->acquired_vtcm_size - vtcm_offset) {
    worker_pool_atomic_inc_return(&(s->error_count));
    worker_pool_synctoken_jobdone(&(s->sync_ctx));
    return;
  }

  uint8_t *vtcm       = s->vtcm_base + vtcm_offset;
  uint8_t *vtcm_limit = vtcm + s->vtcm_size_per_thread;

  hmx_manager_enable_execution();

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(s->task_id)) - 1;
    if (task_id >= s->n_tasks) {
      break;
    }

    int kv_head_idx = task_id;
    const int G = s->n_heads / s->n_kv_heads;
    const bool fast_path = s->mask_format == 1 && s->max_kept_blocks > 0 && G == 1 &&
                           s->block_q == HMX_FP16_TILE_N_ROWS && s->block_k == HMX_FP16_TILE_N_COLS;
    const bool halo_path_requested = s->halo_l2_enabled || s->halo_userdma_enabled;
    bool halo_head_valid = true;
    if (halo_path_requested) {
      const int q_blocks_per_head = s->n_q_blocks / s->n_heads;
      const int k_blocks_per_head = s->n_k_blocks / s->n_kv_heads;
      const int image_blocks = s->image_tokens / s->block_k;
      halo_head_valid = block_sparse_halo_validate_kv_head_rows(
        s->block_mask, kv_head_idx, s->n_kv_heads, q_blocks_per_head,
        k_blocks_per_head, s->max_kept_blocks, image_blocks,
        s->image_block_rows, s->image_block_cols) == 0;
      if (!halo_head_valid) {
        worker_pool_atomic_inc_return(&(s->halo_fallback_count));
      }
    }
    const bool halo_l2_enabled = s->halo_l2_enabled && halo_head_valid;
    const bool halo_userdma_enabled = s->halo_userdma_enabled && halo_head_valid;
    const bool halo_force_chunked = s->halo_force_chunked || (halo_path_requested && !halo_head_valid);

    // Packed input can use the direct-tile chunked fallback when halo planning
    // does not fit VTCM; it must not be sent through raw-layout loaders.

    block_sparse_fa_plan_t plan = {0};
    int plan_result = 0;
    if (halo_l2_enabled || halo_userdma_enabled) {
      if (s->halo_plan.blk_sz_c == 0) {
        plan_result = -1;
      }
      plan.mode = BLOCK_SPARSE_FA_CHUNKED;
      plan.blk_sz_c = s->halo_plan.blk_sz_c;
      plan.required_bytes = s->halo_plan.required_bytes;
    } else if (halo_force_chunked) {
      plan_result = block_sparse_fa_select_chunked_plan(
        &plan, G, s->head_dim, s->kv_len, s->vtcm_size_per_thread);
    } else {
      plan_result = block_sparse_fa_select_plan(
        &plan, fast_path, G, s->head_dim, s->kv_len, s->vtcm_size_per_thread);
    }
    if (plan_result != 0) {
      worker_pool_atomic_inc_return(&(s->error_count));
      continue;
    }
    const bool use_full_kv = plan.mode == BLOCK_SPARSE_FA_FULL_KV;
    const bool use_full_k = plan.mode == BLOCK_SPARSE_FA_FULL_K;
    const char *mode_name = halo_l2_enabled ?
                              (s->input_packed ? "PACKED_HALO_L2_ROWS" : "HALO_L2_PACKED_ROWS") :
                            (halo_userdma_enabled ?
                              (s->input_packed ? "PACKED_HALO_USERDMA_ROWS" : "HALO_USERDMA_ROWS") :
                            (use_full_kv ? "FULL_KV" : (use_full_k ? "FULL_K" : "CHUNKED")));
    if (kv_head_idx == 0) {
      FARF(ALWAYS, "block_sparse_fa worker=%d kv_head=%d mode=%s Br=32 Bc=%u required=%u available=%u",
           worker_index, kv_head_idx, mode_name, (unsigned int) plan.blk_sz_c,
           (unsigned int) plan.required_bytes, (unsigned int) s->vtcm_size_per_thread);
    }
    int result;
    if (use_full_kv) {
      result = block_sparse_flash_attn_f16_core_full_kv(
        kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->block_mask, s->qo_len, s->kv_len, s->n_heads,
        s->n_kv_heads, s->head_dim, s->block_q, s->block_k, s->n_q_blocks, s->n_k_blocks, s->mask_format,
        s->max_kept_blocks, s->with_lse, worker_index);
    } else if (use_full_k) {
      result = block_sparse_flash_attn_f16_core_full_k(
        kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->block_mask, s->qo_len, s->kv_len, s->n_heads,
        s->n_kv_heads, s->head_dim, s->block_q, s->block_k, s->n_q_blocks, s->n_k_blocks, s->mask_format,
        s->max_kept_blocks, s->with_lse, worker_index);
    } else if (halo_l2_enabled) {
      result = s->input_packed ?
        block_sparse_flash_attn_f16_core_halo_l2_packed(
          kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->block_mask, s->qo_len, s->kv_len,
          s->n_heads, s->n_kv_heads, s->head_dim, s->block_q, s->block_k, s->n_q_blocks, s->n_k_blocks,
          s->mask_format, s->max_kept_blocks, s->with_lse, worker_index, &s->halo_plan, s->image_tokens,
          s->text_tokens, s->image_block_rows, s->image_block_cols, s->halo_trace, s->halo_run_trace) :
        block_sparse_flash_attn_f16_core_halo_l2(
          kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->block_mask, s->qo_len, s->kv_len,
          s->n_heads, s->n_kv_heads, s->head_dim, s->block_q, s->block_k, s->n_q_blocks, s->n_k_blocks,
          s->mask_format, s->max_kept_blocks, s->with_lse, worker_index, &s->halo_plan, s->image_tokens,
          s->text_tokens, s->image_block_rows, s->image_block_cols, true, s->halo_trace, s->halo_run_trace);
    } else if (halo_userdma_enabled) {
      result = s->input_packed ?
        block_sparse_flash_attn_f16_core_halo_userdma_packed(
          kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->block_mask, s->qo_len, s->kv_len,
          s->n_heads, s->n_kv_heads, s->head_dim, s->block_q, s->block_k, s->n_q_blocks, s->n_k_blocks,
          s->mask_format, s->max_kept_blocks, s->with_lse, worker_index, &s->halo_plan, s->image_tokens,
          s->text_tokens, s->image_block_rows, s->image_block_cols, s->userdma_scope, s->halo_trace,
          s->halo_run_trace) :
        block_sparse_flash_attn_f16_core_halo_userdma(
          kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->block_mask, s->qo_len, s->kv_len,
          s->n_heads, s->n_kv_heads, s->head_dim, s->block_q, s->block_k, s->n_q_blocks, s->n_k_blocks,
          s->mask_format, s->max_kept_blocks, s->with_lse, worker_index, &s->halo_plan, s->image_tokens,
          s->text_tokens, s->image_block_rows, s->image_block_cols, s->userdma_scope, s->halo_trace,
          s->halo_run_trace);
    } else {
      result = block_sparse_flash_attn_f16_core_chunked(
        kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->block_mask, s->qo_len, s->kv_len, s->n_heads,
        s->n_kv_heads, s->head_dim, s->block_q, s->block_k, s->n_q_blocks, s->n_k_blocks, s->mask_format,
        s->max_kept_blocks, s->with_lse, worker_index, s->input_packed);
    }
    if (result != 0) {
      worker_pool_atomic_inc_return(&(s->error_count));
    }
  }

  hmx_manager_disable_execution();

  worker_pool_synctoken_jobdone(&(s->sync_ctx));
}

static int block_sparse_flash_attn_impl(
    __fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
    const __fp16 *restrict V, const int32_t *restrict block_mask, int qo_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
    int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse, int sparse_policy,
    int image_tokens, int text_tokens, int image_block_rows, int image_block_cols,
    int halo_kv_load_mode, bool input_packed, bool halo_trace,
    block_sparse_halo_run_trace_t *halo_run_trace) {
  if (halo_run_trace != NULL) {
    halo_run_trace->exec_mode = -1;
    halo_run_trace->image_rows_updated = 0;
    halo_run_trace->text_q_image_row_updates = 0;
    halo_run_trace->text_q_image_block_updates = 0;
    halo_run_trace->image_q_batch_updates = 0;
    halo_run_trace->image_q_segment_updates = 0;
  }
  if (head_dim % 64 != 0) {
    FARF(ALWAYS, "block sparse FA not supported: head_dim=%d", head_dim);
    return -1;
  }
  if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
    FARF(ALWAYS, "block sparse FA not supported: n_heads=%d n_kv_heads=%d", n_heads, n_kv_heads);
    return -1;
  }
  if (block_mask == NULL || block_q <= 0 || block_k <= 0 || n_q_blocks <= 0 || n_k_blocks <= 0) {
    return -1;
  }
  if (n_q_blocks % n_heads != 0 || n_k_blocks % n_kv_heads != 0) {
    return -1;
  }
  if (n_q_blocks < n_heads * ceil_div(qo_len, block_q) ||
      n_k_blocks < n_kv_heads * ceil_div(kv_len, block_k)) {
    return -1;
  }
  if (mask_format != 0 && mask_format != 1) {
    return -1;
  }
  if (mask_format == 1 && max_kept_blocks <= 0) {
    return -1;
  }
  if (input_packed &&
      (qo_len != kv_len || !htp_packed_shape_supported(n_heads, n_kv_heads, head_dim, 1) ||
       block_q != HMX_FP16_TILE_N_ROWS || block_k != HMX_FP16_TILE_N_COLS ||
       !((mask_format == 1 && sparse_policy == 1 &&
          (halo_kv_load_mode == 1 || halo_kv_load_mode == 2 || halo_kv_load_mode == 3)) ||
         (mask_format == 0 && sparse_policy == 0 && halo_kv_load_mode == 0)))) {
    return -1;
  }

  const int n_jobs = num_hvx128_contexts;
  const unsigned int vtcm_slice_count = num_workers;
  uint8_t *const vtcm_base = (uint8_t *) vtcm_manager_get_vtcm_base();
  const size_t acquired_vtcm_size = vtcm_manager_get_vtcm_size();
  if (!vtcm_base || acquired_vtcm_size == 0 || n_jobs <= 0 || vtcm_slice_count == 0) {
    FARF(ERROR, "%s: invalid worker counts: jobs=%d pool_workers=%u", __func__, n_jobs, vtcm_slice_count);
    return -1;
  }

  const size_t vtcm_size_per_thread = acquired_vtcm_size / vtcm_slice_count / (64 * 1024) * (64 * 1024);
  if (vtcm_size_per_thread == 0 || (size_t) vtcm_slice_count > SIZE_MAX / vtcm_size_per_thread ||
      (size_t) vtcm_slice_count * vtcm_size_per_thread > acquired_vtcm_size) {
    FARF(ERROR, "%s: invalid VTCM slice: acquired=%u slice=%u pool_workers=%u", __func__,
         (unsigned int) acquired_vtcm_size, (unsigned int) vtcm_size_per_thread, vtcm_slice_count);
    return -1;
  }

  const bool halo_loader_requested = sparse_policy == 1 && halo_kv_load_mode != 0;
  block_sparse_halo_plan_t halo_plan = {0};
  const bool halo_geometry_valid = halo_loader_requested &&
    block_sparse_halo_validate_shape(block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, block_q, block_k,
                                     n_q_blocks, n_k_blocks, mask_format, max_kept_blocks, sparse_policy,
                                     image_tokens, text_tokens, image_block_rows, image_block_cols) == 0;
  const bool halo_l2_geometry_valid = halo_geometry_valid && image_block_rows >= 2;
  block_sparse_halo_plan_t dma_preflight_plan = {0};
  const bool dma_preflight_valid = halo_kv_load_mode != 2 ||
    (halo_geometry_valid && input_packed) ||
    (halo_geometry_valid && !input_packed &&
     block_sparse_halo_select_plan(&dma_preflight_plan, 2, head_dim, text_tokens, image_block_cols,
                                   vtcm_size_per_thread) == 0 &&
     block_sparse_halo_dma_preflight(K, V, vtcm_base, vtcm_size_per_thread, vtcm_slice_count,
                                     n_kv_heads, head_dim, image_tokens, &dma_preflight_plan) == 0);
  const int userdma_scope = dma_preflight_valid && halo_geometry_valid && halo_kv_load_mode == 2 ?
    block_sparse_halo_probe_userdma_scope() : BLOCK_SPARSE_HALO_USERDMA_UNKNOWN;
  const int halo_exec_mode = block_sparse_halo_select_execution(
    &halo_plan, halo_loader_requested ? halo_kv_load_mode : 0,
    halo_kv_load_mode == 1 ? halo_l2_geometry_valid : halo_geometry_valid,
    dma_preflight_valid, userdma_scope,
    head_dim, text_tokens, image_block_cols, vtcm_size_per_thread);
  if (halo_exec_mode < 0) {
    return -1;
  }
  const bool halo_l2_enabled = halo_exec_mode == BLOCK_SPARSE_HALO_EXEC_L2_PREFETCH;
  const bool halo_userdma_enabled = halo_exec_mode == BLOCK_SPARSE_HALO_EXEC_USERDMA_ROWS;
  const bool halo_force_chunked = halo_exec_mode == BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK;
  if (halo_run_trace != NULL) {
    halo_run_trace->exec_mode = halo_exec_mode;
  }
  if (halo_force_chunked) {
    FARF(ALWAYS,
         "block_sparse_fa halo loader fallback=CHUNKED policy=%d loader=%d "
         "geometry=%d dma_preflight=%d "
         "userdma_scope=%d available=%u",
         sparse_policy, halo_kv_load_mode, halo_geometry_valid ? 1 : 0, dma_preflight_valid ? 1 : 0, userdma_scope,
         (unsigned int) vtcm_size_per_thread);
  }

  block_sparse_fa_task_state_t state;
  state.O          = O;
  state.Q          = Q;
  state.K          = K;
  state.V          = V;
  state.block_mask = block_mask;
  state.qo_len     = qo_len;
  state.kv_len     = kv_len;
  state.n_heads    = n_heads;
  state.n_kv_heads = n_kv_heads;
  state.head_dim   = head_dim;
  state.block_q    = block_q;
  state.block_k    = block_k;
  state.n_q_blocks = n_q_blocks;
  state.n_k_blocks = n_k_blocks;
  state.mask_format = mask_format;
  state.max_kept_blocks = max_kept_blocks;
  state.sparse_policy = sparse_policy;
  state.image_tokens = image_tokens;
  state.text_tokens = text_tokens;
  state.image_block_rows = image_block_rows;
  state.image_block_cols = image_block_cols;
  state.halo_kv_load_mode = halo_kv_load_mode;
  state.halo_trace = halo_trace;
  state.halo_l2_enabled = halo_l2_enabled;
  state.halo_userdma_enabled = halo_userdma_enabled;
  state.halo_force_chunked = halo_force_chunked;
  state.input_packed = input_packed;
  state.userdma_scope = userdma_scope;
  state.halo_run_trace = halo_run_trace;
  state.halo_plan = halo_plan;
  state.with_lse = with_lse;

  state.task_id              = 0;
  state.error_count          = 0;
  state.halo_fallback_count  = 0;
  state.n_tasks              = n_kv_heads;
  state.vtcm_base            = vtcm_base;
  state.acquired_vtcm_size   = acquired_vtcm_size;
  state.vtcm_size_per_thread = vtcm_size_per_thread;
  state.vtcm_slice_count     = vtcm_slice_count;

  worker_pool_job_t job;
  job.fptr = block_sparse_flash_attn_worker;
  job.dptr = &state;

  const int64_t worker_begin_us = HAP_perf_get_time_us();

  worker_pool_synctoken_init(&(state.sync_ctx), n_jobs);
  for (int i = 0; i < n_jobs; ++i) {
    const AEEResult submit_result = worker_pool_submit(NULL, job);
    if (submit_result != AEE_SUCCESS) {
      FARF(ERROR, "%s: worker submit failed: job=%d result=0x%x", __func__, i, submit_result);
      worker_pool_atomic_inc_return(&(state.error_count));
      worker_pool_synctoken_jobdone(&(state.sync_ctx));
    }
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  if (state.halo_fallback_count != 0 && halo_run_trace != NULL) {
    halo_run_trace->exec_mode = BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK;
    halo_run_trace->image_rows_updated = 0;
    halo_run_trace->text_q_image_row_updates = 0;
    halo_run_trace->text_q_image_block_updates = 0;
    halo_run_trace->image_q_batch_updates = 0;
    halo_run_trace->image_q_segment_updates = 0;
  }

  if (state.error_count != 0) {
    FARF(ERROR, "%s: worker errors=%u", __func__, state.error_count);
    return -1;
  }

  const int64_t worker_end_us = HAP_perf_get_time_us();
  const int64_t elapsed_us = worker_end_us - worker_begin_us;
  FARF(ALWAYS,
       "%s: %lld us, qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d block_q=%d block_k=%d",
       __func__, elapsed_us, qo_len, kv_len, n_heads, n_kv_heads, head_dim, block_q, block_k);

  return 0;
}

int block_sparse_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                            const __fp16 *restrict V, const int32_t *restrict block_mask, int qo_len, int kv_len,
                            int n_heads, int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                            int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse, int sparse_policy,
                            int image_tokens, int text_tokens, int image_block_rows, int image_block_cols,
                            int halo_kv_load_mode, bool halo_trace,
                            block_sparse_halo_run_trace_t *halo_run_trace) {
  return block_sparse_flash_attn_impl(
    O, Q, K, V, block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, block_q, block_k,
    n_q_blocks, n_k_blocks, mask_format, max_kept_blocks, with_lse, sparse_policy,
    image_tokens, text_tokens, image_block_rows, image_block_cols, halo_kv_load_mode,
    false, halo_trace, halo_run_trace);
}

int block_sparse_flash_attn_packed(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                                   const __fp16 *restrict V, const int32_t *restrict block_mask,
                                   int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                                   int block_q, int block_k, int n_q_blocks, int n_k_blocks,
                                   int mask_format, int max_kept_blocks, bool with_lse, int sparse_policy,
                                   int image_tokens, int text_tokens, int image_block_rows, int image_block_cols,
                                   int halo_kv_load_mode, bool halo_trace,
                                   block_sparse_halo_run_trace_t *halo_run_trace) {
  return block_sparse_flash_attn_impl(
    O, Q, K, V, block_mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, block_q, block_k,
    n_q_blocks, n_k_blocks, mask_format, max_kept_blocks, with_lse, sparse_policy,
    image_tokens, text_tokens, image_block_rows, image_block_cols, halo_kv_load_mode,
    true, halo_trace, halo_run_trace);
}

#define Br 32
#define Bc 256
#define D  128

int naive_flash_attn(float *restrict O, const float *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                     const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
  // 参数校验
  if (n_heads % n_kv_heads != 0) {
    return -1;
  }
  if (head_dim > D) {
    return -1;
  }

  const int gqa_factor = n_heads / n_kv_heads;

  const size_t kv_pad_len = align_up(kv_len, 64);

  const size_t qo_stride = n_heads * head_dim;
  const size_t kv_stride = n_kv_heads * head_dim;

  // 计算特征缩放因子
  const float qk_scale = 1.0f / sqrtf(head_dim) * 1.44269504f;

  for (int h = 0; h < n_heads; ++h) {
    const int h_kv = h / gqa_factor;

    // 分块处理Q序列
    for (int i = 0; i < qo_len; i += Br) {
      const int q_start = i;
      const int q_end   = (i + Br) < qo_len ? (i + Br) : qo_len;
      const int br      = q_end - q_start;

      static float Qi[Br][D];
      static float Oi[Br][D];

      // 加载当前Q块 (fp32)
      const float *Q_src = Q + q_start * qo_stride + h * head_dim;
      for (int r = 0; r < br; ++r) {
        for (int d = 0; d < head_dim; ++d) {
          Qi[r][d] = Q_src[r * qo_stride + d];
        }
      }

      // 初始化输出块和中间状态
      static float mi[Br];
      static float li[Br];
      for (int r = 0; r < br; ++r) {
        mi[r] = -INFINITY;
        li[r] = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          Oi[r][d] = 0.0f;
        }
      }

      // 分块处理KV序列
      for (int j = 0; j < kv_len; j += Bc) {
        const int k_start = j;
        const int k_end   = (j + Bc) < kv_len ? (j + Bc) : kv_len;
        const int bc      = k_end - k_start;

        // 加载当前K块并转换为fp32
        static float  Kj[Bc][D];
        const __fp16 *K_src = K + k_start * kv_stride + h_kv * head_dim;
        for (int c = 0; c < bc; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            Kj[c][d] = (float) K_src[c * kv_stride + d];
          }
        }

        // 计算注意力分数块
        static float Sij[Br][Bc];
        for (int r = 0; r < br; ++r) {
          for (int c = 0; c < bc; ++c) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              sum += Qi[r][d] * Kj[c][d];
            }
            Sij[r][c] = sum * qk_scale;
          }
        }

        // 应用掩码（如果存在）
        if (mask != NULL) {
          for (int r = 0; r < br; ++r) {
            for (int c = 0; c < bc; ++c) {
              const int mask_idx = (q_start + r) * kv_pad_len + (k_start + c);
              Sij[r][c] += (float) mask[mask_idx];
            }
          }
        }

        // 加载当前V块并转换为fp32
        static float  Vj[Bc][D];
        const __fp16 *V_src = V + k_start * kv_stride + h_kv * head_dim;
        for (int c = 0; c < bc; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            Vj[c][d] = (float) V_src[c * kv_stride + d];
          }
        }

        // 在线softmax更新
        for (int r = 0; r < br; ++r) {
          // 当前块最大值
          float m_curr = -INFINITY;
          for (int c = 0; c < bc; ++c) {
            if (Sij[r][c] > m_curr) {
              m_curr = Sij[r][c];
            }
          }

          // 全局最大值
          const float m_new = fmaxf(mi[r], m_curr);

          // 计算指数和
          float exp_sum = 0.0f;
          float exp_values[Bc];
          for (int c = 0; c < bc; ++c) {
            exp_values[c] = exp2f(Sij[r][c] - m_new);
            exp_sum += exp_values[c];
          }

          // 更新累积因子
          const float f = exp2f(mi[r] - m_new);

          const float l_new = li[r] * f + exp_sum;

          // 更新输出值
          for (int d = 0; d < head_dim; ++d) {
            // 校正历史累积值
            Oi[r][d] *= f;

            // 累加新值
            float sum = 0.0f;
            for (int c = 0; c < bc; ++c) {
              sum += exp_values[c] * Vj[c][d];
            }
            Oi[r][d] += sum;
          }

          // 更新中间状态
          mi[r] = m_new;
          li[r] = l_new;
        }
      }

      // 写入最终输出
      float *O_dst = O + q_start * qo_stride + h * head_dim;
      for (int r = 0; r < br; ++r) {
        const float scale = 1.0f / li[r];
        for (int d = 0; d < head_dim; ++d) {
          O_dst[r * qo_stride + d] = Oi[r][d] * scale;
        }
      }
    }
  }
  return 0;
}

#undef Br
#undef Bc
#undef D

static int block_sparse_score_scalar(__fp16 *restrict score, const __fp16 *restrict q_mean,
                                     const __fp16 *restrict k_mean, int head_dim, int n_q_blocks, int n_k_blocks) {
  const size_t block_stride = (size_t) head_dim;

  for (int qb = 0; qb < n_q_blocks; ++qb) {
    for (int kb = 0; kb < n_k_blocks; ++kb) {
      float score_sum = 0.0f;
      const __fp16 *q_row = q_mean + (size_t) qb * block_stride;
      const __fp16 *k_row = k_mean + (size_t) kb * block_stride;
      for (int d = 0; d < head_dim; ++d) {
        score_sum += (float) q_row[d] * (float) k_row[d];
      }
      score[(size_t) qb * n_k_blocks + kb] = (__fp16) score_sum;
    }
  }

  return 0;
}

static inline void block_sparse_score_pack_q_tile(__fp16 *restrict q_tile, const __fp16 *restrict q_mean,
                                                  int head_dim, int n_q_blocks, int qb0) {
  for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
    const int q0 = qb0 + r + 0;
    const int q1 = qb0 + r + 1;

    const HVX_Vector *pv_in0 = q0 < n_q_blocks ? (const HVX_Vector *) (q_mean + (size_t) q0 * head_dim) : NULL;
    const HVX_Vector *pv_in1 = q1 < n_q_blocks ? (const HVX_Vector *) (q_mean + (size_t) q1 * head_dim) : NULL;

    __fp16 *out_base = q_tile + (r / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS * head_dim;

    for (int d = 0; d < head_dim / 64; ++d) {
      const HVX_Vector     v0 = pv_in0 != NULL ? *pv_in0++ : Q6_V_vzero();
      const HVX_Vector     v1 = pv_in1 != NULL ? *pv_in1++ : Q6_V_vzero();
      const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

      __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
      HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r / 2;
      HVX_Vector *pv_out1       = pv_out0 + 16;

      *pv_out0 = Q6_V_lo_W(vp);
      *pv_out1 = Q6_V_hi_W(vp);
    }
  }
}

static inline void block_sparse_score_pack_k_tile(__fp16 *restrict k_tile, const __fp16 *restrict k_mean,
                                                  int head_dim, int n_k_blocks, int kb0) {
  static int32_t transpose_vscatter_indices_base[32] __vec_aligned__;
  for (int i = 0; i < 32; ++i) {
    transpose_vscatter_indices_base[i] = i * 128;
  }

  const HVX_Vector v_step         = Q6_V_vsplat_R(4);
  const HVX_Vector v_offsets_base = vmem(transpose_vscatter_indices_base);
  HVX_Vector       v_offsets      = v_offsets_base;

  for (int r = 0; r < HMX_FP16_TILE_N_COLS; ++r) {
    const int k = kb0 + r;
    if (k >= n_k_blocks) {
      break;
    }

    const HVX_Vector *pv_in = (const HVX_Vector *) (k_mean + (size_t) k * head_dim);
    for (int d = 0; d < head_dim / 64; ++d) {
      __fp16 *out_dual_tile = k_tile + d * HMX_FP16_TILE_N_ELMS * 2;
      Q6_vscatter_RMVwV((size_t) out_dual_tile, HMX_FP16_TILE_SIZE * 2 - 1, v_offsets, *pv_in++);
    }

    v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
  }
}

static inline void block_sparse_score_store_tile(__fp16 *restrict score, const __fp16 *restrict s_tile,
                                                 int n_q_blocks, int n_k_blocks, int qb0, int kb0) {
  __fp16 row0[64] __attribute__((aligned(VLEN)));
  __fp16 row1[64] __attribute__((aligned(VLEN)));

  for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
    const int q0 = qb0 + r + 0;
    const int q1 = qb0 + r + 1;
    if (q0 >= n_q_blocks) {
      break;
    }

    const HVX_Vector v = ((const HVX_Vector *) s_tile)[r / 2];
    const HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v, -2);
    *((HVX_Vector *) row0) = Q6_V_lo_W(vp);
    *((HVX_Vector *) row1) = Q6_V_hi_W(vp);

    const size_t n_cols = smin((size_t) HMX_FP16_TILE_N_COLS, (size_t) (n_k_blocks - kb0));
    memcpy(score + (size_t) q0 * n_k_blocks + kb0, row0, n_cols * sizeof(__fp16));
    if (q1 < n_q_blocks) {
      memcpy(score + (size_t) q1 * n_k_blocks + kb0, row1, n_cols * sizeof(__fp16));
    }
  }
}

int block_sparse_score(__fp16 *restrict score, const __fp16 *restrict q_mean, const __fp16 *restrict k_mean,
                       int head_dim, int n_q_blocks, int n_k_blocks) {
  if (!score || !q_mean || !k_mean || head_dim <= 0 || n_q_blocks <= 0 || n_k_blocks <= 0) {
    return -1;
  }

  if (head_dim % 64 != 0) {
    return block_sparse_score_scalar(score, q_mean, k_mean, head_dim, n_q_blocks, n_k_blocks);
  }

  uint8_t *vtcm_cur = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (vtcm_cur == NULL) {
    return block_sparse_score_scalar(score, q_mean, k_mean, head_dim, n_q_blocks, n_k_blocks);
  }

  const size_t q_tile_size = align_up((size_t) HMX_FP16_TILE_N_ROWS * head_dim * sizeof(__fp16), 4096);
  const size_t k_tile_size = align_up((size_t) HMX_FP16_TILE_N_COLS * head_dim * sizeof(__fp16), 4096);
  const size_t s_tile_size = align_up((size_t) HMX_FP16_TILE_N_ELMS * sizeof(__fp16), 4096);

  __fp16  *q_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, q_tile_size);
  __fp16  *k_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, k_tile_size);
  __fp16  *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, s_tile_size);
  uint8_t *scales = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  for (int qb0 = 0; qb0 < n_q_blocks; qb0 += HMX_FP16_TILE_N_ROWS) {
    hvx_fill_uh(q_tile, 0, q_tile_size);
    block_sparse_score_pack_q_tile(q_tile, q_mean, head_dim, n_q_blocks, qb0);

    for (int kb0 = 0; kb0 < n_k_blocks; kb0 += HMX_FP16_TILE_N_COLS) {
      hvx_fill_uh(k_tile, 0, k_tile_size);
      block_sparse_score_pack_k_tile(k_tile, k_mean, head_dim, n_k_blocks, kb0);

      hmx_unit_acquire();
      hmx_set_output_scales(scales);
      for (int d = 0; d < head_dim / 32; d += 32) {
        const int    offset  = d * HMX_FP16_TILE_N_ELMS;
        const size_t n_tiles = smin((size_t) head_dim / 32 - (size_t) d, (size_t) 32);
        hmx_load_tiles_fp16(q_tile + offset, k_tile + offset, n_tiles);
      }
      hmx_consume_accumulator_fp16(s_tile);
      hmx_unit_release();

      block_sparse_score_store_tile(score, s_tile, n_q_blocks, n_k_blocks, qb0, kb0);
    }
  }

  return 0;
}
