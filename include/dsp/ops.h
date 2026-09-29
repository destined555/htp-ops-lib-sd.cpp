#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/quants.h"
#include "dsp/dma_utils.h"

#ifndef restrict
#  define restrict __restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

int hvx_rms_norm_f32(float *restrict dst, const float *restrict src, int ne0, int ne1);

int hmx_mat_mul_permuted_w16a32(float *restrict dst, const float *activation, const __fp16 *permuted_weight, int m,
                                int k, int n);
int hmx_qkv_mul_rms_packed_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *restrict activation, const __fp16 *restrict permuted_weight,
    const float *restrict bias, const float *restrict q_norm, const float *restrict k_norm,
    int tokens, int input_dim, int n_heads, int head_dim);
int hmx_qkv_mul_rms_packed_dual_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *image_activation, const __fp16 *image_weight, const float *image_bias,
    const float *image_q_norm, const float *image_k_norm, int image_tokens,
    const float *text_activation, const __fp16 *text_weight, const float *text_bias,
    const float *text_q_norm, const float *text_k_norm, int text_tokens,
    int input_dim, int n_heads, int head_dim);

int hmx_qkv_mul_rms_packed_rope_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *restrict activation, const __fp16 *restrict permuted_weight,
    const float *restrict bias, const float *restrict q_norm, const float *restrict k_norm,
    const float *restrict rope_pe, int rope_token_offset, int rope_token_count,
    int tokens, int input_dim, int n_heads, int head_dim);

int hmx_qkv_mul_rms_packed_dual_rope_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *image_activation, const __fp16 *image_weight, const float *image_bias,
    const float *image_q_norm, const float *image_k_norm, int image_tokens,
    const float *text_activation, const __fp16 *text_weight, const float *text_bias,
    const float *text_q_norm, const float *text_k_norm, int text_tokens,
    const float *rope_pe, int image_rope_token_offset, int text_rope_token_offset,
    int input_dim, int n_heads, int head_dim);
int hmx_mat_mul_permuted_qk_0_d16a32(float *restrict dst, const float *activation, const uint8_t *permuted_weight,
                                     int m, int k, int n, enum ggml_type weight_type);

int simple_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                      const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                      bool with_lse);

struct FusedAnchorAttnParams;
typedef struct block_sparse_halo_row block_sparse_halo_row_t;

typedef struct {
  uint64_t q_blocks_decoded;
  uint64_t mask_row_scans;
  uint64_t mask_entry_scans;
  uint64_t k_chunks;
  uint64_t qk_tiles;
  uint64_t dense_pv_tiles;
  uint64_t sparse_selected_tiles;
  uint64_t sparse_skipped_tiles;
  uint64_t sparse_empty_chunks;
  uint64_t text_full_keep_blocks;
} fused_anchor_fa_stats_t;

typedef struct {
  size_t blk_sz_r;
  size_t blk_sz_c;
  size_t required_bytes;
} fused_anchor_fa_plan_t;

#ifndef BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS
#define BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS 9
#endif
#define FUSED_ANCHOR_MAX_SELECTED_SEGMENTS (BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS + 1)

typedef struct {
  int first_tile;
  int tile_count;
  int selected_tiles;
  int skipped_tiles;
  int segment_count;
  int tile_offsets[FUSED_ANCHOR_MAX_SELECTED_SEGMENTS];
  int tile_counts[FUSED_ANCHOR_MAX_SELECTED_SEGMENTS];
  bool is_text_query;
} fused_anchor_keep_plan_t;

size_t fused_anchor_fa_compute_vtcm_usage(int head_dim, int n_rows, int n_cols);
int fused_anchor_fa_find_chunk_size(fused_anchor_fa_plan_t *plan, int head_dim,
                                    int qo_len, int kv_len, size_t available_bytes);
int fused_anchor_block_keep(const int32_t *block_mask, const struct FusedAnchorAttnParams *params,
                            int head, int query_index, int key_block_index);
int fused_anchor_prepare_keep_plan(fused_anchor_keep_plan_t *plan, uint8_t *keep_flags,
                                   int keep_flags_capacity, const block_sparse_halo_row_t *row,
                                   int jc, int n_cols, int kv_len, int block_k);
int fused_anchor_plan_selected_segments(fused_anchor_keep_plan_t *plan, const uint8_t *keep_flags,
                                        int tile_count, int first_tile, bool is_text_query);
int fused_anchor_flash_attn(float *restrict output, const float *restrict q,
                            const __fp16 *restrict k, const __fp16 *restrict v,
                            const __fp16 *restrict dense_mask, const int32_t *restrict block_mask,
                            const struct FusedAnchorAttnParams *params);

enum block_sparse_fa_mode {
  BLOCK_SPARSE_FA_FULL_KV,
  BLOCK_SPARSE_FA_FULL_K,
  BLOCK_SPARSE_FA_CHUNKED,
};

typedef struct {
  enum block_sparse_fa_mode mode;
  size_t                    blk_sz_c;
  size_t                    required_bytes;
} block_sparse_fa_plan_t;

#define BLOCK_SPARSE_HALO_DMA_BANKS 2
#define BLOCK_SPARSE_HALO_DMA_DESCS_PER_BANK (2 * BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS)
#define BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW 2
#define BLOCK_SPARSE_HALO_PACKED_DMA_DESCS_PER_ROW 5
#define BLOCK_SPARSE_HALO_MAX_DMA_DESCS_PER_ROW BLOCK_SPARSE_HALO_PACKED_DMA_DESCS_PER_ROW
#define BLOCK_SPARSE_HALO_ROW_SLOTS 4
#define BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS 3
#define BLOCK_SPARSE_HALO_RAW_STAGING_SLOTS 1
#define BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS 3
#define BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS 4

typedef struct block_sparse_halo_row {
  bool    is_text_query;
  int     image_count;
  int     image_block_total;
  int     first_k_block;
  int     k_blocks_per_head;
  int32_t image_blocks[BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS];
} block_sparse_halo_row_t;

typedef struct {
  int first_active_row;
  int active_row_count;
  int load_row;
} block_sparse_halo_row_schedule_t;

typedef struct {
  uint8_t *raw;
  int      row_id;
  bool     ready;
  bool     outstanding;
} block_sparse_halo_row_slot_t;

typedef struct {
  __fp16 *packed_k;
  __fp16 *packed_v;
  int     row_id;
  bool    ready;
} block_sparse_halo_packed_row_slot_t;

typedef struct {
  int row_count;
  int row_ids[BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS];
  int packed_slots[BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS];
  int next_pending_row;
} block_sparse_halo_text_kv_batch_t;

typedef struct {
  int total_cols;
  int tile_offsets[BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS];
  int tile_counts[BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS];
} block_sparse_halo_update_segment_plan_t;

typedef struct {
  const __fp16 *packed_k;
  const __fp16 *packed_v;
  int           v_tile_stride;
  int           block_count;
} block_sparse_halo_packed_segment_t;

typedef struct {
  int row_id;
  int startup_pack;
  int prefetch_row;
  int evicted_row;
  int text_q_updates;
  int image_block_count;
  int text_q_block_updates;
} block_sparse_halo_packed_row_event_t;

typedef struct {
  int32_t first_block;
  int32_t block_count;
} block_sparse_halo_prefetch_range_t;

typedef struct {
  int range_count;
  int block_count;
  block_sparse_halo_prefetch_range_t ranges[BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS];
} block_sparse_halo_prefetch_plan_t;

typedef struct {
  size_t required_bytes;
  size_t workspace_offset;
  size_t workspace_bytes;
  size_t scratch_col_capacity;
  size_t q_offset;
  size_t o_offsets[2];
  size_t packed_k_offset;
  size_t packed_k_bytes;
  size_t packed_v_offset;
  size_t packed_v_bytes;
  size_t score_offset;
  /* probability_offset aliases score_offset. */
  size_t probability_offset;
  /* o_offsets[1] and diagonal_offset alias o_offsets[0]. */
  size_t diagonal_offset;
  size_t softmax_state_offset;
  size_t softmax_state_bytes;
  size_t row_buffer_offset;
  size_t row_buffer_bytes;
  size_t scales_offset;
  size_t scales_bytes;
  size_t resident_text_offset;
  size_t resident_text_bytes;
  size_t packed_image_row_offsets[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS];
  size_t packed_image_row_stride;
  size_t packed_image_row_bytes;
  int    packed_image_row_slot_count;
  size_t resident_text_q_offset;
  size_t resident_text_q_bytes;
  size_t resident_text_o_offsets[2];
  size_t resident_text_o_bytes;
  size_t resident_text_state_offset;
  size_t resident_text_state_bytes;
  int    resident_text_q_blocks;
  size_t image_row_offsets[BLOCK_SPARSE_HALO_ROW_SLOTS];
  size_t image_row_stride;
  size_t image_row_bytes;
  /* Legacy A/B fields remain in the ABI while the row-ring execution is staged. */
  size_t image_bank_offsets[BLOCK_SPARSE_HALO_DMA_BANKS];
  size_t image_bank_stride;
  size_t image_bank_bytes;
  size_t descriptor_offsets[BLOCK_SPARSE_HALO_DMA_BANKS];
  size_t descriptor_bytes;
  size_t completion_offsets[BLOCK_SPARSE_HALO_DMA_BANKS];
  size_t completion_bytes;
  size_t blk_sz_c;
} block_sparse_halo_plan_t;

typedef struct __attribute__((aligned(64))) {
  uint8_t       *raw;
  dma_desc_2d_t *descs;
  int            block_count;
  int            desc_count;
  int32_t        block_ids[BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS];
  bool           ready;
  bool           outstanding;
  int            submit_result;
  int            wait_result;
} block_sparse_halo_dma_bank_t;

#ifdef __cplusplus
static_assert(sizeof(block_sparse_halo_dma_bank_t) == 64);
#else
_Static_assert(sizeof(block_sparse_halo_dma_bank_t) == 64, "unexpected halo DMA bank state size");
#endif

enum block_sparse_halo_userdma_scope {
  BLOCK_SPARSE_HALO_USERDMA_UNKNOWN = 0,
  BLOCK_SPARSE_HALO_USERDMA_PER_WORKER,
  BLOCK_SPARSE_HALO_USERDMA_SHARED,
  BLOCK_SPARSE_HALO_USERDMA_UNSAFE,
};

enum block_sparse_halo_exec_mode {
  BLOCK_SPARSE_HALO_EXEC_BASELINE = 0,
  BLOCK_SPARSE_HALO_EXEC_L2_PREFETCH,
  BLOCK_SPARSE_HALO_EXEC_USERDMA_ROWS,
  BLOCK_SPARSE_HALO_EXEC_USERDMA_AB = BLOCK_SPARSE_HALO_EXEC_USERDMA_ROWS,
  BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK,
};

/* Per-op DDR->VTCM read accounting for the sparse attention cores.
 *
 * Byte counters are *issued read* counts: everything the kernels hand to a
 * loader (UserDMA descriptors, l2fetch hints, direct pack/copy loops) while
 * reading Q, the resident text K/V and the image K/V rows. Output writes are
 * never counted. Because l2fetch is only a hint, read_bytes is an upper bound
 * on the true DRAM traffic whenever data was already L2 resident.
 *
 * transfer_count is the number of loader issues (one 2D UserDMA descriptor or
 * one K/V tile load counts as one), so read_bytes / transfer_count is the
 * average payload per transfer.
 */
typedef struct block_sparse_attn_metrics {
  uint64_t read_bytes;     /* total DDR->VTCM read bytes (q + text_kv + image_kv + misc) */
  uint64_t q_bytes;
  uint64_t text_kv_bytes;
  uint64_t image_kv_bytes;
  uint64_t misc_bytes;     /* mask, scales and other small constant reads */
  uint64_t transfer_count; /* loader issues; see above */
  uint64_t l2_lines;       /* 128B-aligned read requests, alignment sanity only */
  int64_t  ddr_to_vtcm_us; /* wall time spent waiting for DDR->VTCM data */
  int64_t  overlap_us;     /* part of the transfer window overlapped by compute */
  int64_t  inflight_us;    /* outstanding transfer window: submit + overlap + exposed */
  int64_t  pack_us;        /* in-VTCM rearrange cost, after data has arrived */
  int64_t  submit_us;      /* descriptor build + submit cost */
  int64_t  worker_elapsed_us; /* summed per-task worker wall time; the denominator
                                 for the DDR wait share when several kv heads run
                                 on the same worker */
  int      route_mismatch; /* 1 when the run did not take the requested route */
} block_sparse_attn_metrics_t;

/* Average payload per loader issue; 0 when nothing was transferred. */
static inline uint64_t block_sparse_attn_avg_transfer_bytes(const block_sparse_attn_metrics_t *metrics) {
  if (metrics == NULL || metrics->transfer_count == 0) {
    return 0;
  }
  return metrics->read_bytes / metrics->transfer_count;
}

/* Byte counters travel over the wire as int32 KB, so that one field covers
   every realistic shape. Round up: a non-zero byte count must never read back
   as 0. */
static inline int32_t block_sparse_attn_bytes_to_wire_kb(uint64_t bytes) {
  const uint64_t kb = (bytes + 1023u) / 1024u;
  return kb > (uint64_t) INT32_MAX ? INT32_MAX : (int32_t) kb;
}

/* K/V residency split for the chunked core: a chunk that starts at
   `first_token` covers `token_count` tokens, of which the ones below
   `image_tokens` are image rows and the rest belong to the resident text tail.
   K/V are fp16, so one token contributes head_dim * 2 bytes to each of K and V;
   both are charged here. */
static inline void block_sparse_attn_metrics_add_kv_bytes(block_sparse_attn_metrics_t *metrics,
                                                          int first_token, size_t token_count,
                                                          int head_dim, int image_tokens) {
  if (metrics == NULL || first_token < 0 || token_count == 0 || head_dim <= 0) {
    return;
  }
  size_t image_cols = 0;
  if (first_token < image_tokens) {
    image_cols = (size_t) image_tokens - (size_t) first_token;
    if (image_cols > token_count) {
      image_cols = token_count;
    }
  }
  const uint64_t bytes_per_token = (uint64_t) head_dim * 2u;  /* fp16 K and fp16 V */
  metrics->image_kv_bytes += (uint64_t) image_cols * bytes_per_token;
  metrics->text_kv_bytes += (uint64_t) (token_count - image_cols) * bytes_per_token;
}

/* Wire transport for the microsecond fields; negative and overflowing values
   clamp to 0 / INT32_MAX. */
static inline int32_t block_sparse_attn_us_to_wire(int64_t us) {
  if (us <= 0) {
    return 0;
  }
  return us > (int64_t) INT32_MAX ? INT32_MAX : (int32_t) us;
}

typedef struct {
  int exec_mode;
  int image_rows_updated;
  int text_q_image_row_updates;
  int text_q_image_block_updates;
  int image_q_batch_updates;
  int image_q_segment_updates;
  /* DDR->VTCM read accounting, filled whenever the pointer is non-NULL.
     Compiled in by HTP_ATTN_METRICS; zero when the counters are not built. */
  struct block_sparse_attn_metrics metrics;
} block_sparse_halo_run_trace_t;

int block_sparse_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                            const __fp16 *restrict V, const int32_t *restrict block_mask, int qo_len, int kv_len,
                            int n_heads, int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks,
                            int n_k_blocks, int mask_format, int max_kept_blocks, bool with_lse, int sparse_policy,
                            int image_tokens, int text_tokens, int image_block_rows, int image_block_cols,
                            int halo_kv_load_mode, bool halo_trace, block_sparse_halo_run_trace_t *halo_run_trace);
int block_sparse_flash_attn_packed(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                                   const __fp16 *restrict V, const int32_t *restrict block_mask,
                                   int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                                   int block_q, int block_k, int n_q_blocks, int n_k_blocks,
                                   int mask_format, int max_kept_blocks, bool with_lse, int sparse_policy,
                                   int image_tokens, int text_tokens, int image_block_rows, int image_block_cols,
                                   int halo_kv_load_mode, bool halo_trace,
                                   block_sparse_halo_run_trace_t *halo_run_trace);

int block_sparse_flash_attn_f16_core_halo_l2(
    int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O, const __fp16 *restrict Q,
    const __fp16 *restrict K, const __fp16 *restrict V, const int32_t *restrict block_mask, int qo_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks, int n_k_blocks,
    int mask_format, int max_kept_blocks, bool with_lse, int worker_index, const block_sparse_halo_plan_t *halo_plan,
    int image_tokens, int text_tokens, int image_block_rows, int image_block_cols, bool resident_text_enabled,
    bool halo_trace, block_sparse_halo_run_trace_t *halo_run_trace);

int block_sparse_flash_attn_f16_core_halo_userdma(
    int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O, const __fp16 *restrict Q,
    const __fp16 *restrict K, const __fp16 *restrict V, const int32_t *restrict block_mask, int qo_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim, int block_q, int block_k, int n_q_blocks, int n_k_blocks,
    int mask_format, int max_kept_blocks, bool with_lse, int worker_index, const block_sparse_halo_plan_t *halo_plan,
    int image_tokens, int text_tokens, int image_block_rows, int image_block_cols, int userdma_scope,
    bool halo_trace, block_sparse_halo_run_trace_t *halo_run_trace);

int block_sparse_fa_select_plan(block_sparse_fa_plan_t *plan, bool fast_path, int gqa_factor, int head_dim,
                                int kv_len, size_t available_bytes);
int block_sparse_fa_select_halo_fallback_plan(block_sparse_fa_plan_t *plan, bool fast_path, int gqa_factor,
                                              int head_dim, int kv_len, size_t available_bytes);
int block_sparse_fa_select_chunked_plan(block_sparse_fa_plan_t *plan, int gqa_factor, int head_dim,
                                        int kv_len, size_t available_bytes);

int block_sparse_halo_parse_row(block_sparse_halo_row_t *parsed, const int32_t *row, int max_kept_blocks,
                                int q_block, int q_blocks_per_head, int k_blocks_per_head, int image_blocks);

int block_sparse_halo_schedule_q_row(block_sparse_halo_row_schedule_t *schedule, int q_row, int image_block_rows);
int block_sparse_halo_next_q_block(int current_block, int q_block_count);
int block_sparse_halo_schedule_text_row(block_sparse_halo_row_schedule_t *schedule, int image_row,
                                        int image_block_rows);
int block_sparse_halo_text_prefetch_target(int prefetched_through, int chunk_last_row, int image_block_rows);
int block_sparse_halo_find_ready_row_slot(const block_sparse_halo_row_slot_t slots[BLOCK_SPARSE_HALO_ROW_SLOTS],
                                          int row_id);
int block_sparse_halo_select_staging_slot(const block_sparse_halo_row_slot_t slots[BLOCK_SPARSE_HALO_ROW_SLOTS],
                                          int first_active_row, int active_row_count);
int block_sparse_halo_rotate_row_slots(block_sparse_halo_row_slot_t slots[BLOCK_SPARSE_HALO_ROW_SLOTS],
                                       int evicted_row, int new_row);
int block_sparse_halo_find_packed_row(
  const block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS], int row_id);
int block_sparse_halo_select_packed_row_victim(
  const block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS],
  int first_active_row, int active_row_count);
int block_sparse_halo_bind_packed_row(
  block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS], int slot, int row_id);
int block_sparse_halo_plan_text_kv_batch(
  block_sparse_halo_text_kv_batch_t *batch,
  const block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS],
  int first_pending_row, int latest_packed_row, int image_block_rows, bool flush_tail);
int block_sparse_halo_plan_update_segments(block_sparse_halo_update_segment_plan_t *plan,
                                           const int *real_cols, int count,
                                           size_t scratch_col_capacity);
int block_sparse_halo_get_packed_segment(block_sparse_halo_packed_segment_t *segment,
                                         const block_sparse_halo_packed_row_slot_t *slot,
                                         int col_start, int block_count, int image_block_cols, int head_dim);
int block_sparse_halo_pack_image_row(block_sparse_halo_packed_row_slot_t *slot,
                                     const __fp16 *K, const __fp16 *V,
                                     int n_kv_heads, int kv_head_idx, int head_dim,
                                     int image_block_cols, int row_id);
int block_sparse_halo_pack_raw_row_segment(__fp16 *packed_k, __fp16 *packed_v, size_t packed_bytes,
                                           const block_sparse_halo_row_slot_t *slot, int col_start,
                                           int block_count, int image_block_cols, int head_dim);
int block_sparse_halo_text_o_bank_after_image_row(int current_bank, int image_block_cols);
bool block_sparse_halo_can_merge_index_block(bool row_bounded, int start_block, int next_block,
                                             int image_blocks, int image_block_cols,
                                             int current_cols, int max_cols);
int block_sparse_halo_text_q_blocks(int text_tokens);
int block_sparse_halo_text_q_rows(int text_tokens, int text_q_block);
int block_sparse_halo_text_o_next_bank(int current_bank, int completed_q_blocks, int total_q_blocks);
int block_sparse_halo_plan_packed_row_event(block_sparse_halo_packed_row_event_t *event,
                                            int row_id, int image_block_rows,
                                            int image_block_cols, int text_q_blocks);

int block_sparse_halo_plan_prefetch(block_sparse_halo_prefetch_plan_t *plan,
                                    const block_sparse_halo_row_t *row, int first_image_block,
                                    int image_blocks, int text_group_start);

int block_sparse_halo_build_dma_descs(dma_desc_2d_t *descs, int desc_capacity, const void *k_base,
                                      const void *v_base, void *raw_bank, int n_kv_heads, int kv_head,
                                      int head_dim, int k_blocks_per_head, const block_sparse_halo_row_t *row);
int block_sparse_halo_build_dma_row_descs(dma_desc_2d_t *descs, int desc_capacity, const void *k_base,
                                          const void *v_base, void *raw_row, int n_kv_heads, int kv_head,
                                          int head_dim, int image_block_cols, int row_id);

int block_sparse_halo_probe_userdma_scope(void);
int block_sparse_halo_classify_userdma_pair(const int submit_result[2], const int wait_result[2],
                                            const bool data_ok[2], const bool guard_ok[2],
                                            const int worker_index[2]);
int block_sparse_halo_finalize_userdma_scope(int concurrent_scope, bool serialized_ok);

int block_sparse_halo_validate(const int32_t *block_mask, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                               int head_dim, int block_q, int block_k, int n_q_blocks, int n_k_blocks,
                               int mask_format, int max_kept_blocks, int sparse_policy, int image_tokens,
                               int text_tokens, int image_block_rows, int image_block_cols);
int block_sparse_halo_validate_shape(const int32_t *block_mask, int qo_len, int kv_len,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     int block_q, int block_k, int n_q_blocks, int n_k_blocks,
                                     int mask_format, int max_kept_blocks, int sparse_policy,
                                     int image_tokens, int text_tokens,
                                     int image_block_rows, int image_block_cols);
int block_sparse_halo_validate_kv_head_rows(const int32_t *block_mask, int kv_head_idx,
                                            int n_kv_heads, int q_blocks_per_head,
                                            int k_blocks_per_head, int max_kept_blocks,
                                            int image_blocks, int image_block_rows,
                                            int image_block_cols);

int block_sparse_halo_select_plan(block_sparse_halo_plan_t *plan, int halo_kv_load_mode, int head_dim,
                                  int text_tokens, int image_block_cols, size_t available_bytes);
int block_sparse_halo_select_execution(block_sparse_halo_plan_t *plan, int halo_kv_load_mode,
                                       bool geometry_valid, bool dma_preflight_valid, int userdma_scope, int head_dim,
                                       int text_tokens, int image_block_cols, size_t available_bytes);
int block_sparse_halo_dma_preflight(const void *k_base, const void *v_base, const void *vtcm_base,
                                    size_t vtcm_slice_size, unsigned int vtcm_slice_count,
                                    int n_kv_heads, int head_dim, int image_tokens,
                                    const block_sparse_halo_plan_t *plan);
int block_sparse_packed_qk_tile_offset(size_t *offset, int head, int token_tile, int dim_tile,
                                       int token_tiles, int head_dim);
int block_sparse_packed_v_tile_offset(size_t *offset, int head, int token_tile, int dim_tile,
                                      int token_tiles, int head_dim);
int block_sparse_halo_init_dma_banks(block_sparse_halo_dma_bank_t *banks[BLOCK_SPARSE_HALO_DMA_BANKS],
                                     dma_desc_2d_t *external_descs[BLOCK_SPARSE_HALO_DMA_BANKS],
                                     void *vtcm, size_t vtcm_bytes, const block_sparse_halo_plan_t *plan);

int block_sparse_score(__fp16 *restrict score, const __fp16 *restrict q_mean, const __fp16 *restrict k_mean,
                       int head_dim, int n_q_blocks, int n_k_blocks);

int naive_flash_attn(float *restrict O, const float *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                     const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim);

// micro-benchmark kernels, don't use directly

#define __vtcm  // only a hint, no real effect

int hmx_mat_mul_fp16_core(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a, const __fp16 *restrict __vtcm b,
                          __fp16 *restrict __vtcm scales, int m, int k, int n);

int hvx_mat_mul_fp16_core(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a, const __fp16 *restrict __vtcm b,
                          int m, int k, int n);

int hvx_mat_mul_fp32_core(float *restrict __vtcm c, const float *restrict __vtcm a, const float *restrict __vtcm b,
                          int m, int k, int n);

int hvx_mat_mul_int16_core(int16_t *restrict __vtcm c, const int16_t *restrict __vtcm a,
                           const int16_t *restrict __vtcm b, int m, int k, int n);

int hvx_mat_mul_int32_core(int32_t *restrict __vtcm c, const int32_t *restrict __vtcm a,
                           const int32_t *restrict __vtcm b, int m, int k, int n);

int hvx_mat_mul_fp16_core_mt(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a,
                             const __fp16 *restrict __vtcm b, int M, int K, int N, int n_threads);

/*
 * Times the chunked core's K/V rearrange primitives without running an
 * attention operator: one "block" is the work the kernel does for one consumed
 * K/V block (a K [32, head_dim] transpose plus a V [32, head_dim] pack, 4 KB
 * each). Pass a `variant_mask` built from HTP_ATTN_RB_VARIANT_* and an `src`
 * with at least `rows_total` rows of `row_stride_bytes`. `sum_us` and `blocks`
 * are indexed by variant bit (0 = VTCM, 1 = HOT, 2 = COLD, 3 = COPY) and are
 * summed over the worker slices.
 */
int attn_rearrange_bench(const __fp16 *restrict src, int row_stride_bytes, int head_dim, int n_blocks,
                         int iters, int rows_total, int variant_mask, int32_t *sum_us, int32_t *blocks,
                         int *workers);

#ifdef __cplusplus
}

/* Source-compatible C++ shims for pre-row-ring contract tests. */
static inline int block_sparse_halo_select_plan(block_sparse_halo_plan_t *plan, int halo_kv_load_mode,
                                                int head_dim, int text_tokens, size_t available_bytes) {
  return block_sparse_halo_select_plan(plan, halo_kv_load_mode, head_dim, text_tokens, 9, available_bytes);
}

static inline int block_sparse_halo_select_execution(block_sparse_halo_plan_t *plan, int halo_kv_load_mode,
                                                     bool geometry_valid, bool dma_preflight_valid,
                                                     int userdma_scope, int head_dim, int text_tokens,
                                                     size_t available_bytes) {
  return block_sparse_halo_select_execution(plan, halo_kv_load_mode, geometry_valid, dma_preflight_valid,
                                            userdma_scope, head_dim, text_tokens, 9, available_bytes);
}
#endif

#ifdef __cplusplus

namespace op_utils {

int compare_result(const float *x, const float *y, int n_elems);

}

#endif
