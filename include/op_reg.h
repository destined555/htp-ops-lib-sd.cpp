#pragma once

#include <stddef.h>
#include <stdint.h>

enum HtpSparsePolicy {
  HTP_SPARSE_POLICY_NONE        = 0,
  HTP_SPARSE_POLICY_STATIC_HALO = 1,
};

enum HtpHaloKvLoadMode {
  HTP_HALO_KV_LOAD_BASELINE    = 0,
  HTP_HALO_KV_LOAD_L2_PREFETCH = 1,
  HTP_HALO_KV_LOAD_USERDMA_AB  = 2,
  HTP_HALO_KV_LOAD_CHUNKED     = 3,
  /* Same chunked core, but the in-kernel K/V reuse (fixed image slots plus the
     resident text tail) is turned off, so every q re-fetches and re-arranges its
     K/V blocks. Used to measure the reuse win without rebuilding the skel. */
  HTP_HALO_KV_LOAD_CHUNKED_NO_REUSE = 4,
};

// Keep packed shape admission identical on the HLOS and DSP sides.
static inline int htp_packed_head_count_supported(int n_heads) {
  return n_heads == 24 || n_heads == 30 || n_heads == 38;
}

static inline int htp_packed_head_dim_supported(int head_dim) {
  return head_dim == 64 || head_dim == 128;
}

static inline int htp_packed_shape_supported(int n_heads, int n_kv_heads,
                                              int head_dim, int require_equal_kv) {
  return n_heads > 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0 &&
         htp_packed_head_count_supported(n_heads) &&
         htp_packed_head_dim_supported(head_dim) &&
         (n_heads != 38 || head_dim == 64) &&
         (!require_equal_kv || (n_heads == n_kv_heads &&
                                htp_packed_head_count_supported(n_kv_heads)));
}

enum HtpOpsIndex {
  HTP_OPS_RMS_NORM_F32,
  HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
  HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
  HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16,
  HTP_OPS_BLOCK_SPARSE_SCORE_QO_F32_KM_F16,
  HTP_OPS_QKV_MUL_RMS_PACKED_W16A32,
  HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16,
  HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_W16A32,
  HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32,
  HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32,
  HTP_OPS_FUSED_ANCHOR_ATTN_QO_F32_KV_F16,
  /* Diagnostic op: times the chunked core's K/V rearrange primitives on their
     own, with the source either VTCM-resident, cache-hot or DDR-cold, so the
     rearrange floor can be separated from the exposed fetch time. */
  HTP_OPS_ATTN_REARRANGE_BENCH,
  HTP_OPS_COUNT,
};

struct RpcmemBufAddr {
  int32_t fd;
  int32_t offset;
} __attribute__((packed));

struct RmsNormF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne0;
  int32_t       ne1;
} __attribute__((packed));

struct MatMulParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr activation; // m * k
  struct RpcmemBufAddr weight; // k * n
  int32_t m;
  int32_t k;
  int32_t n;
} __attribute__((packed));

struct QkvMulRmsPackedParams {
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k_out;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr activation;
  struct RpcmemBufAddr weight;
  struct RpcmemBufAddr bias;
  struct RpcmemBufAddr q_norm;
  struct RpcmemBufAddr k_norm;
  int32_t tokens;
  int32_t input_dim;
  int32_t n_heads;
  int32_t head_dim;
  int32_t elapsed_us;
} __attribute__((packed));

struct QkvMulRmsPackedDualParams {
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k_out;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr image_activation;
  struct RpcmemBufAddr image_weight;
  struct RpcmemBufAddr image_bias;
  struct RpcmemBufAddr image_q_norm;
  struct RpcmemBufAddr image_k_norm;
  struct RpcmemBufAddr text_activation;
  struct RpcmemBufAddr text_weight;
  struct RpcmemBufAddr text_bias;
  struct RpcmemBufAddr text_q_norm;
  struct RpcmemBufAddr text_k_norm;
  int32_t image_tokens;
  int32_t text_tokens;
  int32_t input_dim;
  int32_t n_heads;
  int32_t head_dim;
  int32_t elapsed_us;
} __attribute__((packed));

struct QkvMulRmsPackedBufferSizes {
  size_t combined_tokens;
  size_t token_tiles;
  size_t qkv_bytes;
  size_t activation_bytes;
  size_t weight_bytes;
  size_t bias_bytes;
  size_t norm_bytes;
};

struct QkvMulRmsPackedRopeParams {
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k_out;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr activation;
  struct RpcmemBufAddr weight;
  struct RpcmemBufAddr bias;
  struct RpcmemBufAddr q_norm;
  struct RpcmemBufAddr k_norm;
  struct RpcmemBufAddr rope_pe;
  int32_t tokens;
  int32_t input_dim;
  int32_t n_heads;
  int32_t head_dim;
  int32_t rope_token_offset;
  int32_t rope_token_count;
  int32_t elapsed_us;
} __attribute__((packed));

struct QkvMulRmsPackedDualRopeParams {
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k_out;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr image_activation;
  struct RpcmemBufAddr image_weight;
  struct RpcmemBufAddr image_bias;
  struct RpcmemBufAddr image_q_norm;
  struct RpcmemBufAddr image_k_norm;
  struct RpcmemBufAddr text_activation;
  struct RpcmemBufAddr text_weight;
  struct RpcmemBufAddr text_bias;
  struct RpcmemBufAddr text_q_norm;
  struct RpcmemBufAddr text_k_norm;
  struct RpcmemBufAddr rope_pe;
  int32_t image_tokens;
  int32_t text_tokens;
  int32_t input_dim;
  int32_t n_heads;
  int32_t head_dim;
  int32_t image_rope_token_offset;
  int32_t text_rope_token_offset;
  int32_t elapsed_us;
} __attribute__((packed));

#ifdef __cplusplus
static_assert(sizeof(QkvMulRmsPackedParams) == 84);
static_assert(sizeof(QkvMulRmsPackedDualParams) == 128);
static_assert(sizeof(QkvMulRmsPackedRopeParams) == 100);
static_assert(sizeof(QkvMulRmsPackedDualRopeParams) == 144);
#else
_Static_assert(sizeof(struct QkvMulRmsPackedParams) == 84,
               "unexpected QKV MUL payload size");
_Static_assert(sizeof(struct QkvMulRmsPackedRopeParams) == 100,
               "unexpected QKV MUL RoPE payload size");
_Static_assert(sizeof(struct QkvMulRmsPackedDualRopeParams) == 144,
               "unexpected QKV MUL dual RoPE payload size");
#endif

struct FlashAttnParams {
  struct RpcmemBufAddr o;
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr mask;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t impl;
  int32_t with_lse;
  int32_t elapsed_us;
} __attribute__((packed));

struct FusedAnchorAttnParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr dense_mask;
  struct RpcmemBufAddr block_mask;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t block_q;
  int32_t block_k;
  int32_t n_q_blocks;
  int32_t n_k_blocks;
  int32_t max_kept_blocks;
  int32_t mask_format;
  int32_t sparse_policy;
  int32_t image_tokens;
  int32_t text_tokens;
  int32_t image_block_rows;
  int32_t image_block_cols;
  int32_t elapsed_us;
} __attribute__((packed));

#ifdef __cplusplus
static_assert(offsetof(FusedAnchorAttnParams, elapsed_us) == 112);
static_assert(sizeof(FusedAnchorAttnParams) == 116);
#else
_Static_assert(offsetof(struct FusedAnchorAttnParams, elapsed_us) == 112,
               "fused anchor elapsed_us offset changed");
_Static_assert(sizeof(struct FusedAnchorAttnParams) == 116,
               "FusedAnchorAttnParams size changed");
#endif

struct BlockSparseAttnParams {
  struct RpcmemBufAddr o;
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr block_mask;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t block_q;
  int32_t block_k;
  int32_t n_q_blocks;
  int32_t n_k_blocks;
  int32_t mask_format;
  int32_t max_kept_blocks;
  int32_t with_lse;
  int32_t elapsed_us;
  int32_t sparse_policy;
  int32_t image_tokens;
  int32_t text_tokens;
  int32_t halo_kv_load_mode;
  int32_t halo_trace;
  int32_t image_block_rows;
  int32_t image_block_cols;
  int32_t halo_exec_mode;
  int32_t halo_image_rows_updated;
  int32_t halo_text_q_image_row_updates;
  int32_t halo_text_q_image_block_updates;
  int32_t halo_image_q_batch_updates;
  int32_t halo_image_q_segment_updates;
  /* HaloV2: DDR->VTCM read accounting. Byte counters are carried in KB so a
     single int32 covers every realistic shape; the exact byte values stay on
     the DSP side and are printed in the HTP_ATTN_METRICS FARF line. */
  int32_t halo_metrics_present;  /* 1 when the metrics build produced counters */
  int32_t halo_route_mismatch;   /* 1 when the run did not take the requested route */
  int32_t halo_read_kb;          /* total issued DDR->VTCM read bytes, KB */
  int32_t halo_q_kb;
  int32_t halo_text_kv_kb;
  int32_t halo_image_kv_kb;
  int32_t halo_misc_kb;
  int32_t halo_transfer_count;   /* loader issues: DMA descriptors / K-V tile loads */
  int32_t halo_ddr_to_vtcm_us;
  int32_t halo_overlap_us;
  int32_t halo_inflight_us;
  int32_t halo_pack_us;
  int32_t halo_submit_us;
  int32_t halo_worker_elapsed_us; /* summed per-task worker wall time, us */
} __attribute__((packed));

enum HtpHaloExecTraceMode {
  HTP_HALO_EXEC_TRACE_L2_PREFETCH = 1,
  HTP_HALO_EXEC_TRACE_USERDMA_ROWS = 2,
  HTP_HALO_EXEC_TRACE_CHUNKED_FALLBACK = 3,
};

#ifdef __cplusplus
static_assert(offsetof(BlockSparseAttnParams, elapsed_us) == 88);
static_assert(offsetof(BlockSparseAttnParams, sparse_policy) == 92);
static_assert(offsetof(BlockSparseAttnParams, image_block_rows) == 112);
static_assert(offsetof(BlockSparseAttnParams, image_block_cols) == 116);
static_assert(offsetof(BlockSparseAttnParams, halo_exec_mode) == 120);
static_assert(offsetof(BlockSparseAttnParams, halo_image_q_batch_updates) == 136);
static_assert(offsetof(BlockSparseAttnParams, halo_image_q_segment_updates) == 140);
static_assert(offsetof(BlockSparseAttnParams, halo_metrics_present) == 144);
static_assert(offsetof(BlockSparseAttnParams, halo_transfer_count) == 172);
static_assert(offsetof(BlockSparseAttnParams, halo_submit_us) == 192);
static_assert(offsetof(BlockSparseAttnParams, halo_worker_elapsed_us) == 196);
static_assert(sizeof(BlockSparseAttnParams) == 200);
#else
_Static_assert(offsetof(struct BlockSparseAttnParams, elapsed_us) == 88, "elapsed_us offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, sparse_policy) == 92, "sparse_policy offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, image_block_rows) == 112,
               "image_block_rows offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, image_block_cols) == 116,
               "image_block_cols offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, halo_exec_mode) == 120,
               "halo_exec_mode offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, halo_image_q_batch_updates) == 136,
               "halo_image_q_batch_updates offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, halo_image_q_segment_updates) == 140,
               "halo_image_q_segment_updates offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, halo_metrics_present) == 144,
               "halo_metrics_present offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, halo_transfer_count) == 172,
               "halo_transfer_count offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, halo_submit_us) == 192,
               "halo_submit_us offset changed");
_Static_assert(offsetof(struct BlockSparseAttnParams, halo_worker_elapsed_us) == 196,
               "halo_worker_elapsed_us offset changed");
_Static_assert(sizeof(struct BlockSparseAttnParams) == 200, "BlockSparseAttnParams size changed");
#endif

struct BlockSparseScoreParams {
  struct RpcmemBufAddr score;
  struct RpcmemBufAddr q_mean;
  struct RpcmemBufAddr k_mean;
  int32_t head_dim;
  int32_t n_q_blocks;
  int32_t n_k_blocks;
} __attribute__((packed));


/*
 * Rearrange micro-benchmark variants. The chunked core's load window is "issue
 * the prefetch, then load each K/V row and scatter/shuffle it into the HMX tile
 * layout"; l2fetch is only a hint, so the DDR latency is folded into the very
 * loads that feed the rearrange. The bench separates the two by controlling
 * where the source lives instead of adding timers:
 *
 *   VTCM -> pure instruction cost of the rearrange
 *   HOT  -> same loop with cache-resident rows
 *   COLD -> same loop walking fresh DDR cache lines (real fetch)
 *   COPY -> VTCM->VTCM block copy, i.e. what a reuse cache hit costs
 */
enum AttnRearrangeBenchVariant {
  HTP_ATTN_RB_VARIANT_VTCM = 1,
  HTP_ATTN_RB_VARIANT_HOT  = 2,
  HTP_ATTN_RB_VARIANT_COLD = 4,
  HTP_ATTN_RB_VARIANT_COPY = 8,
};

struct AttnRearrangeBenchParams {
  struct RpcmemBufAddr src; /* fp16 rows, row_stride_bytes apart, rows_total rows */
  int32_t              row_stride_bytes;
  int32_t              head_dim;
  int32_t              n_blocks;     /* 32-token blocks per iteration */
  int32_t              iters;        /* iterations per worker, per variant */
  int32_t              rows_total;   /* rows available in src; the cold walk wraps inside */
  int32_t              variant_mask; /* HTP_ATTN_RB_VARIANT_* bitmask */
  int32_t              workers;      /* filled by the DSP: worker slices used */
  int32_t              sum_us[4];    /* filled by the DSP: summed over workers */
  int32_t              blocks[4];    /* filled by the DSP: blocks timed, summed over workers */
} __attribute__((packed));
