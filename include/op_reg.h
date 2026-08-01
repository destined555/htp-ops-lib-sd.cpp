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
};

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

#ifdef __cplusplus
static_assert(sizeof(QkvMulRmsPackedParams) == 84);
#else
_Static_assert(sizeof(struct QkvMulRmsPackedParams) == 84,
               "unexpected QKV MUL payload size");
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
} __attribute__((packed));

enum HtpHaloExecTraceMode {
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
static_assert(sizeof(BlockSparseAttnParams) == 144);
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
_Static_assert(sizeof(struct BlockSparseAttnParams) == 144, "BlockSparseAttnParams size changed");
#endif

struct BlockSparseScoreParams {
  struct RpcmemBufAddr score;
  struct RpcmemBufAddr q_mean;
  struct RpcmemBufAddr k_mean;
  int32_t head_dim;
  int32_t n_q_blocks;
  int32_t n_k_blocks;
} __attribute__((packed));
