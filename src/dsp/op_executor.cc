#include "dsp/op_executor.h"

#include <qurt_memory.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "dsp/mmap_mgr.h"
#include "dsp/ops.h"
#include "op_reg.h"

// debug
#include <HAP_farf.h>
#include <HAP_perf.h>

#include "dsp/hmx_utils.h"

namespace {

size_t align_up_size(size_t v, size_t a) {
  return ((v + a - 1) / a) * a;
}

bool checked_size_add(size_t a, size_t b, size_t *result) {
  if (result == nullptr || a > SIZE_MAX - b) {
    return false;
  }
  *result = a + b;
  return true;
}

bool checked_size_mul(size_t a, size_t b, size_t *result) {
  if (result == nullptr || (a != 0 && b > SIZE_MAX / a)) {
    return false;
  }
  *result = a * b;
  return true;
}

constexpr size_t kPackedZeroBiasFloats = 3u * 30u * 128u;

size_t ggml_super_block_size(enum ggml_type type) {
  // TODO: more types
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_IQ4_NL:
      return sizeof(my_block_q4_0);
    case GGML_TYPE_Q8_0:
      return sizeof(my_block_q8_0);
    default:
      return -1;
  }
}

enum ggml_type matmul_op_to_weight_type(enum HtpOpsIndex op) {
  switch (op) {
    case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
      return GGML_TYPE_F16;
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
      return GGML_TYPE_Q4_0;
    case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
      return GGML_TYPE_Q8_0;
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL:
      return GGML_TYPE_IQ4_NL;
    default:
      return GGML_TYPE_COUNT;  // invalid type
  }
}

constexpr size_t kBlockSparseAttnLegacyPayloadSize = offsetof(BlockSparseAttnParams, sparse_policy);
static_assert(kBlockSparseAttnLegacyPayloadSize == 92);
static_assert(offsetof(BlockSparseAttnParams, elapsed_us) + sizeof(int32_t) ==
              kBlockSparseAttnLegacyPayloadSize);
constexpr size_t kBlockSparseAttnHaloV1PayloadSize =
  offsetof(BlockSparseAttnParams, halo_image_q_batch_updates);
static_assert(kBlockSparseAttnHaloV1PayloadSize == 136);
/* Full pre-metrics struct: every HaloV1 field is present but the sender knows
   nothing about the DDR->VTCM tail. Still accepted, metrics stay untouched. */
constexpr size_t kBlockSparseAttnHaloTracePayloadSize =
  offsetof(BlockSparseAttnParams, halo_metrics_present);
static_assert(kBlockSparseAttnHaloTracePayloadSize == 144);
/* HaloV2 = HaloV1 fields + the metrics tail. */
constexpr size_t kBlockSparseAttnHaloV2PayloadSize = sizeof(BlockSparseAttnParams);
static_assert(kBlockSparseAttnHaloV2PayloadSize == 200);

/* DDR->VTCM counters are only collected when the DSP skel is built with
   HTP_ATTN_METRICS; otherwise the fields stay zero and the wire flag reports
   that no metrics were produced. */
#ifdef HTP_ATTN_METRICS
constexpr bool kBlockSparseAttnMetricsBuilt = true;
#else
constexpr bool kBlockSparseAttnMetricsBuilt = false;
#endif

}  // namespace

extern "C" {

#define IN_PTR(i)  std::get<0>(in_bufs[i])
#define OUT_PTR(i) std::get<0>(out_bufs[i])

int block_sparse_attn_buffer_sizes(const struct BlockSparseAttnParams *params,
                                   struct BlockSparseAttnBufferSizes *sizes) {
  if (params == nullptr || sizes == nullptr || params->qo_len <= 0 || params->kv_len <= 0 ||
      params->n_heads <= 0 || params->n_kv_heads <= 0 || params->head_dim <= 0 ||
      params->n_q_blocks <= 0 || params->n_k_blocks <= 0 ||
      (params->mask_format != 0 && params->mask_format != 1) ||
      (params->mask_format == 1 && params->max_kept_blocks <= 0)) {
    return -1;
  }

  size_t q_elements;
  size_t q_size;
  size_t o_head_dim;
  size_t o_elements;
  size_t o_size;
  size_t kv_elements;
  size_t kv_size;
  size_t block_mask_elements;
  size_t block_mask_size;
  const size_t mask_row_width = params->mask_format == 1 ?
    (size_t) params->max_kept_blocks : (size_t) params->n_k_blocks;
  if (!checked_size_mul((size_t) params->qo_len, (size_t) params->n_heads, &q_elements) ||
      !checked_size_mul(q_elements, (size_t) params->head_dim, &q_elements) ||
      !checked_size_mul(q_elements, sizeof(float), &q_size) ||
      !checked_size_add((size_t) params->head_dim, params->with_lse != 0 ? 1u : 0u, &o_head_dim) ||
      !checked_size_mul((size_t) params->qo_len, (size_t) params->n_heads, &o_elements) ||
      !checked_size_mul(o_elements, o_head_dim, &o_elements) ||
      !checked_size_mul(o_elements, sizeof(float), &o_size) ||
      !checked_size_mul((size_t) params->kv_len, (size_t) params->n_kv_heads, &kv_elements) ||
      !checked_size_mul(kv_elements, (size_t) params->head_dim, &kv_elements) ||
      !checked_size_mul(kv_elements, sizeof(__fp16), &kv_size) ||
      !checked_size_mul((size_t) params->n_q_blocks, mask_row_width, &block_mask_elements) ||
      !checked_size_mul(block_mask_elements, sizeof(int32_t), &block_mask_size)) {
    return -1;
  }

  *sizes = {
    .q = q_size,
    .o = o_size,
    .kv = kv_size,
    .block_mask = block_mask_size,
  };
  return 0;
}

int fused_anchor_attn_buffer_sizes(const struct FusedAnchorAttnParams *params,
                                   struct FusedAnchorAttnBufferSizes *sizes) {
  if (params == nullptr || sizes == nullptr || params->qo_len <= 0 || params->kv_len <= 0 ||
      params->qo_len != params->kv_len || params->n_heads <= 0 ||
      params->n_heads != params->n_kv_heads ||
      (params->head_dim != 64 && params->head_dim != 128) ||
      params->block_q != 32 || params->block_k != 32 ||
      params->mask_format != 1 || params->sparse_policy != HTP_SPARSE_POLICY_STATIC_HALO ||
      params->max_kept_blocks <= 0 || params->image_tokens <= 0 || params->text_tokens < 0 ||
      params->image_tokens % params->block_q != 0 ||
      params->image_tokens > params->qo_len || params->text_tokens != params->qo_len - params->image_tokens ||
      params->image_block_rows <= 0 || params->image_block_cols <= 0) {
    return -1;
  }

  const int q_blocks_per_head = params->qo_len / params->block_q +
                                (params->qo_len % params->block_q != 0);
  const int k_blocks_per_head = params->kv_len / params->block_k +
                                (params->kv_len % params->block_k != 0);
  if (q_blocks_per_head <= 0 || k_blocks_per_head <= 0 ||
      params->n_heads > INT_MAX / q_blocks_per_head ||
      params->n_kv_heads > INT_MAX / k_blocks_per_head ||
      params->n_q_blocks != params->n_heads * q_blocks_per_head ||
      params->n_k_blocks != params->n_kv_heads * k_blocks_per_head ||
      params->max_kept_blocks != k_blocks_per_head ||
      params->image_block_rows > INT_MAX / params->image_block_cols ||
      params->image_block_rows * params->image_block_cols != params->image_tokens / params->block_q) {
    return -1;
  }

  size_t row_count = 0;
  size_t row_width = 0;
  size_t slab_elements = 0;
  size_t slab_size = 0;
  size_t output_size = 0;
  size_t q_elements = 0;
  size_t q_size = 0;
  size_t kv_elements = 0;
  size_t kv_size = 0;
  size_t dense_mask_stride = 0;
  size_t dense_mask_elements = 0;
  size_t dense_mask_size = 0;
  size_t block_mask_elements = 0;
  size_t block_mask_size = 0;
  if (!checked_size_mul((size_t) params->qo_len, (size_t) params->n_heads, &row_count) ||
      !checked_size_add((size_t) params->head_dim, 1u, &row_width) ||
      !checked_size_mul(row_count, row_width, &slab_elements) ||
      !checked_size_mul(slab_elements, sizeof(float), &slab_size) ||
      !checked_size_mul(slab_size, 2u, &output_size) ||
      !checked_size_mul(row_count, (size_t) params->head_dim, &q_elements) ||
      !checked_size_mul(q_elements, sizeof(float), &q_size) ||
      !checked_size_mul((size_t) params->kv_len, (size_t) params->n_kv_heads, &kv_elements) ||
      !checked_size_mul(kv_elements, (size_t) params->head_dim, &kv_elements) ||
      !checked_size_mul(kv_elements, sizeof(__fp16), &kv_size) ||
      !checked_size_add((size_t) params->kv_len, 63u, &dense_mask_stride)) {
    return -1;
  }
  dense_mask_stride &= ~(size_t) 63u;
  if (!checked_size_mul((size_t) params->qo_len, dense_mask_stride, &dense_mask_elements) ||
      !checked_size_mul(dense_mask_elements, sizeof(__fp16), &dense_mask_size) ||
      !checked_size_mul((size_t) params->n_q_blocks, (size_t) params->max_kept_blocks,
                        &block_mask_elements) ||
      !checked_size_mul(block_mask_elements, sizeof(int32_t), &block_mask_size)) {
    return -1;
  }

  *sizes = {
    .output = output_size,
    .slab = slab_size,
    .q = q_size,
    .kv = kv_size,
    .dense_mask = dense_mask_size,
    .block_mask = block_mask_size,
  };
  return 0;
}

int qkv_mul_rms_packed_dual_buffer_sizes(const struct QkvMulRmsPackedDualParams *params,
                                         struct QkvMulRmsPackedBufferSizes *sizes) {
  if (params == nullptr || sizes == nullptr || params->image_tokens <= 0 || params->text_tokens < 0 ||
      params->image_tokens % 32 != 0 ||
      params->input_dim <= 0 || params->input_dim % 32 != 0 ||
      !htp_packed_shape_supported(params->n_heads, params->n_heads, params->head_dim, 1)) {
    return -1;
  }
  const size_t image_tokens = (size_t) params->image_tokens;
  const size_t text_tokens = (size_t) params->text_tokens;
  const size_t combined_tokens = image_tokens + text_tokens;
  const size_t token_tiles = (combined_tokens + 31u) / 32u;
  size_t qkv_elements = 0;
  size_t weight_cols = 0;
  if (!checked_size_mul(token_tiles, (size_t) params->n_heads, &qkv_elements) ||
      !checked_size_mul(qkv_elements, (size_t) params->head_dim / 32u * 1024u, &qkv_elements) ||
      !checked_size_mul(qkv_elements, sizeof(__fp16), &sizes->qkv_bytes) ||
      !checked_size_mul(combined_tokens, (size_t) params->input_dim, &sizes->activation_bytes) ||
      !checked_size_mul(sizes->activation_bytes, sizeof(float), &sizes->activation_bytes) ||
      !checked_size_mul((size_t) params->input_dim, (size_t) params->n_heads * params->head_dim * 3u, &weight_cols) ||
      !checked_size_mul(weight_cols, sizeof(__fp16), &sizes->weight_bytes) ||
      !checked_size_mul((size_t) params->n_heads * params->head_dim * 3u, sizeof(float), &sizes->bias_bytes) ||
      !checked_size_mul((size_t) params->head_dim, sizeof(float), &sizes->norm_bytes)) {
    return -1;
  }
  sizes->combined_tokens = combined_tokens;
  sizes->token_tiles = token_tiles;
  return 0;
}

int execute_op_simple(struct OpComputeRequest *req, size_t req_size) {
  if (req == nullptr || req_size < sizeof(OpComputeRequest)) {
    return -1;
  }

  const size_t payload_size = req_size - sizeof(OpComputeRequest);

  // using FatPointer = std::pair<uint8_t *, size_t>;
  using Buffer = std::tuple<uint8_t *, size_t, bool>;
  std::vector<Buffer> in_bufs, out_bufs;

  auto add_buffer = [](std::vector<Buffer> &bufs, const RpcmemBufAddr &buf_addr, size_t size, bool cached = true) {
    auto base = reinterpret_cast<uint8_t *>(mmap_manager_get_map(buf_addr.fd));
    auto ptr  = base != nullptr ? base + buf_addr.offset : nullptr;
    bufs.push_back({ ptr, size, cached });
  };

  auto validate_in_bufs = [&]() {
    for (auto [ptr, size, cached] : in_bufs) {
      if (ptr && cached) {
        qurt_mem_cache_clean((qurt_addr_t) ptr, size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
      }
    }
  };

  auto validate_out_bufs = [&]() {
    for (auto [ptr, size, cached] : out_bufs) {
      if (ptr && cached) {
        qurt_mem_cache_clean((qurt_addr_t) ptr, size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
      }
    }
  };

  int ret = 0;
  switch (req->op) {
    case HTP_OPS_RMS_NORM_F32:
      {
        if (payload_size < sizeof(RmsNormF32Params)) {
          return -1;
        }
        auto   params = reinterpret_cast<RmsNormF32Params *>(req->payload);
        size_t size   = params->ne0 * params->ne1 * sizeof(float);

        add_buffer(out_bufs, params->dst, size);
        add_buffer(in_bufs, params->src, size);

        validate_in_bufs();
        ret = hvx_rms_norm_f32((float *) OUT_PTR(0), (const float *) IN_PTR(0), params->ne0, params->ne1);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
      {
        if (payload_size < sizeof(MatMulParams)) {
          return -1;
        }
        auto params = reinterpret_cast<MatMulParams *>(req->payload);
        int  m = params->m, k = params->k, n = params->n;

        size_t output_size     = m * n * sizeof(float);
        size_t activation_size = m * k * sizeof(float);
        size_t weight_size     = k * n * sizeof(__fp16);

        add_buffer(out_bufs, params->output, output_size);
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size);

        validate_in_bufs();
        ret = hmx_mat_mul_permuted_w16a32((float *) OUT_PTR(0), (float *) IN_PTR(0), (__fp16 *) IN_PTR(1), m, k, n);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_QKV_MUL_RMS_PACKED_W16A32:
      {
        if (payload_size < sizeof(QkvMulRmsPackedParams)) {
          return -1;
        }
        auto params = reinterpret_cast<QkvMulRmsPackedParams *>(req->payload);
        if (params->tokens <= 0 || params->input_dim <= 0 ||
            !htp_packed_shape_supported(params->n_heads, params->n_heads, params->head_dim, 1) ||
            params->input_dim % 32 != 0) {
          return -1;
        }

        const size_t padded_tokens = align_up_size((size_t) params->tokens, 32);
        const size_t head_tile_elems =
          (padded_tokens / 32) * ((size_t) params->head_dim / 32) * HMX_FP16_TILE_N_ELMS;
        size_t qkv_size = 0;
        size_t activation_size = 0;
        size_t weight_cols = 0;
        size_t weight_size = 0;
        size_t bias_size = 0;
        const bool sizes_ok =
          checked_size_mul((size_t) params->n_heads, head_tile_elems, &qkv_size) &&
          checked_size_mul((size_t) params->tokens, (size_t) params->input_dim, &activation_size) &&
          checked_size_mul((size_t) params->n_heads, (size_t) params->head_dim, &weight_cols) &&
          checked_size_mul(weight_cols, 3, &weight_cols) &&
          checked_size_mul((size_t) params->input_dim, weight_cols, &weight_size) &&
          checked_size_mul(weight_size, sizeof(__fp16), &weight_size) &&
          checked_size_mul(weight_cols, sizeof(float), &bias_size) &&
          checked_size_mul(activation_size, sizeof(float), &activation_size);
        if (!sizes_ok) {
          return -1;
        }

        add_buffer(out_bufs, params->q, qkv_size * sizeof(__fp16));
        add_buffer(out_bufs, params->k_out, qkv_size * sizeof(__fp16));
        add_buffer(out_bufs, params->v, qkv_size * sizeof(__fp16));
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size);
        add_buffer(in_bufs, params->bias, bias_size);
        add_buffer(in_bufs, params->q_norm, (size_t) params->head_dim * sizeof(float));
        add_buffer(in_bufs, params->k_norm, (size_t) params->head_dim * sizeof(float));

        validate_in_bufs();
        // The message-channel receiver owns the HMX execution lock for its loop.
        const int64_t t0 = HAP_perf_get_qtimer_count();
        ret = hmx_qkv_mul_rms_packed_w16a32(
          (__fp16 *) OUT_PTR(0), (__fp16 *) OUT_PTR(1), (__fp16 *) OUT_PTR(2),
          (const float *) IN_PTR(0), (const __fp16 *) IN_PTR(1), (const float *) IN_PTR(2),
          (const float *) IN_PTR(3), (const float *) IN_PTR(4), params->tokens, params->input_dim,
          params->n_heads, params->head_dim);
        params->elapsed_us = (int32_t) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
        FARF(ALWAYS, "HTP_ATTN_OP_BENCH kind=qkv_mul_rms_packed elapsed_us=%d", params->elapsed_us);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_W16A32:
      {
        if (payload_size < sizeof(QkvMulRmsPackedDualParams)) {
          return -1;
        }
        auto params = reinterpret_cast<QkvMulRmsPackedDualParams *>(req->payload);
        QkvMulRmsPackedBufferSizes sizes = {};
        if (qkv_mul_rms_packed_dual_buffer_sizes(params, &sizes) != 0) {
          return -1;
        }
        add_buffer(out_bufs, params->q, sizes.qkv_bytes);
        add_buffer(out_bufs, params->k_out, sizes.qkv_bytes);
        add_buffer(out_bufs, params->v, sizes.qkv_bytes);
        add_buffer(in_bufs, params->image_activation,
                   (size_t) params->image_tokens * params->input_dim * sizeof(float));
        add_buffer(in_bufs, params->image_weight, sizes.weight_bytes);
        if (params->image_bias.fd >= 0) add_buffer(in_bufs, params->image_bias, sizes.bias_bytes);
        else in_bufs.push_back({ nullptr, sizes.bias_bytes, false });
        add_buffer(in_bufs, params->image_q_norm, sizes.norm_bytes);
        add_buffer(in_bufs, params->image_k_norm, sizes.norm_bytes);
        add_buffer(in_bufs, params->text_activation,
                   (size_t) params->text_tokens * params->input_dim * sizeof(float));
        add_buffer(in_bufs, params->text_weight, sizes.weight_bytes);
        if (params->text_bias.fd >= 0) add_buffer(in_bufs, params->text_bias, sizes.bias_bytes);
        else in_bufs.push_back({ nullptr, sizes.bias_bytes, false });
        add_buffer(in_bufs, params->text_q_norm, sizes.norm_bytes);
        add_buffer(in_bufs, params->text_k_norm, sizes.norm_bytes);
        validate_in_bufs();
        const int64_t t0 = HAP_perf_get_qtimer_count();
        static float zero_bias[kPackedZeroBiasFloats] __attribute__((aligned(128))) = {};
        if ((params->image_bias.fd < 0 || params->text_bias.fd < 0) &&
            sizes.bias_bytes > sizeof(zero_bias)) {
          return -1;
        }
        ret = hmx_qkv_mul_rms_packed_dual_w16a32(
            (__fp16 *) OUT_PTR(0), (__fp16 *) OUT_PTR(1), (__fp16 *) OUT_PTR(2),
            (const float *) IN_PTR(0), (const __fp16 *) IN_PTR(1),
            params->image_bias.fd >= 0 ? (const float *) IN_PTR(2) : zero_bias,
            (const float *) IN_PTR(3), (const float *) IN_PTR(4), params->image_tokens,
            (const float *) IN_PTR(5), (const __fp16 *) IN_PTR(6),
            params->text_bias.fd >= 0 ? (const float *) IN_PTR(7) : zero_bias,
            (const float *) IN_PTR(8), (const float *) IN_PTR(9), params->text_tokens,
            params->input_dim, params->n_heads, params->head_dim);
        params->elapsed_us = (int32_t) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
        FARF(ALWAYS, "HTP_ATTN_OP_BENCH kind=qkv_mul_rms_packed_dual elapsed_us=%d", params->elapsed_us);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32:
      {
        if (payload_size < sizeof(QkvMulRmsPackedRopeParams)) {
          return -1;
        }
        auto params = reinterpret_cast<QkvMulRmsPackedRopeParams *>(req->payload);
        if (params->tokens <= 0 || params->input_dim <= 0 ||
            !htp_packed_shape_supported(params->n_heads, params->n_heads, params->head_dim, 1) ||
            params->input_dim % 32 != 0 ||
            params->rope_token_offset < 0 || params->rope_token_count <= 0) {
          return -1;
        }
        const size_t padded_tokens = align_up_size((size_t) params->tokens, 32);
        const size_t head_tile_elems = (padded_tokens / 32) *
          ((size_t) params->head_dim / 32) * HMX_FP16_TILE_N_ELMS;
        size_t qkv_size = 0, activation_size = 0, weight_cols = 0, weight_size = 0, bias_size = 0;
        if (!checked_size_mul((size_t) params->n_heads, head_tile_elems, &qkv_size) ||
            !checked_size_mul((size_t) params->tokens, (size_t) params->input_dim, &activation_size) ||
            !checked_size_mul((size_t) params->n_heads, (size_t) params->head_dim * 3u, &weight_cols) ||
            !checked_size_mul((size_t) params->input_dim, weight_cols, &weight_size) ||
            !checked_size_mul(weight_size, sizeof(__fp16), &weight_size) ||
            !checked_size_mul(weight_cols, sizeof(float), &bias_size) ||
            !checked_size_mul(activation_size, sizeof(float), &activation_size)) {
          return -1;
        }
        const size_t rope_pairs = (size_t) params->head_dim / 2u;
        size_t rope_size = 0;
        if (!checked_size_mul((size_t) params->rope_token_count, rope_pairs * 4u, &rope_size) ||
            !checked_size_mul(rope_size, sizeof(float), &rope_size)) {
          return -1;
        }
        add_buffer(out_bufs, params->q, qkv_size * sizeof(__fp16));
        add_buffer(out_bufs, params->k_out, qkv_size * sizeof(__fp16));
        add_buffer(out_bufs, params->v, qkv_size * sizeof(__fp16));
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size);
        if (params->bias.fd >= 0) add_buffer(in_bufs, params->bias, bias_size);
        else in_bufs.push_back({ nullptr, bias_size, false });
        add_buffer(in_bufs, params->q_norm, (size_t) params->head_dim * sizeof(float));
        add_buffer(in_bufs, params->k_norm, (size_t) params->head_dim * sizeof(float));
        add_buffer(in_bufs, params->rope_pe, rope_size);
        validate_in_bufs();
        const int64_t t0 = HAP_perf_get_qtimer_count();
        static float zero_bias[kPackedZeroBiasFloats] __attribute__((aligned(128))) = {};
        if (params->bias.fd < 0 && bias_size > sizeof(zero_bias)) {
          return -1;
        }
        ret = hmx_qkv_mul_rms_packed_rope_w16a32(
          (__fp16 *) OUT_PTR(0), (__fp16 *) OUT_PTR(1), (__fp16 *) OUT_PTR(2),
          (const float *) IN_PTR(0), (const __fp16 *) IN_PTR(1),
          params->bias.fd >= 0 ? (const float *) IN_PTR(2) : zero_bias,
          (const float *) IN_PTR(3), (const float *) IN_PTR(4), (const float *) IN_PTR(5),
          params->rope_token_offset, params->rope_token_count, params->tokens,
          params->input_dim, params->n_heads, params->head_dim);
        params->elapsed_us = (int32_t) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
        FARF(ALWAYS, "HTP_ATTN_OP_BENCH kind=qkv_mul_rms_packed_rope elapsed_us=%d", params->elapsed_us);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32:
      {
        if (payload_size < sizeof(QkvMulRmsPackedDualRopeParams)) {
          return -1;
        }
        auto params = reinterpret_cast<QkvMulRmsPackedDualRopeParams *>(req->payload);
        QkvMulRmsPackedDualParams base = {};
        base.image_tokens = params->image_tokens;
        base.text_tokens = params->text_tokens;
        base.input_dim = params->input_dim;
        base.n_heads = params->n_heads;
        base.head_dim = params->head_dim;
        QkvMulRmsPackedBufferSizes sizes = {};
        if (qkv_mul_rms_packed_dual_buffer_sizes(&base, &sizes) != 0 ||
            params->image_rope_token_offset < 0 || params->text_rope_token_offset < 0) {
          return -1;
        }
        const size_t rope_tokens = (size_t) std::max(
          params->image_rope_token_offset + params->image_tokens,
          params->text_rope_token_offset + params->text_tokens);
        size_t rope_size = 0;
        if (!checked_size_mul(rope_tokens, (size_t) params->head_dim / 2u * 4u, &rope_size) ||
            !checked_size_mul(rope_size, sizeof(float), &rope_size)) {
          return -1;
        }
        add_buffer(out_bufs, params->q, sizes.qkv_bytes);
        add_buffer(out_bufs, params->k_out, sizes.qkv_bytes);
        add_buffer(out_bufs, params->v, sizes.qkv_bytes);
        add_buffer(in_bufs, params->image_activation, (size_t) params->image_tokens * params->input_dim * sizeof(float));
        add_buffer(in_bufs, params->image_weight, sizes.weight_bytes);
        if (params->image_bias.fd >= 0) add_buffer(in_bufs, params->image_bias, sizes.bias_bytes);
        else in_bufs.push_back({ nullptr, sizes.bias_bytes, false });
        add_buffer(in_bufs, params->image_q_norm, sizes.norm_bytes);
        add_buffer(in_bufs, params->image_k_norm, sizes.norm_bytes);
        add_buffer(in_bufs, params->text_activation, (size_t) params->text_tokens * params->input_dim * sizeof(float));
        add_buffer(in_bufs, params->text_weight, sizes.weight_bytes);
        if (params->text_bias.fd >= 0) add_buffer(in_bufs, params->text_bias, sizes.bias_bytes);
        else in_bufs.push_back({ nullptr, sizes.bias_bytes, false });
        add_buffer(in_bufs, params->text_q_norm, sizes.norm_bytes);
        add_buffer(in_bufs, params->text_k_norm, sizes.norm_bytes);
        add_buffer(in_bufs, params->rope_pe, rope_size);
        validate_in_bufs();
        const int64_t t0 = HAP_perf_get_qtimer_count();
        static float zero_bias[kPackedZeroBiasFloats] __attribute__((aligned(128))) = {};
        if ((params->image_bias.fd < 0 || params->text_bias.fd < 0) &&
            sizes.bias_bytes > sizeof(zero_bias)) {
          return -1;
        }
        ret = hmx_qkv_mul_rms_packed_dual_rope_w16a32(
          (__fp16 *) OUT_PTR(0), (__fp16 *) OUT_PTR(1), (__fp16 *) OUT_PTR(2),
          (const float *) IN_PTR(0), (const __fp16 *) IN_PTR(1),
          params->image_bias.fd >= 0 ? (const float *) IN_PTR(2) : zero_bias,
          (const float *) IN_PTR(3), (const float *) IN_PTR(4), params->image_tokens,
          (const float *) IN_PTR(5), (const __fp16 *) IN_PTR(6),
          params->text_bias.fd >= 0 ? (const float *) IN_PTR(7) : zero_bias,
          (const float *) IN_PTR(8), (const float *) IN_PTR(9), params->text_tokens,
          (const float *) IN_PTR(10), params->image_rope_token_offset,
          params->text_rope_token_offset, params->input_dim, params->n_heads, params->head_dim);
        params->elapsed_us = (int32_t) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
        FARF(ALWAYS, "HTP_ATTN_OP_BENCH kind=qkv_mul_rms_packed_dual_rope elapsed_us=%d", params->elapsed_us);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
    case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL:
      {
        if (payload_size < sizeof(MatMulParams)) {
          return -1;
        }
        auto   weight_type      = matmul_op_to_weight_type(static_cast<HtpOpsIndex>(req->op));
        size_t super_block_size = ggml_super_block_size(weight_type);

        auto params = reinterpret_cast<MatMulParams *>(req->payload);
        int  m = params->m, k = params->k, n = params->n;

        size_t output_size     = m * n * sizeof(float);
        size_t activation_size = m * k * sizeof(float);
        size_t weight_size     = k * n / QK_K * super_block_size;

        add_buffer(out_bufs, params->output, output_size);
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size, false);

        // int64_t t0 = HAP_perf_get_qtimer_count();
        validate_in_bufs();
        // int64_t t1 = HAP_perf_get_qtimer_count();
        ret =
          hmx_mat_mul_permuted_qk_0_d16a32((float *) OUT_PTR(0), (float *) IN_PTR(0), IN_PTR(1), m, k, n, weight_type);
        // int64_t t2 = HAP_perf_get_qtimer_count();
        validate_out_bufs();
        // int64_t t3 = HAP_perf_get_qtimer_count();

        // int64_t mm_time_us  = HAP_perf_qtimer_count_to_us(t2 - t1);
        // int64_t tot_time_us = HAP_perf_qtimer_count_to_us(t3 - t0);

        // FARF(ALWAYS, "mm_time: %lld us, tot_time: %lld us, type: %d, (%d, %d, %d)", mm_time_us, tot_time_us,
        //      weight_type, m, k, n);
        // FARF(ALWAYS, "achieved weight load bandwidth: %.2f GB/s", 1e-3 * weight_size / mm_time_us);
        // FARF(ALWAYS, "achieved GEMM throughput: %.2f GFLOPS", 2e-3 * m * n * k / mm_time_us);
      }
      break;

    case HTP_OPS_FLASH_ATTN_QO_F32_KV_F16:
      {
        if (payload_size < sizeof(FlashAttnParams)) {
          return -1;
        }
        auto params = reinterpret_cast<FlashAttnParams *>(req->payload);

        int qo_len     = params->qo_len;
        int kv_len     = params->kv_len;
        int n_heads    = params->n_heads;
        int n_kv_heads = params->n_kv_heads;
        int head_dim   = params->head_dim;
        bool with_lse  = params->with_lse != 0;

        size_t q_size    = qo_len * n_heads * head_dim * sizeof(float);
        size_t o_size    = qo_len * n_heads * (head_dim + (with_lse ? 1 : 0)) * sizeof(float);
        size_t kv_size   = kv_len * n_kv_heads * head_dim * sizeof(__fp16);
        size_t mask_size = qo_len * align_up_size(kv_len, 64) * sizeof(__fp16);

        add_buffer(out_bufs, params->o, o_size);
        add_buffer(in_bufs, params->q, q_size);
        add_buffer(in_bufs, params->k, kv_size);
        add_buffer(in_bufs, params->v, kv_size);
        add_buffer(in_bufs, params->mask, mask_size);

        constexpr bool check_accuracy = false;

        if (check_accuracy) {
          float *ref_out;
          posix_memalign((void **) &ref_out, 128, o_size);

          validate_in_bufs();
          ret = simple_flash_attn((__fp16 *) ref_out, (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1), (__fp16 *) IN_PTR(2),
                                  (__fp16 *) IN_PTR(3), qo_len, kv_len, n_heads, n_kv_heads, head_dim, with_lse);

          // check logic
          naive_flash_attn((float *) OUT_PTR(0), (float *) IN_PTR(0), (__fp16 *) IN_PTR(1), (__fp16 *) IN_PTR(2),
                           (__fp16 *) IN_PTR(3), qo_len, kv_len, n_heads, n_kv_heads, head_dim);

          op_utils::compare_result((float *) OUT_PTR(0), ref_out, o_size / 4);

          validate_out_bufs();

          free(ref_out);
        } else {
          validate_in_bufs();
          const int64_t t0 = HAP_perf_get_qtimer_count();
          ret =
            simple_flash_attn((__fp16 *) OUT_PTR(0), (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1), (__fp16 *) IN_PTR(2),
                              (__fp16 *) IN_PTR(3), qo_len, kv_len, n_heads, n_kv_heads, head_dim, with_lse);
          const int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
          params->elapsed_us = (int32_t) elapsed_us;
          FARF(ALWAYS, "HTP_ATTN_OP_BENCH kind=simple elapsed_us=%lld", elapsed_us);
          validate_out_bufs();
        }
      }
      break;

    case HTP_OPS_FUSED_ANCHOR_ATTN_QO_F32_KV_F16:
      {
        if (payload_size < sizeof(FusedAnchorAttnParams)) {
          return -1;
        }
        auto *params = reinterpret_cast<FusedAnchorAttnParams *>(req->payload);
        params->elapsed_us = 0;

        FusedAnchorAttnBufferSizes buffer_sizes = {};
        if (fused_anchor_attn_buffer_sizes(params, &buffer_sizes) != 0) {
          return -1;
        }
        add_buffer(out_bufs, params->output, buffer_sizes.output);
        add_buffer(in_bufs, params->q, buffer_sizes.q);
        add_buffer(in_bufs, params->k, buffer_sizes.kv);
        add_buffer(in_bufs, params->v, buffer_sizes.kv);
        add_buffer(in_bufs, params->dense_mask, buffer_sizes.dense_mask);
        add_buffer(in_bufs, params->block_mask, buffer_sizes.block_mask);

        validate_in_bufs();
        const int64_t t0 = HAP_perf_get_qtimer_count();
        ret = fused_anchor_flash_attn(
          (float *) OUT_PTR(0), (const float *) IN_PTR(0), (const __fp16 *) IN_PTR(1),
          (const __fp16 *) IN_PTR(2), (const __fp16 *) IN_PTR(3),
          (const int32_t *) IN_PTR(4), params);
        const int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
        params->elapsed_us = (int32_t) elapsed_us;
        FARF(ALWAYS, "HTP_ATTN_OP_BENCH kind=fused_anchor elapsed_us=%lld", elapsed_us);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_ATTN_REARRANGE_BENCH:
      {
        if (payload_size < sizeof(AttnRearrangeBenchParams)) {
          return -1;
        }
        auto        *params    = reinterpret_cast<AttnRearrangeBenchParams *>(req->payload);
        const size_t src_bytes = (size_t) params->rows_total * (size_t) params->row_stride_bytes;
        add_buffer(in_bufs, params->src, src_bytes);
        validate_in_bufs();

        int32_t sum_us[4] = { 0, 0, 0, 0 };
        int32_t blocks[4] = { 0, 0, 0, 0 };
        int     workers   = 0;
        ret = attn_rearrange_bench((const __fp16 *) IN_PTR(0), params->row_stride_bytes, params->head_dim,
                                   params->n_blocks, params->iters, params->rows_total, params->variant_mask,
                                   sum_us, blocks, &workers);

        /* The wire struct lives in the request payload, which is copied back to
           the host once the op returns. */
        auto *wire_params    = reinterpret_cast<AttnRearrangeBenchParams *>(req->payload);
        wire_params->workers = workers;
        for (int i = 0; i < 4; ++i) {
          wire_params->sum_us[i] = sum_us[i];
          wire_params->blocks[i] = blocks[i];
        }
        FARF(ALWAYS, "HTP_ATTN_REARRANGE_BENCH workers=%d vtcm_us=%d hot_us=%d cold_us=%d copy_us=%d ret=%d",
             workers, sum_us[0], sum_us[1], sum_us[2], sum_us[3], ret);
      }
      break;

    case HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16:
    case HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16:
      {
        const bool input_packed = req->op == HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16;
        const bool legacy_payload = payload_size == kBlockSparseAttnLegacyPayloadSize;
        const bool halo_v1_payload = payload_size == kBlockSparseAttnHaloV1PayloadSize;
        const bool halo_trace_payload = payload_size == kBlockSparseAttnHaloTracePayloadSize;
        const bool halo_v2_payload = payload_size >= kBlockSparseAttnHaloV2PayloadSize;
        if ((!legacy_payload && !halo_v1_payload && !halo_trace_payload && !halo_v2_payload) ||
            (input_packed && payload_size < kBlockSparseAttnHaloTracePayloadSize)) {
          return -1;
        }

        BlockSparseAttnParams decoded_params = {};
        const size_t decoded_size = legacy_payload ? kBlockSparseAttnLegacyPayloadSize :
                                    (halo_v1_payload ? kBlockSparseAttnHaloV1PayloadSize :
                                    (halo_trace_payload ? kBlockSparseAttnHaloTracePayloadSize :
                                                          sizeof(decoded_params)));
        memcpy(&decoded_params, req->payload, decoded_size);
        auto *params = &decoded_params;

        int qo_len     = params->qo_len;
        int kv_len     = params->kv_len;
        int n_heads    = params->n_heads;
        int n_kv_heads = params->n_kv_heads;
        int head_dim   = params->head_dim;
        int mask_format = params->mask_format;
        int max_kept_blocks = params->max_kept_blocks;
        bool with_lse = params->with_lse != 0;

        BlockSparseAttnBufferSizes buffer_sizes = {};
        if (block_sparse_attn_buffer_sizes(params, &buffer_sizes) != 0) {
          return -1;
        }
        if (input_packed) {
          const bool packed_static_halo =
            mask_format == 1 && params->sparse_policy == HTP_SPARSE_POLICY_STATIC_HALO &&
            (params->halo_kv_load_mode == HTP_HALO_KV_LOAD_L2_PREFETCH ||
             params->halo_kv_load_mode == HTP_HALO_KV_LOAD_USERDMA_AB ||
             params->halo_kv_load_mode == HTP_HALO_KV_LOAD_CHUNKED ||
             params->halo_kv_load_mode == HTP_HALO_KV_LOAD_CHUNKED_NO_REUSE);
          const bool packed_full_keep =
            mask_format == 0 && params->sparse_policy == HTP_SPARSE_POLICY_NONE &&
            params->halo_kv_load_mode == HTP_HALO_KV_LOAD_BASELINE;
          if (qo_len != kv_len || !htp_packed_shape_supported(n_heads, n_kv_heads, head_dim, 1) ||
              params->block_q != 32 || params->block_k != 32 ||
              (!packed_static_halo && !packed_full_keep)) {
            return -1;
          }
          const size_t padded_tokens = align_up_size((size_t) qo_len, 32);
          size_t packed_elements = 0;
          size_t packed_size = 0;
          if (!checked_size_mul(padded_tokens, (size_t) n_heads, &packed_elements) ||
              !checked_size_mul(packed_elements, (size_t) head_dim, &packed_elements) ||
              !checked_size_mul(packed_elements, sizeof(__fp16), &packed_size)) {
            return -1;
          }
          buffer_sizes.q = packed_size;
          buffer_sizes.kv = packed_size;
        }

        add_buffer(out_bufs, params->o, buffer_sizes.o);
        add_buffer(in_bufs, params->q, buffer_sizes.q);
        add_buffer(in_bufs, params->k, buffer_sizes.kv);
        add_buffer(in_bufs, params->v, buffer_sizes.kv);
        add_buffer(in_bufs, params->block_mask, buffer_sizes.block_mask);

        validate_in_bufs();
        /* The counters are written into the trace struct whenever the pointer is
           passed, which is the case for trace runs and for metrics builds. */
        const bool halo_trace_enabled = params->halo_trace != 0;
        const bool halo_run_trace_wanted = halo_trace_enabled || kBlockSparseAttnMetricsBuilt;
        block_sparse_halo_run_trace_t halo_run_trace = {};
        const int64_t t0 = HAP_perf_get_qtimer_count();
        ret = input_packed ?
          block_sparse_flash_attn_packed(
            (__fp16 *) OUT_PTR(0), (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1), (__fp16 *) IN_PTR(2),
            (const int32_t *) IN_PTR(3), qo_len, kv_len, n_heads, n_kv_heads, head_dim,
            params->block_q, params->block_k, params->n_q_blocks, params->n_k_blocks,
            mask_format, max_kept_blocks, with_lse, params->sparse_policy, params->image_tokens,
            params->text_tokens, params->image_block_rows, params->image_block_cols,
            params->halo_kv_load_mode, halo_trace_enabled,
            halo_run_trace_wanted ? &halo_run_trace : NULL) :
          block_sparse_flash_attn(
            (__fp16 *) OUT_PTR(0), (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1), (__fp16 *) IN_PTR(2),
            (const int32_t *) IN_PTR(3), qo_len, kv_len, n_heads, n_kv_heads, head_dim,
            params->block_q, params->block_k, params->n_q_blocks, params->n_k_blocks,
            mask_format, max_kept_blocks, with_lse, params->sparse_policy, params->image_tokens,
            params->text_tokens, params->image_block_rows, params->image_block_cols,
            params->halo_kv_load_mode, halo_trace_enabled,
            halo_run_trace_wanted ? &halo_run_trace : NULL);
        const int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
        auto *wire_params = reinterpret_cast<BlockSparseAttnParams *>(req->payload);
        wire_params->elapsed_us = (int32_t) elapsed_us;
        if (!legacy_payload) {
          wire_params->halo_exec_mode = params->halo_trace != 0 ? halo_run_trace.exec_mode : -1;
          wire_params->halo_image_rows_updated =
            params->halo_trace != 0 ? halo_run_trace.image_rows_updated : 0;
          wire_params->halo_text_q_image_row_updates =
            params->halo_trace != 0 ? halo_run_trace.text_q_image_row_updates : 0;
          wire_params->halo_text_q_image_block_updates =
            params->halo_trace != 0 ? halo_run_trace.text_q_image_block_updates : 0;
        }
        if (!legacy_payload && !halo_v1_payload) {
          wire_params->halo_image_q_batch_updates =
            params->halo_trace != 0 ? halo_run_trace.image_q_batch_updates : 0;
          wire_params->halo_image_q_segment_updates =
            params->halo_trace != 0 ? halo_run_trace.image_q_segment_updates : 0;
        }
        if (halo_v2_payload) {
          const block_sparse_attn_metrics_t &attn_metrics = halo_run_trace.metrics;
          wire_params->halo_metrics_present = kBlockSparseAttnMetricsBuilt ? 1 : 0;
          wire_params->halo_route_mismatch = attn_metrics.route_mismatch ? 1 : 0;
          wire_params->halo_read_kb = block_sparse_attn_bytes_to_wire_kb(attn_metrics.read_bytes);
          wire_params->halo_q_kb = block_sparse_attn_bytes_to_wire_kb(attn_metrics.q_bytes);
          wire_params->halo_text_kv_kb = block_sparse_attn_bytes_to_wire_kb(attn_metrics.text_kv_bytes);
          wire_params->halo_image_kv_kb = block_sparse_attn_bytes_to_wire_kb(attn_metrics.image_kv_bytes);
          wire_params->halo_misc_kb = block_sparse_attn_bytes_to_wire_kb(attn_metrics.misc_bytes);
          wire_params->halo_transfer_count = attn_metrics.transfer_count > (uint64_t) INT32_MAX ?
                                               INT32_MAX : (int32_t) attn_metrics.transfer_count;
          wire_params->halo_ddr_to_vtcm_us = block_sparse_attn_us_to_wire(attn_metrics.ddr_to_vtcm_us);
          wire_params->halo_overlap_us = block_sparse_attn_us_to_wire(attn_metrics.overlap_us);
          wire_params->halo_inflight_us = block_sparse_attn_us_to_wire(attn_metrics.inflight_us);
          wire_params->halo_pack_us = block_sparse_attn_us_to_wire(attn_metrics.pack_us);
          wire_params->halo_submit_us = block_sparse_attn_us_to_wire(attn_metrics.submit_us);
          wire_params->halo_worker_elapsed_us =
            block_sparse_attn_us_to_wire(attn_metrics.worker_elapsed_us);
        }
        FARF(ALWAYS, "HTP_ATTN_OP_BENCH kind=%s elapsed_us=%lld",
             input_packed ? "block_sparse_packed" : "block_sparse", elapsed_us);
        if (kBlockSparseAttnMetricsBuilt) {
          const block_sparse_attn_metrics_t &attn_metrics = halo_run_trace.metrics;
          FARF(ALWAYS,
               "HTP_ATTN_METRICS loader=%d route_match=%d read_bytes=%llu q_bytes=%llu text_kv_bytes=%llu "
               "image_kv_bytes=%llu misc_bytes=%llu transfers=%llu l2_lines=%llu ddr_to_vtcm_us=%lld "
               "overlap_us=%lld inflight_us=%lld pack_us=%lld submit_us=%lld worker_us=%lld elapsed_us=%lld",
               (int) params->halo_kv_load_mode, attn_metrics.route_mismatch ? 0 : 1,
               (unsigned long long) attn_metrics.read_bytes, (unsigned long long) attn_metrics.q_bytes,
               (unsigned long long) attn_metrics.text_kv_bytes,
               (unsigned long long) attn_metrics.image_kv_bytes,
               (unsigned long long) attn_metrics.misc_bytes,
               (unsigned long long) attn_metrics.transfer_count,
               (unsigned long long) attn_metrics.l2_lines, (long long) attn_metrics.ddr_to_vtcm_us,
               (long long) attn_metrics.overlap_us, (long long) attn_metrics.inflight_us,
               (long long) attn_metrics.pack_us, (long long) attn_metrics.submit_us,
               (long long) attn_metrics.worker_elapsed_us, (long long) elapsed_us);
        }
        validate_out_bufs();
      }
      break;

    case HTP_OPS_BLOCK_SPARSE_SCORE_QO_F32_KM_F16:
      {
        if (payload_size < sizeof(BlockSparseScoreParams)) {
          return -1;
        }
        auto params = reinterpret_cast<BlockSparseScoreParams *>(req->payload);

        int head_dim   = params->head_dim;
        int n_q_blocks = params->n_q_blocks;
        int n_k_blocks = params->n_k_blocks;

        size_t q_mean_size = n_q_blocks * head_dim * sizeof(__fp16);
        size_t k_mean_size = n_k_blocks * head_dim * sizeof(__fp16);
        size_t score_size = n_q_blocks * n_k_blocks * sizeof(__fp16);

        add_buffer(out_bufs, params->score, score_size);
        add_buffer(in_bufs, params->q_mean, q_mean_size);
        add_buffer(in_bufs, params->k_mean, k_mean_size);

        validate_in_bufs();
        ret = block_sparse_score((__fp16 *) OUT_PTR(0), (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1),
                                 head_dim, n_q_blocks, n_k_blocks);
        validate_out_bufs();
      }
      break;

    default:
      break;
  }
  return ret;
}

#undef IN_PTR
#undef OUT_PTR
}
