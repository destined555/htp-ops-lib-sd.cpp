#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_memory.h>
#include <qurt/qurt_timer.h>

// std headers
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>

#include "dsp/hmx_mgr.h"
#include "dsp/hvx_math.h"
#include "dsp/op_executor.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "op_reg.h"

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

constexpr size_t kBlockSparseAttnLegacyPayloadSize = offsetof(BlockSparseAttnParams, sparse_policy);
static_assert(kBlockSparseAttnLegacyPayloadSize == 92);
static_assert(offsetof(BlockSparseAttnParams, elapsed_us) + sizeof(int32_t) ==
              kBlockSparseAttnLegacyPayloadSize);
constexpr size_t kBlockSparseAttnHaloV1PayloadSize =
  offsetof(BlockSparseAttnParams, halo_image_q_batch_updates);
static_assert(kBlockSparseAttnHaloV1PayloadSize == 136);
static_assert(HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16 ==
              HTP_OPS_QKV_MUL_RMS_PACKED_W16A32 + 1);
static_assert(sizeof(QkvMulRmsPackedDualParams) == 128);

void test_qkv_mul_rms_packed_dual_contracts() {
  QkvMulRmsPackedDualParams params = {};
  params.image_tokens = 256;
  params.text_tokens = 4096;
  params.input_dim = 1536;
  params.n_heads = 24;
  params.head_dim = 64;

  QkvMulRmsPackedBufferSizes sizes = {};
  OP_TEST_ASSERT(qkv_mul_rms_packed_dual_buffer_sizes(&params, &sizes) == 0);
  OP_TEST_ASSERT(sizes.combined_tokens == 4352);
  OP_TEST_ASSERT(sizes.token_tiles == 136);
  OP_TEST_ASSERT(sizes.qkv_bytes == size_t(24) * 136u * 4096u);
  OP_TEST_ASSERT(sizes.activation_bytes == size_t(4352) * 1536u * sizeof(float));
  OP_TEST_ASSERT(sizes.weight_bytes == size_t(1536) * 4608u * sizeof(__fp16));
  OP_TEST_ASSERT(sizes.bias_bytes == size_t(4608) * sizeof(float));
  OP_TEST_ASSERT(sizes.norm_bytes == size_t(64) * sizeof(float));

  params.n_heads = 30;
  params.head_dim = 128;
  OP_TEST_ASSERT(htp_packed_shape_supported(30, 30, 128, 1));
  OP_TEST_ASSERT(qkv_mul_rms_packed_dual_buffer_sizes(&params, &sizes) == 0);
  OP_TEST_ASSERT(sizes.qkv_bytes == size_t(30) * 136u * 8192u);
  OP_TEST_ASSERT(sizes.weight_bytes == size_t(1536) * 11520u * sizeof(__fp16));
  OP_TEST_ASSERT(sizes.bias_bytes == size_t(11520) * sizeof(float));
  OP_TEST_ASSERT(sizes.norm_bytes == size_t(128) * sizeof(float));

  params.image_tokens = 0;
  OP_TEST_ASSERT(qkv_mul_rms_packed_dual_buffer_sizes(&params, &sizes) != 0);
  params.image_tokens = 255;
  OP_TEST_ASSERT(qkv_mul_rms_packed_dual_buffer_sizes(&params, &sizes) != 0);
  params.image_tokens = 256;
  params.n_heads = 16;
  OP_TEST_ASSERT(qkv_mul_rms_packed_dual_buffer_sizes(&params, &sizes) != 0);

  OP_TEST_ASSERT(!htp_packed_shape_supported(16, 16, 128, 1));
  OP_TEST_ASSERT(!htp_packed_shape_supported(30, 24, 128, 1));
  OP_TEST_ASSERT(htp_packed_shape_supported(30, 15, 128, 0));

  FARF(ALWAYS, "qkv_mul_rms_packed_dual_contracts: PASS");
}

void test_block_sparse_30_head_contracts() {
  BlockSparseAttnParams params = {};
  params.qo_len = 512;
  params.kv_len = 512;
  params.n_heads = 30;
  params.n_kv_heads = 30;
  params.head_dim = 128;
  params.block_q = 32;
  params.block_k = 32;
  params.n_q_blocks = 30 * 16;
  params.n_k_blocks = 30 * 16;
  params.mask_format = 1;
  params.max_kept_blocks = 16;
  params.with_lse = 1;

  BlockSparseAttnBufferSizes sizes = {};
  OP_TEST_ASSERT(htp_packed_shape_supported(30, 30, 128, 1));
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(&params, &sizes) == 0);
  OP_TEST_ASSERT(sizes.q == size_t(512) * 30u * 128u * sizeof(float));
  OP_TEST_ASSERT(sizes.o == size_t(512) * 30u * 129u * sizeof(float));
  OP_TEST_ASSERT(sizes.kv == size_t(512) * 30u * 128u * sizeof(__fp16));
  OP_TEST_ASSERT(sizes.block_mask == size_t(30 * 16 * 16) * sizeof(int32_t));
  OP_TEST_ASSERT(!htp_packed_shape_supported(31, 31, 128, 1));
  OP_TEST_ASSERT(htp_packed_shape_supported(30, 15, 128, 0));
  FARF(ALWAYS, "block_sparse_30_head_contracts: PASS");
}

template <size_t PayloadSize>
struct alignas(8) SparseExecutorTestRequest {
  uint8_t compute_request[sizeof(OpComputeRequest) + PayloadSize];
  uint8_t canary[64];
};

static BlockSparseAttnParams make_rejected_sparse_params() {
  BlockSparseAttnParams params = {};
  params.qo_len = 1;
  params.kv_len = 1;
  params.n_heads = 1;
  params.n_kv_heads = 1;
  params.head_dim = 1;
  params.block_q = 32;
  params.block_k = 32;
  params.n_q_blocks = 1;
  params.n_k_blocks = 1;
  params.mask_format = 1;
  params.max_kept_blocks = 1;
  params.elapsed_us = INT32_MIN;
  params.sparse_policy = HTP_SPARSE_POLICY_STATIC_HALO;
  params.image_tokens = 1;
  params.halo_kv_load_mode = HTP_HALO_KV_LOAD_USERDMA_AB;
  params.halo_trace = 1;
  params.image_block_rows = 1;
  params.image_block_cols = 1;
  params.halo_exec_mode = INT32_MIN;
  params.halo_image_rows_updated = INT32_MIN;
  params.halo_text_q_image_row_updates = INT32_MIN;
  params.halo_text_q_image_block_updates = INT32_MIN;
  params.halo_image_q_batch_updates = INT32_MIN;
  params.halo_image_q_segment_updates = INT32_MIN;
  return params;
}

template <size_t PayloadSize>
static SparseExecutorTestRequest<PayloadSize> make_sparse_executor_request() {
  SparseExecutorTestRequest<PayloadSize> storage;
  memset(&storage, 0xa5, sizeof(storage));
  const uint32_t op = HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16;
  memcpy(storage.compute_request, &op, sizeof(op));
  const BlockSparseAttnParams params = make_rejected_sparse_params();
  const size_t copied = PayloadSize < sizeof(params) ? PayloadSize : sizeof(params);
  memcpy(storage.compute_request + sizeof(OpComputeRequest), &params, copied);
  return storage;
}

void test_op_executor_sparse_payload_boundaries() {
  {
    auto request = make_sparse_executor_request<kBlockSparseAttnLegacyPayloadSize>();
    uint8_t canary_before[sizeof(request.canary)];
    memcpy(canary_before, request.canary, sizeof(canary_before));

    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    const auto *params = (const BlockSparseAttnParams *)
      (request.compute_request + sizeof(OpComputeRequest));
    OP_TEST_ASSERT(params->elapsed_us != INT32_MIN);
    OP_TEST_ASSERT(memcmp(request.canary, canary_before, sizeof(canary_before)) == 0);
  }

  {
    auto request = make_sparse_executor_request<kBlockSparseAttnHaloV1PayloadSize>();
    uint8_t canary_before[sizeof(request.canary)];
    memcpy(canary_before, request.canary, sizeof(canary_before));

    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    const auto *params = (const BlockSparseAttnParams *)
      (request.compute_request + sizeof(OpComputeRequest));
    OP_TEST_ASSERT(params->elapsed_us != INT32_MIN);
    OP_TEST_ASSERT(params->halo_exec_mode != INT32_MIN);
    OP_TEST_ASSERT(params->halo_image_rows_updated != INT32_MIN);
    OP_TEST_ASSERT(params->halo_text_q_image_row_updates != INT32_MIN);
    OP_TEST_ASSERT(params->halo_text_q_image_block_updates != INT32_MIN);
    OP_TEST_ASSERT(memcmp(request.canary, canary_before, sizeof(canary_before)) == 0);
  }

  {
    auto request = make_sparse_executor_request<sizeof(BlockSparseAttnParams)>();
    uint8_t canary_before[sizeof(request.canary)];
    memcpy(canary_before, request.canary, sizeof(canary_before));

    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    const auto *params = (const BlockSparseAttnParams *)
      (request.compute_request + sizeof(OpComputeRequest));
    OP_TEST_ASSERT(params->elapsed_us != INT32_MIN);
    OP_TEST_ASSERT(params->halo_exec_mode != INT32_MIN);
    OP_TEST_ASSERT(params->halo_image_rows_updated != INT32_MIN);
    OP_TEST_ASSERT(params->halo_text_q_image_row_updates != INT32_MIN);
    OP_TEST_ASSERT(params->halo_text_q_image_block_updates != INT32_MIN);
    OP_TEST_ASSERT(params->halo_image_q_batch_updates != INT32_MIN);
    OP_TEST_ASSERT(params->halo_image_q_segment_updates != INT32_MIN);
    OP_TEST_ASSERT(memcmp(request.canary, canary_before, sizeof(canary_before)) == 0);
  }

  {
    constexpr size_t kFuturePayloadSize = sizeof(BlockSparseAttnParams) + 16;
    auto request = make_sparse_executor_request<kFuturePayloadSize>();
    auto before = request;

    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    const size_t future_tail_offset = sizeof(OpComputeRequest) + sizeof(BlockSparseAttnParams);
    OP_TEST_ASSERT(memcmp(request.compute_request + future_tail_offset,
                          before.compute_request + future_tail_offset,
                          kFuturePayloadSize - sizeof(BlockSparseAttnParams)) == 0);
    OP_TEST_ASSERT(memcmp(request.canary, before.canary, sizeof(request.canary)) == 0);
  }

  {
    auto request = make_sparse_executor_request<91>();
    auto before = request;
    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    OP_TEST_ASSERT(memcmp(&request, &before, sizeof(request)) == 0);
  }

  {
    auto request = make_sparse_executor_request<93>();
    auto before = request;
    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    OP_TEST_ASSERT(memcmp(&request, &before, sizeof(request)) == 0);
  }

  {
    auto request = make_sparse_executor_request<135>();
    auto before = request;
    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    OP_TEST_ASSERT(memcmp(&request, &before, sizeof(request)) == 0);
  }

  {
    auto request = make_sparse_executor_request<137>();
    auto before = request;
    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    OP_TEST_ASSERT(memcmp(&request, &before, sizeof(request)) == 0);
  }

  {
    auto request = make_sparse_executor_request<143>();
    auto before = request;
    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(request.compute_request)) != 0);
    OP_TEST_ASSERT(memcmp(&request, &before, sizeof(request)) == 0);
  }

  {
    auto request = make_sparse_executor_request<sizeof(BlockSparseAttnParams)>();
    auto before = request;
    OP_TEST_ASSERT(execute_op_simple((OpComputeRequest *) request.compute_request,
                                     sizeof(OpComputeRequest) - 1) != 0);
    OP_TEST_ASSERT(memcmp(&request, &before, sizeof(request)) == 0);
  }

  FARF(ALWAYS, "op_executor_sparse_payload_boundaries: PASS");
}

void test_block_sparse_attn_buffer_size_contracts() {
  BlockSparseAttnParams params = {};
  params.qo_len = 666;
  params.kv_len = 666;
  params.n_heads = 2;
  params.n_kv_heads = 2;
  params.head_dim = 64;
  params.n_q_blocks = 42;
  params.n_k_blocks = 42;
  params.mask_format = 1;
  params.max_kept_blocks = 21;

  BlockSparseAttnBufferSizes sizes = {};
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(&params, &sizes) == 0);
  OP_TEST_ASSERT(sizes.q == 340992 && sizes.o == 340992 &&
                 sizes.kv == 170496 && sizes.block_mask == 3528);

  params.with_lse = 1;
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(&params, &sizes) == 0);
  OP_TEST_ASSERT(sizes.o == 346320);

  params.n_q_blocks = 4;
  params.max_kept_blocks = INT_MAX;
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(&params, &sizes) == -1);

  params.n_q_blocks = 42;
  params.max_kept_blocks = 21;
  params.qo_len = INT_MAX;
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(&params, &sizes) == -1);

  params.qo_len = 666;
  params.mask_format = 2;
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(&params, &sizes) == -1);
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(nullptr, &sizes) == -1);
  OP_TEST_ASSERT(block_sparse_attn_buffer_sizes(&params, nullptr) == -1);

  FARF(ALWAYS, "block_sparse_attn_buffer_size_contracts: PASS");
}

void test_message_header_layout_validation() {
  alignas(8) uint8_t storage[128] = {};
  auto *message = (MessageHeader *) storage;
  size_t total_size = 0;

  auto set_valid_layout = [&]() {
    memset(storage, 0, sizeof(storage));
    message->n_reqs = 2;
    message->req_offsets[0] = (int32_t) message_header_size(message);
    message->req_offsets[1] = message->req_offsets[0] + sizeof(RequestHeader);
    message->req_offsets[2] = message->req_offsets[1] + sizeof(RequestHeader);
    total_size = 0;
  };

  set_valid_layout();
  OP_TEST_ASSERT(message_header_validate_layout(message, sizeof(storage), &total_size));
  OP_TEST_ASSERT(total_size == (size_t) message->req_offsets[2]);

  set_valid_layout();
  message->n_reqs = -1;
  OP_TEST_ASSERT(!message_header_validate_layout(message, sizeof(storage), &total_size));

  set_valid_layout();
  message->n_reqs = INT32_MAX;
  OP_TEST_ASSERT(!message_header_validate_layout(message, sizeof(storage), &total_size));

  set_valid_layout();
  message->req_offsets[0] = (int32_t) message_header_size(message) - 1;
  OP_TEST_ASSERT(!message_header_validate_layout(message, sizeof(storage), &total_size));

  set_valid_layout();
  message->req_offsets[1] = message->req_offsets[0] - 1;
  OP_TEST_ASSERT(!message_header_validate_layout(message, sizeof(storage), &total_size));

  set_valid_layout();
  message->req_offsets[1] = message->req_offsets[0] + sizeof(RequestHeader) - 1;
  OP_TEST_ASSERT(!message_header_validate_layout(message, sizeof(storage), &total_size));

  set_valid_layout();
  message->req_offsets[2] = sizeof(storage) + 1;
  OP_TEST_ASSERT(!message_header_validate_layout(message, sizeof(storage), &total_size));

  FARF(ALWAYS, "message_header_layout_validation: PASS");
}

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

  OP_TEST_ASSERT(block_sparse_fa_select_halo_fallback_plan(&plan, true, 1, 64, 4250, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_FULL_KV);
  OP_TEST_ASSERT(block_sparse_fa_select_halo_fallback_plan(&plan, true, 1, 128, 4250, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_FULL_K);
  OP_TEST_ASSERT(plan.blk_sz_c == 640);
  OP_TEST_ASSERT(plan.required_bytes == 1368064);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, true, 1, 128, 5281, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_CHUNKED);
  OP_TEST_ASSERT(plan.blk_sz_c == 2048);
  OP_TEST_ASSERT(plan.required_bytes == 1349120);

  OP_TEST_ASSERT(block_sparse_fa_select_plan(&plan, false, 1, 64, 410, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_CHUNKED);

  OP_TEST_ASSERT(block_sparse_fa_select_chunked_plan(&plan, 1, 64, 4250, available) == 0);
  OP_TEST_ASSERT(plan.mode == BLOCK_SPARSE_FA_CHUNKED);
  OP_TEST_ASSERT(block_sparse_fa_select_chunked_plan(nullptr, 1, 64, 4250, available) == -1);

  block_sparse_halo_plan_t halo_plan = {};
  OP_TEST_ASSERT(block_sparse_halo_select_execution(
                   &halo_plan, 3, true, true, BLOCK_SPARSE_HALO_USERDMA_UNKNOWN,
                   64, 32, 4, available) == BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);

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

static void fill_static_halo_rows(int32_t *rows, int grid_h, int grid_w, int text_blocks) {
  const int image_blocks = grid_h * grid_w;
  const int blocks_per_head = image_blocks + text_blocks;
  std::fill(rows, rows + blocks_per_head * blocks_per_head, -1);
  for (int qb = 0; qb < image_blocks; ++qb) {
    const int qy = qb / grid_w;
    const int qx = qb % grid_w;
    int32_t *row = rows + qb * blocks_per_head;
    int kept = 0;
    for (int ky = std::max(0, qy - 1); ky <= std::min(grid_h - 1, qy + 1); ++ky) {
      for (int kx = std::max(0, qx - 1); kx <= std::min(grid_w - 1, qx + 1); ++kx) {
        row[kept++] = ky * grid_w + kx;
      }
    }
    for (int kb = image_blocks; kb < blocks_per_head; ++kb) {
      row[kept++] = kb;
    }
  }
  for (int qb = image_blocks; qb < blocks_per_head; ++qb) {
    int32_t *row = rows + qb * blocks_per_head;
    for (int kb = 0; kb < blocks_per_head; ++kb) {
      row[kb] = kb;
    }
  }
}

void test_block_sparse_halo_contracts() {
  static_assert(sizeof(size_t) == 4, "DSP overflow contracts require 32-bit size_t");
  constexpr int grid_h = 4;
  constexpr int grid_w = 4;
  constexpr int image_blocks = grid_h * grid_w;
  constexpr int text_blocks = 2;
  constexpr int blocks_per_head = image_blocks + text_blocks;
  constexpr int image_tokens = image_blocks * 32;
  constexpr int text_tokens = 34;
  constexpr int seq_len = image_tokens + text_tokens;
  int32_t rows[blocks_per_head * blocks_per_head];
  fill_static_halo_rows(rows, grid_h, grid_w, text_blocks);

  block_sparse_halo_row_t parsed = {};
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows, blocks_per_head, 0, blocks_per_head,
                                              blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(!parsed.is_text_query && parsed.image_count == 4);
  OP_TEST_ASSERT(parsed.image_blocks[0] == 0 && parsed.image_blocks[3] == 5);

  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + blocks_per_head, blocks_per_head, 1,
                                              blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(!parsed.is_text_query && parsed.image_count == 6);
  OP_TEST_ASSERT(parsed.image_blocks[0] == 0 && parsed.image_blocks[5] == 6);

  const int interior_qb = grid_w + 1;
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + interior_qb * blocks_per_head, blocks_per_head,
                                              interior_qb, blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(!parsed.is_text_query && parsed.image_count == 9);
  OP_TEST_ASSERT(parsed.image_blocks[0] == 0 && parsed.image_blocks[8] == 10);

  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + image_blocks * blocks_per_head, blocks_per_head,
                                              image_blocks, blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(parsed.is_text_query && parsed.image_count == 0);

  block_sparse_halo_prefetch_plan_t prefetch = {};
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows, blocks_per_head, 0, blocks_per_head,
                                              blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_plan_prefetch(&prefetch, &parsed, 0, image_blocks, 0) == 0);
  OP_TEST_ASSERT(prefetch.block_count == 4 && prefetch.range_count == 2);
  OP_TEST_ASSERT(prefetch.ranges[0].first_block == 0 && prefetch.ranges[0].block_count == 2);
  OP_TEST_ASSERT(prefetch.ranges[1].first_block == 4 && prefetch.ranges[1].block_count == 2);

  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + blocks_per_head, blocks_per_head, 1,
                                              blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_plan_prefetch(&prefetch, &parsed, 0, image_blocks, 0) == 0);
  OP_TEST_ASSERT(prefetch.block_count == 6 && prefetch.range_count == 2);
  OP_TEST_ASSERT(prefetch.ranges[0].first_block == 0 && prefetch.ranges[0].block_count == 3);
  OP_TEST_ASSERT(prefetch.ranges[1].first_block == 4 && prefetch.ranges[1].block_count == 3);

  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + interior_qb * blocks_per_head, blocks_per_head,
                                              interior_qb, blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_plan_prefetch(&prefetch, &parsed, 0, image_blocks, 0) == 0);
  OP_TEST_ASSERT(prefetch.block_count == 9 && prefetch.range_count == 3);
  OP_TEST_ASSERT(prefetch.ranges[0].first_block == 0 && prefetch.ranges[0].block_count == 3);
  OP_TEST_ASSERT(prefetch.ranges[1].first_block == 4 && prefetch.ranges[1].block_count == 3);
  OP_TEST_ASSERT(prefetch.ranges[2].first_block == 8 && prefetch.ranges[2].block_count == 3);

  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + image_blocks * blocks_per_head, blocks_per_head,
                                              image_blocks, blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_plan_prefetch(&prefetch, &parsed, 0, image_blocks, 0) == 0);
  OP_TEST_ASSERT(prefetch.block_count == 9 && prefetch.range_count == 1);
  OP_TEST_ASSERT(prefetch.ranges[0].first_block == 0 && prefetch.ranges[0].block_count == 9);
  OP_TEST_ASSERT(block_sparse_halo_plan_prefetch(&prefetch, &parsed, 0, image_blocks, 9) == 0);
  OP_TEST_ASSERT(prefetch.block_count == 7 && prefetch.range_count == 1);
  OP_TEST_ASSERT(prefetch.ranges[0].first_block == 9 && prefetch.ranges[0].block_count == 7);

  alignas(64) dma_desc_2d_t descs[2 * BLOCK_SPARSE_HALO_MAX_IMAGE_BLOCKS];
  const void *k_fake = reinterpret_cast<const void *>(uintptr_t{0x10000000});
  const void *v_fake = reinterpret_cast<const void *>(uintptr_t{0x20000000});
  void *bank_fake = reinterpret_cast<void *>(uintptr_t{0x30000000});
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows, blocks_per_head, 0, blocks_per_head,
                                              blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_build_dma_descs(descs, 18, k_fake, v_fake, bank_fake, 2, 0, 64,
                                                    blocks_per_head, &parsed) == 8);
  OP_TEST_ASSERT(descs[0].src == 0x10000000u && descs[0].dst == 0x30000000u);
  OP_TEST_ASSERT(descs[1].src == 0x20000000u && descs[1].dst == 0x30001000u);
  OP_TEST_ASSERT(descs[2].src == 0x10002000u && descs[2].dst == 0x30002000u);
  OP_TEST_ASSERT(descs[0].roi_width == 128 && descs[0].roi_height == 32);
  OP_TEST_ASSERT(descs[0].src_stride == 256 && descs[0].dst_stride == 128);
  OP_TEST_ASSERT(descs[0].src_bypass == 1 && descs[0].dst_bypass == 0 && descs[0].ordered == 1);
  OP_TEST_ASSERT(descs[0].cache_alloc == DMA_DESC_CACHEALLOC_NONE);
  OP_TEST_ASSERT(descs[0].next == (uint32_t) (uintptr_t) &descs[1]);
  OP_TEST_ASSERT(descs[7].next == 0);

  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + blocks_per_head, blocks_per_head, 1,
                                              blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_build_dma_descs(descs, 18, k_fake, v_fake, bank_fake, 2, 0, 64,
                                                    blocks_per_head, &parsed) == 12);
  OP_TEST_ASSERT(descs[11].next == 0);
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, rows + interior_qb * blocks_per_head, blocks_per_head,
                                              interior_qb, blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_build_dma_descs(descs, 18, k_fake, v_fake, bank_fake, 2, 0, 64,
                                                    blocks_per_head, &parsed) == 18);
  OP_TEST_ASSERT(descs[17].next == 0);
  OP_TEST_ASSERT(block_sparse_halo_build_dma_descs(descs, 17, k_fake, v_fake, bank_fake, 2, 0, 64,
                                                    blocks_per_head, &parsed) == -1);

  int32_t head1_row[blocks_per_head];
  std::copy(rows, rows + blocks_per_head, head1_row);
  for (int i = 0; i < blocks_per_head && head1_row[i] >= 0; ++i) {
    head1_row[i] += blocks_per_head;
  }
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, head1_row, blocks_per_head, blocks_per_head,
                                              blocks_per_head, blocks_per_head, image_blocks) == 0);
  OP_TEST_ASSERT(block_sparse_halo_build_dma_descs(descs, 18, k_fake, v_fake, bank_fake, 2, 1, 64,
                                                    blocks_per_head, &parsed) == 8);
  OP_TEST_ASSERT(descs[0].src == 0x10000080u);
  OP_TEST_ASSERT(descs[1].src == 0x20000080u);

  dma_desc_2d_t reset_desc;
  std::memset(&reset_desc, 0xff, sizeof(reset_desc));
  OP_TEST_ASSERT(dma_desc_2d_init(&reset_desc, nullptr, k_fake, bank_fake, 128, 32, 256, 128) ==
                 DMA_SUBMIT_OK);
  OP_TEST_ASSERT(reset_desc.next == 0 && reset_desc.type == DMA_DESC_TYPE_2D &&
                 reset_desc.dstate == DMA_DESC_DSTATE_PENDING);
  OP_TEST_ASSERT(dma_desc_2d_init(&reset_desc, nullptr, k_fake, bank_fake, 65536, 32, 65536, 65536) ==
                 DMA_SUBMIT_BAD_DESC);

  OP_TEST_ASSERT(block_sparse_halo_validate(rows, seq_len, seq_len, 1, 1, 64, 32, 32, blocks_per_head,
                                             blocks_per_head, 1, blocks_per_head, 1, image_tokens,
                                             text_tokens, grid_h, grid_w) == 0);
  int32_t wrong_geometry[blocks_per_head * blocks_per_head];
  std::copy(rows, rows + blocks_per_head * blocks_per_head, wrong_geometry);
  wrong_geometry[0] = 1;
  wrong_geometry[1] = 2;
  wrong_geometry[2] = 4;
  wrong_geometry[3] = 5;
  OP_TEST_ASSERT(block_sparse_halo_validate(wrong_geometry, seq_len, seq_len, 1, 1, 64, 32, 32,
                                             blocks_per_head, blocks_per_head, 1, blocks_per_head, 1,
                                             image_tokens, text_tokens, grid_h, grid_w) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate(rows, seq_len, seq_len, 1, 1, 64, 32, 32, blocks_per_head,
                                             blocks_per_head, 1, blocks_per_head, 1, image_tokens,
                                             text_tokens, 0, 0) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate(rows, seq_len, seq_len, 1, 1, 64, 32, 32, blocks_per_head,
                                             blocks_per_head, 1, blocks_per_head, 1, image_tokens,
                                             text_tokens, grid_h, grid_w - 1) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate(rows, seq_len, seq_len, 2, 1, 64, 32, 32,
                                             2 * blocks_per_head, blocks_per_head, 1, blocks_per_head, 1,
                                             image_tokens, text_tokens, grid_h, grid_w) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate(rows, seq_len, seq_len, 1, 1, 64, 16, 32, blocks_per_head,
                                             blocks_per_head, 1, blocks_per_head, 1, image_tokens,
                                             text_tokens, grid_h, grid_w) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate(rows, seq_len, seq_len, 1, 1, 64, 32, 32, blocks_per_head,
                                             blocks_per_head, 1, blocks_per_head, 1, image_tokens - 1,
                                             text_tokens + 1, grid_h, grid_w) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate(rows, seq_len, seq_len - 1, 1, 1, 64, 32, 32,
                                             blocks_per_head, blocks_per_head, 1, blocks_per_head, 1,
                                             image_tokens, text_tokens, grid_h, grid_w) == -1);

  constexpr int two_heads = 2;
  int32_t two_head_rows[two_heads * blocks_per_head * blocks_per_head];
  for (int head = 0; head < two_heads; ++head) {
    int32_t *head_rows = two_head_rows + head * blocks_per_head * blocks_per_head;
    std::copy(rows, rows + blocks_per_head * blocks_per_head, head_rows);
    for (int i = 0; i < blocks_per_head * blocks_per_head; ++i) {
      if (head_rows[i] >= 0) {
        head_rows[i] += head * blocks_per_head;
      }
    }
  }
  OP_TEST_ASSERT(block_sparse_halo_validate_shape(
                   two_head_rows, seq_len, seq_len, two_heads, two_heads, 64, 32, 32,
                   two_heads * blocks_per_head, two_heads * blocks_per_head,
                   1, blocks_per_head, 1, image_tokens, text_tokens, grid_h, grid_w) == 0);
  OP_TEST_ASSERT(block_sparse_halo_validate_shape(
                   rows, 128, 128, 1, 1, 64, 32, 32, 4, 4,
                   1, INT_MAX, 1, 128, 0, 2, 2) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate_kv_head_rows(
                   two_head_rows, 0, two_heads, blocks_per_head, blocks_per_head,
                   blocks_per_head, image_blocks, grid_h, grid_w) == 0);
  OP_TEST_ASSERT(block_sparse_halo_validate_kv_head_rows(
                   two_head_rows, 1, two_heads, blocks_per_head, blocks_per_head,
                   blocks_per_head, image_blocks, grid_h, grid_w) == 0);

  int32_t head1_wrong_geometry[two_heads * blocks_per_head * blocks_per_head];
  std::copy(two_head_rows,
            two_head_rows + two_heads * blocks_per_head * blocks_per_head,
            head1_wrong_geometry);
  int32_t *head1_first_row =
    head1_wrong_geometry + blocks_per_head * blocks_per_head;
  head1_first_row[0] = blocks_per_head + 1;
  head1_first_row[1] = blocks_per_head + 2;
  head1_first_row[2] = blocks_per_head + 4;
  head1_first_row[3] = blocks_per_head + 5;
  OP_TEST_ASSERT(block_sparse_halo_validate_kv_head_rows(
                   head1_wrong_geometry, 0, two_heads, blocks_per_head, blocks_per_head,
                   blocks_per_head, image_blocks, grid_h, grid_w) == 0);
  OP_TEST_ASSERT(block_sparse_halo_validate_kv_head_rows(
                   head1_wrong_geometry, 1, two_heads, blocks_per_head, blocks_per_head,
                   blocks_per_head, image_blocks, grid_h, grid_w) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate(
                   head1_wrong_geometry, seq_len, seq_len, two_heads, two_heads, 64, 32, 32,
                   two_heads * blocks_per_head, two_heads * blocks_per_head,
                   1, blocks_per_head, 1, image_tokens, text_tokens, grid_h, grid_w) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate_kv_head_rows(
                   two_head_rows, two_heads, two_heads, blocks_per_head, blocks_per_head,
                   blocks_per_head, image_blocks, grid_h, grid_w) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate_kv_head_rows(
                   two_head_rows, 1, 2, INT_MAX / 2 + 1, 1, 1, 1, 1, 1) == -1);
  OP_TEST_ASSERT(block_sparse_halo_validate_kv_head_rows(
                   two_head_rows, 0, 2, 1, INT_MAX / 2 + 1,
                   INT_MAX / 2 + 1, 1, 1, 1) == -1);

  int32_t malformed[blocks_per_head];
  std::copy(rows, rows + blocks_per_head, malformed);
  malformed[1] = malformed[0];
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, malformed, blocks_per_head, 0, blocks_per_head,
                                              blocks_per_head, image_blocks) == -1);
  std::copy(rows, rows + blocks_per_head, malformed);
  std::swap(malformed[0], malformed[1]);
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, malformed, blocks_per_head, 0, blocks_per_head,
                                              blocks_per_head, image_blocks) == -1);
  std::copy(rows, rows + blocks_per_head, malformed);
  malformed[5] = -1;
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, malformed, blocks_per_head, 0, blocks_per_head,
                                              blocks_per_head, image_blocks) == -1);
  std::copy(rows, rows + blocks_per_head, malformed);
  malformed[4] = 8;
  malformed[5] = 16;
  malformed[6] = 17;
  OP_TEST_ASSERT(block_sparse_halo_parse_row(&parsed, malformed, blocks_per_head, 0, blocks_per_head,
                                              blocks_per_head, image_blocks) == -1);

  constexpr int resident_text_tokens = 154;
  block_sparse_halo_plan_t l2_plan = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan, 1, 64, resident_text_tokens, grid_w,
                                                std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(l2_plan.blk_sz_c == 128);
  OP_TEST_ASSERT(l2_plan.workspace_bytes == 72704);
  OP_TEST_ASSERT(l2_plan.packed_image_row_slot_count == 3);
  OP_TEST_ASSERT(l2_plan.packed_image_row_stride == 32768);
  OP_TEST_ASSERT(l2_plan.packed_image_row_bytes == 98304);
  OP_TEST_ASSERT(l2_plan.packed_image_row_offsets[0] == 73728);
  OP_TEST_ASSERT(l2_plan.packed_image_row_offsets[1] == 106496);
  OP_TEST_ASSERT(l2_plan.packed_image_row_offsets[2] == 139264);
  OP_TEST_ASSERT(l2_plan.resident_text_offset == 172032);
  OP_TEST_ASSERT(l2_plan.resident_text_bytes == 40960);
  OP_TEST_ASSERT(l2_plan.resident_text_q_offset == 212992);
  OP_TEST_ASSERT(l2_plan.resident_text_q_bytes == 20480);
  OP_TEST_ASSERT(l2_plan.resident_text_o_offsets[0] == 233472);
  OP_TEST_ASSERT(l2_plan.resident_text_o_offsets[1] == 253952);
  OP_TEST_ASSERT(l2_plan.resident_text_o_bytes == 20480);
  OP_TEST_ASSERT(l2_plan.resident_text_state_offset == 274432);
  OP_TEST_ASSERT(l2_plan.resident_text_state_bytes == 2560);
  OP_TEST_ASSERT(l2_plan.resident_text_q_blocks == 5);
  OP_TEST_ASSERT(l2_plan.image_bank_bytes == 0);
  OP_TEST_ASSERT(l2_plan.descriptor_bytes == 0);
  OP_TEST_ASSERT(l2_plan.required_bytes == 276992);
  for (int slot = 0; slot < 3; ++slot) {
    OP_TEST_ASSERT((l2_plan.packed_image_row_offsets[slot] & 4095u) == 0);
  }
  OP_TEST_ASSERT((l2_plan.resident_text_offset & 4095u) == 0);
  OP_TEST_ASSERT((l2_plan.resident_text_q_offset & 4095u) == 0);
  OP_TEST_ASSERT((l2_plan.resident_text_o_offsets[0] & 4095u) == 0);
  OP_TEST_ASSERT((l2_plan.resident_text_o_offsets[1] & 4095u) == 0);
  OP_TEST_ASSERT((l2_plan.resident_text_state_offset & 4095u) == 0);
  OP_TEST_ASSERT(l2_plan.packed_image_row_offsets[2] + l2_plan.packed_image_row_stride <=
                 l2_plan.resident_text_offset);
  OP_TEST_ASSERT(l2_plan.resident_text_offset + l2_plan.resident_text_bytes <=
                 l2_plan.resident_text_q_offset);
  OP_TEST_ASSERT(l2_plan.resident_text_q_offset + l2_plan.resident_text_q_bytes <=
                 l2_plan.resident_text_o_offsets[0]);
  OP_TEST_ASSERT(l2_plan.resident_text_o_offsets[0] + l2_plan.resident_text_o_bytes <=
                 l2_plan.resident_text_o_offsets[1]);
  OP_TEST_ASSERT(l2_plan.resident_text_o_offsets[1] + l2_plan.resident_text_o_bytes <=
                 l2_plan.resident_text_state_offset);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan, 1, 64, resident_text_tokens, grid_w,
                                                276991) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan, 1, 64, resident_text_tokens, grid_w,
                                                276992) == 0);

  block_sparse_halo_plan_t three_col_plan = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&three_col_plan, 1, 64, resident_text_tokens, 3,
                                                std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(three_col_plan.blk_sz_c == 96);

  block_sparse_halo_plan_t l2_plan_d128 = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan_d128, 1, 128, resident_text_tokens, grid_w,
                                                std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(l2_plan_d128.workspace_bytes == 113664);
  OP_TEST_ASSERT(l2_plan_d128.packed_image_row_slot_count == 3);
  OP_TEST_ASSERT(l2_plan_d128.packed_image_row_stride == 65536);
  OP_TEST_ASSERT(l2_plan_d128.packed_image_row_bytes == 196608);
  OP_TEST_ASSERT(l2_plan_d128.packed_image_row_offsets[0] == 114688);
  OP_TEST_ASSERT(l2_plan_d128.packed_image_row_offsets[1] == 180224);
  OP_TEST_ASSERT(l2_plan_d128.packed_image_row_offsets[2] == 245760);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_offset == 311296);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_bytes == 81920);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_q_offset == 393216);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_q_bytes == 40960);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_o_offsets[0] == 434176);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_o_offsets[1] == 475136);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_o_bytes == 40960);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_state_offset == 516096);
  OP_TEST_ASSERT(l2_plan_d128.resident_text_state_bytes == 2560);
  OP_TEST_ASSERT(l2_plan_d128.required_bytes == 518656);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan_d128, 1, 128, resident_text_tokens, grid_w,
                                                518655) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan_d128, 1, 128, resident_text_tokens, grid_w,
                                                518656) == 0);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan, 1, 64, resident_text_tokens, INT_MAX,
                                                std::numeric_limits<size_t>::max()) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&l2_plan, 1, 64, INT_MAX, grid_w,
                                                std::numeric_limits<size_t>::max()) == -1);

  block_sparse_halo_plan_t dma_plan = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&dma_plan, 2, 64, text_tokens, grid_w,
                                                std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(dma_plan.blk_sz_c == 128);
  OP_TEST_ASSERT(dma_plan.workspace_offset == 0);
  OP_TEST_ASSERT(dma_plan.workspace_bytes == 68096);
  OP_TEST_ASSERT(dma_plan.packed_k_offset == 8192 && dma_plan.packed_k_bytes == 16384);
  OP_TEST_ASSERT(dma_plan.packed_v_offset == 24576 && dma_plan.packed_v_bytes == 16384);
  OP_TEST_ASSERT(dma_plan.resident_text_bytes == 16384);
  OP_TEST_ASSERT(dma_plan.resident_text_offset == 200704);
  OP_TEST_ASSERT(dma_plan.packed_image_row_slot_count == 3);
  OP_TEST_ASSERT(dma_plan.packed_image_row_offsets[0] == 69632);
  OP_TEST_ASSERT(dma_plan.packed_image_row_offsets[1] == 102400);
  OP_TEST_ASSERT(dma_plan.packed_image_row_offsets[2] == 135168);
  OP_TEST_ASSERT(dma_plan.packed_image_row_stride == 32768);
  OP_TEST_ASSERT(dma_plan.packed_image_row_bytes == 98304);
  OP_TEST_ASSERT(dma_plan.image_row_offsets[0] == 167936);
  OP_TEST_ASSERT(dma_plan.image_row_stride == 32768);
  OP_TEST_ASSERT(dma_plan.image_row_bytes == 32768);
  OP_TEST_ASSERT(dma_plan.image_bank_bytes == 0);
  OP_TEST_ASSERT(dma_plan.descriptor_bytes == 2 * sizeof(dma_desc_2d_t));
  OP_TEST_ASSERT(dma_plan.completion_bytes == 0);
  OP_TEST_ASSERT(dma_plan.resident_text_q_offset == 217088);
  OP_TEST_ASSERT(dma_plan.resident_text_q_bytes == 8192);
  OP_TEST_ASSERT(dma_plan.resident_text_o_offsets[0] == 225280);
  OP_TEST_ASSERT(dma_plan.resident_text_o_offsets[1] == 233472);
  OP_TEST_ASSERT(dma_plan.resident_text_o_bytes == 8192);
  OP_TEST_ASSERT(dma_plan.resident_text_state_offset == 241664);
  OP_TEST_ASSERT(dma_plan.resident_text_state_bytes == 1024);
  OP_TEST_ASSERT(dma_plan.resident_text_q_blocks == 2);
  OP_TEST_ASSERT(dma_plan.required_bytes == 242688);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&dma_plan, 2, 64, text_tokens, grid_w, 242687) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&dma_plan, 2, 64, text_tokens, grid_w, 242688) == 0);

  block_sparse_halo_plan_t selected_plan = {};
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, 64, text_tokens, grid_w, 242688) ==
                 BLOCK_SPARSE_HALO_EXEC_USERDMA_ROWS);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_SHARED, 64, text_tokens, grid_w, 192512) ==
                 BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_UNSAFE, 64, text_tokens, grid_w, 192512) ==
                 BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, 64, text_tokens, grid_w, 242687) ==
                 BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 2, false, true,
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, 64, text_tokens, grid_w, 192512) ==
                 BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 2, true, false,
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, 64, text_tokens, grid_w, 192512) ==
                 BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 1, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_UNKNOWN, 64, resident_text_tokens, grid_w, 276992) ==
                 BLOCK_SPARSE_HALO_EXEC_L2_PREFETCH);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 1, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_UNKNOWN, 64, 0, grid_w, 276992) ==
                 BLOCK_SPARSE_HALO_EXEC_L2_PREFETCH);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected_plan, 0, false, true,
                   BLOCK_SPARSE_HALO_USERDMA_UNKNOWN, 64, text_tokens, 0) ==
                 BLOCK_SPARSE_HALO_EXEC_BASELINE);

  block_sparse_halo_plan_t dma_plan_d128 = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&dma_plan_d128, 2, 128, text_tokens, grid_w,
                                                std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(dma_plan_d128.workspace_bytes == 109056);
  OP_TEST_ASSERT(dma_plan_d128.packed_k_offset == 16384 && dma_plan_d128.packed_k_bytes == 32768);
  OP_TEST_ASSERT(dma_plan_d128.packed_v_offset == 49152 && dma_plan_d128.packed_v_bytes == 32768);
  OP_TEST_ASSERT(dma_plan_d128.packed_image_row_slot_count == 3);
  OP_TEST_ASSERT(dma_plan_d128.packed_image_row_offsets[0] == 110592);
  OP_TEST_ASSERT(dma_plan_d128.packed_image_row_offsets[1] == 176128);
  OP_TEST_ASSERT(dma_plan_d128.packed_image_row_offsets[2] == 241664);
  OP_TEST_ASSERT(dma_plan_d128.packed_image_row_stride == 65536);
  OP_TEST_ASSERT(dma_plan_d128.packed_image_row_bytes == 196608);
  OP_TEST_ASSERT(dma_plan_d128.image_row_offsets[0] == 307200);
  OP_TEST_ASSERT(dma_plan_d128.image_row_stride == 65536);
  OP_TEST_ASSERT(dma_plan_d128.image_row_bytes == 65536);
  OP_TEST_ASSERT(dma_plan_d128.image_bank_bytes == 0);
  OP_TEST_ASSERT(dma_plan_d128.descriptor_bytes == 2 * sizeof(dma_desc_2d_t));
  OP_TEST_ASSERT(dma_plan_d128.completion_bytes == 0);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_offset == 372736);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_bytes == 32768);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_q_offset == 405504);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_q_bytes == 16384);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_o_offsets[0] == 421888);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_o_offsets[1] == 438272);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_o_bytes == 16384);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_state_offset == 454656);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_state_bytes == 1024);
  OP_TEST_ASSERT(dma_plan_d128.resident_text_q_blocks == 2);
  OP_TEST_ASSERT(dma_plan_d128.required_bytes == 455680);

  const void *k_preflight = reinterpret_cast<const void *>(0x10000000u);
  const void *v_preflight = reinterpret_cast<const void *>(0x20000000u);
  const void *vtcm_preflight = reinterpret_cast<const void *>(0x30000000u);
  OP_TEST_ASSERT(block_sparse_halo_dma_preflight(
                   k_preflight, v_preflight, vtcm_preflight, dma_plan.required_bytes, 2,
                   2, 64, image_tokens, &dma_plan) == 0);
  OP_TEST_ASSERT(block_sparse_halo_dma_preflight(
                   k_preflight, v_preflight, vtcm_preflight, dma_plan.required_bytes, 2,
                   512, 64, image_tokens, &dma_plan) == -1);
  OP_TEST_ASSERT(block_sparse_halo_dma_preflight(
                   reinterpret_cast<const void *>(UINT32_MAX - 127u), v_preflight, vtcm_preflight,
                   dma_plan.required_bytes, 2, 1, 64, image_tokens, &dma_plan) == -1);
  OP_TEST_ASSERT(block_sparse_halo_dma_preflight(
                   k_preflight, v_preflight, reinterpret_cast<const void *>(UINT32_MAX - 127u),
                   dma_plan.required_bytes, 2, 1, 64, image_tokens, &dma_plan) == -1);

  block_sparse_halo_dma_bank_t *banks[2] = {};
  alignas(64) dma_desc_2d_t external_descs[2][BLOCK_SPARSE_HALO_DMA_DESCS_PER_BANK];
  dma_desc_2d_t *external_desc_ptrs[2] = {external_descs[0], external_descs[1]};
  void *layout_storage = nullptr;
  OP_TEST_ASSERT(posix_memalign(&layout_storage, 4096, dma_plan.required_bytes) == 0);
  OP_TEST_ASSERT(block_sparse_halo_init_dma_banks(
                   banks, external_desc_ptrs, layout_storage, dma_plan.required_bytes, &dma_plan) == -1);
  std::free(layout_storage);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&dma_plan, 2, std::numeric_limits<int>::max(),
                                                std::numeric_limits<int>::max(),
                                                std::numeric_limits<size_t>::max()) == -1);

  FARF(ALWAYS, "block_sparse_halo_contracts: PASS");
}

void test_block_sparse_halo_image_only_contracts() {
  constexpr int image_block_cols = 9;
  constexpr size_t unlimited = std::numeric_limits<size_t>::max();

  for (int head_dim : {64, 128}) {
    block_sparse_halo_plan_t plan = {};
    OP_TEST_ASSERT(block_sparse_halo_select_plan(&plan, 1, head_dim, 0, image_block_cols, unlimited) == 0);
    OP_TEST_ASSERT(plan.required_bytes > 0);
    OP_TEST_ASSERT(plan.resident_text_bytes == 0);
    OP_TEST_ASSERT(plan.resident_text_q_bytes == 0);
    OP_TEST_ASSERT(plan.resident_text_o_bytes == 0);
    OP_TEST_ASSERT(plan.resident_text_state_bytes == 0);
    OP_TEST_ASSERT(plan.packed_image_row_slot_count == BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS);

    block_sparse_halo_plan_t selected = {};
    OP_TEST_ASSERT(block_sparse_halo_select_execution(
                     &selected, 1, true, true, BLOCK_SPARSE_HALO_USERDMA_UNKNOWN,
                     head_dim, 0, image_block_cols, plan.required_bytes) == BLOCK_SPARSE_HALO_EXEC_L2_PREFETCH);
    OP_TEST_ASSERT(selected.required_bytes == plan.required_bytes);

    block_sparse_halo_plan_t dma_selected = {};
    OP_TEST_ASSERT(block_sparse_halo_select_execution(
                     &dma_selected, 2, true, true, BLOCK_SPARSE_HALO_USERDMA_PER_WORKER,
                     head_dim, 0, image_block_cols, unlimited) == BLOCK_SPARSE_HALO_EXEC_USERDMA_ROWS);
    OP_TEST_ASSERT(dma_selected.required_bytes > 0);
    OP_TEST_ASSERT(dma_selected.resident_text_bytes == 0);
    OP_TEST_ASSERT(dma_selected.resident_text_q_bytes == 0);
    OP_TEST_ASSERT(dma_selected.resident_text_o_bytes == 0);
    OP_TEST_ASSERT(dma_selected.resident_text_state_bytes == 0);
  }

  FARF(ALWAYS, "block_sparse_halo_image_only_contracts: PASS");
}

void test_block_sparse_halo_row_ring_contracts() {
  block_sparse_halo_row_schedule_t schedule = {};
  OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, 0, 4) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == 0 && schedule.active_row_count == 2 && schedule.load_row == 2);
  OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, 1, 4) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == 0 && schedule.active_row_count == 3 && schedule.load_row == 3);
  OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, 2, 4) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == 1 && schedule.active_row_count == 3 && schedule.load_row == -1);
  OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, 3, 4) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == 2 && schedule.active_row_count == 2 && schedule.load_row == -1);
  OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, -1, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, 0, 0) == -1);
  OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, INT_MAX - 1, INT_MAX) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == INT_MAX - 2 && schedule.active_row_count == 2 && schedule.load_row == -1);

  OP_TEST_ASSERT(block_sparse_halo_schedule_text_row(&schedule, 0, 4) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == 0 && schedule.active_row_count == 1 && schedule.load_row == 1);
  OP_TEST_ASSERT(block_sparse_halo_schedule_text_row(&schedule, 2, 4) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == 2 && schedule.active_row_count == 1 && schedule.load_row == 3);
  OP_TEST_ASSERT(block_sparse_halo_schedule_text_row(&schedule, 3, 4) == 0);
  OP_TEST_ASSERT(schedule.first_active_row == 3 && schedule.active_row_count == 1 && schedule.load_row == -1);
  OP_TEST_ASSERT(block_sparse_halo_schedule_text_row(&schedule, -1, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_schedule_text_row(&schedule, 4, 4) == -1);

  OP_TEST_ASSERT(block_sparse_halo_text_prefetch_target(0, 0, 4) == 1);
  OP_TEST_ASSERT(block_sparse_halo_text_prefetch_target(0, 1, 4) == 2);
  OP_TEST_ASSERT(block_sparse_halo_text_prefetch_target(2, 1, 4) == 2);
  OP_TEST_ASSERT(block_sparse_halo_text_prefetch_target(2, 3, 4) == 3);
  OP_TEST_ASSERT(block_sparse_halo_text_prefetch_target(-1, 0, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_text_prefetch_target(0, 4, 4) == -1);

  int image_query_rows_loaded = 2;
  for (int q_row = 0; q_row < 4; ++q_row) {
    OP_TEST_ASSERT(block_sparse_halo_schedule_q_row(&schedule, q_row, 4) == 0);
    image_query_rows_loaded += schedule.load_row >= 0 ? 1 : 0;
  }
  OP_TEST_ASSERT(image_query_rows_loaded == 4);

  int text_query_rows_loaded = 0;
  for (int text_q = 0; text_q < 5; ++text_q) {
    for (int image_row = 0; image_row < 4; ++image_row) {
      OP_TEST_ASSERT(block_sparse_halo_schedule_text_row(&schedule, image_row, 4) == 0);
      ++text_query_rows_loaded;
    }
  }
  OP_TEST_ASSERT(text_query_rows_loaded == 20);

  block_sparse_halo_row_slot_t slots[BLOCK_SPARSE_HALO_ROW_SLOTS] = {};
  for (int i = 0; i < BLOCK_SPARSE_HALO_ROW_SLOTS; ++i) {
    slots[i].row_id = -1;
  }
  slots[0].row_id = 0;
  slots[0].ready = true;
  slots[1].row_id = 1;
  slots[1].ready = true;
  slots[2].row_id = 2;
  slots[2].ready = true;
  slots[3].row_id = 3;
  slots[3].ready = false;
  OP_TEST_ASSERT(block_sparse_halo_find_ready_row_slot(slots, 0) == 0);
  OP_TEST_ASSERT(block_sparse_halo_find_ready_row_slot(slots, 3) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_staging_slot(slots, 0, 3) == 3);
  OP_TEST_ASSERT(block_sparse_halo_rotate_row_slots(slots, 0, 4) == 0);
  OP_TEST_ASSERT(slots[0].row_id == 4 && !slots[0].ready && !slots[0].outstanding);
  OP_TEST_ASSERT(block_sparse_halo_find_ready_row_slot(slots, 0) == -1);
  OP_TEST_ASSERT(block_sparse_halo_rotate_row_slots(slots, 1, 2) == -1);

  block_sparse_halo_packed_row_slot_t packed_slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS] = {};
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
    packed_slots[slot].row_id = -1;
  }
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, 0, 0) == 0);
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, 1, 1) == 0);
  OP_TEST_ASSERT(block_sparse_halo_find_packed_row(packed_slots, 0) == 0);
  OP_TEST_ASSERT(block_sparse_halo_find_packed_row(packed_slots, 1) == 1);
  OP_TEST_ASSERT(block_sparse_halo_find_packed_row(packed_slots, 2) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_packed_row_victim(packed_slots, 0, 2) == 2);
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, 2, 2) == 0);
  OP_TEST_ASSERT(block_sparse_halo_select_packed_row_victim(packed_slots, 1, 2) == 0);
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, 0, 3) == 0);
  OP_TEST_ASSERT(block_sparse_halo_find_packed_row(packed_slots, 0) == -1);
  OP_TEST_ASSERT(block_sparse_halo_find_packed_row(packed_slots, 3) == 0);
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, 1, 3) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_packed_row_victim(packed_slots, 1, 3) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_packed_row_victim(packed_slots, -1, 2) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_packed_row_victim(packed_slots, INT_MAX, 1) == -1);
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, -1, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, 3, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(packed_slots, 0, -1) == -1);

  block_sparse_halo_packed_row_slot_t packed = {};
  packed.packed_k = reinterpret_cast<__fp16 *>(uintptr_t{0x10000000});
  packed.packed_v = reinterpret_cast<__fp16 *>(uintptr_t{0x20000000});
  packed.row_id = 2;
  packed.ready = true;
  block_sparse_halo_packed_segment_t segment = {};
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 1, 2, 4, 64) == 0);
  OP_TEST_ASSERT(reinterpret_cast<uintptr_t>(segment.packed_k) == 0x10001000u);
  OP_TEST_ASSERT(reinterpret_cast<uintptr_t>(segment.packed_v) == 0x20000800u);
  OP_TEST_ASSERT(segment.v_tile_stride == 4 && segment.block_count == 2);
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 0, 1, 4, 64) == 0);
  OP_TEST_ASSERT(segment.packed_k == packed.packed_k && segment.packed_v == packed.packed_v);
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 0, 3, 4, 64) == 0);
  OP_TEST_ASSERT(segment.block_count == 3);
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 0, 4, 4, 64) == 0);
  OP_TEST_ASSERT(segment.block_count == 4);
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 0, 0, 4, 64) == -1);
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 3, 2, 4, 64) == -1);
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 0, 1, 0, 64) == -1);
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 0, 1, 4, INT_MAX) == -1);
  packed.ready = false;
  OP_TEST_ASSERT(block_sparse_halo_get_packed_segment(&segment, &packed, 0, 1, 4, 64) == -1);

  OP_TEST_ASSERT(block_sparse_halo_can_merge_index_block(true, 1, 2, 16, 4, 32, 64));
  OP_TEST_ASSERT(block_sparse_halo_can_merge_index_block(true, 1, 3, 16, 4, 64, 96));
  OP_TEST_ASSERT(!block_sparse_halo_can_merge_index_block(true, 3, 4, 16, 4, 32, 64));
  OP_TEST_ASSERT(!block_sparse_halo_can_merge_index_block(true, 15, 16, 16, 4, 32, 64));
  OP_TEST_ASSERT(!block_sparse_halo_can_merge_index_block(true, 2, 4, 16, 4, 32, 64));
  OP_TEST_ASSERT(!block_sparse_halo_can_merge_index_block(true, 1, 3, 16, 4, 64, 64));
  OP_TEST_ASSERT(block_sparse_halo_can_merge_index_block(false, 3, 4, 16, 4, 32, 64));
  OP_TEST_ASSERT(block_sparse_halo_can_merge_index_block(false, 15, 16, 16, 4, 32, 64));
  OP_TEST_ASSERT(!block_sparse_halo_can_merge_index_block(false, 1, 2, 16, 4, 0, 64));
  OP_TEST_ASSERT(!block_sparse_halo_can_merge_index_block(false, 1, 2, 16, 4, 32, 0));

  OP_TEST_ASSERT(block_sparse_halo_text_q_blocks(154) == 5);
  OP_TEST_ASSERT(block_sparse_halo_text_q_rows(154, 0) == 32);
  OP_TEST_ASSERT(block_sparse_halo_text_q_rows(154, 1) == 32);
  OP_TEST_ASSERT(block_sparse_halo_text_q_rows(154, 2) == 32);
  OP_TEST_ASSERT(block_sparse_halo_text_q_rows(154, 3) == 32);
  OP_TEST_ASSERT(block_sparse_halo_text_q_rows(154, 4) == 26);
  OP_TEST_ASSERT(block_sparse_halo_text_q_blocks(0) == 0);
  OP_TEST_ASSERT(block_sparse_halo_text_q_blocks(-1) == -1);
  OP_TEST_ASSERT(block_sparse_halo_text_q_rows(154, 5) == -1);

  int text_o_bank = 0;
  OP_TEST_ASSERT(block_sparse_halo_text_o_next_bank(text_o_bank, 4, 5) == -1);
  text_o_bank = block_sparse_halo_text_o_next_bank(text_o_bank, 5, 5);
  OP_TEST_ASSERT(text_o_bank == 1);
  text_o_bank = block_sparse_halo_text_o_next_bank(text_o_bank, 5, 5);
  OP_TEST_ASSERT(text_o_bank == 0);
  text_o_bank = block_sparse_halo_text_o_next_bank(text_o_bank, 5, 5);
  OP_TEST_ASSERT(text_o_bank == 1);
  OP_TEST_ASSERT(block_sparse_halo_text_o_next_bank(2, 5, 5) == -1);
  OP_TEST_ASSERT(block_sparse_halo_text_o_next_bank(0, 0, 0) == -1);

  for (int image_rows : {4, 6}) {
    int startup_packs = 0;
    int prefetches = 0;
    int text_row_updates = 0;
    int text_q_updates = 0;
    int packed_image_blocks = 0;
    int text_q_block_updates = 0;
    for (int row = 0; row < image_rows; ++row) {
      block_sparse_halo_packed_row_event_t event = {};
      OP_TEST_ASSERT(block_sparse_halo_plan_packed_row_event(&event, row, image_rows, 4, 5) == 0);
      OP_TEST_ASSERT(event.row_id == row);
      OP_TEST_ASSERT(event.startup_pack == (row < 2));
      OP_TEST_ASSERT(event.prefetch_row == (row >= 2 ? row : -1));
      OP_TEST_ASSERT(event.evicted_row == (row >= 3 ? row - 3 : -1));
      OP_TEST_ASSERT(event.text_q_updates == 5);
      OP_TEST_ASSERT(event.image_block_count == 4);
      OP_TEST_ASSERT(event.text_q_block_updates == 20);
      startup_packs += event.startup_pack;
      prefetches += event.prefetch_row >= 0;
      ++text_row_updates;
      text_q_updates += event.text_q_updates;
      packed_image_blocks += event.image_block_count;
      text_q_block_updates += event.text_q_block_updates;
    }
    OP_TEST_ASSERT(startup_packs == 2);
    OP_TEST_ASSERT(prefetches == image_rows - 2);
    OP_TEST_ASSERT(text_row_updates == image_rows);
    OP_TEST_ASSERT(text_q_updates == image_rows * 5);
    OP_TEST_ASSERT(packed_image_blocks == image_rows * 4);
    OP_TEST_ASSERT(text_q_block_updates == image_rows * 5 * 4);
    if (image_rows == 4) {
      OP_TEST_ASSERT(packed_image_blocks == 16 && prefetches == 2 &&
                     text_q_updates == 20 && text_q_block_updates == 80);
    } else {
      OP_TEST_ASSERT(packed_image_blocks == 24 && prefetches == 4 &&
                     text_q_updates == 30 && text_q_block_updates == 120);
    }
  }

  OP_TEST_ASSERT(block_sparse_halo_next_q_block(0, 4) == 1);
  OP_TEST_ASSERT(block_sparse_halo_next_q_block(1, 4) == 2);
  OP_TEST_ASSERT(block_sparse_halo_next_q_block(2, 4) == 3);
  OP_TEST_ASSERT(block_sparse_halo_next_q_block(3, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_next_q_block(0, 1) == -1);
  OP_TEST_ASSERT(block_sparse_halo_next_q_block(-1, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_next_q_block(4, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_next_q_block(0, 0) == -1);

  block_sparse_halo_packed_row_event_t clipped_event = {};
  OP_TEST_ASSERT(block_sparse_halo_plan_packed_row_event(&clipped_event, 4, 4, 4, 5) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_packed_row_event(&clipped_event, -1, 4, 4, 5) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_packed_row_event(&clipped_event, 0, 0, 4, 5) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_packed_row_event(&clipped_event, 0, 4, 0, 5) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_packed_row_event(&clipped_event, 0, 4, 4, 0) == -1);

  FARF(ALWAYS, "block_sparse_halo_row_ring_contracts: PASS");
}

void test_block_sparse_halo_text_kv_batch_contracts() {
  block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS] = {};
  auto set_slot = [&](int slot, int row_id, bool ready = true) {
    slots[slot].row_id = row_id;
    slots[slot].ready = ready;
  };

  block_sparse_halo_text_kv_batch_t batch = {};

  // Row order is logical, even when the packed ring is physically permuted.
  set_slot(0, 2);
  set_slot(1, 0);
  set_slot(2, 1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 0, 2, 6, false) == 0);
  OP_TEST_ASSERT(batch.row_count == 3 && batch.next_pending_row == 3);
  OP_TEST_ASSERT(batch.row_ids[0] == 0 && batch.row_ids[1] == 1 && batch.row_ids[2] == 2);
  OP_TEST_ASSERT(batch.packed_slots[0] == 1 && batch.packed_slots[1] == 2 && batch.packed_slots[2] == 0);

  set_slot(0, 5);
  set_slot(1, 3);
  set_slot(2, 4);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 5, 6, false) == 0);
  OP_TEST_ASSERT(batch.row_count == 3 && batch.next_pending_row == 6);
  OP_TEST_ASSERT(batch.row_ids[0] == 3 && batch.row_ids[1] == 4 && batch.row_ids[2] == 5);
  OP_TEST_ASSERT(batch.packed_slots[0] == 1 && batch.packed_slots[1] == 2 && batch.packed_slots[2] == 0);

  // A normal update defers until three rows are available.
  set_slot(0, 0);
  set_slot(1, 1);
  set_slot(2, -1, false);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 2, 1, 6, false) == 0);
  OP_TEST_ASSERT(batch.row_count == 0 && batch.next_pending_row == 2);
  set_slot(0, 2);
  set_slot(1, 3);
  set_slot(2, 4);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 4, 6, false) == 0);
  OP_TEST_ASSERT(batch.row_count == 0 && batch.next_pending_row == 3);

  // The final flush consumes a one- or two-row tail.
  set_slot(0, 1);
  set_slot(1, 2);
  set_slot(2, 3);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 3, 4, true) == 0);
  OP_TEST_ASSERT(batch.row_count == 1 && batch.row_ids[0] == 3 && batch.packed_slots[0] == 2 &&
                 batch.next_pending_row == 4);

  set_slot(0, 4);
  set_slot(1, 2);
  set_slot(2, 3);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 4, 5, true) == 0);
  OP_TEST_ASSERT(batch.row_count == 2 && batch.row_ids[0] == 3 && batch.row_ids[1] == 4);
  OP_TEST_ASSERT(batch.packed_slots[0] == 2 && batch.packed_slots[1] == 0 && batch.next_pending_row == 5);

  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 5, 4, 5, true) == 0);
  OP_TEST_ASSERT(batch.row_count == 0 && batch.next_pending_row == 5);

  // Once a later row is known packed, every pending row must still be ready in the ring.
  set_slot(0, 2);
  set_slot(1, 0);
  set_slot(2, 1, false);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 0, 2, 6, false) == -1);
  set_slot(0, 3);
  set_slot(1, 1);
  set_slot(2, 2);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 0, 3, 6, false) == -1);
  set_slot(0, 4);
  set_slot(1, 2);
  set_slot(2, -1, false);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 4, 6, false) == -1);
  set_slot(0, 3);
  set_slot(1, 3);
  set_slot(2, 4);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 4, 6, false) == -1);

  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(nullptr, slots, 0, 2, 6, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, nullptr, 0, 2, 6, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, -1, 2, 6, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 0, -2, 6, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 0, 2, 0, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 7, 5, 6, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 0, 6, 6, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 1, 6, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(&batch, slots, 3, 3, 5, true) == -1);

  // The largest valid two-row tail reaches INT_MAX without overflowing.
  set_slot(0, INT_MAX - 1);
  set_slot(1, INT_MAX - 2);
  set_slot(2, INT_MAX - 3);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(
                   &batch, slots, INT_MAX - 2, INT_MAX - 1, INT_MAX, true) == 0);
  OP_TEST_ASSERT(batch.row_count == 2 && batch.row_ids[0] == INT_MAX - 2 &&
                 batch.row_ids[1] == INT_MAX - 1 && batch.next_pending_row == INT_MAX);

  // Reject out-of-range INT endpoints before any +1/+batch-width arithmetic.
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(
                   &batch, slots, INT_MAX - 1, INT_MAX, INT_MAX, false) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(
                   &batch, slots, INT_MAX, INT_MAX - 1, INT_MAX, true) == 0);
  OP_TEST_ASSERT(batch.row_count == 0 && batch.next_pending_row == INT_MAX);

  FARF(ALWAYS, "block_sparse_halo_text_kv_batch_contracts: PASS");
}

void test_block_sparse_halo_text_kv_batch_cadence() {
  const int image_rows_values[] = {3, 4, 5, 6, 7};
  const int expected_batch_counts[][3] = {
    {3, 0, 0},
    {3, 1, 0},
    {3, 2, 0},
    {3, 3, 0},
    {3, 3, 1},
  };
  const int expected_batch_totals[] = {1, 2, 2, 2, 3};

  // L2 and UserDMA share this packed-row victim and batch cadence.
  for (int test_case = 0; test_case < 5; ++test_case) {
    const int image_rows = image_rows_values[test_case];
    block_sparse_halo_packed_row_slot_t slots[BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS] = {};
    for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
      slots[slot].row_id = -1;
    }

    int first_pending_row = 0;
    int actual_batch_counts[3] = {};
    int actual_batch_total = 0;
    for (int row = 0; row < image_rows; ++row) {
      const int first_active_row = row >= 2 ? row - 2 : 0;
      const int active_row_count = row >= 2 ? 2 : 1;
      const int victim = block_sparse_halo_select_packed_row_victim(
        slots, first_active_row, active_row_count);
      OP_TEST_ASSERT(victim >= 0);
      OP_TEST_ASSERT(slots[victim].row_id < first_pending_row);
      OP_TEST_ASSERT(block_sparse_halo_bind_packed_row(slots, victim, row) == 0);

      block_sparse_halo_text_kv_batch_t batch = {};
      OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(
                       &batch, slots, first_pending_row, row, image_rows, false) == 0);
      if (batch.row_count > 0) {
        OP_TEST_ASSERT(batch.row_count == BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS);
        for (int batch_row = 0; batch_row < batch.row_count; ++batch_row) {
          OP_TEST_ASSERT(batch.row_ids[batch_row] == first_pending_row + batch_row);
          OP_TEST_ASSERT(batch.packed_slots[batch_row] ==
                         block_sparse_halo_find_packed_row(slots, batch.row_ids[batch_row]));
        }
        OP_TEST_ASSERT(actual_batch_total < 3);
        actual_batch_counts[actual_batch_total++] = batch.row_count;
        first_pending_row = batch.next_pending_row;
      }
    }

    block_sparse_halo_text_kv_batch_t tail = {};
    OP_TEST_ASSERT(block_sparse_halo_plan_text_kv_batch(
                     &tail, slots, first_pending_row, image_rows - 1, image_rows, true) == 0);
    if (tail.row_count > 0) {
      OP_TEST_ASSERT(actual_batch_total < 3);
      actual_batch_counts[actual_batch_total++] = tail.row_count;
      first_pending_row = tail.next_pending_row;
    }

    OP_TEST_ASSERT(first_pending_row == image_rows);
    OP_TEST_ASSERT(actual_batch_total == expected_batch_totals[test_case]);
    for (int batch = 0; batch < actual_batch_total; ++batch) {
      OP_TEST_ASSERT(actual_batch_counts[batch] == expected_batch_counts[test_case][batch]);
    }
  }

  FARF(ALWAYS, "block_sparse_halo_text_kv_batch_cadence: PASS");
}

void test_block_sparse_halo_row_workspace_contracts() {
  constexpr int text_tokens = 34;
  constexpr int image_block_cols = 4;
  constexpr size_t d64_row_bytes = 4u * 32u * 64u * sizeof(__fp16) * 2u;
  constexpr size_t d64_packed_rows_bytes = d64_row_bytes * BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS;

  block_sparse_halo_plan_t d64 = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&d64, 2, 64, text_tokens, image_block_cols,
                                                std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(d64.packed_image_row_slot_count == BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS);
  OP_TEST_ASSERT(d64.packed_image_row_stride == d64_row_bytes);
  OP_TEST_ASSERT(d64.packed_image_row_bytes == d64_packed_rows_bytes);
  OP_TEST_ASSERT(d64.image_row_stride == d64_row_bytes);
  OP_TEST_ASSERT(d64.image_row_bytes == d64_row_bytes);
  OP_TEST_ASSERT(d64.image_bank_bytes == 0);
  OP_TEST_ASSERT(d64.completion_bytes == 0);
  OP_TEST_ASSERT(d64.descriptor_bytes == 2 * sizeof(dma_desc_2d_t));
  for (int slot = 0; slot < BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS; ++slot) {
    OP_TEST_ASSERT((d64.packed_image_row_offsets[slot] & 63u) == 0);
    if (slot > 0) {
      OP_TEST_ASSERT(d64.packed_image_row_offsets[slot] >=
                     d64.packed_image_row_offsets[slot - 1] + d64.packed_image_row_stride);
    }
  }
  OP_TEST_ASSERT(d64.image_row_offsets[0] >=
                 d64.packed_image_row_offsets[2] + d64.packed_image_row_stride);
  OP_TEST_ASSERT(d64.image_row_offsets[0] + d64.image_row_bytes <= d64.resident_text_offset);
  OP_TEST_ASSERT(d64.required_bytes == 242688);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&d64, 2, 64, text_tokens, image_block_cols,
                                                d64.required_bytes - 1) == -1);
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&d64, 2, 64, text_tokens, image_block_cols,
                                                d64.required_bytes) == 0);

  constexpr size_t d128_row_bytes = 4u * 32u * 128u * sizeof(__fp16) * 2u;
  block_sparse_halo_plan_t d128 = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(&d128, 2, 128, text_tokens, image_block_cols,
                                                std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(d128.packed_image_row_slot_count == BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS);
  OP_TEST_ASSERT(d128.packed_image_row_stride == d128_row_bytes);
  OP_TEST_ASSERT(d128.packed_image_row_bytes == d128_row_bytes * BLOCK_SPARSE_HALO_PACKED_ROW_SLOTS);
  OP_TEST_ASSERT(d128.image_row_stride == d128_row_bytes);
  OP_TEST_ASSERT(d128.image_row_bytes == d128_row_bytes);
  OP_TEST_ASSERT(d128.resident_text_offset == 372736);
  OP_TEST_ASSERT(d128.required_bytes == 455680);

  block_sparse_halo_plan_t selected = {};
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, 64, text_tokens, image_block_cols,
                   d64.required_bytes) == BLOCK_SPARSE_HALO_EXEC_USERDMA_ROWS);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_SHARED, 64, text_tokens, image_block_cols,
                   d64.required_bytes) == BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_UNSAFE, 64, text_tokens, image_block_cols,
                   d64.required_bytes) == BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, 64, text_tokens, image_block_cols,
                   d64.required_bytes - 1) == BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(&selected, 2, true, true,
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, 64, text_tokens, INT_MAX,
                   std::numeric_limits<size_t>::max()) == BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);

  alignas(64) dma_desc_2d_t row_descs[BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW];
  const void *k_fake = reinterpret_cast<const void *>(uintptr_t{0x10000000});
  const void *v_fake = reinterpret_cast<const void *>(uintptr_t{0x20000000});
  void *raw_fake = reinterpret_cast<void *>(uintptr_t{0x30000000});
  OP_TEST_ASSERT(block_sparse_halo_build_dma_row_descs(
                   row_descs, BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW, k_fake, v_fake, raw_fake,
                   2, 1, 64, image_block_cols, 2) == BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW);
  OP_TEST_ASSERT(row_descs[0].src == 0x10010080u && row_descs[1].src == 0x20010080u);
  OP_TEST_ASSERT(row_descs[0].dst == 0x30000000u && row_descs[1].dst == 0x30004000u);
  OP_TEST_ASSERT(row_descs[0].roi_width == 128 && row_descs[0].roi_height == 128);
  OP_TEST_ASSERT(row_descs[0].src_stride == 256 && row_descs[0].dst_stride == 128);
  OP_TEST_ASSERT(row_descs[0].next == (uint32_t) (uintptr_t) &row_descs[1] && row_descs[1].next == 0);
  OP_TEST_ASSERT(block_sparse_halo_build_dma_row_descs(
                   row_descs, BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW - 1, k_fake, v_fake, raw_fake,
                   2, 1, 64, image_block_cols, 2) == -1);
  constexpr uintptr_t raw_edge = UINT32_MAX - (2u * 4u * 32u * 64u * sizeof(__fp16) - 1u);
  OP_TEST_ASSERT(block_sparse_halo_build_dma_row_descs(
                   row_descs, BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW, k_fake, v_fake,
                   reinterpret_cast<void *>(raw_edge), 1, 0, 64, image_block_cols, 0) ==
                 BLOCK_SPARSE_HALO_DMA_DESCS_PER_ROW);

  FARF(ALWAYS, "block_sparse_halo_row_workspace_contracts: PASS");
}

void test_block_sparse_halo_update_segment_plan_contracts() {
  static_assert(BLOCK_SPARSE_HALO_MAX_UPDATE_SEGMENTS == 4);

  block_sparse_halo_update_segment_plan_t plan = {};
  const int three_rows_and_text[] = {96, 96, 96, 154};
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, three_rows_and_text, 4, 448) == 0);
  OP_TEST_ASSERT(plan.total_cols == 442);
  OP_TEST_ASSERT(plan.tile_offsets[0] == 0 && plan.tile_counts[0] == 3);
  OP_TEST_ASSERT(plan.tile_offsets[1] == 3 && plan.tile_counts[1] == 3);
  OP_TEST_ASSERT(plan.tile_offsets[2] == 6 && plan.tile_counts[2] == 3);
  OP_TEST_ASSERT(plan.tile_offsets[3] == 9 && plan.tile_counts[3] == 5);

  const int two_block_rows_and_text[] = {64, 64, 64, 154};
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, two_block_rows_and_text, 4, 352) == 0);
  OP_TEST_ASSERT(plan.total_cols == 346);
  OP_TEST_ASSERT(plan.tile_offsets[0] == 0 && plan.tile_counts[0] == 2);
  OP_TEST_ASSERT(plan.tile_offsets[1] == 2 && plan.tile_counts[1] == 2);
  OP_TEST_ASSERT(plan.tile_offsets[2] == 4 && plan.tile_counts[2] == 2);
  OP_TEST_ASSERT(plan.tile_offsets[3] == 6 && plan.tile_counts[3] == 5);

  const int partial_non_final[] = {96, 95, 96, 154};
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, partial_non_final, 4, 448) == -1);

  const int five_segments[] = {32, 32, 32, 32, 32};
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, five_segments, 5, 160) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, three_rows_and_text, 4, 441) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, three_rows_and_text, 4, 416) == -1);

  const int zero_cols[] = {96, 96, 0, 154};
  const int negative_cols[] = {96, 96, -32, 154};
  const int zero_final_cols[] = {96, 96, 96, 0};
  const int negative_final_cols[] = {96, 96, 96, -1};
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, zero_cols, 4, 448) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, negative_cols, 4, 448) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, zero_final_cols, 4, 448) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, negative_final_cols, 4, 448) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, three_rows_and_text, 0, 448) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, three_rows_and_text, -1, 448) == -1);

  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   nullptr, three_rows_and_text, 4, 448) == -1);
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, nullptr, 4, 448) == -1);

  const int overflowing_cols[] = {INT_MAX - 31, 32};
  OP_TEST_ASSERT(block_sparse_halo_plan_update_segments(
                   &plan, overflowing_cols, 2,
                   size_t(INT_MAX) + 1u) == -1);

  FARF(ALWAYS, "block_sparse_halo_update_segment_plan_contracts: PASS");
}

void test_block_sparse_halo_fused_scratch_contracts() {
  constexpr int text_tokens = 154;
  const int image_block_cols_values[] = {2, 3, 4, 8, 16, 32};
  const size_t expected_scratch_cols[] = {352, 448, 448, 768, 1536, 3072};
  const int load_modes[] = {1, 2};
  const int head_dims[] = {64, 128};
  const size_t expected_required_bytes[2][2][6] = {
    {
      {215552, 244224, 276992, 428544, 743936, 1374720},
      {399872, 453120, 518656, 801280, 1378816, 2533888},
    },
    {
      {231936, 268800, 309760, 494080, 875008, 1636864},
      {432640, 502272, 584192, 932352, 1640960, 3058176},
    },
  };

  for (int load_mode_case = 0; load_mode_case < 2; ++load_mode_case) {
    const int load_mode = load_modes[load_mode_case];
    for (int head_dim_case = 0; head_dim_case < 2; ++head_dim_case) {
      const int head_dim = head_dims[head_dim_case];
      for (int col_case = 0; col_case < 6; ++col_case) {
        const int image_block_cols = image_block_cols_values[col_case];
        block_sparse_halo_plan_t plan = {};
        OP_TEST_ASSERT(block_sparse_halo_select_plan(
                         &plan, load_mode, head_dim, text_tokens, image_block_cols,
                         std::numeric_limits<size_t>::max()) == 0);

        const size_t three_row_cols = size_t(BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS) *
                                      size_t(image_block_cols) * 32u;
        const size_t image_q_cols = size_t(BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS) *
                                    size_t(std::min(3, image_block_cols)) * 32u;
        const size_t padded_text_cols = (size_t(text_tokens) + 31u) / 32u * 32u;
        OP_TEST_ASSERT(plan.scratch_col_capacity == expected_scratch_cols[col_case]);
        OP_TEST_ASSERT(plan.scratch_col_capacity >= three_row_cols);
        OP_TEST_ASSERT(plan.scratch_col_capacity >= image_q_cols + padded_text_cols);
        const size_t row_vec_bytes_unaligned =
          plan.scratch_col_capacity * sizeof(__fp16);
        const size_t expected_row_vec_bytes =
          (row_vec_bytes_unaligned + 255u) / 256u * 256u;
        OP_TEST_ASSERT(plan.probability_offset == plan.score_offset);
        OP_TEST_ASSERT(plan.o_offsets[1] == plan.o_offsets[0]);
        OP_TEST_ASSERT(plan.diagonal_offset == plan.o_offsets[0]);
        OP_TEST_ASSERT(plan.softmax_state_bytes == 2 * 256u);
        OP_TEST_ASSERT(plan.row_buffer_bytes == 2 * expected_row_vec_bytes);

        const size_t exact_bytes =
          expected_required_bytes[load_mode_case][head_dim_case][col_case];
        OP_TEST_ASSERT(plan.required_bytes == exact_bytes);
        OP_TEST_ASSERT(exact_bytes > 0);
        OP_TEST_ASSERT(block_sparse_halo_select_plan(
                         &plan, load_mode, head_dim, text_tokens, image_block_cols, exact_bytes - 1) == -1);
        OP_TEST_ASSERT(block_sparse_halo_select_plan(
                         &plan, load_mode, head_dim, text_tokens, image_block_cols, exact_bytes) == 0);
        OP_TEST_ASSERT(plan.required_bytes == exact_bytes &&
                       plan.scratch_col_capacity == expected_scratch_cols[col_case]);
      }
    }
  }

  // Image-Q fusion needs the local image halo and padded text at the same time.
  block_sparse_halo_plan_t text_dominant = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(
                   &text_dominant, 1, 64, 410, 4,
                   std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(text_dominant.scratch_col_capacity == 704);
  OP_TEST_ASSERT(text_dominant.probability_offset == text_dominant.score_offset);
  OP_TEST_ASSERT(text_dominant.diagonal_offset == text_dominant.o_offsets[0]);
  OP_TEST_ASSERT(text_dominant.row_buffer_bytes == 3072);

  // The D64 4-column image halo has 442 real columns in a 448-column
  // physical scratch row. End-to-end static-halo tests exercise this partial
  // final tile while these assertions lock the in-place S/P lifetime contract.
  constexpr int real_image_q_cols = 3 * 3 * 32 + text_tokens;
  constexpr int physical_image_q_cols = 448;
  block_sparse_halo_plan_t tail_plan = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(
                   &tail_plan, 1, 64, text_tokens, 4,
                   std::numeric_limits<size_t>::max()) == 0);
  OP_TEST_ASSERT(real_image_q_cols == 442);
  OP_TEST_ASSERT(physical_image_q_cols == 448);
  OP_TEST_ASSERT(tail_plan.scratch_col_capacity == physical_image_q_cols);
  OP_TEST_ASSERT(tail_plan.probability_offset == tail_plan.score_offset);

  // The fused kernel passes the combined column count through an int.
  constexpr int max_fused_image_block_cols =
    INT_MAX / (BLOCK_SPARSE_HALO_TEXT_KV_BATCH_ROWS * 32);
  static_assert(max_fused_image_block_cols + 1 <= INT_MAX / 32);
  block_sparse_halo_plan_t int_boundary = {};
  OP_TEST_ASSERT(block_sparse_halo_select_plan(
                   &int_boundary, 1, 64, 0, max_fused_image_block_cols + 1,
                   std::numeric_limits<size_t>::max()) == -1);

  constexpr size_t worker_bytes = 1376256;
  block_sparse_halo_plan_t selection = {};
  OP_TEST_ASSERT(block_sparse_halo_select_execution(
                   &selection, 1, true, true, BLOCK_SPARSE_HALO_USERDMA_UNKNOWN,
                   64, text_tokens, 32, worker_bytes) == BLOCK_SPARSE_HALO_EXEC_L2_PREFETCH);
  OP_TEST_ASSERT(selection.required_bytes == 1374720);
  OP_TEST_ASSERT(block_sparse_halo_select_execution(
                   &selection, 2, true, true, BLOCK_SPARSE_HALO_USERDMA_PER_WORKER,
                   64, text_tokens, 32, worker_bytes) == BLOCK_SPARSE_HALO_EXEC_CHUNKED_FALLBACK);

  FARF(ALWAYS, "block_sparse_halo_fused_scratch_contracts: PASS");
}

static void test_block_sparse_halo_raw_row_pack_contracts_worker_body() {
  constexpr int head_dim = 64;
  constexpr int image_block_cols = 4;
  constexpr int block_tokens = 32;
  constexpr int tile_elems = 32 * 32;
  constexpr int row_tokens = image_block_cols * block_tokens;
  constexpr size_t row_half_elems = size_t(row_tokens) * head_dim;
  constexpr size_t row_half_bytes = row_half_elems * sizeof(__fp16);
  constexpr size_t temp_bytes = size_t(2 * block_tokens) * head_dim * sizeof(__fp16);

  uint8_t *vtcm = static_cast<uint8_t *>(vtcm_manager_get_vtcm_base());
  const size_t required_bytes = 4 * row_half_bytes + 2 * temp_bytes;
  OP_TEST_ASSERT(vtcm != nullptr && vtcm_manager_get_vtcm_size() >= required_bytes);
  void *raw_storage = vtcm;
  void *full_k_storage = vtcm + 2 * row_half_bytes;
  void *full_v_storage = vtcm + 3 * row_half_bytes;
  void *temp_k_storage = vtcm + 4 * row_half_bytes;
  void *temp_v_storage = vtcm + 4 * row_half_bytes + temp_bytes;

  auto *raw_k = static_cast<__fp16 *>(raw_storage);
  auto *raw_v = raw_k + row_half_elems;
  for (size_t i = 0; i < row_half_elems; ++i) {
    raw_k[i] = (__fp16) (float((i * 13 + 7) % 251) / 32.0f - 4.0f);
    raw_v[i] = (__fp16) (float((i * 17 + 11) % 241) / 32.0f - 3.0f);
  }

  block_sparse_halo_row_slot_t raw_slot = {
    .raw = static_cast<uint8_t *>(raw_storage),
    .row_id = 0,
    .ready = true,
    .outstanding = false,
  };
  block_sparse_halo_packed_row_slot_t full_row = {
    .packed_k = static_cast<__fp16 *>(full_k_storage),
    .packed_v = static_cast<__fp16 *>(full_v_storage),
  };
  OP_TEST_ASSERT(block_sparse_halo_pack_image_row(
                   &full_row, raw_k, raw_v, 1, 0, head_dim, image_block_cols, 0) == 0);

  auto *temp_k = static_cast<__fp16 *>(temp_k_storage);
  auto *temp_v = static_cast<__fp16 *>(temp_v_storage);
  for (int block_count : {1, 2}) {
    constexpr int col_start = 1;
    OP_TEST_ASSERT(block_sparse_halo_pack_raw_row_segment(
                     temp_k, temp_v, temp_bytes, &raw_slot, col_start, block_count,
                     image_block_cols, head_dim) == 0);
    const size_t k_elems = size_t(block_count) * block_tokens * head_dim;
    OP_TEST_ASSERT(std::memcmp(
                     temp_k,
                     full_row.packed_k + size_t(col_start) * block_tokens * head_dim,
                     k_elems * sizeof(__fp16)) == 0);
    for (int d_tile = 0; d_tile < head_dim / 32; ++d_tile) {
      for (int block = 0; block < block_count; ++block) {
        const __fp16 *temp_tile = temp_v + size_t(d_tile * 2 + block) * tile_elems;
        const __fp16 *full_tile = full_row.packed_v +
          size_t(d_tile * image_block_cols + col_start + block) * tile_elems;
        OP_TEST_ASSERT(std::memcmp(temp_tile, full_tile, tile_elems * sizeof(__fp16)) == 0);
      }
      if (block_count == 1) {
        const __fp16 *padding_tile = temp_v + size_t(d_tile * 2 + 1) * tile_elems;
        for (int i = 0; i < tile_elems; ++i) {
          OP_TEST_ASSERT(padding_tile[i] == (__fp16) 0.0f);
        }
      }
    }
  }

  raw_slot.ready = false;
  OP_TEST_ASSERT(block_sparse_halo_pack_raw_row_segment(
                   temp_k, temp_v, temp_bytes, &raw_slot, 0, 1,
                   image_block_cols, head_dim) == -1);
  raw_slot.ready = true;
  raw_slot.outstanding = true;
  OP_TEST_ASSERT(block_sparse_halo_pack_raw_row_segment(
                   temp_k, temp_v, temp_bytes, &raw_slot, 0, 1,
                   image_block_cols, head_dim) == -1);
  raw_slot.outstanding = false;
  OP_TEST_ASSERT(block_sparse_halo_pack_raw_row_segment(
                   temp_k, temp_v, temp_bytes - sizeof(__fp16), &raw_slot, 0, 2,
                   image_block_cols, head_dim) == -1);

  OP_TEST_ASSERT(block_sparse_halo_text_o_bank_after_image_row(0, 1) == 1);
  OP_TEST_ASSERT(block_sparse_halo_text_o_bank_after_image_row(0, 2) == 1);
  OP_TEST_ASSERT(block_sparse_halo_text_o_bank_after_image_row(0, 3) == 1);
  OP_TEST_ASSERT(block_sparse_halo_text_o_bank_after_image_row(1, 4) == 0);
  OP_TEST_ASSERT(block_sparse_halo_text_o_bank_after_image_row(2, 4) == -1);
  OP_TEST_ASSERT(block_sparse_halo_text_o_bank_after_image_row(0, 0) == -1);

  FARF(ALWAYS, "block_sparse_halo_raw_row_pack_contracts: PASS");
}

typedef struct {
  worker_synctoken_t sync;
} block_sparse_halo_raw_row_pack_test_state_t;

static void test_block_sparse_halo_raw_row_pack_contracts_worker(void *data, int worker_index) {
  (void) worker_index;
  auto *state = static_cast<block_sparse_halo_raw_row_pack_test_state_t *>(data);
  hmx_manager_enable_execution();
  test_block_sparse_halo_raw_row_pack_contracts_worker_body();
  hmx_manager_disable_execution();
  worker_pool_synctoken_jobdone(&state->sync);
}

void test_block_sparse_halo_raw_row_pack_contracts() {
  block_sparse_halo_raw_row_pack_test_state_t state = {};
  worker_pool_synctoken_init(&state.sync, 1);
  worker_pool_job_t job = {
    .fptr = test_block_sparse_halo_raw_row_pack_contracts_worker,
    .dptr = &state,
  };
  OP_TEST_ASSERT(worker_pool_submit(nullptr, job) == AEE_SUCCESS);
  worker_pool_synctoken_wait(&state.sync);
}

void test_block_sparse_packed_layout_contracts() {
  constexpr int token_tiles = 7;
  constexpr int head_dim = 64;
  constexpr size_t tile_elems = 32 * 32;
  constexpr size_t head_elems = token_tiles * (head_dim / 32) * tile_elems;
  size_t offset = 0;

  OP_TEST_ASSERT(block_sparse_packed_qk_tile_offset(
                   &offset, 2, 3, 1, token_tiles, head_dim) == 0);
  OP_TEST_ASSERT(offset == 2 * head_elems + (3 * 2 + 1) * tile_elems);

  OP_TEST_ASSERT(block_sparse_packed_v_tile_offset(
                   &offset, 2, 3, 1, token_tiles, head_dim) == 0);
  OP_TEST_ASSERT(offset == 2 * head_elems + (1 * token_tiles + 3) * tile_elems);

  OP_TEST_ASSERT(block_sparse_packed_qk_tile_offset(
                   &offset, -1, 0, 0, token_tiles, head_dim) == -1);
  OP_TEST_ASSERT(block_sparse_packed_qk_tile_offset(
                   &offset, 0, token_tiles, 0, token_tiles, head_dim) == -1);
  OP_TEST_ASSERT(block_sparse_packed_v_tile_offset(
                   &offset, 0, 0, head_dim / 32, token_tiles, head_dim) == -1);
  OP_TEST_ASSERT(block_sparse_packed_v_tile_offset(
                   &offset, 0, 0, 0, token_tiles, 48) == -1);
}

void test_block_sparse_halo_userdma_scope() {
  const int ok[2] = {DMA_SUBMIT_OK, DMA_SUBMIT_OK};
  const int busy[2] = {DMA_SUBMIT_OK, DMA_SUBMIT_BUSY};
  const int wait_ok[2] = {DMA_SUBMIT_OK, DMA_SUBMIT_OK};
  const bool valid[2] = {true, true};
  const bool second_invalid[2] = {true, false};
  const int distinct_workers[2] = {0, 1};
  const int same_worker[2] = {0, 0};

  OP_TEST_ASSERT(block_sparse_halo_classify_userdma_pair(
                   ok, wait_ok, valid, valid, distinct_workers) == BLOCK_SPARSE_HALO_USERDMA_PER_WORKER);
  OP_TEST_ASSERT(block_sparse_halo_classify_userdma_pair(
                   busy, wait_ok, second_invalid, valid, distinct_workers) == BLOCK_SPARSE_HALO_USERDMA_SHARED);
  OP_TEST_ASSERT(block_sparse_halo_classify_userdma_pair(
                   ok, wait_ok, valid, valid, same_worker) == BLOCK_SPARSE_HALO_USERDMA_UNSAFE);
  OP_TEST_ASSERT(block_sparse_halo_classify_userdma_pair(
                   ok, wait_ok, valid, second_invalid, distinct_workers) == BLOCK_SPARSE_HALO_USERDMA_UNSAFE);
  OP_TEST_ASSERT(block_sparse_halo_finalize_userdma_scope(
                   BLOCK_SPARSE_HALO_USERDMA_PER_WORKER, true) == BLOCK_SPARSE_HALO_USERDMA_PER_WORKER);
  OP_TEST_ASSERT(block_sparse_halo_finalize_userdma_scope(
                   BLOCK_SPARSE_HALO_USERDMA_SHARED, true) == BLOCK_SPARSE_HALO_USERDMA_SHARED);
  OP_TEST_ASSERT(block_sparse_halo_finalize_userdma_scope(
                   BLOCK_SPARSE_HALO_USERDMA_UNSAFE, true) == BLOCK_SPARSE_HALO_USERDMA_SHARED);
  OP_TEST_ASSERT(block_sparse_halo_finalize_userdma_scope(
                   BLOCK_SPARSE_HALO_USERDMA_UNSAFE, false) == BLOCK_SPARSE_HALO_USERDMA_UNSAFE);

  const int scope = block_sparse_halo_probe_userdma_scope();
  OP_TEST_ASSERT(scope >= BLOCK_SPARSE_HALO_USERDMA_PER_WORKER &&
                 scope <= BLOCK_SPARSE_HALO_USERDMA_UNSAFE);
  FARF(ALWAYS, "block_sparse_halo_userdma_scope: CLASSIFIED scope=%d", scope);
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

void test_fp32_vector_log() {
  alignas(VLEN) float input[VLEN / sizeof(float)];
  alignas(VLEN) float output[VLEN / sizeof(float)];
  for (int i = 0; i < VLEN / (int) sizeof(float); ++i) {
    input[i] = exp2f(-1.0f + 13.0f * (float) i /
                    (float) (VLEN / sizeof(float) - 1));
  }

  vmem(output) = hvx_my_log_vsf(vmem(input));
  for (int i = 0; i < VLEN / (int) sizeof(float); ++i) {
    if (fabsf(output[i] - logf(input[i])) > 2e-5f) {
      FARF(ALWAYS, "%s: i=%d input=%g output=%g reference=%g", __func__, i,
           input[i], output[i], logf(input[i]));
      abort();
    }
  }
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

  test_qkv_mul_rms_packed_dual_contracts();
  test_block_sparse_30_head_contracts();
  test_block_sparse_fa_select_plan();
  test_block_sparse_halo_contracts();
  test_block_sparse_halo_image_only_contracts();
  test_block_sparse_halo_row_ring_contracts();
  test_block_sparse_halo_text_kv_batch_contracts();
  test_block_sparse_halo_text_kv_batch_cadence();
  test_block_sparse_halo_row_workspace_contracts();
  test_block_sparse_halo_update_segment_plan_contracts();
  test_block_sparse_halo_fused_scratch_contracts();
  test_block_sparse_halo_raw_row_pack_contracts();
  test_block_sparse_packed_layout_contracts();
  test_block_sparse_halo_userdma_scope();
  test_block_sparse_attn_buffer_size_contracts();
  test_op_executor_sparse_payload_boundaries();
  test_message_header_layout_validation();
  test_fp32_vector_log();

  // test_int16_fp16_conversion();
  // test_fp16_exp2();

  // benchmark_hmx_gemm();
  // benchmark_hvx_gemm();
  // benchmark_vtcm_bandwidth();
}
}
