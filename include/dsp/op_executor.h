#pragma once

#include "message.h"
#include "op_reg.h"

struct BlockSparseAttnBufferSizes {
  size_t q;
  size_t o;
  size_t kv;
  size_t block_mask;
};

#ifdef __cplusplus
extern "C" {
#endif

int block_sparse_attn_buffer_sizes(const struct BlockSparseAttnParams *params,
                                   struct BlockSparseAttnBufferSizes *sizes);
int qkv_mul_rms_packed_dual_buffer_sizes(const struct QkvMulRmsPackedDualParams *params,
                                         struct QkvMulRmsPackedBufferSizes *sizes);
int execute_op_simple(struct OpComputeRequest *req, size_t req_size);

#ifdef __cplusplus
}
#endif
