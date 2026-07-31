#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "host/session.h"
#include "htp_ops.h"  // auto-generated
#include "message.h"
#include "op_reg.h"

static inline int64_t get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000L + ts.tv_nsec / 1000;
}

static inline int align_up(size_t size, size_t align) {
  return (size + align - 1) / align * align;
}

static inline double rand_01() {
  return ((double) rand()) / RAND_MAX;
}

// assert p_buf, p_fd and size are always valid
int alloc_shared_mem_buf(void **p_buf, int *p_fd, size_t size) {
  void *buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
  if (!buf) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_alloc failed\n");
    return -1;
  }

  int fd = rpcmem_to_fd(buf);
  if (fd < 0) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_to_fd failed\n");
    return -1;
  }

  // map buffer to the DSP
  int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf, 0, size, FASTRPC_MAP_FD);
  if (err) {
    fprintf(stderr, "alloc_shared_mem_buf: fastrpc_mmap failed, err: %d\n", err);
    return -1;
  }

  *p_buf = buf;
  *p_fd  = fd;
  return 0;
}

void free_shared_mem_buf(void *buf, int fd, size_t size) {
  fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf, size);
  rpcmem_free(buf);
}

static void rms_norm_f32_ref(float *dst, const float *src, int ne0, int ne1) {
  const float eps = 1e-5;

  for (int j = 0; j < ne1; ++j) {
    const float *x = src + j * ne0;
    float       *y = dst + j * ne0;

    float sum = 0;
    for (int i = 0; i < ne0; ++i) {
      sum += x[i] * x[i];
    }

    float mean  = sum / ne0;
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < ne0; ++i) {
      y[i] = x[i] * scale;
    }

    printf("%s: sum: %.5f mean: %.5f scale: %.5f\n", __func__, sum, mean, scale);
  }
}

static void test_rms_norm_f32_rpc(remote_handle64 handle, int ne0) {
  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  // fill data, [0, 20000] -> [-20, 20]
  for (int i = 0; i < ne0; ++i) {
    src[i] = (rand() % 20000) * 2e-3f - 20.0f;
  }

  int64_t t0             = get_time_us();
  err                    = htp_ops_rms_norm_f32(handle, fd_dst, 0, fd_src, 0, ne0, 1);
  int64_t rpc_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 RPC took %ld us\n", rpc_elapsed_us);

  if (err != 0) {
    fprintf(stderr, "%s: RPC failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
  return;
}

static void test_rms_norm_f32_chan(void *chan, int ne0) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;

  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  // fill data, [0, 20000] -> [-20, 20]
  for (int i = 0; i < ne0; ++i) {
    src[i] = (rand() % 20000) * 2e-3f - 20.0f;
  }

  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_OP_COMPUTE,
    };
    struct OpComputeRequest compute_req = {
      .op = HTP_OPS_RMS_NORM_F32,
    };
    struct RmsNormF32Params params = {
      .dst = { .fd = fd_dst, .offset = 0, },
      .src = { .fd = fd_src, .offset = 0, },
      .ne0 = ne0,
      .ne1 = 1,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct OpComputeRequest *) p = compute_req;
    p += sizeof(struct OpComputeRequest);
    *(struct RmsNormF32Params *) p = params;
    p += sizeof(struct RmsNormF32Params);
  }

  int64_t t0      = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    // usleep(10);
  }
  int64_t chan_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 CHAN took %ld us\n", chan_elapsed_us);

  err = message_header_get_request_ptr(msg, 0)->state;
  if (err != 0) {
    fprintf(stderr, "%s: CHAN failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

  // extra test: trigger DSP-side mapping reclaimation
  // fprintf(stderr, "manually unmap fd %d, %d\n", fd_dst, fd_src);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_dst, NULL, 0);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_src, NULL, 0);
  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_RPCMEM_MAP,
    };
    struct RpcmemMapRequest map_req = {
      .n_puts = 2,
      .n_gets = 0,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(map_req) + 2 * sizeof(int);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct RpcmemMapRequest *) p = map_req;
    p += sizeof(struct RpcmemMapRequest);

    // fill in fd data
    *(int *) p = fd_dst;
    p += sizeof(int);
    *(int *) p = fd_src;
    p += sizeof(int);
  }

  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
}

static int test_block_sparse_legacy_channel(void *chan) {
  enum {
    HEAD_DIM = 64,
    TOKENS = 32,
    LEGACY_PAYLOAD_SIZE = offsetof(struct BlockSparseAttnParams, sparse_policy),
    CANARY_FD_COUNT = 7,
  };
  _Static_assert(LEGACY_PAYLOAD_SIZE == 92, "legacy block-sparse payload size changed");
  _Static_assert(offsetof(struct BlockSparseAttnParams, elapsed_us) + sizeof(int32_t) == LEGACY_PAYLOAD_SIZE,
                 "legacy elapsed_us boundary changed");

  const size_t q_size = TOKENS * HEAD_DIM * sizeof(float);
  const size_t kv_size = TOKENS * HEAD_DIM * sizeof(__fp16);
  const size_t o_size = q_size;
  const size_t q_offset = 0;
  const size_t k_offset = align_up(q_offset + q_size, 128);
  const size_t v_offset = align_up(k_offset + kv_size, 128);
  const size_t o_offset = align_up(v_offset + kv_size, 128);
  const size_t mask_offset = align_up(o_offset + o_size, 128);
  const size_t tensor_bytes = align_up(mask_offset + sizeof(int32_t), 128);

  uint8_t *tensor_data = NULL;
  int tensor_fd = -1;
  if (alloc_shared_mem_buf((void **) &tensor_data, &tensor_fd, tensor_bytes) != 0) {
    return -1;
  }

  float *q = (float *) (tensor_data + q_offset);
  __fp16 *k = (__fp16 *) (tensor_data + k_offset);
  __fp16 *v = (__fp16 *) (tensor_data + v_offset);
  float *o = (float *) (tensor_data + o_offset);
  int32_t *mask = (int32_t *) (tensor_data + mask_offset);
  for (int i = 0; i < TOKENS * HEAD_DIM; ++i) {
    q[i] = (float) ((i % 17) - 8) * 0.03125f;
    k[i] = (__fp16) ((float) ((i % 13) - 6) * 0.02734375f);
    v[i] = (__fp16) ((float) ((i % 11) - 5) * 0.0234375f);
    o[i] = 0.0f;
  }
  mask[0] = 0;

  struct BlockSparseAttnParams params = {
    .o = { .fd = tensor_fd, .offset = (int32_t) o_offset },
    .q = { .fd = tensor_fd, .offset = (int32_t) q_offset },
    .k = { .fd = tensor_fd, .offset = (int32_t) k_offset },
    .v = { .fd = tensor_fd, .offset = (int32_t) v_offset },
    .block_mask = { .fd = tensor_fd, .offset = (int32_t) mask_offset },
    .qo_len = TOKENS,
    .kv_len = TOKENS,
    .n_heads = 1,
    .n_kv_heads = 1,
    .head_dim = HEAD_DIM,
    .block_q = TOKENS,
    .block_k = TOKENS,
    .n_q_blocks = 1,
    .n_k_blocks = 1,
    .mask_format = 1,
    .max_kept_blocks = 1,
    .with_lse = 0,
    .elapsed_us = INT32_MIN,
  };
  const struct RequestHeader compute_header = {
    .state = 0,
    .type = REQUEST_TYPE_OP_COMPUTE,
  };
  const struct OpComputeRequest compute_request = {
    .op = HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16,
  };
  const struct RequestHeader map_header = {
    .state = 0,
    .type = REQUEST_TYPE_RPCMEM_MAP,
  };
  const struct RpcmemMapRequest map_request = {
    .n_puts = 1,
    .n_gets = 0,
  };
  const int32_t canary_fds[CANARY_FD_COUNT] = {
    tensor_fd, 0x20212223, 0x30313233, 0x40414243,
    0x50515253, 0x60616263, 0x70717273,
  };

  const size_t compute_wire_size =
    sizeof(compute_header) + sizeof(compute_request) + LEGACY_PAYLOAD_SIZE;
  const size_t map_wire_size = sizeof(map_header) + sizeof(map_request) + sizeof(canary_fds);
  struct MessageHeader *msg = (struct MessageHeader *) chan;
  msg->state.d = 0;
  msg->n_reqs = 2;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + compute_wire_size;
  msg->req_offsets[2] = msg->req_offsets[1] + map_wire_size;

  uint8_t *p = (uint8_t *) message_header_get_request_ptr(msg, 0);
  memcpy(p, &compute_header, sizeof(compute_header));
  p += sizeof(compute_header);
  memcpy(p, &compute_request, sizeof(compute_request));
  p += sizeof(compute_request);
  memcpy(p, &params, LEGACY_PAYLOAD_SIZE);

  p = (uint8_t *) message_header_get_request_ptr(msg, 1);
  memcpy(p, &map_header, sizeof(map_header));
  p += sizeof(map_header);
  memcpy(p, &map_request, sizeof(map_request));
  p += sizeof(map_request);
  memcpy(p, canary_fds, sizeof(canary_fds));

  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }

  const struct RequestHeader *computed = message_header_get_request_ptr(msg, 0);
  const struct BlockSparseAttnParams *returned_params =
    (const struct BlockSparseAttnParams *)
      (computed->data + sizeof(struct OpComputeRequest));
  const struct RequestHeader *mapped = message_header_get_request_ptr(msg, 1);
  const struct RpcmemMapRequest *returned_map = (const struct RpcmemMapRequest *) mapped->data;
  const int32_t *returned_fds = returned_map->fds;
  const int passed = computed->state == 0 && returned_params->elapsed_us != INT32_MIN &&
    mapped->state == 0 && mapped->type == REQUEST_TYPE_RPCMEM_MAP &&
    returned_map->n_puts == 1 && returned_map->n_gets == 0 &&
    memcmp(returned_fds, canary_fds, sizeof(canary_fds)) == 0 && isfinite(o[0]);

  free_shared_mem_buf(tensor_data, tensor_fd, tensor_bytes);
  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
  return passed ? 0 : -1;
}

static void test_mat_mul_rpc(remote_handle64 handle) {
  float *activation, *output;
  __fp16 *weight;

  int output_fd, activation_fd, weight_fd;

  int m = 1;
  int k = 1024;
  // int n = 608; // 576 | 608
  int n = 1024;

  alloc_shared_mem_buf((void **) &output, &output_fd, m * n * sizeof(float));
  alloc_shared_mem_buf((void **) &activation, &activation_fd, m * k * sizeof(float));
  alloc_shared_mem_buf((void **) &weight, &weight_fd, k * n * sizeof(__fp16));

  float *weight_ref = (float *) malloc(n * k * sizeof(float));
  float *output_ref = (float *) malloc(m * n * sizeof(float));
  memset(output_ref, 0, m * n * sizeof(float));

  __fp16 *output_f16 = (__fp16 *) malloc(m * n * sizeof(__fp16));
  memset(output_f16, 0, m * n * sizeof(__fp16));

  float *output_mix = (float *) malloc(m * n * sizeof(float));
  memset(output_mix, 0, m * n * sizeof(float));

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      activation[i * k + j] = rand_01();
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      float x = rand_01();

      int i0 = i / 32, i1 = i % 32;
      int j0 = j / 32, j1 = j % 32;

      int tile_idx = j0 * (k / 32) + i0;
      __fp16 *tile = weight + tile_idx * 1024;
      tile[(i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = (__fp16) x;
      weight_ref[i * n + j] = x;
    }
  }

  htp_ops_mat_mul_permuted_w16a32(handle, output_fd, 0, activation_fd, 0, weight_fd, 0, m, k, n);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        output_ref[i * n + j] += activation[i * k + l] * weight_ref[l * n + j];
        output_f16[i * n + j] += (__fp16)(((__fp16) activation[i * k + l]) * ((__fp16) weight_ref[l * n + j]));
        output_mix[i * n + j] += (float)((__fp16) activation[i * k + l] * ((__fp16) weight_ref[l * n + j]));
      }
    }
  }

  for (int i = 0; i < m * n; ++i)
    printf("#%d hmx: %g, f32: %g, f16: %g, mix: %g\n", i, output[i], output_ref[i], output_f16[i], output_mix[i]);

  free(weight_ref);
  free(output_ref);
  free(output_f16);
  free(output_mix);

  free_shared_mem_buf(output, output_fd, m * n * sizeof(float));
  free_shared_mem_buf(activation, activation_fd, m * k * sizeof(float));
  free_shared_mem_buf(weight, weight_fd, k * n * sizeof(__fp16));
}

int main(int argc, char **argv) {
  int err = open_dsp_session(CDSP_DOMAIN_ID, 1);
  if (err != 0) {
    fprintf(stderr, "Open DSP session failed\n");
    return 1;
  }

  init_htp_backend();

  // test_mat_mul_rpc(get_global_handle());

  int userdma_scope = 0;
  err = htp_ops_test_ops(get_global_handle(), &userdma_scope);
  if (err != 0) {
    fprintf(stderr, "DSP op tests failed: 0x%x\n", err);
    goto skip1;
  }
  if (userdma_scope < 1 || userdma_scope > 3) {
    fprintf(stderr, "DSP returned invalid UserDMA scope: %d\n", userdma_scope);
    err = 1;
    goto skip1;
  }
  printf("HTP_USERDMA_SCOPE=%s\n",
         userdma_scope == 1 ? "PER_WORKER" : (userdma_scope == 2 ? "SHARED" : "UNSAFE"));
  if (userdma_scope == 3) {
    fprintf(stderr, "UserDMA probe classified this device as unsafe\n");
    err = 1;
    goto skip1;
  }

  void *chan = NULL;
  int chan_fd = -1;
  const size_t max_msg_size = 4096;
  err = alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size);
  if (err != 0) {
    fprintf(stderr, "Cannot allocate rpcmem for message channel\n");
    goto skip1;
  }
  err = htp_ops_create_channel(get_global_handle(), chan_fd, max_msg_size);
  if (err != 0) {
    fprintf(stderr, "Create channel failed\n");
    goto free_channel;
  }
  err = test_block_sparse_legacy_channel(chan);
  htp_ops_destroy_channel(get_global_handle());

free_channel:
  free_shared_mem_buf(chan, chan_fd, max_msg_size);
  if (err != 0) {
    goto skip1;
  }

  /*
  test_rms_norm_f32_rpc(get_global_handle(), 60000);

  void        *chan;
  int          chan_fd;
  const size_t max_msg_size = 4096;

  err = alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Cannot allocate rpcmem for message channel\n");
    goto skip1;
  }

  err = htp_ops_create_channel(get_global_handle(), chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Create channel failed\n");
    goto skip2;
  }

  test_rms_norm_f32_chan(chan, 60000);

  htp_ops_destroy_channel(get_global_handle());

skip2:
  free_shared_mem_buf(chan, chan_fd, max_msg_size);
  */

skip1:
  close_dsp_session();
  return err == 0 ? 0 : 1;
}
