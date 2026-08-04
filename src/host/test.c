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

static int env_positive_int(const char *name, int fallback) {
  const char *value = getenv(name);
  if (value == NULL || *value == '\0') {
    return fallback;
  }
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
    return fallback;
  }
  return (int) parsed;
}

static int env_nonnegative_int(const char *name, int fallback) {
  const char *value = getenv(name);
  if (value == NULL || *value == '\0') {
    return fallback;
  }
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
    return fallback;
  }
  return (int) parsed;
}

static int env_enabled(const char *name) {
  const char *value = getenv(name);
  return value != NULL && strcmp(value, "0") != 0;
}

static float qkv_bench_weight_value(int row, int col) {
  return (float) (((row * 13 + col * 7) % 101) - 50) * 0.001f;
}

static float qkv_bench_activation_value(int row, int col) {
  return (float) (((row * 11 + col * 5) % 37) - 18) * 0.01f;
}

static int test_qkv_mul_channel(void *chan) {
  enum { N_HEADS = 24 };
  const int HEAD_DIM = env_positive_int("HTP_QKV_HEAD_DIM", 64);
  const int tokens = env_positive_int("HTP_QKV_TOKENS", 32);
  const int input_dim = env_positive_int("HTP_QKV_INPUT_DIM", 1536);
  if (input_dim % 32 != 0) {
    fprintf(stderr, "qkv benchmark requires HTP_QKV_INPUT_DIM %% 32 == 0\n");
    return -1;
  }

  const int output_cols = 3 * N_HEADS * HEAD_DIM;
  const size_t token_tiles = (size_t) (tokens + 31) / 32;
  const size_t head_tile_elems = token_tiles * (HEAD_DIM / 32) * 1024;
  const size_t qkv_size = (size_t) N_HEADS * head_tile_elems * sizeof(__fp16);
  const size_t activation_size = (size_t) tokens * input_dim * sizeof(float);
  const size_t weight_size = (size_t) input_dim * output_cols * sizeof(__fp16);
  const size_t bias_size = (size_t) output_cols * sizeof(float);
  const size_t norm_size = HEAD_DIM * sizeof(float);

  const size_t activation_offset = 0;
  const size_t weight_offset = align_up(activation_offset + activation_size, 128);
  const size_t bias_offset = align_up(weight_offset + weight_size, 128);
  const size_t q_norm_offset = align_up(bias_offset + bias_size, 128);
  const size_t k_norm_offset = align_up(q_norm_offset + norm_size, 128);
  const size_t q_offset = align_up(k_norm_offset + norm_size, 128);
  const size_t k_offset = align_up(q_offset + qkv_size, 128);
  const size_t v_offset = align_up(k_offset + qkv_size, 128);
  const size_t tensor_size = align_up(v_offset + qkv_size, 128);
  uint8_t *tensor_data = NULL;
  int tensor_fd = -1;
  float *activation = NULL;
  __fp16 *weight = NULL;
  float *bias = NULL;
  float *q_norm = NULL;
  float *k_norm = NULL;
  __fp16 *q = NULL;
  __fp16 *k = NULL;
  __fp16 *v = NULL;
  float *weight_ref = NULL;
  float *q_raw = NULL;
  float *k_raw = NULL;
  int result = -1;

  if (alloc_shared_mem_buf((void **) &tensor_data, &tensor_fd, tensor_size) != 0) {
    fprintf(stderr, "qkv benchmark shared allocation failed\n");
    goto cleanup;
  }
  activation = (float *) (tensor_data + activation_offset);
  weight = (__fp16 *) (tensor_data + weight_offset);
  bias = (float *) (tensor_data + bias_offset);
  q_norm = (float *) (tensor_data + q_norm_offset);
  k_norm = (float *) (tensor_data + k_norm_offset);
  q = (__fp16 *) (tensor_data + q_offset);
  k = (__fp16 *) (tensor_data + k_offset);
  v = (__fp16 *) (tensor_data + v_offset);

  weight_ref = (float *) malloc((size_t) input_dim * output_cols * sizeof(float));
  if (weight_ref == NULL) {
    fprintf(stderr, "qkv benchmark host reference allocation failed\n");
    goto cleanup;
  }
  for (int r = 0; r < tokens; ++r) {
    for (int c = 0; c < input_dim; ++c) {
      activation[(size_t) r * input_dim + c] = qkv_bench_activation_value(r, c);
    }
  }
  for (int c = 0; c < output_cols; ++c) {
    bias[c] = (float) ((c % 17) - 8) * 0.01f;
    weight_ref[c] = 0.0f;
  }
  for (int d = 0; d < HEAD_DIM; ++d) {
    q_norm[d] = 1.0f + (float) (d % 7) * 0.01f;
    k_norm[d] = 1.0f - (float) (d % 5) * 0.01f;
  }
  memset(q, 0, qkv_size);
  memset(k, 0, qkv_size);
  memset(v, 0, qkv_size);

  for (int r = 0; r < input_dim; ++r) {
    for (int c = 0; c < output_cols; ++c) {
      const float value = qkv_bench_weight_value(r, c);
      weight_ref[(size_t) r * output_cols + c] = value;
      const int r0 = r / 32;
      const int r1 = r % 32;
      const int c0 = c / 32;
      const int c1 = c % 32;
      const int tile_idx = c0 * (input_dim / 32) + r0;
      __fp16 *tile = weight + (size_t) tile_idx * 1024;
      tile[(r1 & ~1) * 32 + c1 * 2 + (r1 & 1)] = (__fp16) value;
    }
  }

  struct MessageHeader *msg = (struct MessageHeader *) chan;
  const struct RequestHeader req_hdr = {
    .state = 0,
    .type = REQUEST_TYPE_OP_COMPUTE,
  };
  const struct OpComputeRequest compute_req = {
    .op = HTP_OPS_QKV_MUL_RMS_PACKED_W16A32,
  };
  struct QkvMulRmsPackedParams params = {
    .q = { .fd = tensor_fd, .offset = (int32_t) q_offset },
    .k_out = { .fd = tensor_fd, .offset = (int32_t) k_offset },
    .v = { .fd = tensor_fd, .offset = (int32_t) v_offset },
    .activation = { .fd = tensor_fd, .offset = (int32_t) activation_offset },
    .weight = { .fd = tensor_fd, .offset = (int32_t) weight_offset },
    .bias = { .fd = tensor_fd, .offset = (int32_t) bias_offset },
    .q_norm = { .fd = tensor_fd, .offset = (int32_t) q_norm_offset },
    .k_norm = { .fd = tensor_fd, .offset = (int32_t) k_norm_offset },
    .tokens = tokens,
    .input_dim = input_dim,
    .n_heads = N_HEADS,
    .head_dim = HEAD_DIM,
    .elapsed_us = -1,
  };
  const size_t req_size = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
  msg->state.d = 0;
  msg->n_reqs = 1;
  msg->req_offsets[0] = (int32_t) message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + (int32_t) req_size;
  uint8_t *p = (uint8_t *) message_header_get_request_ptr(msg, 0);
  memcpy(p, &req_hdr, sizeof(req_hdr));
  p += sizeof(req_hdr);
  memcpy(p, &compute_req, sizeof(compute_req));
  p += sizeof(compute_req);
  memcpy(p, &params, sizeof(params));

  const int64_t t0 = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }
  const int64_t channel_elapsed_us = get_time_us() - t0;
  const struct RequestHeader *computed = message_header_get_request_ptr(msg, 0);
  const struct QkvMulRmsPackedParams *returned =
    (const struct QkvMulRmsPackedParams *) (computed->data + sizeof(struct OpComputeRequest));

  float expected_q0 = 0.0f;
  float expected_k0 = 0.0f;
  float expected_v0 = 0.0f;
  float expected_k1 = 0.0f;
  float expected_v1 = 0.0f;
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  q_raw = (float *) malloc((size_t) HEAD_DIM * sizeof(float));
  k_raw = (float *) malloc((size_t) HEAD_DIM * sizeof(float));
  if (q_raw == NULL || k_raw == NULL) {
    goto cleanup;
  }
  for (int d = 0; d < HEAD_DIM; ++d) {
    float q_value = bias[d];
    float k_value = bias[N_HEADS * HEAD_DIM + d];
    for (int c = 0; c < input_dim; ++c) {
      const float a = activation[c];
      q_value += a * weight_ref[(size_t) c * output_cols + d];
      k_value += a * weight_ref[(size_t) c * output_cols + N_HEADS * HEAD_DIM + d];
    }
    q_raw[d] = q_value;
    k_raw[d] = k_value;
    q_sum += q_value * q_value;
    k_sum += k_value * k_value;
  }
  const float q_inv = 1.0f / sqrtf(q_sum / HEAD_DIM + 1.0e-6f);
  const float k_inv = 1.0f / sqrtf(k_sum / HEAD_DIM + 1.0e-6f);
  expected_q0 = q_raw[0] * q_inv * q_norm[0];
  expected_k0 = k_raw[0] * k_inv * k_norm[0];
  expected_k1 = k_raw[1] * k_inv * k_norm[1];
  expected_v0 = bias[2 * N_HEADS * HEAD_DIM];
  expected_v1 = bias[2 * N_HEADS * HEAD_DIM + 1];
  for (int c = 0; c < input_dim; ++c) {
    expected_v0 += activation[c] * weight_ref[(size_t) c * output_cols + 2 * N_HEADS * HEAD_DIM];
    expected_v1 += activation[c] * weight_ref[(size_t) c * output_cols + 2 * N_HEADS * HEAD_DIM + 1];
  }

  const int finite_outputs = isfinite((float) q[0]) && isfinite((float) k[0]) && isfinite((float) v[0]);
  const int nonzero_outputs = ((float) q[0] != 0.0f) || ((float) k[0] != 0.0f) || ((float) v[0] != 0.0f);
  const int reference_match = fabsf((float) q[0] - expected_q0) < 0.01f &&
                              fabsf((float) k[0] - expected_k0) < 0.01f &&
                              fabsf((float) v[0] - expected_v0) < 0.01f &&
                              fabsf((float) k[1] - expected_k1) < 0.01f &&
                              fabsf((float) v[2] - expected_v1) < 0.01f;
  fprintf(stderr,
          "qkv_mul_rms_packed: tokens=%d input_dim=%d channel_us=%ld dsp_us=%d "
          "q0=%g/%g k0=%g/%g v0=%g/%g k1=%g/%g v1=%g/%g\n",
          tokens, input_dim, (long) channel_elapsed_us, returned->elapsed_us, (float) q[0], expected_q0,
          (float) k[0], expected_k0, (float) v[0], expected_v0,
          (float) k[1], expected_k1, (float) v[2], expected_v1);
  result = computed->state == 0 && returned->elapsed_us >= 0 && finite_outputs &&
           nonzero_outputs && reference_match ? 0 : -1;

cleanup:
  if (q_raw) free(q_raw);
  if (k_raw) free(k_raw);
  if (weight_ref) free(weight_ref);
  if (tensor_data) free_shared_mem_buf(tensor_data, tensor_fd, tensor_size);
  fprintf(stderr, result == 0 ? "%s passed\n" : "%s failed\n", __func__);
  return result;
}

static void pack_attention_qkv(const float *q_raw, const __fp16 *k_raw, const __fp16 *v_raw,
                               __fp16 *q_packed, __fp16 *k_packed, __fp16 *v_packed,
                               int tokens, int n_heads, int head_dim) {
  const int token_tiles = (tokens + 31) / 32;
  const size_t head_elems = (size_t) token_tiles * (head_dim / 32) * 1024;
  memset(q_packed, 0, (size_t) n_heads * head_elems * sizeof(__fp16));
  memset(k_packed, 0, (size_t) n_heads * head_elems * sizeof(__fp16));
  memset(v_packed, 0, (size_t) n_heads * head_elems * sizeof(__fp16));
  for (int h = 0; h < n_heads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      const int tt = t / 32;
      const int tr = t % 32;
      for (int d = 0; d < head_dim; ++d) {
        const int dt = d / 32;
        const int dc = d % 32;
        const size_t raw = ((size_t) t * n_heads + h) * head_dim + d;
        const size_t q_tile = (size_t) h * head_elems + ((size_t) tt * (head_dim / 32) + dt) * 1024;
        const size_t v_tile = (size_t) h * head_elems + ((size_t) dt * token_tiles + tt) * 1024;
        const size_t qv_in_tile = (size_t) (tr & ~1) * 32 + dc * 2 + (tr & 1);
        const size_t k_in_tile = (size_t) (dc & ~1) * 32 + tr * 2 + (dc & 1);
        q_packed[q_tile + qv_in_tile] = (__fp16) q_raw[raw];
        k_packed[q_tile + k_in_tile] = k_raw[raw];
        v_packed[v_tile + qv_in_tile] = v_raw[raw];
      }
    }
  }
}

static float packed_q_value(const __fp16 *packed, int tokens, int head_dim,
                            int token, int head, int dim) {
  const int token_tiles = (tokens + 31) / 32;
  const size_t head_elems = (size_t) token_tiles * (head_dim / 32) * 1024;
  const size_t tile = (size_t) head * head_elems +
    ((size_t) (token / 32) * (head_dim / 32) + dim / 32) * 1024;
  const size_t in_tile = (size_t) ((token % 32) & ~1) * 32 +
    (dim % 32) * 2 + (token & 1);
  return (float) packed[tile + in_tile];
}

static float packed_k_value(const __fp16 *packed, int tokens, int head_dim,
                            int token, int head, int dim) {
  const int token_tiles = (tokens + 31) / 32;
  const size_t head_elems = (size_t) token_tiles * (head_dim / 32) * 1024;
  const size_t tile = (size_t) head * head_elems +
    ((size_t) (token / 32) * (head_dim / 32) + dim / 32) * 1024;
  const size_t in_tile = (size_t) ((dim % 32) & ~1) * 32 +
    (token % 32) * 2 + (dim & 1);
  return (float) packed[tile + in_tile];
}

static float packed_v_value(const __fp16 *packed, int tokens, int head_dim,
                            int token, int head, int dim) {
  const int token_tiles = (tokens + 31) / 32;
  const size_t head_elems = (size_t) token_tiles * (head_dim / 32) * 1024;
  const size_t tile = (size_t) head * head_elems +
    ((size_t) (dim / 32) * token_tiles + token / 32) * 1024;
  const size_t in_tile = (size_t) ((token % 32) & ~1) * 32 +
    (dim % 32) * 2 + (token & 1);
  return (float) packed[tile + in_tile];
}

static int run_qkv_rope_request(void *chan, enum HtpOpsIndex op,
                                const void *params, size_t params_size,
                                void *returned, size_t returned_size);

/* Keep this producer-only check independent from sparse attention.  It catches
 * packed RoPE/indexing errors before the attention consumer can hide them. */
static int test_qkv_rope_channel(void *chan) {
  enum { N_HEADS = 24, TOKENS = 64, INPUT_DIM = 1536 };
  const int head_dim = env_positive_int("HTP_QKV_ROPE_HEAD_DIM", 128);
  const int dual = env_enabled("HTP_QKV_ROPE_DUAL");
  if ((head_dim != 64 && head_dim != 128) || INPUT_DIM % 32 != 0) {
    return -1;
  }
  const int output_cols = 3 * N_HEADS * head_dim;
  const size_t token_tiles = (TOKENS + 31) / 32;
  const size_t head_tile_elems = token_tiles * (head_dim / 32) * 1024;
  const size_t qkv_bytes = (size_t) N_HEADS * head_tile_elems * sizeof(__fp16);
  const size_t activation_bytes = (size_t) TOKENS * INPUT_DIM * sizeof(float);
  const size_t weight_bytes = (size_t) INPUT_DIM * output_cols * sizeof(__fp16);
  const size_t bias_bytes = (size_t) output_cols * sizeof(float);
  const size_t norm_bytes = (size_t) head_dim * sizeof(float);
  const size_t rope_bytes = (size_t) TOKENS * (head_dim / 2) * 4 * sizeof(float);
  size_t activation_offset = 0;
  size_t weight_offset = align_up(activation_offset + activation_bytes, 128);
  size_t bias_offset = align_up(weight_offset + weight_bytes, 128);
  size_t q_norm_offset = align_up(bias_offset + bias_bytes, 128);
  size_t k_norm_offset = align_up(q_norm_offset + norm_bytes, 128);
  size_t rope_offset = align_up(k_norm_offset + norm_bytes, 128);
  size_t q_offset = align_up(rope_offset + rope_bytes, 128);
  size_t k_offset = align_up(q_offset + qkv_bytes, 128);
  size_t v_offset = align_up(k_offset + qkv_bytes, 128);
  size_t total_bytes = align_up(v_offset + qkv_bytes, 128);
  uint8_t *data = NULL;
  int fd = -1;
  float *weight_ref = NULL;
  int result = -1;
  if (alloc_shared_mem_buf((void **) &data, &fd, total_bytes) != 0) {
    return -1;
  }
  float *activation = (float *) (data + activation_offset);
  __fp16 *weight = (__fp16 *) (data + weight_offset);
  float *bias = (float *) (data + bias_offset);
  float *q_norm = (float *) (data + q_norm_offset);
  float *k_norm = (float *) (data + k_norm_offset);
  float *rope_pe = (float *) (data + rope_offset);
  __fp16 *q = (__fp16 *) (data + q_offset);
  __fp16 *k = (__fp16 *) (data + k_offset);
  __fp16 *v = (__fp16 *) (data + v_offset);
  weight_ref = (float *) malloc((size_t) INPUT_DIM * output_cols * sizeof(float));
  if (weight_ref == NULL) {
    goto cleanup;
  }
  for (int t = 0; t < TOKENS; ++t) {
    for (int c = 0; c < INPUT_DIM; ++c) {
      activation[(size_t) t * INPUT_DIM + c] = qkv_bench_activation_value(t, c);
    }
  }
  for (int c = 0; c < output_cols; ++c) {
    bias[c] = (float) ((c % 19) - 9) * 0.007f;
  }
  for (int d = 0; d < head_dim; ++d) {
    q_norm[d] = 0.9f + (float) (d % 11) * 0.013f;
    k_norm[d] = 1.1f - (float) (d % 7) * 0.009f;
  }
  for (int t = 0; t < TOKENS; ++t) {
    for (int d = 0; d < head_dim / 2; ++d) {
      const float angle = 0.013f * (float) (t + 1) * (float) (d + 1);
      float *m = rope_pe + ((size_t) t * (head_dim / 2) + d) * 4;
      m[0] = cosf(angle);
      m[1] = -sinf(angle);
      m[2] = sinf(angle);
      m[3] = cosf(angle);
    }
  }
  for (int r = 0; r < INPUT_DIM; ++r) {
    for (int c = 0; c < output_cols; ++c) {
      const float value = qkv_bench_weight_value(r, c);
      weight_ref[(size_t) r * output_cols + c] = value;
      const int r0 = r / 32;
      const int r1 = r % 32;
      const int c0 = c / 32;
      const int c1 = c % 32;
      __fp16 *tile = weight + (size_t) (c0 * (INPUT_DIM / 32) + r0) * 1024;
      tile[(r1 & ~1) * 32 + c1 * 2 + (r1 & 1)] = (__fp16) value;
    }
  }
  memset(q, 0, qkv_bytes);
  memset(k, 0, qkv_bytes);
  memset(v, 0, qkv_bytes);
  struct QkvMulRmsPackedRopeParams params = {
    .q = { fd, (int32_t) q_offset }, .k_out = { fd, (int32_t) k_offset },
    .v = { fd, (int32_t) v_offset }, .activation = { fd, (int32_t) activation_offset },
    .weight = { fd, (int32_t) weight_offset }, .bias = { fd, (int32_t) bias_offset },
    .q_norm = { fd, (int32_t) q_norm_offset }, .k_norm = { fd, (int32_t) k_norm_offset },
    .rope_pe = { fd, (int32_t) rope_offset }, .tokens = TOKENS,
    .input_dim = INPUT_DIM, .n_heads = N_HEADS, .head_dim = head_dim,
    .rope_token_offset = 0, .rope_token_count = TOKENS, .elapsed_us = -1,
  };
  struct QkvMulRmsPackedRopeParams returned = {};
  struct QkvMulRmsPackedDualRopeParams dual_params = {
    .q = { fd, (int32_t) q_offset }, .k_out = { fd, (int32_t) k_offset },
    .v = { fd, (int32_t) v_offset },
    .image_activation = { fd, (int32_t) activation_offset },
    .image_weight = { fd, (int32_t) weight_offset },
    .image_bias = { fd, (int32_t) bias_offset },
    .image_q_norm = { fd, (int32_t) q_norm_offset },
    .image_k_norm = { fd, (int32_t) k_norm_offset },
    .text_activation = { fd, (int32_t) (activation_offset + 32u * INPUT_DIM * sizeof(float)) },
    .text_weight = { fd, (int32_t) weight_offset },
    .text_bias = { fd, (int32_t) bias_offset },
    .text_q_norm = { fd, (int32_t) q_norm_offset },
    .text_k_norm = { fd, (int32_t) k_norm_offset },
    .rope_pe = { fd, (int32_t) rope_offset }, .image_tokens = 32, .text_tokens = 32,
    .input_dim = INPUT_DIM, .n_heads = N_HEADS, .head_dim = head_dim,
    .image_rope_token_offset = 0, .text_rope_token_offset = 32, .elapsed_us = -1,
  };
  struct QkvMulRmsPackedDualRopeParams dual_returned = {};
  const int state = dual ?
    run_qkv_rope_request(chan, HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32,
                         &dual_params, sizeof(dual_params), &dual_returned, sizeof(dual_returned)) :
    run_qkv_rope_request(chan, HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32,
                         &params, sizeof(params), &returned, sizeof(returned));
  const int dsp_us = dual ? dual_returned.elapsed_us : returned.elapsed_us;
  float max_q = 0.0f, max_k = 0.0f, max_v = 0.0f;
  for (int t = 0; t < TOKENS; ++t) {
    for (int h = 0; h < N_HEADS; ++h) {
      float q_raw[128];
      float k_raw[128];
      float q_sum = 0.0f;
      float k_sum = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        float qv = bias[h * head_dim + d];
        float kv = bias[N_HEADS * head_dim + h * head_dim + d];
        float vv = bias[2 * N_HEADS * head_dim + h * head_dim + d];
        const int source_t = t;
        for (int c = 0; c < INPUT_DIM; ++c) {
          qv += activation[(size_t) source_t * INPUT_DIM + c] *
                weight_ref[(size_t) c * output_cols + h * head_dim + d];
          kv += activation[(size_t) source_t * INPUT_DIM + c] *
                weight_ref[(size_t) c * output_cols + N_HEADS * head_dim + h * head_dim + d];
          vv += activation[(size_t) source_t * INPUT_DIM + c] *
                weight_ref[(size_t) c * output_cols + 2 * N_HEADS * head_dim + h * head_dim + d];
        }
        q_raw[d] = qv;
        k_raw[d] = kv;
        q_sum += qv * qv;
        k_sum += kv * kv;
        (void) vv;
      }
      const float q_inv = 1.0f / sqrtf(q_sum / head_dim + 1.0e-6f);
      const float k_inv = 1.0f / sqrtf(k_sum / head_dim + 1.0e-6f);
      for (int d = 0; d < head_dim; ++d) {
        q_raw[d] *= q_inv * q_norm[d];
        k_raw[d] *= k_inv * k_norm[d];
      }
      for (int d = 0; d < head_dim / 2; ++d) {
        const float x0 = q_raw[2 * d], x1 = q_raw[2 * d + 1];
        const float *m = rope_pe + ((size_t) t * (head_dim / 2) + d) * 4;
        q_raw[2 * d] = x0 * m[0] + x1 * m[1];
        q_raw[2 * d + 1] = x0 * m[2] + x1 * m[3];
        const float y0 = k_raw[2 * d], y1 = k_raw[2 * d + 1];
        k_raw[2 * d] = y0 * m[0] + y1 * m[1];
        k_raw[2 * d + 1] = y0 * m[2] + y1 * m[3];
      }
      for (int d = 0; d < head_dim; ++d) {
        const float q_err = fabsf((float) packed_q_value(q, TOKENS, head_dim, t, h, d) - (float) (__fp16) q_raw[d]);
        const float k_err = fabsf((float) packed_k_value(k, TOKENS, head_dim, t, h, d) - (float) (__fp16) k_raw[d]);
        float v_ref = bias[2 * N_HEADS * head_dim + h * head_dim + d];
        for (int c = 0; c < INPUT_DIM; ++c) {
          const int source_t = t;
          v_ref += activation[(size_t) source_t * INPUT_DIM + c] *
                   weight_ref[(size_t) c * output_cols + 2 * N_HEADS * head_dim + h * head_dim + d];
        }
        const float v_err = fabsf((float) packed_v_value(v, TOKENS, head_dim, t, h, d) - (float) (__fp16) v_ref);
        if (q_err > max_q) max_q = q_err;
        if (k_err > max_k) max_k = k_err;
        if (v_err > max_v) max_v = v_err;
      }
    }
  }
  fprintf(stderr, "qkv_rope: mode=%s state=%d dsp_us=%d max_q=%g max_k=%g max_v=%g\n",
          dual ? "dual" : "single", state, dsp_us, max_q, max_k, max_v);
  result = state == 0 && dsp_us >= 0 && max_q <= 0.03f &&
           max_k <= 0.03f && max_v <= 0.03f ? 0 : -1;
cleanup:
  if (weight_ref != NULL) free(weight_ref);
  if (data != NULL) free_shared_mem_buf(data, fd, total_bytes);
  fprintf(stderr, result == 0 ? "%s passed\n" : "%s failed\n", __func__);
  return result;
}

static void fill_static_halo_index(int32_t *index, int blocks_per_head, int max_kept,
                                   int n_heads, int image_block_rows, int image_block_cols) {
  const int image_blocks = image_block_rows * image_block_cols;
  memset(index, 0xff, (size_t) n_heads * blocks_per_head * max_kept * sizeof(int32_t));
  for (int h = 0; h < n_heads; ++h) {
    for (int qb = 0; qb < blocks_per_head; ++qb) {
      int32_t *row = index + ((size_t) h * blocks_per_head + qb) * max_kept;
      int count = 0;
      if (qb >= image_blocks) {
        for (int kb = 0; kb < blocks_per_head; ++kb) {
          row[count++] = h * blocks_per_head + kb;
        }
      } else {
        const int qy = qb / image_block_cols;
        const int qx = qb % image_block_cols;
        const int y0 = qy > 0 ? qy - 1 : 0;
        const int y1 = qy + 1 < image_block_rows ? qy + 1 : image_block_rows - 1;
        const int x0 = qx > 0 ? qx - 1 : 0;
        const int x1 = qx + 1 < image_block_cols ? qx + 1 : image_block_cols - 1;
        for (int ky = y0; ky <= y1; ++ky) {
          for (int kx = x0; kx <= x1; ++kx) {
            row[count++] = h * blocks_per_head + ky * image_block_cols + kx;
          }
        }
        for (int kb = image_blocks; kb < blocks_per_head; ++kb) {
          row[count++] = h * blocks_per_head + kb;
        }
      }
    }
  }
}

static int run_sparse_attention_request(void *chan, enum HtpOpsIndex op,
                                        const struct BlockSparseAttnParams *params,
                                        struct BlockSparseAttnParams *returned) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;
  const struct RequestHeader req_hdr = { .state = 0, .type = REQUEST_TYPE_OP_COMPUTE };
  const struct OpComputeRequest compute_req = { .op = op };
  const size_t req_size = sizeof(req_hdr) + sizeof(compute_req) + sizeof(*params);
  msg->state.d = 0;
  msg->n_reqs = 1;
  msg->req_offsets[0] = (int32_t) message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + (int32_t) req_size;
  uint8_t *p = (uint8_t *) message_header_get_request_ptr(msg, 0);
  memcpy(p, &req_hdr, sizeof(req_hdr));
  p += sizeof(req_hdr);
  memcpy(p, &compute_req, sizeof(compute_req));
  p += sizeof(compute_req);
  memcpy(p, params, sizeof(*params));
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }
  const struct RequestHeader *computed = message_header_get_request_ptr(msg, 0);
  memcpy(returned, computed->data + sizeof(struct OpComputeRequest), sizeof(*returned));
  return computed->state;
}

static int run_qkv_request(void *chan, const struct QkvMulRmsPackedParams *params,
                           struct QkvMulRmsPackedParams *returned) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;
  const struct RequestHeader req_hdr = { .state = 0, .type = REQUEST_TYPE_OP_COMPUTE };
  const struct OpComputeRequest compute_req = { .op = HTP_OPS_QKV_MUL_RMS_PACKED_W16A32 };
  const size_t req_size = sizeof(req_hdr) + sizeof(compute_req) + sizeof(*params);
  msg->state.d = 0;
  msg->n_reqs = 1;
  msg->req_offsets[0] = (int32_t) message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + (int32_t) req_size;
  uint8_t *p = (uint8_t *) message_header_get_request_ptr(msg, 0);
  memcpy(p, &req_hdr, sizeof(req_hdr));
  p += sizeof(req_hdr);
  memcpy(p, &compute_req, sizeof(compute_req));
  p += sizeof(compute_req);
  memcpy(p, params, sizeof(*params));
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }
  const struct RequestHeader *computed = message_header_get_request_ptr(msg, 0);
  memcpy(returned, computed->data + sizeof(struct OpComputeRequest), sizeof(*returned));
  return computed->state;
}

static int run_qkv_rope_request(void *chan, enum HtpOpsIndex op,
                                const void *params, size_t params_size,
                                void *returned, size_t returned_size) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;
  const struct RequestHeader req_hdr = { .state = 0, .type = REQUEST_TYPE_OP_COMPUTE };
  const struct OpComputeRequest compute_req = { .op = op };
  const size_t req_size = sizeof(req_hdr) + sizeof(compute_req) + params_size;
  msg->state.d = 0;
  msg->n_reqs = 1;
  msg->req_offsets[0] = (int32_t) message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + (int32_t) req_size;
  uint8_t *p = (uint8_t *) message_header_get_request_ptr(msg, 0);
  memcpy(p, &req_hdr, sizeof(req_hdr));
  p += sizeof(req_hdr);
  memcpy(p, &compute_req, sizeof(compute_req));
  p += sizeof(compute_req);
  memcpy(p, params, params_size);
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }
  const struct RequestHeader *computed = message_header_get_request_ptr(msg, 0);
  if (returned != NULL && returned_size > 0) {
    memcpy(returned, computed->data + sizeof(struct OpComputeRequest), returned_size);
  }
  return computed->state;
}

static int compare_i64(const void *a, const void *b) {
  const int64_t lhs = *(const int64_t *) a;
  const int64_t rhs = *(const int64_t *) b;
  return lhs < rhs ? -1 : (lhs > rhs ? 1 : 0);
}

static int test_qkv_sparse_e2e_channel(void *chan) {
  enum { N_HEADS = 24, INPUT_DIM = 1536, TEXT_TOKENS = 154, MAX_RUNS = 100 };
  const int HEAD_DIM = env_positive_int("HTP_QKV_HEAD_DIM", 64);
  const int resolution = env_positive_int("HTP_QKV_E2E_RESOLUTION", 256);
  const int warmups = env_positive_int("HTP_QKV_E2E_WARMUPS", 3);
  const int runs = env_positive_int("HTP_QKV_E2E_RUNS", 10);
  if (resolution < 256 || resolution % 16 != 0 || runs > MAX_RUNS) {
    return -1;
  }
  const int token_h = resolution / 16;
  const int token_w = resolution / 16;
  if (token_h % 4 != 0 || token_w % 8 != 0) {
    return -1;
  }
  const int image_tokens = token_h * token_w;
  const int tokens = image_tokens + TEXT_TOKENS;
  const int image_block_rows = token_h / 4;
  const int image_block_cols = token_w / 8;
  const int blocks = (tokens + 31) / 32;
  const int output_cols = 3 * N_HEADS * HEAD_DIM;
  const int load_mode = env_enabled("HTP_QKV_E2E_USERDMA") ?
    HTP_HALO_KV_LOAD_USERDMA_AB : HTP_HALO_KV_LOAD_L2_PREFETCH;
  const size_t activation_size = (size_t) tokens * INPUT_DIM * sizeof(float);
  const size_t weight_size = (size_t) INPUT_DIM * output_cols * sizeof(__fp16);
  const size_t params_size = (size_t) output_cols * sizeof(float) + 2 * HEAD_DIM * sizeof(float);
  const size_t packed_size = (size_t) blocks * 32 * N_HEADS * HEAD_DIM * sizeof(__fp16);
  const size_t mask_size = (size_t) N_HEADS * blocks * blocks * sizeof(int32_t);
  const size_t output_size = (size_t) tokens * N_HEADS * (HEAD_DIM + 1) * sizeof(float);
  void *activation_mem = NULL;
  void *weight_mem = NULL;
  void *params_mem = NULL;
  void *q_mem = NULL;
  void *k_mem = NULL;
  void *v_mem = NULL;
  void *mask_mem = NULL;
  void *output_mem = NULL;
  int activation_fd = -1;
  int weight_fd = -1;
  int params_fd = -1;
  int q_fd = -1;
  int k_fd = -1;
  int v_fd = -1;
  int mask_fd = -1;
  int output_fd = -1;
  int result = -1;
#define ALLOC(mem, fd, bytes) \
  do { if (alloc_shared_mem_buf(&(mem), &(fd), (bytes)) != 0) goto cleanup; } while (0)
  ALLOC(activation_mem, activation_fd, activation_size);
  ALLOC(weight_mem, weight_fd, weight_size);
  ALLOC(params_mem, params_fd, params_size);
  ALLOC(q_mem, q_fd, packed_size);
  ALLOC(k_mem, k_fd, packed_size);
  ALLOC(v_mem, v_fd, packed_size);
  ALLOC(mask_mem, mask_fd, mask_size);
  ALLOC(output_mem, output_fd, output_size);
#undef ALLOC
  float *activation = (float *) activation_mem;
  __fp16 *weight = (__fp16 *) weight_mem;
  float *bias = (float *) params_mem;
  float *q_norm = bias + output_cols;
  float *k_norm = q_norm + HEAD_DIM;
  for (int t = 0; t < tokens; ++t) {
    for (int c = 0; c < INPUT_DIM; ++c) {
      activation[(size_t) t * INPUT_DIM + c] = qkv_bench_activation_value(t, c);
    }
  }
  for (int c = 0; c < output_cols; ++c) {
    bias[c] = (float) ((c % 17) - 8) * 0.01f;
  }
  for (int d = 0; d < HEAD_DIM; ++d) {
    q_norm[d] = 1.0f + (float) (d % 7) * 0.01f;
    k_norm[d] = 1.0f - (float) (d % 5) * 0.01f;
  }
  for (int r = 0; r < INPUT_DIM; ++r) {
    for (int c = 0; c < output_cols; ++c) {
      const int r0 = r / 32;
      const int r1 = r % 32;
      const int c0 = c / 32;
      const int c1 = c % 32;
      __fp16 *tile = weight + (size_t) (c0 * (INPUT_DIM / 32) + r0) * 1024;
      tile[(r1 & ~1) * 32 + c1 * 2 + (r1 & 1)] = (__fp16) qkv_bench_weight_value(r, c);
    }
  }
  fill_static_halo_index((int32_t *) mask_mem, blocks, blocks, N_HEADS,
                         image_block_rows, image_block_cols);
  memset(q_mem, 0, packed_size);
  memset(k_mem, 0, packed_size);
  memset(v_mem, 0, packed_size);
  memset(output_mem, 0, output_size);

  const struct QkvMulRmsPackedParams qkv_params = {
    .q = { q_fd, 0 }, .k_out = { k_fd, 0 }, .v = { v_fd, 0 },
    .activation = { activation_fd, 0 }, .weight = { weight_fd, 0 },
    .bias = { params_fd, 0 },
    .q_norm = { params_fd, (int32_t) (output_cols * sizeof(float)) },
    .k_norm = { params_fd, (int32_t) ((output_cols + HEAD_DIM) * sizeof(float)) },
    .tokens = tokens, .input_dim = INPUT_DIM, .n_heads = N_HEADS, .head_dim = HEAD_DIM,
  };
  const struct BlockSparseAttnParams attn_params = {
    .o = { output_fd, 0 }, .q = { q_fd, 0 }, .k = { k_fd, 0 }, .v = { v_fd, 0 },
    .block_mask = { mask_fd, 0 }, .qo_len = tokens, .kv_len = tokens,
    .n_heads = N_HEADS, .n_kv_heads = N_HEADS, .head_dim = HEAD_DIM,
    .block_q = 32, .block_k = 32, .n_q_blocks = N_HEADS * blocks,
    .n_k_blocks = N_HEADS * blocks, .mask_format = 1, .max_kept_blocks = blocks,
    .with_lse = 1, .sparse_policy = HTP_SPARSE_POLICY_STATIC_HALO,
    .image_tokens = image_tokens, .text_tokens = TEXT_TOKENS,
    .halo_kv_load_mode = load_mode, .halo_trace = 1,
    .image_block_rows = image_block_rows, .image_block_cols = image_block_cols,
  };
  int64_t samples[MAX_RUNS];
  struct QkvMulRmsPackedParams qkv_returned = {};
  struct BlockSparseAttnParams attn_returned = {};
  for (int run = -warmups; run < runs; ++run) {
    const int64_t t0 = get_time_us();
    const int qkv_state = run_qkv_request(chan, &qkv_params, &qkv_returned);
    const int attn_state = qkv_state == 0 ? run_sparse_attention_request(
      chan, HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16, &attn_params, &attn_returned) : -1;
    const int64_t elapsed = get_time_us() - t0;
    if (qkv_state != 0 || attn_state != 0) {
      fprintf(stderr,
              "qkv_sparse_e2e_packed request failed: resolution=%d run=%d qkv_state=%d attn_state=%d\n",
              resolution, run, qkv_state, attn_state);
      goto cleanup;
    }
    if (run >= 0) {
      samples[run] = elapsed;
    }
  }
  int64_t sorted[MAX_RUNS];
  memcpy(sorted, samples, (size_t) runs * sizeof(int64_t));
  qsort(sorted, (size_t) runs, sizeof(int64_t), compare_i64);
  int64_t sum = 0;
  for (int i = 0; i < runs; ++i) sum += samples[i];
  const double median = runs % 2 ? sorted[runs / 2] :
    0.5 * (double) (sorted[runs / 2 - 1] + sorted[runs / 2]);
  const float *output = (const float *) output_mem;
  const int expected_exec = load_mode == HTP_HALO_KV_LOAD_USERDMA_AB ? 2 : 1;
  double q_sum = 0.0;
  double k_sum = 0.0;
  double v_sum = 0.0;
  double q_sumsq = 0.0;
  double k_sumsq = 0.0;
  double v_sumsq = 0.0;
  float ref_o0 = 0.0f;
  float ref_lse0 = 0.0f;
  if (resolution == 256) {
    for (int t = 0; t < tokens; ++t) {
      for (int h = 0; h < N_HEADS; ++h) {
        for (int d = 0; d < HEAD_DIM; ++d) {
          const float qv = packed_q_value((const __fp16 *) q_mem, tokens, HEAD_DIM, t, h, d);
          const float kv = packed_k_value((const __fp16 *) k_mem, tokens, HEAD_DIM, t, h, d);
          const float vv = packed_v_value((const __fp16 *) v_mem, tokens, HEAD_DIM, t, h, d);
          q_sum += qv; k_sum += kv; v_sum += vv;
          q_sumsq += qv * qv; k_sumsq += kv * kv; v_sumsq += vv * vv;
        }
      }
    }
    const int32_t *row = (const int32_t *) mask_mem;
    float max_score = -INFINITY;
    for (int pos = 0; pos < blocks && row[pos] >= 0; ++pos) {
      const int kb = row[pos] % blocks;
      const int begin = kb * 32;
      const int end = begin + 32 < tokens ? begin + 32 : tokens;
      for (int kt = begin; kt < end; ++kt) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
          score += packed_q_value((const __fp16 *) q_mem, tokens, HEAD_DIM, 0, 0, d) *
                   packed_k_value((const __fp16 *) k_mem, tokens, HEAD_DIM, kt, 0, d);
        }
        score /= sqrtf((float) HEAD_DIM);
        if (score > max_score) max_score = score;
      }
    }
    float denom = 0.0f;
    float numerator = 0.0f;
    for (int pos = 0; pos < blocks && row[pos] >= 0; ++pos) {
      const int kb = row[pos] % blocks;
      const int begin = kb * 32;
      const int end = begin + 32 < tokens ? begin + 32 : tokens;
      for (int kt = begin; kt < end; ++kt) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
          score += packed_q_value((const __fp16 *) q_mem, tokens, HEAD_DIM, 0, 0, d) *
                   packed_k_value((const __fp16 *) k_mem, tokens, HEAD_DIM, kt, 0, d);
        }
        const float probability = expf(score / sqrtf((float) HEAD_DIM) - max_score);
        denom += probability;
        numerator += probability * packed_v_value(
          (const __fp16 *) v_mem, tokens, HEAD_DIM, kt, 0, 0);
      }
    }
    ref_o0 = numerator / denom;
    ref_lse0 = max_score + logf(denom);
  }
  fprintf(stderr,
          "qkv_sparse_e2e_packed: resolution=%d tokens=%d loader=%s avg_us=%.1f "
          "median_us=%.1f min_us=%lld qkv_us=%d attn_us=%d exec=%d o0=%g lse0=%g "
          "q0=%g k0=%g v0=%g q_h1=%g k_h1=%g v_h1=%g q_t1=%g k_t1=%g v_t1=%g "
          "q_sum=%.9g k_sum=%.9g v_sum=%.9g q_sumsq=%.9g k_sumsq=%.9g v_sumsq=%.9g "
          "ref_o0=%g ref_lse0=%g samples_us=",
          resolution, tokens, load_mode == HTP_HALO_KV_LOAD_USERDMA_AB ? "userdma" : "l2fetch",
          (double) sum / runs, median, (long long) sorted[0], qkv_returned.elapsed_us,
          attn_returned.elapsed_us, attn_returned.halo_exec_mode, output[0], output[HEAD_DIM],
          packed_q_value((const __fp16 *) q_mem, tokens, HEAD_DIM, 0, 0, 0),
          packed_k_value((const __fp16 *) k_mem, tokens, HEAD_DIM, 0, 0, 0),
          packed_v_value((const __fp16 *) v_mem, tokens, HEAD_DIM, 0, 0, 0),
          packed_q_value((const __fp16 *) q_mem, tokens, HEAD_DIM, 0, 1, 0),
          packed_k_value((const __fp16 *) k_mem, tokens, HEAD_DIM, 0, 1, 0),
          packed_v_value((const __fp16 *) v_mem, tokens, HEAD_DIM, 0, 1, 0),
          packed_q_value((const __fp16 *) q_mem, tokens, HEAD_DIM, 1, 0, 0),
          packed_k_value((const __fp16 *) k_mem, tokens, HEAD_DIM, 1, 0, 0),
          packed_v_value((const __fp16 *) v_mem, tokens, HEAD_DIM, 1, 0, 0),
          q_sum, k_sum, v_sum, q_sumsq, k_sumsq, v_sumsq, ref_o0, ref_lse0);
  for (int i = 0; i < runs; ++i) {
    fprintf(stderr, "%s%lld", i == 0 ? "" : ",", (long long) samples[i]);
  }
  fprintf(stderr, "\n");
  result = attn_returned.halo_exec_mode == expected_exec &&
           isfinite(output[0]) && isfinite(output[HEAD_DIM]) ? 0 : -1;

cleanup:
#define FREE_BUF(mem, fd, bytes) do { if ((mem) != NULL) free_shared_mem_buf((mem), (fd), (bytes)); } while (0)
  FREE_BUF(output_mem, output_fd, output_size);
  FREE_BUF(mask_mem, mask_fd, mask_size);
  FREE_BUF(v_mem, v_fd, packed_size);
  FREE_BUF(k_mem, k_fd, packed_size);
  FREE_BUF(q_mem, q_fd, packed_size);
  FREE_BUF(params_mem, params_fd, params_size);
  FREE_BUF(weight_mem, weight_fd, weight_size);
  FREE_BUF(activation_mem, activation_fd, activation_size);
#undef FREE_BUF
  fprintf(stderr, result == 0 ? "%s passed\n" : "%s failed\n", __func__);
  return result;
}

static int test_packed_sparse_attention_channel(void *chan) {
  enum { N_HEADS = 24, DEFAULT_TEXT_TOKENS = 154 };
  const int HEAD_DIM = env_positive_int("HTP_PACKED_ATTN_HEAD_DIM", 64);
  const int token_h = env_positive_int("HTP_PACKED_ATTN_TOKEN_H", 16);
  const int token_w = env_positive_int("HTP_PACKED_ATTN_TOKEN_W", 16);
  const int text_tokens = env_nonnegative_int("HTP_PACKED_ATTN_TEXT_TOKENS", DEFAULT_TEXT_TOKENS);
  const int image_tokens = token_h * token_w;
  const int tokens = image_tokens + text_tokens;
  if (token_h <= 0 || token_w <= 0 || token_h % 4 != 0 || token_w % 8 != 0) {
    return -1;
  }
  const int image_block_rows = token_h / 4;
  const int image_block_cols = token_w / 8;
  const int blocks = (tokens + 31) / 32;
  const int max_kept = blocks;
  const int load_mode = env_enabled("HTP_PACKED_ATTN_USERDMA") ?
    HTP_HALO_KV_LOAD_USERDMA_AB : HTP_HALO_KV_LOAD_L2_PREFETCH;
  const size_t raw_elements = (size_t) tokens * N_HEADS * HEAD_DIM;
  const size_t packed_elements = (size_t) blocks * 32 * N_HEADS * HEAD_DIM;
  const size_t output_elements = (size_t) tokens * N_HEADS * (HEAD_DIM + 1);
  const size_t mask_elements = (size_t) N_HEADS * blocks * max_kept;
  const size_t q_raw_offset = 0;
  const size_t k_raw_offset = align_up(q_raw_offset + raw_elements * sizeof(float), 128);
  const size_t v_raw_offset = align_up(k_raw_offset + raw_elements * sizeof(__fp16), 128);
  const size_t q_packed_offset = align_up(v_raw_offset + raw_elements * sizeof(__fp16), 128);
  const size_t k_packed_offset = align_up(q_packed_offset + packed_elements * sizeof(__fp16), 128);
  const size_t v_packed_offset = align_up(k_packed_offset + packed_elements * sizeof(__fp16), 128);
  const size_t mask_offset = align_up(v_packed_offset + packed_elements * sizeof(__fp16), 128);
  const size_t raw_o_offset = align_up(mask_offset + mask_elements * sizeof(int32_t), 128);
  const size_t packed_o_offset = align_up(raw_o_offset + output_elements * sizeof(float), 128);
  const size_t total_size = align_up(packed_o_offset + output_elements * sizeof(float), 128);
  uint8_t *data = NULL;
  int fd = -1;
  int result = -1;
  if (alloc_shared_mem_buf((void **) &data, &fd, total_size) != 0) {
    return -1;
  }
  float *q_raw = (float *) (data + q_raw_offset);
  __fp16 *k_raw = (__fp16 *) (data + k_raw_offset);
  __fp16 *v_raw = (__fp16 *) (data + v_raw_offset);
  __fp16 *q_packed = (__fp16 *) (data + q_packed_offset);
  __fp16 *k_packed = (__fp16 *) (data + k_packed_offset);
  __fp16 *v_packed = (__fp16 *) (data + v_packed_offset);
  int32_t *mask = (int32_t *) (data + mask_offset);
  float *raw_o = (float *) (data + raw_o_offset);
  float *packed_o = (float *) (data + packed_o_offset);
  for (size_t i = 0; i < raw_elements; ++i) {
    q_raw[i] = (float) ((int) (i % 29) - 14) * 0.015f;
    k_raw[i] = (__fp16) ((float) ((int) (i % 31) - 15) * 0.012f);
    v_raw[i] = (__fp16) ((float) ((int) (i % 37) - 18) * 0.01f);
  }
  pack_attention_qkv(q_raw, k_raw, v_raw, q_packed, k_packed, v_packed,
                     tokens, N_HEADS, HEAD_DIM);
  fill_static_halo_index(mask, blocks, max_kept, N_HEADS,
                         image_block_rows, image_block_cols);
  memset(raw_o, 0, output_elements * sizeof(float));
  memset(packed_o, 0, output_elements * sizeof(float));

#define BUF(off) { .fd = fd, .offset = (int32_t) (off) }
  struct BlockSparseAttnParams raw_params = {
    .o = BUF(raw_o_offset), .q = BUF(q_raw_offset), .k = BUF(k_raw_offset), .v = BUF(v_raw_offset),
    .block_mask = BUF(mask_offset), .qo_len = tokens, .kv_len = tokens,
    .n_heads = N_HEADS, .n_kv_heads = N_HEADS, .head_dim = HEAD_DIM,
    .block_q = 32, .block_k = 32, .n_q_blocks = N_HEADS * blocks,
    .n_k_blocks = N_HEADS * blocks, .mask_format = 1, .max_kept_blocks = max_kept,
    .with_lse = 1, .sparse_policy = HTP_SPARSE_POLICY_STATIC_HALO,
    .image_tokens = image_tokens, .text_tokens = text_tokens,
    .halo_kv_load_mode = load_mode, .halo_trace = 1,
    .image_block_rows = image_block_rows, .image_block_cols = image_block_cols,
  };
  struct BlockSparseAttnParams packed_params = raw_params;
  packed_params.o = (struct RpcmemBufAddr) BUF(packed_o_offset);
  packed_params.q = (struct RpcmemBufAddr) BUF(q_packed_offset);
  packed_params.k = (struct RpcmemBufAddr) BUF(k_packed_offset);
  packed_params.v = (struct RpcmemBufAddr) BUF(v_packed_offset);
#undef BUF
  struct BlockSparseAttnParams raw_returned;
  struct BlockSparseAttnParams packed_returned;
  const int raw_state = run_sparse_attention_request(
    chan, HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16, &raw_params, &raw_returned);
  const int packed_state = run_sparse_attention_request(
    chan, HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16, &packed_params, &packed_returned);
  float max_abs = 0.0f;
  size_t worst = 0;
  for (size_t i = 0; i < output_elements; ++i) {
    const float diff = fabsf(raw_o[i] - packed_o[i]);
    if (diff > max_abs) {
      max_abs = diff;
      worst = i;
    }
  }
  // Non-standard image geometry cannot use the halo workspace and is expected
  // to execute through the chunked fallback (mode 3).
  const int expected_exec = (image_block_rows < 2 || image_block_cols <= 0) ?
    HTP_HALO_EXEC_TRACE_CHUNKED_FALLBACK :
    (load_mode == HTP_HALO_KV_LOAD_USERDMA_AB ? 2 : 1);
  fprintf(stderr,
          "packed_sparse_attention: tokens=%d loader=%s raw_us=%d packed_us=%d "
          "raw_exec=%d packed_exec=%d max_abs=%g worst=%zu raw=%g packed=%g\n",
          tokens, load_mode == HTP_HALO_KV_LOAD_USERDMA_AB ? "userdma" : "l2fetch",
          raw_returned.elapsed_us, packed_returned.elapsed_us,
          raw_returned.halo_exec_mode, packed_returned.halo_exec_mode,
          max_abs, worst, raw_o[worst], packed_o[worst]);
  result = raw_state == 0 && packed_state == 0 &&
           raw_returned.halo_exec_mode == expected_exec &&
           packed_returned.halo_exec_mode == expected_exec && max_abs <= 0.006f ? 0 : -1;
  free_shared_mem_buf(data, fd, total_size);
  fprintf(stderr, result == 0 ? "%s passed\n" : "%s failed\n", __func__);
  return result;
}

static void unpack_packed_attention_qkv(const __fp16 *q, const __fp16 *k,
                                        const __fp16 *v, float *raw_q,
                                        __fp16 *raw_k, __fp16 *raw_v,
                                        int tokens, int n_heads, int head_dim) {
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < n_heads; ++h) {
      for (int d = 0; d < head_dim; ++d) {
        const size_t raw = ((size_t) t * n_heads + h) * head_dim + d;
        raw_q[raw] = packed_q_value(q, tokens, head_dim, t, h, d);
        raw_k[raw] = (__fp16) packed_k_value(k, tokens, head_dim, t, h, d);
        raw_v[raw] = (__fp16) packed_v_value(v, tokens, head_dim, t, h, d);
      }
    }
  }
}

static int test_flux2_attention_benchmark_channel(void *chan) {
  enum {
    N_HEADS = 24,
    HEAD_DIM = 128,
    INPUT_DIM = 3072,
    IMAGE_TOKENS = 256,
    TEXT_TOKENS = 512,
    TOKENS = IMAGE_TOKENS + TEXT_TOKENS,
    MAX_RUNS = 100,
  };
  const int dual = env_enabled("HTP_FLUX2_ATTN_DUAL");
  const int reorder = env_enabled("HTP_FLUX2_ATTN_REORDER");
  const int use_userdma = env_enabled("HTP_FLUX2_ATTN_USERDMA");
  const int warmups = env_positive_int("HTP_FLUX2_ATTN_WARMUPS", 10);
  const int requested_runs = env_positive_int("HTP_FLUX2_ATTN_RUNS", 50);
  const int runs = requested_runs > MAX_RUNS ? MAX_RUNS : requested_runs;
  const int blocks = (TOKENS + 31) / 32;
  const int image_block_rows = 4;
  const int image_block_cols = 2;
  const int output_cols = 3 * N_HEADS * HEAD_DIM;
  const int load_mode = use_userdma ? HTP_HALO_KV_LOAD_USERDMA_AB : HTP_HALO_KV_LOAD_L2_PREFETCH;
  const size_t activation_bytes = (size_t) TOKENS * INPUT_DIM * sizeof(float);
  const size_t weight_bytes = (size_t) INPUT_DIM * output_cols * sizeof(__fp16);
  const size_t bias_bytes = (size_t) output_cols * sizeof(float);
  const size_t norm_bytes = (size_t) HEAD_DIM * sizeof(float);
  const size_t rope_bytes = (size_t) TOKENS * (HEAD_DIM / 2) * 4 * sizeof(float);
  const size_t head_elems = (size_t) blocks * (HEAD_DIM / 32) * 1024;
  const size_t qkv_bytes = (size_t) N_HEADS * head_elems * sizeof(__fp16);
  const size_t raw_elements = (size_t) TOKENS * N_HEADS * HEAD_DIM;
  const size_t mask_elements = (size_t) N_HEADS * blocks * blocks;
  const size_t output_elements = (size_t) TOKENS * N_HEADS * (HEAD_DIM + 1);

  size_t activation_offset = 0;
  size_t image_weight_offset = align_up(activation_offset + activation_bytes, 128);
  size_t text_weight_offset = align_up(image_weight_offset + weight_bytes, 128);
  size_t image_bias_offset = align_up(text_weight_offset + weight_bytes, 128);
  size_t text_bias_offset = align_up(image_bias_offset + bias_bytes, 128);
  size_t image_q_norm_offset = align_up(text_bias_offset + bias_bytes, 128);
  size_t image_k_norm_offset = align_up(image_q_norm_offset + norm_bytes, 128);
  size_t text_q_norm_offset = align_up(image_k_norm_offset + norm_bytes, 128);
  size_t text_k_norm_offset = align_up(text_q_norm_offset + norm_bytes, 128);
  size_t rope_offset = align_up(text_k_norm_offset + norm_bytes, 128);
  size_t q_offset = align_up(rope_offset + rope_bytes, 128);
  size_t k_offset = align_up(q_offset + qkv_bytes, 128);
  size_t v_offset = align_up(k_offset + qkv_bytes, 128);
  size_t mask_offset = align_up(v_offset + qkv_bytes, 128);
  size_t raw_q_offset = align_up(mask_offset + mask_elements * sizeof(int32_t), 128);
  size_t raw_k_offset = align_up(raw_q_offset + raw_elements * sizeof(float), 128);
  size_t raw_v_offset = align_up(raw_k_offset + raw_elements * sizeof(__fp16), 128);
  size_t raw_output_offset = align_up(raw_v_offset + raw_elements * sizeof(__fp16), 128);
  size_t packed_output_offset = align_up(raw_output_offset + output_elements * sizeof(float), 128);
  size_t total_bytes = align_up(packed_output_offset + output_elements * sizeof(float), 128);

  uint8_t *data = NULL;
  int fd = -1;
  int result = -1;
  if (alloc_shared_mem_buf((void **) &data, &fd, total_bytes) != 0) {
    fprintf(stderr, "Cannot allocate Flux2 benchmark rpcmem (%zu bytes)\n", total_bytes);
    return -1;
  }

  float *activation = (float *) (data + activation_offset);
  __fp16 *image_weight = (__fp16 *) (data + image_weight_offset);
  __fp16 *text_weight = (__fp16 *) (data + text_weight_offset);
  float *image_bias = (float *) (data + image_bias_offset);
  float *text_bias = (float *) (data + text_bias_offset);
  float *image_q_norm = (float *) (data + image_q_norm_offset);
  float *image_k_norm = (float *) (data + image_k_norm_offset);
  float *text_q_norm = (float *) (data + text_q_norm_offset);
  float *text_k_norm = (float *) (data + text_k_norm_offset);
  float *rope_pe = (float *) (data + rope_offset);
  __fp16 *q = (__fp16 *) (data + q_offset);
  __fp16 *k = (__fp16 *) (data + k_offset);
  __fp16 *v = (__fp16 *) (data + v_offset);
  int32_t *mask = (int32_t *) (data + mask_offset);
  float *raw_q = (float *) (data + raw_q_offset);
  __fp16 *raw_k = (__fp16 *) (data + raw_k_offset);
  __fp16 *raw_v = (__fp16 *) (data + raw_v_offset);
  float *raw_output = (float *) (data + raw_output_offset);
  float *packed_output = (float *) (data + packed_output_offset);

  for (int t = 0; t < TOKENS; ++t) {
    for (int c = 0; c < INPUT_DIM; ++c) {
      activation[(size_t) t * INPUT_DIM + c] = qkv_bench_activation_value(t, c);
    }
  }
  for (int c = 0; c < output_cols; ++c) {
    image_bias[c] = (float) ((c % 19) - 9) * 0.007f;
    text_bias[c] = (float) ((c % 23) - 11) * 0.006f;
  }
  for (int d = 0; d < HEAD_DIM; ++d) {
    image_q_norm[d] = 0.9f + (float) (d % 11) * 0.013f;
    image_k_norm[d] = 1.1f - (float) (d % 7) * 0.009f;
    text_q_norm[d] = 1.0f + (float) (d % 13) * 0.008f;
    text_k_norm[d] = 0.95f - (float) (d % 5) * 0.007f;
  }
  for (int t = 0; t < TOKENS; ++t) {
    for (int d = 0; d < HEAD_DIM / 2; ++d) {
      const float angle = 0.013f * (float) (t + 1) * (float) (d + 1);
      float *m = rope_pe + ((size_t) t * (HEAD_DIM / 2) + d) * 4;
      m[0] = cosf(angle);
      m[1] = -sinf(angle);
      m[2] = sinf(angle);
      m[3] = cosf(angle);
    }
  }
  for (int r = 0; r < INPUT_DIM; ++r) {
    for (int c = 0; c < output_cols; ++c) {
      const int r0 = r / 32;
      const int r1 = r % 32;
      const int c0 = c / 32;
      const int c1 = c % 32;
      const size_t tile = (size_t) (c0 * (INPUT_DIM / 32) + r0) * 1024;
      const size_t in_tile = (size_t) (r1 & ~1) * 32 + c1 * 2 + (r1 & 1);
      image_weight[tile + in_tile] = (__fp16) qkv_bench_weight_value(r, c);
      text_weight[tile + in_tile] = (__fp16) qkv_bench_weight_value(r + 17, c + 11);
    }
  }
  fill_static_halo_index(mask, blocks, blocks, N_HEADS,
                         image_block_rows, image_block_cols);
  memset(q, 0, qkv_bytes);
  memset(k, 0, qkv_bytes);
  memset(v, 0, qkv_bytes);
  memset(raw_output, 0, output_elements * sizeof(float));
  memset(packed_output, 0, output_elements * sizeof(float));

  struct QkvMulRmsPackedRopeParams single_params = {
    .q = { fd, (int32_t) q_offset }, .k_out = { fd, (int32_t) k_offset },
    .v = { fd, (int32_t) v_offset }, .activation = { fd, (int32_t) activation_offset },
    .weight = { fd, (int32_t) image_weight_offset }, .bias = { fd, (int32_t) image_bias_offset },
    .q_norm = { fd, (int32_t) image_q_norm_offset }, .k_norm = { fd, (int32_t) image_k_norm_offset },
    .rope_pe = { fd, (int32_t) rope_offset }, .tokens = TOKENS,
    .input_dim = INPUT_DIM, .n_heads = N_HEADS, .head_dim = HEAD_DIM,
    .rope_token_offset = 0, .rope_token_count = TOKENS, .elapsed_us = -1,
  };
  struct QkvMulRmsPackedDualRopeParams dual_params = {
    .q = { fd, (int32_t) q_offset }, .k_out = { fd, (int32_t) k_offset },
    .v = { fd, (int32_t) v_offset },
    .image_activation = { fd, (int32_t) activation_offset },
    .image_weight = { fd, (int32_t) image_weight_offset },
    .image_bias = { fd, (int32_t) image_bias_offset },
    .image_q_norm = { fd, (int32_t) image_q_norm_offset },
    .image_k_norm = { fd, (int32_t) image_k_norm_offset },
    .text_activation = { fd, (int32_t) (activation_offset + (size_t) IMAGE_TOKENS * INPUT_DIM * sizeof(float)) },
    .text_weight = { fd, (int32_t) text_weight_offset },
    .text_bias = { fd, (int32_t) text_bias_offset },
    .text_q_norm = { fd, (int32_t) text_q_norm_offset },
    .text_k_norm = { fd, (int32_t) text_k_norm_offset },
    .rope_pe = { fd, (int32_t) rope_offset },
    .image_tokens = IMAGE_TOKENS, .text_tokens = TEXT_TOKENS,
    .input_dim = INPUT_DIM, .n_heads = N_HEADS, .head_dim = HEAD_DIM,
    .image_rope_token_offset = 0, .text_rope_token_offset = IMAGE_TOKENS, .elapsed_us = -1,
  };

#define BUF_AT(off) { .fd = fd, .offset = (int32_t) (off) }
  struct BlockSparseAttnParams packed_attn_params = {
    .o = BUF_AT(packed_output_offset), .q = BUF_AT(q_offset),
    .k = BUF_AT(k_offset), .v = BUF_AT(v_offset), .block_mask = BUF_AT(mask_offset),
    .qo_len = TOKENS, .kv_len = TOKENS, .n_heads = N_HEADS, .n_kv_heads = N_HEADS,
    .head_dim = HEAD_DIM, .block_q = 32, .block_k = 32,
    .n_q_blocks = N_HEADS * blocks, .n_k_blocks = N_HEADS * blocks,
    .mask_format = 1, .max_kept_blocks = blocks, .with_lse = 1,
    .sparse_policy = HTP_SPARSE_POLICY_STATIC_HALO,
    .image_tokens = IMAGE_TOKENS, .text_tokens = TEXT_TOKENS,
    .halo_kv_load_mode = load_mode, .halo_trace = 1,
    .image_block_rows = image_block_rows, .image_block_cols = image_block_cols,
  };
  struct BlockSparseAttnParams raw_attn_params = packed_attn_params;
  raw_attn_params.o = (struct RpcmemBufAddr) BUF_AT(raw_output_offset);
  raw_attn_params.q = (struct RpcmemBufAddr) BUF_AT(raw_q_offset);
  raw_attn_params.k = (struct RpcmemBufAddr) BUF_AT(raw_k_offset);
  raw_attn_params.v = (struct RpcmemBufAddr) BUF_AT(raw_v_offset);
#undef BUF_AT

  int64_t samples[MAX_RUNS];
  int64_t qkv_dsp_total = 0;
  int64_t attn_dsp_total = 0;
  struct QkvMulRmsPackedRopeParams single_returned = {};
  struct QkvMulRmsPackedDualRopeParams dual_returned = {};
  struct BlockSparseAttnParams packed_returned = {};
  struct BlockSparseAttnParams raw_returned = {};
  int qkv_state = 0;
  int timed_attn_state = 0;
  for (int run = -warmups; run < runs; ++run) {
    const int64_t start = get_time_us();
    qkv_state = dual ?
      run_qkv_rope_request(chan, HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32,
                           &dual_params, sizeof(dual_params), &dual_returned, sizeof(dual_returned)) :
      run_qkv_rope_request(chan, HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32,
                           &single_params, sizeof(single_params), &single_returned, sizeof(single_returned));
    if (qkv_state == 0 && reorder) {
      unpack_packed_attention_qkv(q, k, v, raw_q, raw_k, raw_v,
                                  TOKENS, N_HEADS, HEAD_DIM);
      timed_attn_state = run_sparse_attention_request(
        chan, HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16,
        &raw_attn_params, &raw_returned);
    } else if (qkv_state == 0) {
      timed_attn_state = run_sparse_attention_request(
        chan, HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16,
        &packed_attn_params, &packed_returned);
    } else {
      timed_attn_state = -1;
    }
    const int64_t elapsed = get_time_us() - start;
    if (qkv_state != 0 || timed_attn_state != 0) {
      fprintf(stderr, "Flux2 %s benchmark dispatch failed qkv=%d attn=%d\n",
              dual ? "double" : "single", qkv_state, timed_attn_state);
      goto cleanup;
    }
    if (run >= 0) {
      samples[run] = elapsed;
      qkv_dsp_total += dual ? dual_returned.elapsed_us : single_returned.elapsed_us;
      attn_dsp_total += reorder ? raw_returned.elapsed_us : packed_returned.elapsed_us;
    }
  }

  unpack_packed_attention_qkv(q, k, v, raw_q, raw_k, raw_v,
                              TOKENS, N_HEADS, HEAD_DIM);
  const int raw_state = run_sparse_attention_request(
    chan, HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16, &raw_attn_params, &raw_returned);
  const int packed_state = run_sparse_attention_request(
    chan, HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16, &packed_attn_params, &packed_returned);
  float max_abs = 0.0f;
  size_t worst = 0;
  for (size_t i = 0; i < output_elements; ++i) {
    const float diff = fabsf(raw_output[i] - packed_output[i]);
    if (diff > max_abs) {
      max_abs = diff;
      worst = i;
    }
  }

  qsort(samples, runs, sizeof(samples[0]), compare_i64);
  int64_t total_us = 0;
  for (int i = 0; i < runs; ++i) total_us += samples[i];
  const double avg_us = (double) total_us / runs;
  const double median_us = runs % 2 ? (double) samples[runs / 2] :
    0.5 * (double) (samples[runs / 2 - 1] + samples[runs / 2]);
  const int p95_index = (95 * runs + 99) / 100 - 1;
  fprintf(stderr,
          "flux2_attention_e2e: device=8gen3 mode=%s layout=%s loader=%s image_tokens=%d text_tokens=%d "
          "heads=%d head_dim=%d warmups=%d runs=%d avg_us=%.1f median_us=%.1f "
          "p95_us=%lld min_us=%lld qkv_dsp_avg_us=%.1f attn_dsp_avg_us=%.1f "
          "timed_exec=%d raw_exec=%d packed_exec=%d max_abs=%g worst=%zu raw=%g packed=%g\n",
          dual ? "double" : "single", reorder ? "reordered" : "packed",
          use_userdma ? "userdma" : "l2fetch",
          IMAGE_TOKENS, TEXT_TOKENS, N_HEADS, HEAD_DIM, warmups, runs,
          avg_us, median_us, (long long) samples[p95_index], (long long) samples[0],
          (double) qkv_dsp_total / runs, (double) attn_dsp_total / runs,
          reorder ? raw_returned.halo_exec_mode : packed_returned.halo_exec_mode,
          raw_returned.halo_exec_mode, packed_returned.halo_exec_mode,
          max_abs, worst, raw_output[worst], packed_output[worst]);

  const int expected_exec = use_userdma ? 2 : 1;
  result = raw_state == 0 && packed_state == 0 &&
           raw_returned.halo_exec_mode == expected_exec &&
           packed_returned.halo_exec_mode == expected_exec &&
           isfinite(raw_output[0]) && isfinite(packed_output[0]) && max_abs <= 0.006f ? 0 : -1;

cleanup:
  if (data != NULL) free_shared_mem_buf(data, fd, total_bytes);
  fprintf(stderr, result == 0 ? "%s passed\n" : "%s failed\n", __func__);
  return result;
}

static int test_qkv_separate_projection_microbenchmark(void *chan) {
  enum {
    N_HEADS = 24,
    HEAD_DIM = 64,
  };
  const int tokens = env_positive_int("HTP_QKV_TOKENS", 32);
  const int input_dim = env_positive_int("HTP_QKV_INPUT_DIM", 1536);
  const int output_cols = N_HEADS * HEAD_DIM;
  const int total_output_cols = 3 * output_cols;
  if (input_dim % 32 != 0) {
    fprintf(stderr, "qkv separate projection microbenchmark requires HTP_QKV_INPUT_DIM %% 32 == 0\n");
    return -1;
  }

  const size_t activation_size = (size_t) tokens * input_dim * sizeof(float);
  const size_t weight_size = (size_t) input_dim * total_output_cols * sizeof(__fp16);
  const size_t projection_size = (size_t) tokens * output_cols * sizeof(float);
  const size_t activation_offset = 0;
  const size_t weight_offset = align_up(activation_offset + activation_size, 128);
  const size_t q_raw_offset = align_up(weight_offset + weight_size, 128);
  const size_t k_raw_offset = align_up(q_raw_offset + projection_size, 128);
  const size_t v_raw_offset = align_up(k_raw_offset + projection_size, 128);
  const size_t q_norm_offset = align_up(v_raw_offset + projection_size, 128);
  const size_t k_norm_offset = align_up(q_norm_offset + projection_size, 128);
  const size_t tensor_size = align_up(k_norm_offset + projection_size, 128);
  uint8_t *tensor_data = NULL;
  int tensor_fd = -1;
  float *activation = NULL;
  __fp16 *weight = NULL;
  float *q_raw = NULL;
  float *k_raw = NULL;
  float *v_raw = NULL;
  float *q_norm = NULL;
  float *k_norm = NULL;
  int result = -1;

  if (alloc_shared_mem_buf((void **) &tensor_data, &tensor_fd, tensor_size) != 0) {
    fprintf(stderr, "qkv separate projection microbenchmark shared allocation failed\n");
    goto cleanup;
  }
  activation = (float *) (tensor_data + activation_offset);
  weight = (__fp16 *) (tensor_data + weight_offset);
  q_raw = (float *) (tensor_data + q_raw_offset);
  k_raw = (float *) (tensor_data + k_raw_offset);
  v_raw = (float *) (tensor_data + v_raw_offset);
  q_norm = (float *) (tensor_data + q_norm_offset);
  k_norm = (float *) (tensor_data + k_norm_offset);

  for (int r = 0; r < tokens; ++r) {
    for (int c = 0; c < input_dim; ++c) {
      activation[(size_t) r * input_dim + c] = qkv_bench_activation_value(r, c);
    }
  }
  memset(q_raw, 0, projection_size);
  memset(k_raw, 0, projection_size);
  memset(v_raw, 0, projection_size);
  memset(q_norm, 0, projection_size);
  memset(k_norm, 0, projection_size);
  for (int r = 0; r < input_dim; ++r) {
    for (int c = 0; c < total_output_cols; ++c) {
      const int r0 = r / 32;
      const int r1 = r % 32;
      const int c0 = c / 32;
      const int c1 = c % 32;
      const int tile_idx = c0 * (input_dim / 32) + r0;
      __fp16 *tile = weight + (size_t) tile_idx * 1024;
      tile[(r1 & ~1) * 32 + c1 * 2 + (r1 & 1)] =
        (__fp16) qkv_bench_weight_value(r, c);
    }
  }

  struct MessageHeader *msg = (struct MessageHeader *) chan;
  const struct RequestHeader req_hdr = {
    .state = 0,
    .type = REQUEST_TYPE_OP_COMPUTE,
  };
  const struct OpComputeRequest matmul_req = {
    .op = HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  };
  const struct OpComputeRequest rms_req = {
    .op = HTP_OPS_RMS_NORM_F32,
  };
  const struct MatMulParams matmul_params[3] = {
    { .output = { .fd = tensor_fd, .offset = (int32_t) q_raw_offset },
      .activation = { .fd = tensor_fd, .offset = (int32_t) activation_offset },
      .weight = { .fd = tensor_fd, .offset = (int32_t) weight_offset },
      .m = tokens, .k = input_dim, .n = output_cols },
    { .output = { .fd = tensor_fd, .offset = (int32_t) k_raw_offset },
      .activation = { .fd = tensor_fd, .offset = (int32_t) activation_offset },
      .weight = { .fd = tensor_fd, .offset = (int32_t) (weight_offset + (size_t) output_cols * input_dim * sizeof(__fp16)) },
      .m = tokens, .k = input_dim, .n = output_cols },
    { .output = { .fd = tensor_fd, .offset = (int32_t) v_raw_offset },
      .activation = { .fd = tensor_fd, .offset = (int32_t) activation_offset },
      .weight = { .fd = tensor_fd, .offset = (int32_t) (weight_offset + (size_t) output_cols * input_dim * sizeof(__fp16) * 2) },
      .m = tokens, .k = input_dim, .n = output_cols },
  };
  const struct RmsNormF32Params rms_params[2] = {
    { .dst = { .fd = tensor_fd, .offset = (int32_t) q_norm_offset },
      .src = { .fd = tensor_fd, .offset = (int32_t) q_raw_offset },
      .ne0 = HEAD_DIM, .ne1 = tokens * N_HEADS },
    { .dst = { .fd = tensor_fd, .offset = (int32_t) k_norm_offset },
      .src = { .fd = tensor_fd, .offset = (int32_t) k_raw_offset },
      .ne0 = HEAD_DIM, .ne1 = tokens * N_HEADS },
  };
  const size_t req_sizes[5] = {
    sizeof(req_hdr) + sizeof(matmul_req) + sizeof(matmul_params[0]),
    sizeof(req_hdr) + sizeof(matmul_req) + sizeof(matmul_params[1]),
    sizeof(req_hdr) + sizeof(matmul_req) + sizeof(matmul_params[2]),
    sizeof(req_hdr) + sizeof(rms_req) + sizeof(rms_params[0]),
    sizeof(req_hdr) + sizeof(rms_req) + sizeof(rms_params[1]),
  };
  msg->state.d = 0;
  msg->n_reqs = 5;
  msg->req_offsets[0] = (int32_t) message_header_size(msg);
  for (int i = 0; i < 5; ++i) {
    msg->req_offsets[i + 1] = msg->req_offsets[i] + (int32_t) req_sizes[i];
  }
  for (int i = 0; i < 5; ++i) {
    uint8_t *p = (uint8_t *) message_header_get_request_ptr(msg, i);
    memcpy(p, &req_hdr, sizeof(req_hdr));
    p += sizeof(req_hdr);
    if (i < 3) {
      memcpy(p, &matmul_req, sizeof(matmul_req));
      p += sizeof(matmul_req);
      memcpy(p, &matmul_params[i], sizeof(matmul_params[i]));
    } else {
      memcpy(p, &rms_req, sizeof(rms_req));
      p += sizeof(rms_req);
      memcpy(p, &rms_params[i - 3], sizeof(rms_params[i - 3]));
    }
  }

  const int64_t t0 = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }
  const int64_t channel_elapsed_us = get_time_us() - t0;
  int failed = 0;
  for (int i = 0; i < 5; ++i) {
    if (message_header_get_request_ptr(msg, i)->state != 0) {
      failed = 1;
    }
  }
  const int finite_outputs = isfinite(q_norm[0]) && isfinite(k_norm[0]) && isfinite(v_raw[0]);
  const int nonzero_outputs = q_norm[0] != 0.0f || k_norm[0] != 0.0f || v_raw[0] != 0.0f;
  fprintf(stderr,
          "qkv_separate_matmul_rms_microbench: tokens=%d input_dim=%d channel_us=%ld failed=%d q0=%g k0=%g v0=%g\n",
          tokens, input_dim, (long) channel_elapsed_us, failed, q_norm[0], k_norm[0], v_raw[0]);
  result = !failed && finite_outputs && nonzero_outputs ? 0 : -1;

cleanup:
  if (tensor_data) {
    free_shared_mem_buf(tensor_data, tensor_fd, tensor_size);
  }
  fprintf(stderr, result == 0 ? "%s passed\n" : "%s failed\n", __func__);
  return result;
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
  if (getenv("HTP_SKIP_OP_TESTS") == NULL) {
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
  } else {
    fprintf(stderr, "Skipping DSP op tests by HTP_SKIP_OP_TESTS\n");
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
  int qkv_bench_err = 0;
  if (env_enabled("HTP_QKV_SEPARATE_MATMUL_BENCH")) {
    qkv_bench_err = test_qkv_separate_projection_microbenchmark(chan);
    if (qkv_bench_err != 0) {
      fprintf(stderr, "QKV separate projection microbenchmark failed\n");
    }
  }
  if (env_enabled("HTP_QKV_MUL_BENCH")) {
    int fused_err = test_qkv_mul_channel(chan);
    if (fused_err != 0) {
      fprintf(stderr, "QKV MUL benchmark failed\n");
    }
    if (qkv_bench_err == 0) {
      qkv_bench_err = fused_err;
    }
  }
  if (env_enabled("HTP_QKV_ROPE_BENCH")) {
    int rope_err = test_qkv_rope_channel(chan);
    if (rope_err != 0) {
      fprintf(stderr, "QKV RoPE benchmark failed\n");
    }
    if (qkv_bench_err == 0) {
      qkv_bench_err = rope_err;
    }
  }
  if (env_enabled("HTP_QKV_PACKED_ATTN_BENCH")) {
    int packed_attn_err = test_packed_sparse_attention_channel(chan);
    if (packed_attn_err != 0) {
      fprintf(stderr, "Packed QKV sparse attention benchmark failed\n");
    }
    if (qkv_bench_err == 0) {
      qkv_bench_err = packed_attn_err;
    }
  }
  if (env_enabled("HTP_QKV_E2E_BENCH")) {
    int e2e_err = test_qkv_sparse_e2e_channel(chan);
    if (e2e_err != 0) {
      fprintf(stderr, "Packed QKV sparse end-to-end benchmark failed\n");
    }
    if (qkv_bench_err == 0) {
      qkv_bench_err = e2e_err;
    }
  }
  if (env_enabled("HTP_FLUX2_ATTN_BENCH")) {
    int flux2_err = test_flux2_attention_benchmark_channel(chan);
    if (flux2_err != 0) {
      fprintf(stderr, "Flux2 attention benchmark failed\n");
    }
    if (qkv_bench_err == 0) {
      qkv_bench_err = flux2_err;
    }
  }
  if (getenv("HTP_SKIP_BLOCK_SPARSE_TEST") == NULL) {
    err = test_block_sparse_legacy_channel(chan);
  } else {
    err = 0;
    fprintf(stderr, "Skipping block-sparse channel test by HTP_SKIP_BLOCK_SPARSE_TEST\n");
  }
  if (err == 0 && qkv_bench_err != 0) {
    err = qkv_bench_err;
  }
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
