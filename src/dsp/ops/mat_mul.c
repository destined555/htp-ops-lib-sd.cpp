#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/f16_gemm_plan.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"
#include "dsp/quants.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "op_reg.h"

// debug & profile
#include "HAP_farf.h"
#include "HAP_perf.h"

#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define ACTIVATION_AREA_SIZE (1 * 1024 * 1024)
#define OUTPUT_AREA_SIZE     (1 * 1024 * 1024)
#define SCRATCH_AREA_SIZE    (1 * 1024 * 1024)

static const __fp16 q4_0_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

static const __fp16 iq4_nl_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
  -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
  1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

static const uint32_t common_layout_vscatter_offsets_base[32] __attribute__((aligned(VLEN))) = {
  0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
};

static inline void hvx_fill_uh(void *p, uint16_t v, size_t size) {
  assert(size % VLEN == 0);
  assert(is_aligned(p, VLEN));
  const HVX_Vector v_fill = Q6_Vh_vsplat_R(v);
  HVX_Vector *out = (HVX_Vector *) p;
  for (size_t i = 0; i < size / VLEN; ++i) {
    *out++ = v_fill;
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

static inline void swap_ptr(void **p1, void **p2) {
  void *t = *p1;
  *p1     = *p2;
  *p2     = t;
}

static inline size_t get_super_block_size(enum ggml_type weight_type) {
  switch (weight_type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_IQ4_NL:
      return sizeof(my_block_q4_0);
    case GGML_TYPE_Q8_0:
      return sizeof(my_block_q8_0);
    default:
      return 0;
  }
}

static inline int dma_issue_load_from_ddr(dma_desc_1d_t *desc, void *vtcm_dst, const void *src, size_t size) {
  dma_wait_for_idle();

  desc->next       = 0;
  desc->length     = size;
  desc->type       = DMA_DESC_TYPE_1D;
  desc->src_bypass = 1;
  desc->dst_bypass = 0;
  desc->ordered    = 1;
  desc->dstate     = DMA_DESC_DSTATE_PENDING;
  desc->src        = (uint32_t) src;
  desc->dst        = (uint32_t) vtcm_dst;

  return dma_submit_one(desc);
}

static void find_chunk_size(size_t x_max, size_t y_max, size_t xy_max, size_t x_unit, size_t y_unit, size_t *x_out,
                            size_t *y_out) {
  int64_t best_xy = 0;
  size_t  best_x = 0, best_y = 0;

  for (size_t x = x_max; x > 0; x -= x_unit) {
    size_t  y  = smin(align_down(xy_max / x, y_unit), y_max);
    int64_t xy = x * y;
    if (best_xy < xy) {
      best_xy = xy;
      best_x = x, best_y = y;
    }
  }
  *x_out = best_x, *y_out = best_y;
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                                   int k_block, int k_stride) {
  assert(k_block % HMX_FP16_TILE_N_COLS == 0 && k_stride % HMX_FP16_TILE_N_COLS == 0);
  assert(VLEN == 32 * sizeof(float));

  for (int r = 0; r < n_rows; r += 2) {
    int prefetch_row_idx = r + 2;
    if (prefetch_row_idx < n_rows) {
      const float *prefetch_addr = src + prefetch_row_idx * k_stride;
      // NOTE(hzx): prefetch 2 rows at a time
      l2fetch(prefetch_addr, k_stride * sizeof(float), k_block * sizeof(float), 2, 0);
    }

    int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
    int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

    const bool next_row_valid = (r + 1) < n_rows;

    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * k_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * k_stride);
    for (int c = 0; c < k_block; c += 32) {
      HVX_Vector v0 = *pv_in0++;
      HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

      HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

      // compute output position
      int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
      int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  int           k;
  __fp16       *dst;
  const __fp16 *src;
} permuted_weight_transfer_fp16_task_state_t;

static void transfer_permuted_weight_fp16_task(__fp16 *restrict vtcm_dst, const __fp16 *restrict src, int k,
                                               int n_col_tiles) {
  // transfer logical K*(32*n_col_tiles) weight block, direct copy, no extra computation
  size_t size   = k * n_col_tiles * HMX_FP16_TILE_N_COLS * sizeof(__fp16);
  int    n_vecs = size / VLEN;

  const size_t PREFETCH_SIZE   = 4096;
  const int    PREFETCH_N_VECS = PREFETCH_SIZE / VLEN;

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        size_t prefetch_n_vecs = smin(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + PREFETCH_N_VECS, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    *pv_out++ = *pv_in++;
  }
}

static void transfer_permuted_weight_fp16_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_transfer_fp16_task_state_t *state = (permuted_weight_transfer_fp16_task_state_t *) data;

  int k = state->k;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    int           c        = chunk_idx * HMX_FP16_TILE_N_COLS;
    __fp16       *vtcm_dst = state->dst + c * k;
    const __fp16 *src      = state->src + c * k;
    transfer_permuted_weight_fp16_task(vtcm_dst, src, k, chunk_size);
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void transfer_permuted_weight_chunk_fp16(__fp16 *vtcm_dst, const __fp16 *src, int n_cols, int k) {
  // NOTE(hzx): weight matrix is already transposed. n_cols actually turns into n_rows
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  // The message-channel path passes weights through fastrpc-mapped RPCMEM. The DMA load path currently produces zero
  // F16 W16 outputs for this buffer source on device, while the vector copy path is correct.
  const bool use_dma = false;

  if (use_dma) {
    size_t size = n_cols * k * sizeof(__fp16);

    dma_desc_1d_t desc;
    dma_issue_load_from_ddr(&desc, vtcm_dst, src, size);
    dma_wait_for_idle();

    return;
  }

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = n_cols / HMX_FP16_TILE_N_COLS;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);
  // size_t n_chunks_per_task = 1;

  permuted_weight_transfer_fp16_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.k   = k;
  state.dst = vtcm_dst;
  state.src = src;

  worker_pool_job_t job;
  job.fptr = transfer_permuted_weight_fp16_worker_loop;
  job.dptr = &state;

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  // NOTE: n_tot_chunks = number of total super-blocks
  __fp16        *dst;
  const void    *src;
  enum ggml_type quant_type;
  bool           src_in_vtcm;
  int            k;  // NOTE(hzx): only used in non-pre-permuted (common) weight case
} permuted_weight_dequantize_qk_0_hvx_task_state_t;

#define EXPAND_QK_0_VEC_SCALES_COMPUTATION(blk, vs0_c, vs1_c, vs2_c, vs3_c) \
  do {                                                                      \
    __fp16 s0 = blk.scales[0];                                              \
    __fp16 s1 = blk.scales[1];                                              \
    __fp16 s2 = blk.scales[2];                                              \
    __fp16 s3 = blk.scales[3];                                              \
    __fp16 s4 = blk.scales[4];                                              \
    __fp16 s5 = blk.scales[5];                                              \
    __fp16 s6 = blk.scales[6];                                              \
    __fp16 s7 = blk.scales[7];                                              \
                                                                            \
    HVX_Vector vs0 = Q6_Vh_vsplat_R(fp16_to_bits(&s0));                     \
    HVX_Vector vs1 = Q6_Vh_vsplat_R(fp16_to_bits(&s1));                     \
    HVX_Vector vs2 = Q6_Vh_vsplat_R(fp16_to_bits(&s2));                     \
    HVX_Vector vs3 = Q6_Vh_vsplat_R(fp16_to_bits(&s3));                     \
    HVX_Vector vs4 = Q6_Vh_vsplat_R(fp16_to_bits(&s4));                     \
    HVX_Vector vs5 = Q6_Vh_vsplat_R(fp16_to_bits(&s5));                     \
    HVX_Vector vs6 = Q6_Vh_vsplat_R(fp16_to_bits(&s6));                     \
    HVX_Vector vs7 = Q6_Vh_vsplat_R(fp16_to_bits(&s7));                     \
                                                                            \
    vs0_c = Q6_V_valign_VVR(vs1, vs0, 64);                                  \
    vs1_c = Q6_V_valign_VVR(vs3, vs2, 64);                                  \
    vs2_c = Q6_V_valign_VVR(vs5, vs4, 64);                                  \
    vs3_c = Q6_V_valign_VVR(vs7, vs6, 64);                                  \
  } while (0)

static inline HVX_Vector dequantize_single_q4_0_group(const block_q4_0 *group, const HVX_Vector vlut_cvt) {
  HVX_Vector vq = vmemu(&(group->quants));
  HVX_Vector vs = vmemu(&(group->scale));

  HVX_Vector v_scales = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), vs, 0));

  HVX_Vector v_qs_lo = vq;
  HVX_Vector v_qs_hi = Q6_Vub_vlsr_VubR(vq, 4);

  // concat lo & hi --> 32 elements in a group
  HVX_Vector v_lo_rot = Q6_V_vror_VR(v_qs_lo, 16);
  HVX_Vector v_quants = Q6_V_vlalign_VVR(v_qs_hi, v_lo_rot, 16);

  // convert INT4 -> FP16
  HVX_VectorPair vp = Q6_Wh_vlut16_VbVhR_nomatch(v_quants, vlut_cvt, 0);

  HVX_Vector v_group_hf = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -2));

  // // convert INT4 -> FP16
  // HVX_VectorPair vp_q0 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_lo, vlut_cvt, 0);
  // HVX_VectorPair vp_q1 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_hi, vlut_cvt, 0);

  // HVX_Vector v0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp_q0), Q6_V_lo_W(vp_q0), -2));  // 16 valid elements
  // HVX_Vector v1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp_q1), Q6_V_hi_W(vp_q1), -2));  // 16 valid elements

  // // concat v0 & v1 --> 32 elements in a group
  // HVX_Vector v0_rot     = Q6_V_vror_VR(v0, 32);
  // HVX_Vector v_group_hf = Q6_V_vlalign_VVR(v1, v0_rot, 32);

  // dequantize: quants(FP16) * values(FP16)
  v_group_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_group_hf, v_scales));
  return v_group_hf;
}

static inline HVX_Vector dequantize_single_q8_0_group(const block_q8_0 *group) {
  HVX_Vector vq = vmemu(&(group->quants));
  HVX_Vector vs = vmemu(&(group->scale));

  HVX_Vector v_scales = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), vs, 0));

  HVX_Vector v0         = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vq));
  HVX_Vector v_group_hf = Q6_Vhf_equals_Vh(v0);

  // dequantize: quants(FP16) * values(FP16)
  v_group_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_group_hf, v_scales));
  return v_group_hf;
}

void dequantize_permuted_weight_q4_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const my_block_q4_0 *restrict src,
                                                      int n_blocks, bool src_in_vtcm, bool is_iq4_nl) {
  const int L2_PREFETCH_N_BLOCKS = 32;  // ~ 4K
  const int DC_PREFETCH_N_BLOCKS = 4;

  const bool no_group_coalesce = false;
  const bool no_dequantization = false;

  const HVX_Vector vlut_cvt = is_iq4_nl ? vmem(iq4_nl_to_fp16_lut) : vmem(q4_0_to_fp16_lut);

  static const uint8_t vlut_scales_idx_data[128] __attribute__((aligned(VLEN))) = {
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
  };
  const HVX_Vector vlut_scales_idx0 = vmem(vlut_scales_idx_data);
  const HVX_Vector vlut_scales_idx1 = Q6_Vb_vadd_VbVb(vlut_scales_idx0, Q6_Vb_vsplat_R(4));

  HVX_Vector *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_blocks; ++i) {
    if (!src_in_vtcm && false) {
      if (i % L2_PREFETCH_N_BLOCKS == 0) {
        int prefetch_idx = i + L2_PREFETCH_N_BLOCKS;
        if (prefetch_idx < n_blocks) {
          size_t prefetch_n_blocks = smin(n_blocks - prefetch_idx, L2_PREFETCH_N_BLOCKS);
          l2fetch(src + prefetch_idx, sizeof(my_block_q4_0), sizeof(my_block_q4_0), prefetch_n_blocks, 0);
        }
      }

      if (i + DC_PREFETCH_N_BLOCKS < n_blocks) {
        Q6_dcfetch_A((void *) &(src[i + DC_PREFETCH_N_BLOCKS].scales));
      }
    }

    if (no_group_coalesce) {
      const block_q4_0 *groups = (const block_q4_0 *) (src + i);

      HVX_Vector v_g0 = dequantize_single_q4_0_group(groups + 0, vlut_cvt);
      HVX_Vector v_g1 = dequantize_single_q4_0_group(groups + 1, vlut_cvt);
      HVX_Vector v_g2 = dequantize_single_q4_0_group(groups + 2, vlut_cvt);
      HVX_Vector v_g3 = dequantize_single_q4_0_group(groups + 3, vlut_cvt);
      HVX_Vector v_g4 = dequantize_single_q4_0_group(groups + 4, vlut_cvt);
      HVX_Vector v_g5 = dequantize_single_q4_0_group(groups + 5, vlut_cvt);
      HVX_Vector v_g6 = dequantize_single_q4_0_group(groups + 6, vlut_cvt);
      HVX_Vector v_g7 = dequantize_single_q4_0_group(groups + 7, vlut_cvt);

      HVX_Vector v_g0_rot = Q6_V_vror_VR(v_g0, 64);
      HVX_Vector v_g2_rot = Q6_V_vror_VR(v_g2, 64);
      HVX_Vector v_g4_rot = Q6_V_vror_VR(v_g4, 64);
      HVX_Vector v_g6_rot = Q6_V_vror_VR(v_g6, 64);

      HVX_Vector v0 = Q6_V_vlalign_VVR(v_g1, v_g0_rot, 64);
      HVX_Vector v1 = Q6_V_vlalign_VVR(v_g3, v_g2_rot, 64);
      HVX_Vector v2 = Q6_V_vlalign_VVR(v_g5, v_g4_rot, 64);
      HVX_Vector v3 = Q6_V_vlalign_VVR(v_g7, v_g6_rot, 64);

      *pv_out++ = v0;
      *pv_out++ = v1;
      *pv_out++ = v2;
      *pv_out++ = v3;
      continue;
    }

    HVX_Vector qs = vmemu(src[i].quants);

    if (no_dequantization) {
      *pv_out++ = qs;
      *pv_out++ = qs;
      *pv_out++ = qs;
      *pv_out++ = qs;
      continue;
    }

    HVX_Vector v_qs_lo = qs;  // no need to mask out high 4 bits in each byte since vlut will do that for us
    HVX_Vector v_qs_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair vp_q0 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_lo, vlut_cvt, 0);
    HVX_VectorPair vp_q1 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_hi, vlut_cvt, 0);

    // NOTE(hzx): the previous scalar->vector scales implementation is faster when src resides in DDR memory
    // HVX_Vector vs0_c, vs1_c, vs2_c, vs3_c;
    // EXPAND_QK_0_VEC_SCALES_COMPUTATION(src[i], vs0_c, vs1_c, vs2_c, vs3_c);

    HVX_Vector v_packed_scales = vmemu(src[i].scales);
    HVX_Vector vlut_scales     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_packed_scales));

    HVX_VectorPair vp_s0 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx0, vlut_scales, 0);
    HVX_VectorPair vp_s1 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx1, vlut_scales, 0);

    HVX_Vector vs0_c = Q6_V_lo_W(vp_s0), vs1_c = Q6_V_hi_W(vp_s0);
    HVX_Vector vs2_c = Q6_V_lo_W(vp_s1), vs3_c = Q6_V_hi_W(vp_s1);

    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_q0), vs0_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_q0), vs1_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_q1), vs2_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_q1), vs3_c));
  }
}

void dequantize_permuted_weight_q8_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const my_block_q8_0 *restrict src,
                                                      int n_blocks, bool src_in_vtcm) {
  const int L2_PREFETCH_N_BLOCKS = 16;  // ~ 4K
  const int DC_PREFETCH_N_BLOCKS = 4;

  const bool no_group_coalesce = false;
  const bool no_dequantization = false;

  static const uint8_t vlut_scales_idx_data[128] __attribute__((aligned(VLEN))) = {
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
  };
  const HVX_Vector vlut_scales_idx0 = vmem(vlut_scales_idx_data);
  const HVX_Vector vlut_scales_idx1 = Q6_Vb_vadd_VbVb(vlut_scales_idx0, Q6_Vb_vsplat_R(4));

  HVX_Vector *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_blocks; ++i) {
    if (!src_in_vtcm && false) {
      if (i % L2_PREFETCH_N_BLOCKS == 0) {
        int prefetch_idx = i + L2_PREFETCH_N_BLOCKS;
        if (prefetch_idx < n_blocks) {
          size_t prefetch_n_blocks = smin(n_blocks - prefetch_idx, L2_PREFETCH_N_BLOCKS);
          l2fetch(src + prefetch_idx, sizeof(my_block_q8_0), sizeof(my_block_q8_0), prefetch_n_blocks, 0);
        }
      }

      if (i + DC_PREFETCH_N_BLOCKS < n_blocks) {
        Q6_dcfetch_A((void *) &(src[i + DC_PREFETCH_N_BLOCKS].scales));
      }
    }

    if (no_group_coalesce) {
      const block_q8_0 *groups = (const block_q8_0 *) (src + i);

      HVX_Vector v_g0 = dequantize_single_q8_0_group(groups + 0);
      HVX_Vector v_g1 = dequantize_single_q8_0_group(groups + 1);
      HVX_Vector v_g2 = dequantize_single_q8_0_group(groups + 2);
      HVX_Vector v_g3 = dequantize_single_q8_0_group(groups + 3);
      HVX_Vector v_g4 = dequantize_single_q8_0_group(groups + 4);
      HVX_Vector v_g5 = dequantize_single_q8_0_group(groups + 5);
      HVX_Vector v_g6 = dequantize_single_q8_0_group(groups + 6);
      HVX_Vector v_g7 = dequantize_single_q8_0_group(groups + 7);

      HVX_Vector v_g0_rot = Q6_V_vror_VR(v_g0, 64);
      HVX_Vector v_g2_rot = Q6_V_vror_VR(v_g2, 64);
      HVX_Vector v_g4_rot = Q6_V_vror_VR(v_g4, 64);
      HVX_Vector v_g6_rot = Q6_V_vror_VR(v_g6, 64);

      HVX_Vector v0 = Q6_V_vlalign_VVR(v_g1, v_g0_rot, 64);
      HVX_Vector v1 = Q6_V_vlalign_VVR(v_g3, v_g2_rot, 64);
      HVX_Vector v2 = Q6_V_vlalign_VVR(v_g5, v_g4_rot, 64);
      HVX_Vector v3 = Q6_V_vlalign_VVR(v_g7, v_g6_rot, 64);

      *pv_out++ = v0;
      *pv_out++ = v1;
      *pv_out++ = v2;
      *pv_out++ = v3;
      continue;
    }

    HVX_Vector vq0 = vmemu(src[i].quants);
    HVX_Vector vq1 = vmemu(src[i].quants + VLEN);

    if (no_dequantization) {
      *pv_out++ = vq0;
      *pv_out++ = vq0;
      *pv_out++ = vq1;
      *pv_out++ = vq1;
      continue;
    }

    HVX_VectorPair vp0 = Q6_Wh_vunpack_Vb(vq0);
    HVX_VectorPair vp1 = Q6_Wh_vunpack_Vb(vq1);

    HVX_Vector v0 = Q6_Vhf_equals_Vh(Q6_V_lo_W(vp0));
    HVX_Vector v1 = Q6_Vhf_equals_Vh(Q6_V_hi_W(vp0));
    HVX_Vector v2 = Q6_Vhf_equals_Vh(Q6_V_lo_W(vp1));
    HVX_Vector v3 = Q6_Vhf_equals_Vh(Q6_V_hi_W(vp1));

    // HVX_Vector vs0_c, vs1_c, vs2_c, vs3_c;
    // EXPAND_QK_0_VEC_SCALES_COMPUTATION(src[i], vs0_c, vs1_c, vs2_c, vs3_c);

    HVX_Vector v_packed_scales = vmemu(src[i].scales);
    HVX_Vector vlut_scales     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_packed_scales));

    HVX_VectorPair vp_s0 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx0, vlut_scales, 0);
    HVX_VectorPair vp_s1 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx1, vlut_scales, 0);

    HVX_Vector vs0_c = Q6_V_lo_W(vp_s0), vs1_c = Q6_V_hi_W(vp_s0);
    HVX_Vector vs2_c = Q6_V_lo_W(vp_s1), vs3_c = Q6_V_hi_W(vp_s1);

    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, vs0_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v1, vs1_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v2, vs2_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v3, vs3_c));
  }
}

static void dequantize_permuted_weight_qk_0_to_fp16_hvx_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_dequantize_qk_0_hvx_task_state_t *state = (permuted_weight_dequantize_qk_0_hvx_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    __fp16 *vtcm_dst = state->dst + chunk_idx * QK_K;

    if (state->quant_type == GGML_TYPE_Q4_0 || state->quant_type == GGML_TYPE_IQ4_NL) {
      const my_block_q4_0 *src = ((const my_block_q4_0 *) state->src) + chunk_idx;
      dequantize_permuted_weight_q4_0_to_fp16_hvx_task(vtcm_dst, src, chunk_size, state->src_in_vtcm,
                                                       state->quant_type == GGML_TYPE_IQ4_NL);
    } else if (state->quant_type == GGML_TYPE_Q8_0) {
      const my_block_q8_0 *src = ((const my_block_q8_0 *) state->src) + chunk_idx;
      dequantize_permuted_weight_q8_0_to_fp16_hvx_task(vtcm_dst, src, chunk_size, state->src_in_vtcm);
    }
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

void dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(__fp16 *vtcm_dst, const void *src, int ne, int k,
                                                       enum ggml_type type, void *vtcm_scratch) {
  assert(ne % QK_K == 0);
  (void) k;

  const bool src_in_vtcm = true;

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = ne / QK_K;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);

  permuted_weight_dequantize_qk_0_hvx_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.dst         = vtcm_dst;
  state.src         = src_in_vtcm ? vtcm_scratch : src;
  state.quant_type  = type;
  state.src_in_vtcm = src_in_vtcm;

  worker_pool_job_t job;
  job.fptr = dequantize_permuted_weight_qk_0_to_fp16_hvx_worker_loop;
  job.dptr = &state;

  // int64_t t0 = HAP_perf_get_qtimer_count();

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  // int64_t e = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  // FARF(ALWAYS, "QK_0 dequantize: ne: %d time: %lld us", ne, e);
}

void dequantize_common_weight_q4_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const block_q4_0 *restrict src,
                                                    int start_group_idx, int end_group_idx, int k, bool src_in_vtcm,
                                                    bool is_iq4_nl) {
  assert(src_in_vtcm);

  const size_t GROUP_SIZE = QK_0;
  assert(GROUP_SIZE == HMX_FP16_TILE_N_ROWS);

  const size_t N_GROUPS_PER_SCALAR_COLUMN = k / GROUP_SIZE;
  const size_t N_GROUPS_PER_TILE_COLUMN   = N_GROUPS_PER_SCALAR_COLUMN * HMX_FP16_TILE_N_COLS;

  const HVX_Vector vlut_cvt       = is_iq4_nl ? vmem(iq4_nl_to_fp16_lut) : vmem(q4_0_to_fp16_lut);
  const HVX_Vector v_offsets_base = vmem(common_layout_vscatter_offsets_base);

  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

  for (int g_idx = start_group_idx; g_idx < end_group_idx; ++g_idx) {
    const block_q4_0 *group = src + g_idx;

    HVX_Vector v_group_hf = dequantize_single_q4_0_group(group, vlut_cvt);

    // prepare for scatter
    int i0 = g_idx / N_GROUPS_PER_TILE_COLUMN;
    int i1 = g_idx % N_GROUPS_PER_TILE_COLUMN;

    int gr = i1 % N_GROUPS_PER_SCALAR_COLUMN;
    int gc = i1 / N_GROUPS_PER_SCALAR_COLUMN;

    int     tile_idx  = i0 * (k / HMX_FP16_TILE_N_ROWS) + gr;
    __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

    HVX_Vector v_offsets_delta = Q6_V_vsplat_R(gc * 4);
    HVX_Vector v_offsets       = Q6_Vw_vadd_VwVw(v_offsets_base, v_offsets_delta);
    Q6_vscatter_QRMVwV(q_32_elems_mask, (size_t) tile_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_group_hf);
  }
}

void dequantize_common_weight_q8_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const block_q8_0 *restrict src,
                                                    int start_group_idx, int end_group_idx, int k, bool src_in_vtcm) {
  const size_t GROUP_SIZE = QK_0;
  assert(GROUP_SIZE == HMX_FP16_TILE_N_ROWS);

  const size_t N_GROUPS_PER_SCALAR_COLUMN = k / GROUP_SIZE;
  const size_t N_GROUPS_PER_TILE_COLUMN   = N_GROUPS_PER_SCALAR_COLUMN * HMX_FP16_TILE_N_COLS;

  const HVX_Vector v_offsets_base = vmem(common_layout_vscatter_offsets_base);

  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

  for (int g_idx = start_group_idx; g_idx < end_group_idx; ++g_idx) {
    const block_q8_0 *group = src + g_idx;

    HVX_Vector v_group_hf = dequantize_single_q8_0_group(group);

    // prepare for scatter
    int i0 = g_idx / N_GROUPS_PER_TILE_COLUMN;
    int i1 = g_idx % N_GROUPS_PER_TILE_COLUMN;

    int gr = i1 % N_GROUPS_PER_SCALAR_COLUMN;
    int gc = i1 / N_GROUPS_PER_SCALAR_COLUMN;

    int     tile_idx  = i0 * (k / HMX_FP16_TILE_N_ROWS) + gr;
    __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

    HVX_Vector v_offsets_delta = Q6_V_vsplat_R(gc * 4);
    HVX_Vector v_offsets       = Q6_Vw_vadd_VwVw(v_offsets_base, v_offsets_delta);
    Q6_vscatter_QRMVwV(q_32_elems_mask, (size_t) tile_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_group_hf);
  }
}

static void dequantize_common_weight_chunk_qk_0_to_fp16_hvx_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_dequantize_qk_0_hvx_task_state_t *state = (permuted_weight_dequantize_qk_0_hvx_task_state_t *) data;

  const int     k        = state->k;
  __fp16 *const vtcm_dst = state->dst;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    size_t start_idx = task_id * state->n_chunks_per_task;
    size_t end_idx   = smin(start_idx + state->n_chunks_per_task, state->n_tot_chunks);

    if (state->quant_type == GGML_TYPE_Q4_0 || state->quant_type == GGML_TYPE_IQ4_NL) {
      const block_q4_0 *src = (const block_q4_0 *) state->src;
      dequantize_common_weight_q4_0_to_fp16_hvx_task(vtcm_dst, src, start_idx, end_idx, k, state->src_in_vtcm,
                                                     state->quant_type == GGML_TYPE_IQ4_NL);
    } else if (state->quant_type == GGML_TYPE_Q8_0) {
      const block_q8_0 *src = (const block_q8_0 *) state->src;
      dequantize_common_weight_q8_0_to_fp16_hvx_task(vtcm_dst, src, start_idx, end_idx, k, state->src_in_vtcm);
    }
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

void dequantize_common_weight_chunk_qk_0_to_fp16_hvx(__fp16 *vtcm_dst, const void *src, int ne, int k,
                                                     enum ggml_type type, void *vtcm_scratch) {
  assert(ne % QK_0 == 0);
  assert(k % QK_0 == 0);

  const bool src_in_vtcm = true;

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = ne / QK_0;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);

  // NOTE: reuse the task state type for now
  permuted_weight_dequantize_qk_0_hvx_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.dst         = vtcm_dst;
  state.src         = src_in_vtcm ? vtcm_scratch : src;
  state.quant_type  = type;
  state.src_in_vtcm = src_in_vtcm;
  state.k           = k;

  worker_pool_job_t job;
  job.fptr = dequantize_common_weight_chunk_qk_0_to_fp16_hvx_worker_loop;
  job.dptr = &state;

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

static void core_dot_chunk_fp16(__fp16 *output, const __fp16 *activation, const __fp16 *weight, const __fp16 *scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
  hmx_unit_acquire();

  asm volatile("mxclracc.hf");
  hmx_set_output_scales(scales);

  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < n_col_tiles; ++c) {
      const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < n_dot_tiles; k += 32) {
        int    offset  = k * HMX_FP16_TILE_N_ELMS;
        size_t n_tiles = smin(n_dot_tiles - k, 32);
        hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
      }

      __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(out_tile);
    }
  }

  hmx_unit_release();
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, int n_rows,
                                               int n_cols, int n) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;
    int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      int c0 = c / HMX_FP16_TILE_N_COLS;

      const __fp16 *tile = vtcm_src + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;

      HVX_Vector v_src = ((const HVX_Vector *) tile)[r1 / 2];

      HVX_VectorPair vp = hvx_my_vhf_to_wsf(v_src);

      HVX_Vector *pv_out0 = (HVX_Vector *) (dst + (r * n + c + 0));
      HVX_Vector *pv_out1 = (HVX_Vector *) (dst + (r * n + c + n));  // next row in global memory

      *pv_out0 = Q6_V_lo_W(vp);
      if (r + 1 < n_rows) {
        *pv_out1 = Q6_V_hi_W(vp);
      }
    }
  }
}

// Apply the epilogue while the HMX tile is still in VTCM; no DDR intermediate.
static inline HVX_Vector epilogue_mul(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));
}
static inline HVX_Vector epilogue_const(float x) {
  union { float f; int i; } u = { .f=x };
  return Q6_V_vsplat_R(u.i);
}
static HVX_INLINE_ALWAYS HVX_Vector epilogue_activation(HVX_Vector x, int mode) {
  if (mode != 1 && mode != 2 && mode != 4 && mode != 5) return x;
  // GELU(x)=x*sigmoid(2*sqrt(2/pi)*(x+0.044715*x^3)).
  HVX_Vector z = x;
  if (mode == 1) {
    HVX_Vector cubic = epilogue_mul(epilogue_mul(x,x),x);
    z = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x,epilogue_mul(cubic,epilogue_const(.044715f))));
  }
  z = epilogue_mul(z,epilogue_const(mode == 1 ? 1.5957691216057308f : (mode==4 || mode==5) ? 1.f : 1.702f));
  // exp2 helper requires non-positive arguments. Stable sigmoid avoids overflow.
  HVX_Vector absz = Q6_V_vand_VV(z,Q6_V_vsplat_R(0x7fffffff));
  HVX_Vector neg = epilogue_mul(absz,epilogue_const(-1.4426950408889634f));
  neg = Q6_Vsf_vmax_VsfVsf(neg,epilogue_const(-126.f));
  HVX_Vector e = hvx_my_exp2_vsf(neg);
  HVX_Vector denom = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(e,epilogue_const(1.f)));
  // denom is in [1,2]. Three Newton steps from (1.5 - .5*denom)
  // give float precision without a general-purpose reciprocal LUT/gather.
  HVX_Vector inv_q = Q6_Vqf32_vsub_VsfVsf(epilogue_const(1.5f),epilogue_mul(denom,epilogue_const(.5f)));
  HVX_Vector denom_q=Q6_Vqf32_vadd_VsfVsf(denom,Q6_V_vzero());
  HVX_Vector two_q=Q6_Vqf32_vadd_VsfVsf(epilogue_const(2.f),Q6_V_vzero());
  for (int step=0;step<3;++step) {
    HVX_Vector product=Q6_Vqf32_vmpy_Vqf32Vqf32(inv_q,denom_q);
    HVX_Vector correction=Q6_Vqf32_vsub_Vqf32Vqf32(two_q,product);
    inv_q=Q6_Vqf32_vmpy_Vqf32Vqf32(inv_q,correction);
  }
  HVX_Vector inv = Q6_Vsf_equals_Vqf32(inv_q);
  if(mode==5) {
    // Bound the sigmoid to F32 precision between reciprocal iterations.
    // Keeping all iterations in qfloat loses accuracy near denominator=1
    // for wide-range DiT gates; leave legacy epilogues unchanged.
    inv=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(epilogue_const(1.5f),epilogue_mul(denom,epilogue_const(.5f))));
    for(int step=0;step<4;++step) {
      HVX_Vector correction=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(epilogue_const(2.f),epilogue_mul(denom,inv)));
      inv=epilogue_mul(inv,correction);
    }
  }
  HVX_Vector sig = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(),z),epilogue_mul(e,inv),inv);
  return epilogue_mul(x,sig);
}

// Scatter only within VTCM, then write contiguous spatial vectors to DDR.
// This avoids one DDR cache-line write for every output channel/position pair.
static void transfer_vae_output(float *dst, const __fp16 *src, const float *bias,
    int rows, int cols, int m, int mr, int nc, int channels, float *scratch) {
  static const int32_t offsets[32] __attribute__((aligned(128))) = {
    0,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,
    2048,2176,2304,2432,2560,2688,2816,2944,3072,3200,3328,3456,3584,3712,3840,3968
  };
  const HVX_Vector index=vmem(offsets);
  for (int r=0;r<rows;r+=32) {
    int valid=smin(32,rows-r);
    for (int c=0;c<cols && nc+c<channels;c+=32) {
      const __fp16 *tile=src+((r/32)*(cols/32)+c/32)*1024;
      for (int rr=0;rr<valid;rr+=2) {
        HVX_VectorPair v=hvx_my_vhf_to_wsf(((const HVX_Vector *)tile)[rr/2]);
        Q6_vscatter_RMVwV((size_t)(scratch+rr),4095-rr*4,index,Q6_V_lo_W(v));
        if (rr+1<valid) Q6_vscatter_RMVwV((size_t)(scratch+rr+1),4095-(rr+1)*4,index,Q6_V_hi_W(v));
      }
      asm volatile("vmem(%0 + #0):scatter_release" :: "r"(scratch) : "memory");
      for (int ch=0;ch<32 && nc+c+ch<channels;++ch) {
        HVX_Vector v=vmem(scratch+ch*32);
        if (bias) v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,epilogue_const(bias[nc+c+ch])));
        // memcpy also handles non-vector-aligned chunk and channel strides.
        float *out=dst+(size_t)(nc+c+ch)*m+mr+r;
        if (valid==32) {
          vmemu(out)=v;
        } else {
          float values[32] __attribute__((aligned(128)));
          vmem(values)=v;
          memcpy(out,values,valid*sizeof(float));
        }
      }
    }
  }
}

static void transfer_fused_output(float *dst, const __fp16 *src, const float *bias,
    int rows, int cols, int m, int n, int mr, int nc, int channels, int mode, float *scratch) {
  if (mode==3) { transfer_vae_output(dst,src,bias,rows,cols,m,mr,nc,channels,scratch); return; }
  int tiles = cols/32;
  for (int r=0;r<rows;r+=2) {
    for (int c=0;c<cols;c+=32) {
      const __fp16 *tile=src+((r/32)*tiles+c/32)*1024;
      HVX_VectorPair pair=hvx_my_vhf_to_wsf(((const HVX_Vector *)tile)[(r%32)/2]);
      for (int rr=0;rr<2 && r+rr<rows;++rr) {
        HVX_Vector v=rr ? Q6_V_hi_W(pair) : Q6_V_lo_W(pair);
        // Text channels are multiples of 32; the cropped VAE path above
        // handles its own tails. Avoid a per-vector stack copy here.
        if (bias) v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,vmemu(bias+nc+c)));
        v=epilogue_activation(v,mode);
        vmem(dst+(size_t)(mr+r+rr)*n+nc+c)=v;
      }
    }
  }
}

typedef struct {
  worker_synctoken_t sync;
  unsigned int next;
  float *dst, *scratch;
  const __fp16 *src;
  const float *bias;
  int rows, cols, m, n, mr, nc, channels, mode, tasks;
} epilogue_job_t;

static void epilogue_worker(void *data, int worker_index) {
  epilogue_job_t *s=(epilogue_job_t *)data;
  if (worker_index<0 || (unsigned int)worker_index>=num_workers) {
    worker_pool_synctoken_jobdone(&s->sync);
    return;
  }
  for (;;) {
    unsigned int task=worker_pool_atomic_inc_return(&s->next)-1;
    if (task>=(unsigned int)s->tasks) break;
    int rt=task/(s->cols/32), ct=task%(s->cols/32);
    int rows=smin(32,s->rows-rt*32);
    const __fp16 *tile=s->src+(rt*(s->cols/32)+ct)*1024;
    transfer_fused_output(s->dst,tile,s->bias,rows,32,s->m,s->n,
        s->mr+rt*32,s->nc+ct*32,s->channels,s->mode,
        s->scratch ? s->scratch+(size_t)worker_index*1024 : NULL);
  }
  worker_pool_synctoken_jobdone(&s->sync);
}

static int transfer_parallel_epilogue(float *dst, const __fp16 *src, const float *bias,
    int rows, int cols, int m, int n, int mr, int nc, int channels, int mode, float *scratch) {
  int tasks=ceil_div(rows,32)*(cols/32);
  int workers=smin(num_workers,num_hvx128_contexts);
  // Short text projections lose more to worker wakeups than they gain from
  // parallel epilogues. Reserve the worker pool for spatial VAE output.
  if (mode!=3 || tasks<16 || workers<2) {
    transfer_fused_output(dst,src,bias,rows,cols,m,n,mr,nc,channels,mode,scratch);
    return 0;
  }
  epilogue_job_t state={.next=0,.dst=dst,.scratch=scratch,.src=src,.bias=bias,
    .rows=rows,.cols=cols,.m=m,.n=n,.mr=mr,.nc=nc,.channels=channels,.mode=mode,.tasks=tasks};
  worker_pool_synctoken_init(&state.sync,workers);
  worker_pool_job_t job={.fptr=epilogue_worker,.dptr=&state};
  int failed=0;
  for (int i=0;i<workers;++i) {
    if (worker_pool_submit(NULL,job)!=AEE_SUCCESS) {
      worker_pool_synctoken_jobdone(&state.sync);
      failed=1;
    }
  }
  worker_pool_synctoken_wait(&state.sync);
  return failed ? -1 : 0;
}

int hmx_mat_mul_f16_epilogue(float *restrict dst, const float *restrict activation,
                                const __fp16 *restrict permuted_weight, const float *bias, int m, int k, int n, int channels, int mode) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -10;
  }
  if (m < 1 || k < 1 || n < 1 || channels < 1 || channels > n || mode < 0 || mode > 3 ||
      (mode != 3 && channels != n) || k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -11;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -12;
  }

  const size_t scratch_size = mode == 3 ? 4096 * smax(1, num_workers) : 0;
  f16_gemm_plan_t plan;
  if (!f16_gemm_make_plan(m, k, n, vtcm_manager_get_vtcm_size(), 256 + scratch_size, &plan)) {
    return -14;
  }

  // VTCM layout: weight | activation | output | scales
  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, plan.weight_bytes);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, plan.activation_bytes);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, plan.output_bytes);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  float *transpose_scratch = mode==3 ? (float *)vtcm_seq_alloc(&vtcm_ptr,scratch_size) : NULL;

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  const size_t m_chunk_n_rows = plan.rows, n_chunk_n_cols = plan.cols;

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
    // transfer activation matrix chunk into VTCM
    size_t n_rows = smin(m - mr, m_chunk_n_rows);

    // int64_t act_t0 = HAP_perf_get_qtimer_count();
    {
      const float *activation_chunk = activation + mr * k;
      transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
    }
    // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

    // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

    for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
      size_t n_cols = smin(n - nc, n_chunk_n_cols);

      // int64_t wei_t0 = HAP_perf_get_qtimer_count();
      {
        const __fp16 *permuted_weight_chunk = permuted_weight + nc * k;
        transfer_permuted_weight_chunk_fp16(vtcm_weight, permuted_weight_chunk, n_cols, k);
      }
      // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

      // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

      // int64_t core_t0 = HAP_perf_get_qtimer_count();
      {
        const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
        const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
        core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
      }
      // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

      // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

      // int64_t out_t0 = HAP_perf_get_qtimer_count();
      {
        if (!bias && mode == 0) {
          transfer_output_chunk_fp16_to_fp32(dst + mr*n + nc, vtcm_output, n_rows, n_cols, n);
        } else {
          if (transfer_parallel_epilogue(dst, vtcm_output, bias, n_rows, n_cols,
              m, n, mr, nc, channels, mode, transpose_scratch)!=0) return -13;
        }
      }
      // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

      // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
    }
  }

  static int matmul_debug_once = 0;
  if (m == 768 && k == 3072 && (n == 9216 || n == 27648) &&
      __atomic_exchange_n(&matmul_debug_once, 1, __ATOMIC_ACQ_REL) == 0) {
    FARF(ALWAYS, "QKV_DEBUG matmul_raw q0=%g q1=%g q2=%g q3=%g q4=%g q5=%g q6=%g q7=%g",
         dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
  }

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n * sizeof(__fp16);
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

  return 0;
}

int hmx_mat_mul_permuted_w16a32(float *restrict dst, const float *restrict activation,
                                const __fp16 *restrict weight, int m, int k, int n) {
  return hmx_mat_mul_f16_epilogue(dst,activation,weight,NULL,m,k,n,n,0);
}

static void qkv_pack_q_tile(__fp16 *dst, const float *src, int n_rows, int head_dim) {
  assert(src != NULL && dst != NULL);
  assert(n_rows > 0 && n_rows <= HMX_FP16_TILE_N_ROWS);
  assert(head_dim > 0 && head_dim % 32 == 0);
  if (n_rows < HMX_FP16_TILE_N_ROWS) {
    hvx_fill_uh(dst, 0, (size_t) (head_dim / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_ELMS *
                         sizeof(__fp16));
  }
  for (int r = 0; r < n_rows; r += 2) {
    const bool next_row_valid = r + 1 < n_rows;
    const HVX_Vector *pv0 = (const HVX_Vector *) (src + (size_t) r * head_dim);
    const HVX_Vector *pv1 = next_row_valid ?
      (const HVX_Vector *) (src + (size_t) (r + 1) * head_dim) : pv0;
    for (int d = 0; d < head_dim / 32; ++d) {
      const HVX_Vector v0 = *pv0++;
      const HVX_Vector v1 = next_row_valid ? *pv1++ : Q6_V_vzero();
      HVX_Vector *out_tile =
        (HVX_Vector *) (dst + (size_t) d * HMX_FP16_TILE_N_ELMS);
      out_tile[r / 2] = hvx_my_wsf_to_vhf(v1, v0);
    }
  }
}

static inline bool qkv_supported_head_dim(int head_dim) {
  return htp_packed_head_dim_supported(head_dim) != 0;
}

static void qkv_pack_k_tile(__fp16 *dst, const __fp16 *src, int n_rows, int head_dim) {
  static const int32_t transpose_vscatter_indices_base[32] __attribute__((aligned(VLEN))) = {
    0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
    2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968,
  };
  assert(src != NULL && dst != NULL);
  assert(n_rows > 0 && n_rows <= HMX_FP16_TILE_N_COLS);
  assert(head_dim > 0 && head_dim % 64 == 0);
  if (n_rows < HMX_FP16_TILE_N_COLS) {
    hvx_fill_uh(dst, 0, (size_t) head_dim * HMX_FP16_TILE_N_COLS * sizeof(__fp16));
  }
  HVX_Vector v_offsets = vmem(transpose_vscatter_indices_base);
  const HVX_Vector v_step = Q6_V_vsplat_R(4);
  for (int r = 0; r < n_rows; ++r) {
    const HVX_Vector *pv_in = (const HVX_Vector *) (src + (size_t) r * head_dim);
#pragma unroll
    for (int d = 0; d < head_dim / 64; ++d) {
      __fp16 *out_dual_tile = dst + (size_t) d * HMX_FP16_TILE_N_ELMS * 2;
      Q6_vscatter_RMVwV((size_t) out_dual_tile, HMX_FP16_TILE_SIZE * 2 - 1,
                        v_offsets, *pv_in++);
    }
    v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
  }
}

static void qkv_pack_v_tile(__fp16 *dst, const __fp16 *src, int token_tile, int total_token_tiles,
                            int head_dim) {
  assert(src != NULL && dst != NULL);
  assert(token_tile >= 0 && token_tile < total_token_tiles);
  assert(head_dim > 0 && head_dim % 64 == 0);
  for (int r = 0; r < HMX_FP16_TILE_N_COLS; r += 2) {
    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (size_t) r * head_dim);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (size_t) (r + 1) * head_dim);
#pragma unroll
    for (int c = 0; c < head_dim; c += 64) {
      const HVX_VectorPair vp = Q6_W_vshuff_VVR(*pv_in1++, *pv_in0++, -2);
      const int d_tile_idx = c / HMX_FP16_TILE_N_COLS;
      __fp16 *tile_base0 =
        dst + ((size_t) (d_tile_idx + 0) * total_token_tiles + token_tile) * HMX_FP16_TILE_N_ELMS;
      __fp16 *tile_base1 =
        dst + ((size_t) (d_tile_idx + 1) * total_token_tiles + token_tile) * HMX_FP16_TILE_N_ELMS;
      HVX_Vector *pv_out0 = ((HVX_Vector *) tile_base0) + r / 2;
      HVX_Vector *pv_out1 = ((HVX_Vector *) tile_base1) + r / 2;
      *pv_out0 = Q6_V_lo_W(vp);
      *pv_out1 = Q6_V_hi_W(vp);
    }
  }
}

static inline float qkv_reduce_sum(HVX_Vector v_sum) {
  const HVX_Vector v_zero = Q6_V_vzero();
  for (int shift = 64; shift >= 4; shift >>= 1) {
    v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_V_vlalign_VVR(v_sum, v_zero, shift));
  }
  float tmp[32] __attribute__((aligned(VLEN)));
  vmem(tmp) = Q6_Vsf_equals_Vqf32(v_sum);
  return tmp[31];
}

static inline HVX_Vector qkv_rsqrt_vsf(HVX_Vector x) {
  // Three Newton iterations, matching the Qualcomm qhmath HVX rsqrt implementation.
  const HVX_Vector v_zero = Q6_V_vzero();
  const HVX_Vector v_half = Q6_V_vsplat_R(0x3f000000);
  const HVX_Vector v_three_halves_qf32 = Q6_V_vsplat_R(0x6000007f);
  const HVX_Vector v_magic = Q6_V_vsplat_R(0x5f3759df);
  const HVX_Vector x_half = Q6_Vqf32_vadd_Vqf32Vsf(
    Q6_Vqf32_vmpy_VsfVsf(x, v_half), v_zero);
  HVX_Vector y = Q6_Vw_vsub_VwVw(v_magic, Q6_Vw_vasr_VwR(x, 1));
  HVX_Vector y_squared = Q6_Vqf32_vadd_Vqf32Vsf(
    Q6_Vqf32_vmpy_VsfVsf(y, y), v_zero);
  HVX_Vector correction = Q6_Vqf32_vsub_Vqf32Vqf32(
    v_three_halves_qf32, Q6_Vqf32_vmpy_Vqf32Vqf32(x_half, y_squared));
  HVX_Vector next = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(correction));

  y = Q6_Vqf32_vadd_Vqf32Vsf(next, v_zero);
  y_squared = Q6_Vqf32_vadd_Vqf32Vsf(
    Q6_Vqf32_vmpy_Vqf32Vqf32(y, y), v_zero);
  correction = Q6_Vqf32_vsub_Vqf32Vqf32(
    v_three_halves_qf32, Q6_Vqf32_vmpy_Vqf32Vqf32(x_half, y_squared));
  next = Q6_Vqf32_vmpy_Vqf32Vqf32(y, correction);

  y = Q6_Vqf32_vadd_Vqf32Vsf(next, v_zero);
  y_squared = Q6_Vqf32_vadd_Vqf32Vsf(
    Q6_Vqf32_vmpy_Vqf32Vqf32(y, y), v_zero);
  correction = Q6_Vqf32_vsub_Vqf32Vqf32(
    v_three_halves_qf32, Q6_Vqf32_vmpy_Vqf32Vqf32(x_half, y_squared));
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(y, correction));
}

static void qkv_rms_bias(float *rows, int n_rows, int head_dim, const float *bias, const float *gamma) {
  assert(qkv_supported_head_dim(head_dim));
  assert(is_aligned(rows, VLEN) && is_aligned(bias, VLEN) && is_aligned(gamma, VLEN));
  const HVX_Vector *bias_vec = (const HVX_Vector *) bias;
  const HVX_Vector *gamma_vec = (const HVX_Vector *) gamma;
  float inverse_scales[32] __attribute__((aligned(VLEN)));
  for (int r = 0; r < n_rows; ++r) {
    float *row = rows + (size_t) r * head_dim;
    HVX_Vector v_sum = Q6_V_vzero();
    HVX_Vector *row_vec = (HVX_Vector *) row;
    for (int d = 0; d < head_dim / 64; ++d) {
      HVX_Vector v0 = row_vec[d * 2 + 0];
      HVX_Vector v1 = row_vec[d * 2 + 1];
      v0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v0, bias_vec[d * 2 + 0]));
      v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v1, bias_vec[d * 2 + 1]));
      v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vmpy_VsfVsf(v0, v0));
      v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vmpy_VsfVsf(v1, v1));
      row_vec[d * 2 + 0] = v0;
      row_vec[d * 2 + 1] = v1;
    }
    inverse_scales[r] = qkv_reduce_sum(v_sum) / (float) head_dim + 1.0e-6f;
  }
  for (int r = n_rows; r < 32; ++r) {
    inverse_scales[r] = 1.0f;
  }
  vmem(inverse_scales) = qkv_rsqrt_vsf(vmem(inverse_scales));
  for (int r = 0; r < n_rows; ++r) {
    float *row = rows + (size_t) r * head_dim;
    const HVX_Vector v_inv = Q6_V_vsplat_R(*(const int32_t *) &inverse_scales[r]);
    HVX_Vector *row_vec = (HVX_Vector *) row;
    for (int d = 0; d < head_dim / 64; ++d) {
      HVX_Vector v0 = row_vec[d * 2 + 0];
      HVX_Vector v1 = row_vec[d * 2 + 1];
      v0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, v_inv));
      v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, v_inv));
      v0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, gamma_vec[d * 2 + 0]));
      v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, gamma_vec[d * 2 + 1]));
      row_vec[d * 2 + 0] = v0;
      row_vec[d * 2 + 1] = v1;
    }
  }
}

static void qkv_apply_rope(float *rows, int n_rows, int head_dim,
                           const float *pe, int token_offset, int token_count) {
  if (pe == NULL || token_offset < 0 || token_count <= 0) {
    return;
  }
  const size_t pair_stride = (size_t) (head_dim / 2) * 4;
  for (int r = 0; r < n_rows; ++r) {
    const int token = token_offset + r;
    if (token >= token_count) {
      break;
    }
    const float *token_pe = pe + (size_t) token * pair_stride;
    float *row = rows + (size_t) r * head_dim;
    for (int d = 0; d < head_dim / 2; ++d) {
      const float x0 = row[2 * d + 0];
      const float x1 = row[2 * d + 1];
      const float *matrix = token_pe + (size_t) d * 4;
      row[2 * d + 0] = x0 * matrix[0] + x1 * matrix[1];
      row[2 * d + 1] = x0 * matrix[2] + x1 * matrix[3];
    }
  }
}

static void qkv_cast_f32_to_f16(__fp16 *dst, const float *src, int n_rows, int head_dim) {
  assert(qkv_supported_head_dim(head_dim));
  assert(is_aligned(dst, VLEN) && is_aligned(src, VLEN));
  for (int r = 0; r < HMX_FP16_TILE_N_ROWS; ++r) {
    HVX_Vector *out = (HVX_Vector *) (dst + (size_t) r * head_dim);
    for (int d = 0; d < head_dim / 64; ++d) {
      HVX_Vector value = Q6_V_vzero();
      if (r < n_rows) {
        const HVX_Vector *in = (const HVX_Vector *) (src + (size_t) r * head_dim);
        value = Q6_Vh_vdeal_Vh(hvx_my_wsf_to_vhf(in[d * 2 + 1], in[d * 2 + 0]));
      }
      out[d] = value;
    }
  }
}

static void qkv_bias_cast(__fp16 *dst, const float *src, int n_rows, int head_dim, const float *bias) {
  assert(qkv_supported_head_dim(head_dim));
  assert(is_aligned(dst, VLEN) && is_aligned(src, VLEN) && is_aligned(bias, VLEN));
  const HVX_Vector *bias_vec = (const HVX_Vector *) bias;
  for (int r = 0; r < HMX_FP16_TILE_N_ROWS; ++r) {
    HVX_Vector *out = (HVX_Vector *) (dst + (size_t) r * head_dim);
    for (int d = 0; d < head_dim / 64; ++d) {
      HVX_Vector value = Q6_V_vzero();
      if (r < n_rows) {
        const HVX_Vector *in = (const HVX_Vector *) (src + (size_t) r * head_dim);
        const HVX_Vector v0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(in[d * 2 + 0], bias_vec[d * 2 + 0]));
        const HVX_Vector v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(in[d * 2 + 1], bias_vec[d * 2 + 1]));
        value = Q6_Vh_vdeal_Vh(hvx_my_wsf_to_vhf(v1, v0));
      }
      out[d] = value;
    }
  }
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  __fp16       *q;
  __fp16       *k_out;
  __fp16       *v;
  const __fp16 *vtcm_output;
  const float  *bias;
  const float  *q_norm;
  const float  *k_norm;
  float        *raw_scratch;
  __fp16       *raw16_scratch;
  __fp16       *pack_scratch;
  size_t        mr;
  size_t        nc;
  size_t        head_tile_elems;
  size_t        pack_tile_bytes;
  int           n_rows;
  int           n_row_tiles;
  int           n_col_tiles;
  int           n_cols;
  int           q_cols;
  int           token_tiles;
  int           output_token_offset;
  int           head_dim;
  const float  *rope_pe;
  int           rope_token_offset;
  int           rope_token_count;
} qkv_postprocess_task_state_t;

static void qkv_postprocess_worker(void *data, int worker_index) {
  qkv_postprocess_task_state_t *state = (qkv_postprocess_task_state_t *) data;
  if (worker_index < 0 || (unsigned int) worker_index >= num_workers) {
    worker_pool_synctoken_jobdone(&state->sync_ctx);
    return;
  }
  const size_t raw_stride = (size_t) HMX_FP16_TILE_N_ROWS * (size_t) state->head_dim;
  float *raw = state->raw_scratch + (size_t) worker_index * raw_stride;
  __fp16 *raw16 = state->raw16_scratch + (size_t) worker_index * raw_stride;
  __fp16 *vtcm_pack = state->pack_scratch +
    (size_t) worker_index * state->pack_tile_bytes / sizeof(__fp16);
  const int heads_in_chunk = state->n_cols / state->head_dim;

  while (1) {
    const unsigned int task_id = worker_pool_atomic_inc_return(&state->task_id) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }
    const int row_tile = task_id / heads_in_chunk;
    const int col = (task_id % heads_in_chunk) * state->head_dim;
    const int rows_in_tile = smin(
      state->n_rows - row_tile * HMX_FP16_TILE_N_ROWS, HMX_FP16_TILE_N_ROWS);
    const int token_tile = (state->output_token_offset + (int) state->mr) /
                             HMX_FP16_TILE_N_ROWS + row_tile;
    const int global_col = (int) state->nc + col;
    const bool is_q = global_col < state->q_cols;
    const bool is_k = global_col >= state->q_cols && global_col < 2 * state->q_cols;
    const int local_col = is_q ? global_col :
      (is_k ? global_col - state->q_cols : global_col - 2 * state->q_cols);
    const int head = local_col / state->head_dim;
    const __fp16 *output_tile = state->vtcm_output +
      ((size_t) row_tile * state->n_col_tiles +
       (size_t) col / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_ELMS;
    transfer_output_chunk_fp16_to_fp32(
      raw, output_tile, rows_in_tile, state->head_dim, state->head_dim);
    static int qkv_debug_once = 0;
    if (state->mr == 0 && state->nc == 0 &&
        __atomic_exchange_n(&qkv_debug_once, 1, __ATOMIC_ACQ_REL) == 0) {
      FARF(ALWAYS, "QKV_DEBUG packed_raw q0=%g q1=%g q2=%g q3=%g q4=%g q5=%g q6=%g q7=%g",
           raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
    }
    const float *segment_bias = state->bias + global_col;

    if (is_q || is_k) {
      qkv_rms_bias(raw, rows_in_tile, state->head_dim, segment_bias,
                   is_q ? state->q_norm : state->k_norm);
      qkv_apply_rope(raw, rows_in_tile, state->head_dim, state->rope_pe,
                     state->rope_token_offset + (int) state->mr + row_tile * HMX_FP16_TILE_N_ROWS,
                     state->rope_token_count);
      if (is_k) {
        qkv_cast_f32_to_f16(raw16, raw, rows_in_tile, state->head_dim);
      }
      __fp16 *dst = (is_q ? state->q : state->k_out) +
        (size_t) head * state->head_tile_elems +
        (size_t) token_tile * state->head_dim * HMX_FP16_TILE_N_ROWS;
      if (is_q) {
        qkv_pack_q_tile(vtcm_pack, raw, rows_in_tile, state->head_dim);
      } else {
        qkv_pack_k_tile(vtcm_pack, raw16, rows_in_tile, state->head_dim);
      }
      hvx_copy(dst, vtcm_pack, state->pack_tile_bytes);
    } else {
      qkv_bias_cast(raw16, raw, rows_in_tile, state->head_dim, segment_bias);
      __fp16 *dst = state->v + (size_t) head * state->head_tile_elems;
      qkv_pack_v_tile(vtcm_pack, raw16, 0, 1, state->head_dim);
      for (int d_tile = 0; d_tile < state->head_dim / HMX_FP16_TILE_N_COLS; ++d_tile) {
        __fp16 *dst_tile = dst +
          ((size_t) d_tile * state->token_tiles + token_tile) * HMX_FP16_TILE_N_ELMS;
        hvx_copy(dst_tile, vtcm_pack + (size_t) d_tile * HMX_FP16_TILE_N_ELMS,
                 HMX_FP16_TILE_N_ELMS * sizeof(__fp16));
      }
    }
  }
  worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static int hmx_qkv_mul_rms_packed_w16a32_impl(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *restrict activation, const __fp16 *restrict permuted_weight,
    const float *restrict bias, const float *restrict q_norm, const float *restrict k_norm,
    int tokens, int input_dim, int n_heads, int head_dim,
    int output_token_tiles, int output_token_offset,
    const float *rope_pe, int rope_token_offset, int rope_token_count) {
  if (q == NULL || k_out == NULL || v == NULL || activation == NULL || permuted_weight == NULL ||
      bias == NULL || q_norm == NULL || k_norm == NULL || tokens <= 0 || input_dim <= 0 ||
      !htp_packed_shape_supported(n_heads, n_heads, head_dim, 1) || input_dim % 32 != 0) {
    return -1;
  }
  if (!is_aligned(q, VLEN) || !is_aligned(k_out, VLEN) || !is_aligned(v, VLEN) ||
      !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -1;
  }
  const int q_cols = n_heads * head_dim;
  const int total_cols = q_cols * 3;
  if (total_cols % 64 != 0) {
    return -1;
  }

  const int token_tiles = ceil_div(tokens, HMX_FP16_TILE_N_ROWS);
  if (output_token_tiles < token_tiles || output_token_offset < 0 ||
      output_token_offset % HMX_FP16_TILE_N_ROWS != 0) {
    return -1;
  }
  const size_t head_tile_elems = (size_t) output_token_tiles * (size_t) (head_dim / HMX_FP16_TILE_N_COLS) *
                                 HMX_FP16_TILE_N_ELMS;
  enum { ROW_CHUNK = HMX_FP16_TILE_N_ROWS };
  uint8_t *vtcm_ptr = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16 *vtcm_weight = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, WEIGHT_AREA_SIZE);
  __fp16 *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, ACTIVATION_AREA_SIZE);
  __fp16 *vtcm_output = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, OUTPUT_AREA_SIZE);
  __fp16 *vtcm_scales = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  const size_t postprocess_rows = (size_t) num_workers * ROW_CHUNK * (size_t) head_dim;
  float *raw_scratch = (float *) vtcm_seq_alloc(&vtcm_ptr, postprocess_rows * sizeof(float));
  __fp16 *raw16_scratch = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, postprocess_rows * sizeof(__fp16));
  const size_t pack_tile_bytes = (size_t) (head_dim / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_ELMS *
                                 sizeof(__fp16);
  __fp16 *pack_scratch = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) num_workers * pack_tile_bytes);
  if (vtcm_weight == NULL || vtcm_activation == NULL || vtcm_output == NULL || vtcm_scales == NULL ||
      num_workers == 0 || raw_scratch == NULL || raw16_scratch == NULL || pack_scratch == NULL) {
    return -1;
  }
  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

  const size_t vec_dot_size = (size_t) input_dim * sizeof(__fp16);
  const size_t m_chunk_max_n_rows = align_down(ACTIVATION_AREA_SIZE / vec_dot_size,
                                                HMX_FP16_TILE_N_ROWS);
  const size_t n_chunk_max_n_cols = align_down(WEIGHT_AREA_SIZE / vec_dot_size,
                                                HMX_FP16_TILE_N_COLS);
  size_t m_chunk_n_rows = 0;
  size_t n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols,
                  OUTPUT_AREA_SIZE / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);
  // Q/K/V post-processing consumes complete heads, so keep output chunks head aligned.
  n_chunk_n_cols = align_down(n_chunk_n_cols, (size_t) head_dim);
  if (m_chunk_n_rows == 0 || n_chunk_n_cols == 0) {
    return -1;
  }
  for (size_t mr = 0; mr < (size_t) tokens; mr += m_chunk_n_rows) {
    const int n_rows = smin((size_t) tokens - mr, m_chunk_n_rows);
    const float *activation_chunk = activation + mr * (size_t) input_dim;
    transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, input_dim, input_dim);
    const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);

    for (size_t nc = 0; nc < (size_t) total_cols; nc += n_chunk_n_cols) {
      const int n_cols = smin((size_t) total_cols - nc, n_chunk_n_cols);
      if (n_cols <= 0 || n_cols % head_dim != 0) {
        return -1;
      }
      transfer_permuted_weight_chunk_fp16(vtcm_weight, permuted_weight + nc * (size_t) input_dim,
                                          n_cols, input_dim);
      const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
      core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales,
                          n_row_tiles, n_col_tiles, input_dim / 32);

      const size_t n_postprocess_tasks = (size_t) n_row_tiles * (n_cols / head_dim);
      const int n_postprocess_workers = smin((size_t) num_hvx128_contexts, n_postprocess_tasks);
      if (n_postprocess_workers <= 0) {
        return -1;
      }
      const size_t n_tot_chunks = n_postprocess_tasks;
      const size_t n_chunks_per_task = 1;
      qkv_postprocess_task_state_t state;
      INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
      state.q = q;
      state.k_out = k_out;
      state.v = v;
      state.vtcm_output = vtcm_output;
      state.bias = bias;
      state.q_norm = q_norm;
      state.k_norm = k_norm;
      state.raw_scratch = raw_scratch;
      state.raw16_scratch = raw16_scratch;
      state.pack_scratch = pack_scratch;
      state.mr = mr;
      state.nc = nc;
      state.head_tile_elems = head_tile_elems;
      state.pack_tile_bytes = pack_tile_bytes;
      state.n_rows = n_rows;
      state.n_row_tiles = n_row_tiles;
      state.n_col_tiles = n_col_tiles;
      state.n_cols = n_cols;
      state.q_cols = q_cols;
      state.token_tiles = output_token_tiles;
      state.output_token_offset = output_token_offset;
      state.head_dim = head_dim;
      state.rope_pe = rope_pe;
      state.rope_token_offset = rope_token_offset;
      state.rope_token_count = rope_token_count;
      const worker_pool_job_t job = { .fptr = qkv_postprocess_worker, .dptr = &state };
      worker_pool_synctoken_init(&state.sync_ctx, n_postprocess_workers);
      int submit_failed = 0;
      for (int i = 0; i < n_postprocess_workers; ++i) {
        if (worker_pool_submit(NULL, job) != AEE_SUCCESS) {
          submit_failed = 1;
          worker_pool_synctoken_jobdone(&state.sync_ctx);
        }
      }
      worker_pool_synctoken_wait(&state.sync_ctx);
      if (submit_failed) {
        return -1;
      }
    }
  }
  return 0;
}

int hmx_qkv_mul_rms_packed_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *restrict activation, const __fp16 *restrict permuted_weight,
    const float *restrict bias, const float *restrict q_norm, const float *restrict k_norm,
    int tokens, int input_dim, int n_heads, int head_dim) {
  return hmx_qkv_mul_rms_packed_w16a32_impl(
      q, k_out, v, activation, permuted_weight, bias, q_norm, k_norm,
      tokens, input_dim, n_heads, head_dim, ceil_div(tokens, HMX_FP16_TILE_N_ROWS), 0,
      NULL, 0, 0);
}

int hmx_qkv_mul_rms_packed_dual_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *image_activation, const __fp16 *image_weight, const float *image_bias,
    const float *image_q_norm, const float *image_k_norm, int image_tokens,
    const float *text_activation, const __fp16 *text_weight, const float *text_bias,
    const float *text_q_norm, const float *text_k_norm, int text_tokens,
    int input_dim, int n_heads, int head_dim) {
  if (image_tokens <= 0 || text_tokens <= 0) {
    return -1;
  }
  const int output_token_tiles = ceil_div(image_tokens + text_tokens, HMX_FP16_TILE_N_ROWS);
  if (hmx_qkv_mul_rms_packed_w16a32_impl(
          q, k_out, v, image_activation, image_weight, image_bias,
          image_q_norm, image_k_norm, image_tokens, input_dim, n_heads, head_dim,
          output_token_tiles, 0, NULL, 0, 0) != 0) {
    return -1;
  }
  return hmx_qkv_mul_rms_packed_w16a32_impl(
      q, k_out, v, text_activation, text_weight, text_bias,
      text_q_norm, text_k_norm, text_tokens, input_dim, n_heads, head_dim,
      output_token_tiles,
      (image_tokens + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS * HMX_FP16_TILE_N_ROWS,
      NULL, 0, 0);
}

int hmx_qkv_mul_rms_packed_rope_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *restrict activation, const __fp16 *restrict permuted_weight,
    const float *restrict bias, const float *restrict q_norm, const float *restrict k_norm,
    const float *restrict rope_pe, int rope_token_offset, int rope_token_count,
    int tokens, int input_dim, int n_heads, int head_dim) {
  return hmx_qkv_mul_rms_packed_w16a32_impl(
      q, k_out, v, activation, permuted_weight, bias, q_norm, k_norm,
      tokens, input_dim, n_heads, head_dim, ceil_div(tokens, HMX_FP16_TILE_N_ROWS), 0,
      rope_pe, rope_token_offset, rope_token_count);
}

int hmx_qkv_mul_rms_packed_dual_rope_w16a32(
    __fp16 *restrict q, __fp16 *restrict k_out, __fp16 *restrict v,
    const float *image_activation, const __fp16 *image_weight, const float *image_bias,
    const float *image_q_norm, const float *image_k_norm, int image_tokens,
    const float *text_activation, const __fp16 *text_weight, const float *text_bias,
    const float *text_q_norm, const float *text_k_norm, int text_tokens,
    const float *rope_pe, int image_rope_token_offset, int text_rope_token_offset,
    int input_dim, int n_heads, int head_dim) {
  if (image_tokens <= 0 || text_tokens <= 0) {
    return -1;
  }
  const int output_token_tiles = ceil_div(image_tokens + text_tokens, HMX_FP16_TILE_N_ROWS);
  if (hmx_qkv_mul_rms_packed_w16a32_impl(
          q, k_out, v, image_activation, image_weight, image_bias,
          image_q_norm, image_k_norm, image_tokens, input_dim, n_heads, head_dim,
          output_token_tiles, 0, rope_pe, image_rope_token_offset,
          image_tokens + text_tokens) != 0) {
    return -1;
  }
  return hmx_qkv_mul_rms_packed_w16a32_impl(
      q, k_out, v, text_activation, text_weight, text_bias,
      text_q_norm, text_k_norm, text_tokens, input_dim, n_heads, head_dim,
      output_token_tiles,
      (image_tokens + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS * HMX_FP16_TILE_N_ROWS,
      rope_pe, text_rope_token_offset,
      image_tokens + text_tokens);
}

extern worker_pool_context_t hmx_worker_pool_ctx;

typedef struct {
  __fp16            *c;
  const __fp16      *a, *b, *s;
  int                n_row_tiles, n_col_tiles, n_dot_tiles;
  worker_synctoken_t sync_ctx;
} core_dot_fp16_task_state_t;

static void core_dot_fp16_hmx_worker_fn(void *data, int _worker_index) {
  (void) _worker_index;
  core_dot_fp16_task_state_t *st = (core_dot_fp16_task_state_t *) data;

  core_dot_chunk_fp16(st->c, st->a, st->b, st->s, st->n_row_tiles, st->n_col_tiles, st->n_dot_tiles);

  worker_pool_synctoken_jobdone(&st->sync_ctx);
}

int mat_mul_qk_0_d16a32_out_stationary(float *restrict out, const float *restrict x, const uint8_t *restrict w, int m,
                                       int k, int n, enum ggml_type w_type);

int hmx_mat_mul_permuted_qk_0_d16a32(float *restrict dst, const float *restrict activation,
                                     const uint8_t *restrict permuted_weight, int m, int k, int n,
                                     enum ggml_type weight_type) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -1;
  }

  // for large m, k (e.g. prefill FFN Down), use out-staionary version
  if (m >= 128 && k > n && n > 1024) {
    return mat_mul_qk_0_d16a32_out_stationary(dst, activation, permuted_weight, m, k, n, weight_type);
  }

  size_t super_block_size = get_super_block_size(weight_type);
  if (super_block_size == 0) {
    return -1;
  }

  const size_t weight_area_size     = WEIGHT_AREA_SIZE;
  const size_t activation_area_size = ACTIVATION_AREA_SIZE;
  const size_t output_area_size     = OUTPUT_AREA_SIZE;
  const size_t scratch_area_size    = SCRATCH_AREA_SIZE;

  // VTCM layout: weight | activation | output | scales
  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch2   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  const bool use_pipeline = (m >= 128) && (k <= n);
  // const bool use_pipeline = false;

  if (!use_pipeline) {
    // NOTE(hzx): In this simple implementation, load-matmul-store are executed sequentially
    // only DMA load and dequantization process are overlapped during the load stage

    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      // transfer activation matrix chunk into VTCM
      size_t n_rows = smin(m - mr, m_chunk_n_rows);

      // int64_t act_t0 = HAP_perf_get_qtimer_count();
      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
      }
      // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

      // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

      void *buf_curr = vtcm_scratch0;
      void *buf_next = vtcm_scratch1;

      static dma_desc_1d_t desc
        __attribute__((aligned(64)));  // NOTE(hzx): make sure the DMA descriptor's lifetime is long enough

      // issue async DDR data transfer for the first weight chunk
      {
        const size_t n_cols_first            = smin(n, n_chunk_n_cols);
        const size_t first_weight_chunk_size = n_cols_first * k / QK_K * super_block_size;

        dma_issue_load_from_ddr(&desc, buf_curr, permuted_weight, first_weight_chunk_size);
      }

      for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
        size_t n_cols = smin(n - nc, n_chunk_n_cols);

        // int64_t wei_t0 = HAP_perf_get_qtimer_count();
        {
          dma_wait_for_idle();  // wait until current weight chunk become ready

          const size_t nc_next = nc + n_chunk_n_cols;
          if (nc_next < n) {
            const size_t n_cols_next = smin(n - nc_next, n_chunk_n_cols);

            const size_t   next_weight_chunk_size = n_cols_next * k / QK_K * super_block_size;
            const uint8_t *next_weight_chunk      = permuted_weight + nc_next * k / QK_K * super_block_size;

            dma_issue_load_from_ddr(&desc, buf_next, next_weight_chunk, next_weight_chunk_size);
          }

          const uint8_t *permuted_weight_chunk = permuted_weight + (nc * k / QK_K) * super_block_size;
          dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight, permuted_weight_chunk, n_cols * k, k,
                                                            weight_type, buf_curr);

          swap_ptr(&buf_curr, &buf_next);
        }
        // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

        // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

        // int64_t core_t0 = HAP_perf_get_qtimer_count();
        {
          const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
          core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
        }
        // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

        // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

        // int64_t out_t0 = HAP_perf_get_qtimer_count();
        {
          float *output = dst + (mr * n + nc);
          transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
        }
        // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

        // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
      }
    }
  } else {
    // 4-stage pipeline: DMA load (A), dequantize (B), HMX matmul (C), store (D)
    // stage B and D (dequantize and store) are expected to be on the critical path

    // A --> B: vtcm_qweight, 1 buffer
    // B --> C: vtcm_weight0/vtcm_weight1, 2 buffers
    // C --> D: vtcm_output0/vtcm_output1, 2 buffers

    //
    // LD ||A3|  | B3 ||
    // MM ||    C2    ||
    // ST || D1 |     ||

    static dma_desc_1d_t _Alignas(64) dma_desc;
    static core_dot_fp16_task_state_t mm_task_state;
    static worker_pool_job_t          mm_task_job;

    mm_task_job.dptr = &mm_task_state;
    mm_task_job.fptr = &core_dot_fp16_hmx_worker_fn;

    int n_chunk_cnt = ceil_div(n, n_chunk_n_cols);
    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      const size_t n_rows = smin(m - mr, m_chunk_n_rows);

      void *vtcm_qweight        = vtcm_weight;
      void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
      void *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

      // prologue: A0
      const size_t n_cols_A0 = smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
      {
        const size_t chunk_size_A0 = n_cols_A0 * k / QK_K * super_block_size;

        const uint8_t *qweight_chunk_A0 = permuted_weight;
        dma_issue_load_from_ddr(&dma_desc, vtcm_qweight, qweight_chunk_A0, chunk_size_A0);
      }

      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
      }

      // prologue: B0, A1, C0, B1
      {
        // B0
        dma_wait_for_idle();
        dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[0], NULL, n_cols_A0 * k, k, weight_type,
                                                          vtcm_qweight);

        // A1
        const size_t n_cols_A1 = smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
        if (1 < n_chunk_cnt) {
          const size_t chunk_size_A1 = n_cols_A1 * k / QK_K * super_block_size;

          const uint8_t *qweight_chunk_A1 = permuted_weight + n_chunk_n_cols * k / QK_K * super_block_size;
          dma_issue_load_from_ddr(&dma_desc, vtcm_qweight, qweight_chunk_A1, chunk_size_A1);
        }

        // C0
        {
          core_dot_fp16_task_state_t *s = &mm_task_state;

          s->c = (__fp16 *) vtcm_output_bufs[0];
          s->a = (__fp16 *) vtcm_activation;
          s->b = (__fp16 *) vtcm_weight_bufs[0];
          s->s = vtcm_scales;

          s->n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          s->n_col_tiles = ceil_div(n_cols_A0, HMX_FP16_TILE_N_COLS);
          s->n_dot_tiles = k / HMX_FP16_TILE_N_ROWS;

          worker_pool_synctoken_init(&s->sync_ctx, 1);
          worker_pool_submit(hmx_worker_pool_ctx, mm_task_job);
        }

        // B1
        if (1 < n_chunk_cnt) {
          dma_wait_for_idle();
          dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[1], NULL, n_cols_A1 * k, k, weight_type,
                                                            vtcm_qweight);
        }
      }

      // main loop
      for (int i = 0; i < n_chunk_cnt; ++i) {
        const size_t nc    = i * n_chunk_n_cols;
        const size_t nc_p1 = nc + 1 * n_chunk_n_cols;
        const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

        const size_t n_cols    = smin(n - nc, n_chunk_n_cols);
        const size_t n_cols_p1 = smin(n - nc_p1, n_chunk_n_cols);
        const size_t n_cols_p2 = smin(n - nc_p2, n_chunk_n_cols);

        // issue A_{i+2}
        if (i + 2 < n_chunk_cnt) {
          const size_t   chunk_size_p2    = n_cols_p2 * k / QK_K * super_block_size;
          const uint8_t *qweight_chunk_p2 = permuted_weight + nc_p2 * k / QK_K * super_block_size;
          dma_issue_load_from_ddr(&dma_desc, vtcm_qweight, qweight_chunk_p2, chunk_size_p2);
        }

        // wait for HMX (C_{i})
        worker_pool_synctoken_wait(&mm_task_state.sync_ctx);

        // result of B_{i+1} (input of C_{i+1}) should be ready now

        // issue C_{i+1}
        if (i + 1 < n_chunk_cnt) {
          core_dot_fp16_task_state_t *s = &mm_task_state;

          s->c = (__fp16 *) vtcm_output_bufs[(i + 1) % 2];
          s->a = (__fp16 *) vtcm_activation;
          s->b = (__fp16 *) vtcm_weight_bufs[(i + 1) % 2];
          s->s = vtcm_scales;

          s->n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          s->n_col_tiles = ceil_div(n_cols_p1, HMX_FP16_TILE_N_COLS);
          s->n_dot_tiles = k / HMX_FP16_TILE_N_ROWS;

          worker_pool_synctoken_init(&s->sync_ctx, 1);
          worker_pool_submit(hmx_worker_pool_ctx, mm_task_job);
        }

        // compute D_{i}
        float *output_chunk = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output_chunk, vtcm_output_bufs[i % 2], n_rows, n_cols, n);

        // wait for DMA (A_{i+2}), compute B_{i+2}
        if (i + 2 < n_chunk_cnt) {
          dma_wait_for_idle();
          dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[(i + 2) % 2], NULL, n_cols_p2 * k, k,
                                                            weight_type, vtcm_qweight);
        }
      }
    }
  }

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n / QK_K * super_block_size;
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

  return 0;
}

// C += AB
void core_mma_chunk_fp16(__fp16 *c, const __fp16 *a, const __fp16 *b, const __fp16 *col_scales, const __fp16 *eye_tile,
                         int n_row_tiles, int n_col_tiles, int n_dot_tiles, bool zero_init) {
  hmx_unit_acquire();

  asm volatile("mxclracc.hf");
  hmx_set_output_scales(col_scales);

  for (int i = 0; i < n_row_tiles; ++i) {
    for (int j = 0; j < n_col_tiles; ++j) {
      const __fp16 *row_tiles = a + i * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = b + j * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      __fp16 *accum_tile = c + (i * n_col_tiles + j) * HMX_FP16_TILE_N_ELMS;
      if (!zero_init) {
        hmx_load_tiles_fp16(accum_tile, eye_tile, 1);
      }

      for (int k = 0; k < n_dot_tiles; k += 32) {
        int    offset  = k * HMX_FP16_TILE_N_ELMS;
        size_t n_tiles = smin(n_dot_tiles - k, 32);
        hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
      }

      hmx_consume_accumulator_fp16(accum_tile);
    }
  }

  hmx_unit_release();
}

typedef struct {
  uint8_t           *dst;
  const uint8_t     *src;
  size_t             height;
  size_t             width;   // bytes
  size_t             stride;  // bytes
  worker_synctoken_t sync_ctx;
} qweight_fetch_task_state_t;

// Synchronously load 2D data from DDR to VTCM via DMA (using sequential 1D DMA transfers)
static void dma_load_2d_sync(uint8_t *dst, const uint8_t *src, size_t dst_stride, size_t src_stride, size_t height,
                             size_t width) {
  static dma_desc_1d_t desc __attribute__((aligned(64))) = { 0 };
  for (size_t i = 0; i < height; ++i) {
    size_t src_offset = i * src_stride;
    size_t dst_offset = i * dst_stride;
    dma_issue_load_from_ddr(&desc, dst + dst_offset, src + src_offset, width);
    dma_wait_for_idle();  // wait for the current row to be ready
  }
}

static void qweight_fetch_worker_fn(void *data, int _worker_index) {
  (void) _worker_index;
  qweight_fetch_task_state_t *st = (qweight_fetch_task_state_t *) data;

  dma_load_2d_sync(st->dst, st->src, st->width, st->stride, st->height, st->width);

  worker_pool_synctoken_jobdone(&st->sync_ctx);
}

// Only slightly faster than the common version (with L2 prefetch enabled) when doing VTCM to VTCM transfer
void transfer_activation_chunk_no_prefetch(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                           int k_block, int k_stride) {
  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
    int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

    const bool next_row_valid = (r + 1) < n_rows;

    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * k_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * k_stride);
    for (int c = 0; c < k_block; c += 32) {
      HVX_Vector v0 = *pv_in0++;
      HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

      HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

      // compute output position
      int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
      int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  __fp16      *dst;
  const float *src;
  int          k_block, k_stride;
} activation_transfer_task_state_t;

static void transfer_activation_chunk_worker_fn(void *data, int _worker_index) {
  (void) _worker_index;
  activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&st->task_id) - 1;
    if (task_id >= st->n_tasks) {
      break;
    }
    // one chunk: one row
    int    chunk_idx  = task_id * st->n_chunks_per_task;
    size_t chunk_size = smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

    __fp16      *dst = st->dst + chunk_idx * st->k_block;
    const float *src = st->src + chunk_idx * st->k_stride;
    transfer_activation_chunk_no_prefetch(dst, src, chunk_size, st->k_block, st->k_stride);
  }

  worker_pool_synctoken_jobdone(&st->sync_ctx);
}

void transfer_activation_chunk_multithread(__fp16 *dst, const float *src, int n_rows, int k_block, int k_stride) {
  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = n_rows;
  size_t n_chunks_per_task = 32;  // NOTE(hzx): must be multiple of 32 to ensure correct destination address

  activation_transfer_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.dst      = dst;
  state.src      = src;
  state.k_block  = k_block;
  state.k_stride = k_stride;

  worker_pool_job_t job;
  job.dptr = &state;
  job.fptr = &transfer_activation_chunk_worker_fn;

  worker_pool_synctoken_init(&state.sync_ctx, n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&state.sync_ctx);
}

int mat_mul_qk_0_d16a32_out_stationary(float *restrict out, const float *restrict x, const uint8_t *restrict w, int m,
                                       int k, int n, enum ggml_type weight_type) {
  // NOTE(hzx): this constraint on k originates from 2D DMA, consider alternative ways to load activation
  assert(k < 16384);
  // assume k % 32 == 0 && n % 32 == 0
  const size_t super_block_size = get_super_block_size(weight_type);
  if (super_block_size == 0) {
    return -1;
  }

  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, WEIGHT_AREA_SIZE);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, ACTIVATION_AREA_SIZE);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, OUTPUT_AREA_SIZE);
  uint8_t *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, SCRATCH_AREA_SIZE);
  uint8_t *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, SCRATCH_AREA_SIZE * 2);
  __fp16  *vtcm_eye_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  // initialize eye tile (32x32 identity matrix)
  {
    HVX_Vector v;
    v = Q6_V_vzero();
    v = Q6_Vw_vinsert_VwR(v, 0x3c000000);
    v = Q6_V_vror_VR(v, VLEN - 4);
    v = Q6_Vw_vinsert_VwR(v, 0x00003c00);
    for (int i = 0; i < 16; ++i) {
      ((HVX_Vector *) vtcm_eye_tile)[i] = v;

      v = Q6_V_vror_VR(v, VLEN - 8);
    }
  }
  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  // 704, 512
  const size_t M_BLOCK_SIZE = 512;
  const size_t N_BLOCK_SIZE = 512;
  const size_t K_BLOCK_SIZE = 512;

  static worker_pool_context_t      fetch_task_worker_pool_ctx;
  static qweight_fetch_task_state_t fetch_task_state;
  static worker_pool_job_t          fetch_task_job;

  worker_pool_init_ex(&fetch_task_worker_pool_ctx, 4096, 1, 0);
  fetch_task_job.dptr = &fetch_task_state;
  fetch_task_job.fptr = &qweight_fetch_worker_fn;

  int64_t t_a, t_b, t_c;
  t_a = t_b = t_c = 0;

  for (size_t mr = 0; mr < m; mr += M_BLOCK_SIZE) {
    size_t m_blk_sz = smin(m - mr, M_BLOCK_SIZE);
    for (size_t nc = 0; nc < n; nc += N_BLOCK_SIZE) {
      size_t n_blk_sz = smin(n - nc, N_BLOCK_SIZE);

      const int n_row_tiles = ceil_div(m_blk_sz, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = ceil_div(n_blk_sz, HMX_FP16_TILE_N_COLS);

      // TODO(hzx): fully pipelined loop
      for (size_t kk = 0; kk < k; kk += K_BLOCK_SIZE) {
        size_t k_blk_sz = smin(k - kk, K_BLOCK_SIZE);

        int64_t t0 = HAP_perf_get_qtimer_count();
        // fetch activation block into VTCN
        {
          const float *activation_block = x + mr * k + kk;

          _Alignas(64) dma_desc_2d_t desc = { 0 };

          desc.next       = 0;
          desc.length     = 0;
          desc.type       = DMA_DESC_TYPE_2D;
          desc.src_bypass = 1;
          desc.dst_bypass = 0;
          desc.ordered    = 1;
          desc.dstate     = DMA_DESC_DSTATE_PENDING;

          desc.src              = (uint32_t) activation_block;
          desc.dst              = (uint32_t) vtcm_scratch1;
          desc.roi_width        = k_blk_sz * sizeof(float);
          desc.roi_height       = m_blk_sz;
          desc.src_stride       = k * sizeof(float);
          desc.dst_stride       = k_blk_sz * sizeof(float);
          desc.src_width_offset = 0;
          desc.dst_width_offset = 0;

          dma_wait_for_idle();
          dma_submit_one((dma_desc_1d_t *) &desc);
          dma_wait_for_idle();
        }

        // fetch weight block into VTCM
        {
          qweight_fetch_task_state_t *s = &fetch_task_state;

          size_t width_ne  = HMX_FP16_TILE_N_COLS * k_blk_sz;
          size_t stride_ne = HMX_FP16_TILE_N_COLS * k;
          size_t width     = width_ne / QK_K * super_block_size;
          size_t stride    = stride_ne / QK_K * super_block_size;

          s->dst    = vtcm_scratch0;
          s->src    = w + (nc * k + HMX_FP16_TILE_N_COLS * kk) / QK_K * super_block_size;
          s->height = n_col_tiles;
          s->width  = width;
          s->stride = stride;

          worker_pool_synctoken_init(&s->sync_ctx, 1);
          worker_pool_submit(fetch_task_worker_pool_ctx, fetch_task_job);
        }
        t_a += HAP_perf_get_qtimer_count() - t0;

        int64_t t1 = HAP_perf_get_qtimer_count();
        // load activation block
        {
          // const float *activation_block = x + mr * k + kk;
          // transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_block, m_blk_sz, k_blk_sz, k);
          // transfer_activation_chunk_multithread(vtcm_activation, activation_block, m_blk_sz, k_blk_sz, k);

          // NOTE(hzx): This code assumes that the activation block already resides in VTCM
          // transfer_activation_chunk_no_prefetch(vtcm_activation, (float *) vtcm_scratch1, m_blk_sz, k_blk_sz, k_blk_sz);
          transfer_activation_chunk_multithread(vtcm_activation, (float *) vtcm_scratch1, m_blk_sz, k_blk_sz, k_blk_sz);
        }

        // dequantize weight block
        {
          // vtcm_scratch0 is used to store the qweight chunk
          worker_pool_synctoken_wait(&fetch_task_state.sync_ctx);
          dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight, NULL /*unused*/, n_blk_sz * k_blk_sz,
                                                            -1 /*unused*/, weight_type, vtcm_scratch0);
        }
        t_b += HAP_perf_get_qtimer_count() - t1;

        // core mma
        int64_t t2 = HAP_perf_get_qtimer_count();
        {
          core_mma_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, vtcm_eye_tile, n_row_tiles,
                              n_col_tiles, k_blk_sz / HMX_FP16_TILE_N_COLS, kk == 0);
        }
        t_c += HAP_perf_get_qtimer_count() - t2;
      }

      // store output block
      {
        float *output_block = out + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output_block, vtcm_output, m_blk_sz, n_blk_sz, n);
      }
    }
  }

  FARF(ALWAYS, "t_a: %lld us, t_b: %lld us, t_c: %lld us", HAP_perf_qtimer_count_to_us(t_a),
       HAP_perf_qtimer_count_to_us(t_b), HAP_perf_qtimer_count_to_us(t_c));

  worker_pool_deinit(&fetch_task_worker_pool_ctx);
  return 0;
}
// Pack one spatial x K tile directly into the interleaved FP16 HMX layout.
// Each worker owns a private 4 KiB scratch tile; no full F32 im2col is materialized.
typedef struct {
  worker_synctoken_t sync; unsigned next;
  __fp16 *activation; HVX_Vector *scratch;
  const float *input; const struct NativeConvParams *p;
  int mr, rows, padded_rows;
} conv_gather_job;

static void conv_pack_transpose(HVX_Vector *m, HVX_Vector *out) {
  // Keep the first three butterfly stages in registers, rather than
  // repeatedly writing full intermediate matrices to VTCM.
  for(int g=0;g<32;g+=8) {
    HVX_Vector *v=m+g;
    HVX_VectorPair a0=Q6_W_vshuff_VVR(v[1],v[0],-4);
    HVX_VectorPair a1=Q6_W_vshuff_VVR(v[3],v[2],-4);
    HVX_VectorPair a2=Q6_W_vshuff_VVR(v[5],v[4],-4);
    HVX_VectorPair a3=Q6_W_vshuff_VVR(v[7],v[6],-4);
    HVX_VectorPair b0=Q6_W_vshuff_VVR(Q6_V_lo_W(a1),Q6_V_lo_W(a0),-8);
    HVX_VectorPair b1=Q6_W_vshuff_VVR(Q6_V_hi_W(a1),Q6_V_hi_W(a0),-8);
    HVX_VectorPair b2=Q6_W_vshuff_VVR(Q6_V_lo_W(a3),Q6_V_lo_W(a2),-8);
    HVX_VectorPair b3=Q6_W_vshuff_VVR(Q6_V_hi_W(a3),Q6_V_hi_W(a2),-8);
    HVX_VectorPair c0=Q6_W_vshuff_VVR(Q6_V_lo_W(b2),Q6_V_lo_W(b0),-16);
    HVX_VectorPair c1=Q6_W_vshuff_VVR(Q6_V_hi_W(b2),Q6_V_hi_W(b0),-16);
    HVX_VectorPair c2=Q6_W_vshuff_VVR(Q6_V_lo_W(b3),Q6_V_lo_W(b1),-16);
    HVX_VectorPair c3=Q6_W_vshuff_VVR(Q6_V_hi_W(b3),Q6_V_hi_W(b1),-16);
    v[0]=Q6_V_lo_W(c0);v[1]=Q6_V_hi_W(c0);
    v[2]=Q6_V_lo_W(c1);v[3]=Q6_V_hi_W(c1);
    v[4]=Q6_V_lo_W(c2);v[5]=Q6_V_hi_W(c2);
    v[6]=Q6_V_lo_W(c3);v[7]=Q6_V_hi_W(c3);
  }
  for(int k=0;k<8;++k) {
    HVX_VectorPair d0=Q6_W_vshuff_VVR(m[k+8],m[k],-32);
    HVX_VectorPair d1=Q6_W_vshuff_VVR(m[k+24],m[k+16],-32);
    HVX_VectorPair e0=Q6_W_vshuff_VVR(Q6_V_lo_W(d1),Q6_V_lo_W(d0),-64);
    HVX_VectorPair e1=Q6_W_vshuff_VVR(Q6_V_hi_W(d1),Q6_V_hi_W(d0),-64);
    // Pair spatial rows into the HMX halfword-interleaved layout.
    out[2*k]=Q6_V_vor_VV(Q6_V_lo_W(e0),Q6_Vw_vasl_VwR(Q6_V_hi_W(e0),16));
    out[2*k+1]=Q6_V_vor_VV(Q6_V_lo_W(e1),Q6_Vw_vasl_VwR(Q6_V_hi_W(e1),16));
  }
}

static void conv_gather_worker(void *data,int worker_index) {
  conv_gather_job *s=(conv_gather_job *)data; const struct NativeConvParams *p=s->p;
  HVX_Vector *m=s->scratch+worker_index*32;
  float tail[32] __attribute__((aligned(128)));
  const HVX_Vector zero=Q6_V_vzero();
  int ktiles=p->kp/32;
  for(;;) {
    unsigned block=__atomic_fetch_add(&s->next,1,__ATOMIC_RELAXED);
    if(block>=(unsigned)ktiles) break;
    int kxs[32],kys[32],cis[32];
    for(int j=0;j<32;++j) {
      int col=block*32+j;
      kxs[j]=(col%p->kw)*p->dx;kys[j]=((col/p->kw)%p->kh)*p->dy;
      cis[j]=col/(p->kw*p->kh);
    }
    for(int r=0;r<s->padded_rows;r+=32) {
      int pos=s->mr+r, ox=pos%p->ow,oy=pos/p->ow;
      for(int j=0;j<32;++j) {
        int kx=kxs[j],ky=kys[j],ci=cis[j];
        int ix=ox*p->sx-p->px+kx,iy=oy*p->sy-p->py+ky;
        HVX_Vector val=Q6_V_vzero();
        if(ci<p->ic && r+32<=s->rows && ox+32<=p->ow && p->sx==1 && p->upscale==1 &&
           ix>=0 && ix+32<=p->iw && iy>=0 && iy<p->ih) {
          val=vmemu(s->input+((size_t)ci*p->ih+iy)*p->iw+ix);
        } else if(ci<p->ic && r+32<=s->rows && ox+32<=p->ow && p->sx==1 && p->upscale==2 &&
                  ix>=0 && ix+32<=2*p->iw && iy>=0 && iy<2*p->ih) {
          const float *plane=s->input+((size_t)ci*p->ih+(iy>>1))*p->iw;
          int first=ix>>1;
          HVX_Vector src;
          if(first+32<=p->iw) src=vmemu(plane+first);
          else {
            memset(tail,0,sizeof(tail));
            memcpy(tail,plane+first,(16+(ix&1))*sizeof(float));
            src=vmem(tail);
          }
          HVX_VectorPair dup=Q6_W_vshuff_VVR(src,src,-4);
          val=(ix&1)?Q6_V_valign_VVR(Q6_V_hi_W(dup),Q6_V_lo_W(dup),4):Q6_V_lo_W(dup);
        } else if(ci<p->ic && r+32<=s->rows && ox+32<=p->ow) {
          // Border vectors share the same source row. Avoid 32 dynamic integer
          // divisions per K column, including the common nearest-upsample case.
          memset(tail,0,sizeof(tail));
          if(iy>=0 && iy<p->ih*p->upscale) {
            const float *plane=s->input+((size_t)ci*p->ih+(p->upscale==2?iy>>1:iy))*p->iw;
            if(p->sx==1 && p->upscale==1) {
              int first=ix<0?-ix:0, last=p->iw-ix<32?p->iw-ix:32;
              if(last>first) memcpy(tail+first,plane+ix+first,(last-first)*4);
            } else for(int j=0;j<32;++j) {
              int x=ix+j*p->sx;
              if(x>=0 && x<p->iw*p->upscale) tail[j]=plane[p->upscale==2?x>>1:x];
            }
          }
          val=vmem(tail);
        } else if(ci<p->ic) {
          for(int j=0;j<32;++j) {
            int t=s->mr+r+j,x=(t%p->ow)*p->sx-p->px+kx,y=(t/p->ow)*p->sy-p->py+ky;
            tail[j]=(r+j<s->rows && x>=0 && x<p->iw*p->upscale && y>=0 && y<p->ih*p->upscale) ?
              s->input[((size_t)ci*p->ih+y/p->upscale)*p->iw+x/p->upscale] : 0.f;
          }
          val=vmem(tail);
        }
        m[j]=hvx_my_wsf_to_vhf(zero,val);
      }
      conv_pack_transpose(m,(HVX_Vector *)(s->activation+((size_t)(r/32)*ktiles+block)*1024));
    }
  }
  worker_pool_synctoken_jobdone(&s->sync);
}
int hmx_conv2d_nchw_f16(float *dst,const float *input,const __fp16 *weight,const float *bias,struct NativeConvParams *p) {
  int64_t gather_t=0,weight_t=0,hmx_t=0,store_t=0;
  size_t total=vtcm_manager_get_vtcm_size();
  size_t reserved=WEIGHT_AREA_SIZE+OUTPUT_AREA_SIZE+256+4096*smax(1,num_workers);
  if(total<=reserved) return -20;
  // Only the final FP16 activation occupies the spatial chunk allocation.
  int mrmax=align_down((total-reserved)/(2*p->kp),32);
  mrmax=smin(mrmax,1024);
  int ncmax=align_down(WEIGHT_AREA_SIZE/(2*p->kp),32);
  if(mrmax<32 || ncmax<32) return -21;
  ncmax=smin(ncmax,p->np);
  while((size_t)mrmax*ncmax*2>OUTPUT_AREA_SIZE) ncmax-=32;
  uint8_t *vp=(uint8_t *)vtcm_manager_get_vtcm_base();
  __fp16 *w=(__fp16 *)vtcm_seq_alloc(&vp,WEIGHT_AREA_SIZE);
  __fp16 *o=(__fp16 *)vtcm_seq_alloc(&vp,OUTPUT_AREA_SIZE);
  // HMX tile addresses must be 2 KiB aligned; allocate all tile matrices
  // before the 256-byte scales area (HMX otherwise drops low address bits).
  __fp16 *a=(__fp16 *)vtcm_seq_alloc(&vp,(size_t)mrmax*p->kp*2);
  __fp16 *scales=(__fp16 *)vtcm_seq_alloc(&vp,256);
  float *scratch=(float *)vtcm_seq_alloc(&vp,4096*smax(1,num_workers));
  hmx_init_column_scales(scales,Q6_V_vsplat_R(0x3c00));
  int spatial=p->ow*p->oh,workers=smin(num_workers,num_hvx128_contexts);
  if(workers<1) return -22;
  for(int batch=0;batch<p->batch;++batch) for(int mr=0;mr<spatial;mr+=mrmax) {
    int rows=smin(mrmax,spatial-mr),padded=align_up(rows,32);
    uint64_t tick=HAP_perf_get_qtimer_count();
    conv_gather_job state={.next=0,.activation=a,.scratch=(HVX_Vector *)scratch,.input=input+(size_t)batch*p->iw*p->ih*p->ic,.p=p,.mr=mr,.rows=rows,.padded_rows=padded};
    worker_pool_synctoken_init(&state.sync,workers);
    worker_pool_job_t job={.fptr=conv_gather_worker,.dptr=&state};
    int fail=0;
    for(int i=0;i<workers;++i) if(worker_pool_submit(NULL,job)!=AEE_SUCCESS) {worker_pool_synctoken_jobdone(&state.sync);fail=1;}
    worker_pool_synctoken_wait(&state.sync);if(fail) return -23;
    gather_t+=HAP_perf_get_qtimer_count()-tick;
    for(int nc=0;nc<p->np;nc+=ncmax) {
      int cols=smin(ncmax,p->np-nc);
      tick=HAP_perf_get_qtimer_count();
      transfer_permuted_weight_chunk_fp16(w,weight+(size_t)nc*p->kp,cols,p->kp);
      weight_t+=HAP_perf_get_qtimer_count()-tick;tick=HAP_perf_get_qtimer_count();
      core_dot_chunk_fp16(o,a,w,scales,padded/32,cols/32,p->kp/32);
      hmx_t+=HAP_perf_get_qtimer_count()-tick;tick=HAP_perf_get_qtimer_count();
      if(transfer_parallel_epilogue(dst+(size_t)batch*spatial*p->oc,o,bias,rows,cols,
          spatial,p->np,mr,nc,p->oc,3,scratch)) return -24;
      store_t+=HAP_perf_get_qtimer_count()-tick;
    }
  }
  p->gather_us=gather_t/19.2;p->weight_us=weight_t/19.2;p->hmx_us=hmx_t/19.2;p->store_us=store_t/19.2;
  return 0;
}

// F32 model auxiliaries. Explicit epsilon and masked tails avoid the legacy
// RMS kernel's fixed epsilon and full-vector tail stores.
typedef struct {
 worker_synctoken_t sync; unsigned next; float *dst;
 const float *x,*w,*b;const struct NativeAuxParams *p;int tasks;
} model_aux_job;
static HVX_Vector aux_load(const float *p,int n) {
 if(n==32) return vmemu(p);
 float tail[32] __attribute__((aligned(128)))={0};memcpy(tail,p,n*4);return vmem(tail);
}
static void aux_store(float *p,HVX_Vector v,int n) {
 if(n==32) {vmemu(p)=v;return;}
 float tail[32] __attribute__((aligned(128)));vmem(tail)=v;memcpy(p,tail,n*4);
}
static double aux_sum(HVX_Vector v) {
 float values[32] __attribute__((aligned(128)));vmem(values)=Q6_Vsf_equals_Vqf32(v);
 double sum=0;for(int i=0;i<32;++i) sum+=values[i];return sum;
}
static void model_aux_worker(void *data,int worker_index) {
 (void)worker_index;model_aux_job *s=data;const struct NativeAuxParams *p=s->p;
 for(;;) {
  int task=__atomic_fetch_add(&s->next,1,__ATOMIC_RELAXED);if(task>=s->tasks) break;
  if(p->mode==8) {
   int nt=ceil_div(p->spatial,32),mt=ceil_div(p->width,32);
   int batch=task/(mt*nt),rt=(task/nt)%mt,ct=task%nt;
   int nr=smin(32,p->width-rt*32),nc=smin(32,p->spatial-ct*32);
   float *scratch=(float *)vtcm_manager_get_vtcm_base()+worker_index*1024;
   int32_t offsets[32] __attribute__((aligned(128)));
   for(int j=0;j<32;++j) offsets[j]=j*128;HVX_Vector index=vmem(offsets);
   for(int row=0;row<nr;++row) {
    const float *src=s->x+((size_t)batch*p->width+rt*32+row)*p->spatial+ct*32;
    Q6_vscatter_RMVwV((size_t)(scratch+row),4095-row*4,index,aux_load(src,nc));
   }
   asm volatile("vmem(%0 + #0):scatter_release" :: "r"(scratch) : "memory");
   for(int col=0;col<nc;++col) aux_store(s->dst+((size_t)batch*p->spatial+ct*32+col)*p->width+rt*32,vmem(scratch+col*32),nr);
  } else if(p->mode==11) {
   // Read the compact 2x2 position matrices directly. Each task writes one
   // head/token row in attention's output layout, without REPEAT or transpose.
   // Reuse each position matrix across all heads. Preparing its lane
   // vectors once avoids heads-fold scalar unpacking and atomic job claims.
   int token=task%p->spatial,batch=task/p->spatial,dim=p->width;
   const float *pe=s->w+((size_t)(p->groups==1?0:batch)*p->spatial+token)*dim*2;
   for(int i=0;i<dim;i+=32) {
    int valid=smin(32,dim-i);
    float diagonal[32] __attribute__((aligned(128)))={0},off[32] __attribute__((aligned(128)))={0};
    for(int j=0;j<valid;j+=2) {
     const float *m=pe+2*(i+j);
     diagonal[j]=m[0];diagonal[j+1]=m[3];off[j]=m[1];off[j+1]=m[2];
    }
    HVX_Vector diagonal_v=vmem(diagonal),off_v=vmem(off);
    for(int head=0;head<p->channels;++head) {
     const float *x=s->x+((size_t)batch*p->spatial*p->channels+token*p->channels+head)*dim;
     float *dst=s->dst+((size_t)(batch*p->channels+head)*p->spatial+token)*dim;
     HVX_Vector v=aux_load(x+i,valid),swapped=Q6_V_vdelta_VV(v,Q6_V_vsplat_R(0x04040404));
     HVX_Vector a=epilogue_mul(v,diagonal_v),b=epilogue_mul(swapped,off_v);
     aux_store(dst+i,Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a,b)),valid);
    }
   }
  } else if(p->mode<=2 || p->mode==9) {
   int count=p->mode==2?p->spatial*(p->channels/p->groups):p->width;
   size_t offset=(size_t)task*count;const float *x=s->x+offset;
   float mean=0;HVX_Vector sum=Q6_V_vzero();
   if(p->mode!=1) {
    for(int i=0;i<count;i+=32) sum=Q6_Vqf32_vadd_Vqf32Vsf(sum,aux_load(x+i,smin(32,count-i)));
    mean=aux_sum(sum)/count;
   }
   HVX_Vector vmean=epilogue_const(mean);sum=Q6_V_vzero();
   for(int i=0;i<count;i+=32) {
    int valid=smin(32,count-i);HVX_Vector v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(aux_load(x+i,valid),vmean));
    if(valid<32) v=Q6_V_vmux_QVV(Q6_Q_vsetq_R(valid*4),v,Q6_V_vzero());
    sum=Q6_Vqf32_vadd_Vqf32Vqf32(sum,Q6_Vqf32_vmpy_VsfVsf(v,v));
   }
   float inv=1.f/sqrtf(aux_sum(sum)/count+p->eps);HVX_Vector scale=epilogue_const(inv);
   for(int i=0;i<count;i+=32) {
    int valid=smin(32,count-i);HVX_Vector v=epilogue_mul(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(aux_load(x+i,valid),vmean)),scale);
    if(p->mode==2) {
     float weights[32] __attribute__((aligned(128))),biases[32] __attribute__((aligned(128)));
     int ch=((offset+i)/p->spatial)%p->channels;
     if((offset+i)%p->spatial+valid<=p->spatial) {
      if(s->w) v=epilogue_mul(v,epilogue_const(s->w[ch]));
      if(s->b) v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,epilogue_const(s->b[ch])));
     } else {
      for(int j=0;j<32;++j) {ch=((offset+i+j)/p->spatial)%p->channels;weights[j]=s->w?s->w[ch]:1.f;biases[j]=s->b?s->b[ch]:0.f;}
      v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(epilogue_mul(v,vmem(weights)),vmem(biases)));
     }
    } else if(p->mode==9) {
     int batch=task/p->spatial,param_batch=p->other_count==p->width?0:batch;
     const float *w=s->w+(size_t)param_batch*p->width+i,*b=s->b+(size_t)param_batch*p->width+i;
     HVX_Vector product=epilogue_mul(v,aux_load(w,valid));
     v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,product));
     v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,aux_load(b,valid)));
    } else {
     if(s->w) v=epilogue_mul(v,aux_load(s->w+i,valid));
     if(s->b) v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,aux_load(s->b+i,valid)));
    }
    if(p->activation) v=epilogue_activation(v,4);
    aux_store(s->dst+offset+i,v,valid);
   }
  } else if(p->mode==13) {
   // One task owns a token row, preserving attention bits and writing SwiGLU
   // into its final concatenated location without a separate MLP tensor.
   int width=p->width,attention=p->groups,out_width=width+attention;
   float *dst=s->dst+(size_t)task*out_width;
   const float *a=s->w+(size_t)task*attention;
   for(int i=0;i<attention;i+=32) {
    int valid=smin(32,attention-i);
    aux_store(dst+i,aux_load(a+i,valid),valid);
   }
   const float *x=s->x+(size_t)task*p->channels;
   for(int i=0;i<width;i+=32) {
    int valid=smin(32,width-i);
    const float *gate=x+i+(p->activation?0:width),*up=x+i+(p->activation?width:0);
    aux_store(dst+attention+i,epilogue_mul(epilogue_activation(aux_load(gate,valid),5),aux_load(up,valid)),valid);
   }
  } else if(p->mode==12) {
   int first=task*4096,last=smin(p->count,first+4096);
   for(int i=first;i<last;) {
    int column=i%p->width,row=i/p->width,valid=smin(32,smin(last-i,p->width-column));
    const float *a=s->x+(size_t)row*p->channels+column;
    const float *gate=a+(p->activation?0:p->width),*up=a+(p->activation?p->width:0);
    HVX_Vector v=epilogue_mul(epilogue_activation(aux_load(gate,valid),5),aux_load(up,valid));
    aux_store(s->dst+i,v,valid);i+=valid;
   }
  } else if(p->mode==10) {
   // Compact gates vary by channel/batch and broadcast only across tokens.
   int first=task*4096,last=smin(p->count,first+4096);
   for(int i=first;i<last;) {
    int channel=i%p->width,batch=i/(p->width*p->spatial);
    int valid=smin(32,smin(last-i,p->width-channel));
    int param_batch=p->other_count==p->width?0:batch;
    HVX_Vector v=epilogue_mul(aux_load(s->x+i,valid),aux_load(s->w+(size_t)param_batch*p->width+channel,valid));
    v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,aux_load(s->b+i,valid)));
    aux_store(s->dst+i,v,valid);i+=valid;
   }
  } else {
   int first=task*4096,last=smin(p->count,first+4096);
   for(int i=first;i<last;i+=32) {
    int valid=smin(32,last-i);HVX_Vector v=aux_load(s->x+i,valid);
    if(p->mode==3||p->mode==4) {
     int index=i%p->other_count;HVX_Vector w;
     if(index+valid<=p->other_count) w=aux_load(s->w+index,valid);
     else {float tail[32] __attribute__((aligned(128)));for(int j=0;j<32;++j) tail[j]=s->w[(index+j)%p->other_count];w=vmem(tail);}
     v=p->mode==3?Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v,w)):epilogue_mul(v,w);
    } else v=epilogue_activation(v,p->mode==5?4:p->mode==6?1:2);
    aux_store(s->dst+i,v,valid);
   }
  }
 }
 worker_pool_synctoken_jobdone(&s->sync);
}
int hvx_model_aux_f32(float *dst,const float *x,const float *w,const float *b,const struct NativeAuxParams *p) {
 model_aux_job state={.next=0,.dst=dst,.x=x,.w=w,.b=b,.p=p};
 state.tasks=p->mode==13?p->spatial*p->batch:p->mode==11?p->spatial*p->batch:p->mode==8?ceil_div(p->width,32)*ceil_div(p->spatial,32)*p->batch:p->mode==2?p->groups*p->batch:(p->mode<=1 || p->mode==9)?p->count/p->width:ceil_div(p->count,4096);
 int jobs=smin(smin(num_workers,num_hvx128_contexts),state.tasks);if(jobs<1) return -40;
 if(p->mode==8 && vtcm_manager_get_vtcm_size()<(size_t)num_workers*4096) return -42;
 // Short encoder rows are cheaper on the dispatch thread than waking the
 // worker pool. Spatial VAE tensors retain parallel execution.
 if(p->mode!=8 && p->count<=512*1024) {
  worker_pool_synctoken_init(&state.sync,1);model_aux_worker(&state,0);return 0;
 }
 worker_pool_synctoken_init(&state.sync,jobs);worker_pool_job_t job={.fptr=model_aux_worker,.dptr=&state};
 int failed=0;for(int i=0;i<jobs;++i) if(worker_pool_submit(NULL,job)!=AEE_SUCCESS) {worker_pool_synctoken_jobdone(&state.sync);failed=1;}
 worker_pool_synctoken_wait(&state.sync);return failed?-41:0;
}
