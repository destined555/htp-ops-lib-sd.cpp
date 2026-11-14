#include <math.h>
#include <stdint.h>
#include <string.h>
#include <hexagon_protos.h>
#include <hmx_hexagon_protos.h>
#include <HAP_perf.h>
#include <qurt_memory.h>
#include "dsp/hmx_mgr.h"
#include "dsp/hvx_internal.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "op_reg.h"

typedef HVX_Vector unaligned_vector __attribute__((aligned(1)));

#if __HEXAGON_ARCH__ >= 75

#include "linear_w8a8_allocation_table.h"

// v75-compatible vector arithmetic. QFloat arithmetic followed by IEEE conversion.
static inline HVX_Vector qfloat_mul(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static inline HVX_Vector qfloat_add(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

static HVX_Vector splat_float(float value) {
    int32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return Q6_V_vsplat_R(bits);
}

static inline float row_scale(HVX_Vector maximum) {
    for (int shift = 64; shift >= 4; shift /= 2)
        maximum = Q6_Vsf_vmax_VsfVsf(maximum, Q6_V_vror_VR(maximum, shift));
    union { int32_t bits; float value; } reduced;
    reduced.bits = Q6_R_vextract_VR(maximum, 0);
    return fmaxf(reduced.value/127, 1e-12f);
}

static inline HVX_Vector quantize_scaled(HVX_Vector x, HVX_Vector scale,
                                         HVX_Vector offset) {
    x = qfloat_add(qfloat_mul(x, scale), offset);
    // QFloat loses low bits in the large-magic-add trick. Round the IEEE
    // bits of the small, positive [1,255] value using integer HVX instead.
    const HVX_Vector one = Q6_V_vsplat_R(1);
    x = Q6_Vw_vmax_VwVw(x, splat_float(1));
    x = Q6_Vw_vmin_VwVw(x, splat_float(255));
    HVX_Vector exponent = Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(x,23), Q6_V_vsplat_R(127));
    HVX_Vector shift = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(23), exponent);
    HVX_Vector mantissa = Q6_V_vor_VV(Q6_V_vand_VV(x,Q6_V_vsplat_R(0x7fffff)),Q6_V_vsplat_R(0x800000));
    HVX_Vector integer = Q6_Vw_vasr_VwVw(mantissa,shift);
    HVX_Vector half = Q6_Vw_vasl_VwVw(one,Q6_Vw_vsub_VwVw(shift,one));
    // Positive mantissas have at most 24 bits. Adding half-1 plus the retained
    // low bit gives exact round-to-nearest-even without predicates or muxes.
    HVX_Vector bias = Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(half,one), Q6_V_vand_VV(integer,one));
    return Q6_Vw_vasr_VwVw(Q6_Vw_vadd_VwVw(mantissa,bias),shift);
}

static inline HVX_Vector load_input(const float *p, int aligned) {
    return aligned ? *(const HVX_Vector *)p : *(const unaligned_vector *)p;
}

static inline __attribute__((always_inline)) void prepare_tile_impl(uint8_t *packed, float *scales, const float *input,
                         const float *inverse, float *scratch, int rows, int k, int aligned) {
    const HVX_Vector signmask = Q6_V_vsplat_R(0x7fffffff);
    const HVX_Vector offset = splat_float(128);
    const HVX_Vector zero = Q6_V_vzero();
    // Four independent rows hide HVX QFloat dependency latency. Cache already
    // scaled FP32 values in VTCM; the second pass does not reread DDR or repeat
    // channel scaling. QFloat operations round before storing IEEE FP32 values.
    for (int r = 0; r < rows; r += 4) {
        const int valid = rows-r < 4 ? rows-r : 4;
        HVX_Vector m0=zero, m1=zero, m2=zero, m3=zero;
        // Prefetch bounded slices across all valid rows, then advance one
        // 2 KiB slice ahead while the current slice is being scaled.
        l2fetch(input+r*k, k*4, k < 512 ? k*4 : 2048, valid, 0);
        for (int c = 0; c < k; c += 32) {
            if (c%512 == 0 && c+512 < k)
                l2fetch(input+r*k+c+512, k*4,
                        k-c-512 < 512 ? (k-c-512)*4 : 2048, valid, 0);
            const HVX_Vector inv = load_input(inverse+c, aligned);
            HVX_Vector x0 = qfloat_mul(load_input(input+r*k+c, aligned), inv);
            HVX_Vector x1 = valid>1 ? qfloat_mul(load_input(input+(r+1)*k+c, aligned), inv) : zero;
            HVX_Vector x2 = valid>2 ? qfloat_mul(load_input(input+(r+2)*k+c, aligned), inv) : zero;
            HVX_Vector x3 = valid>3 ? qfloat_mul(load_input(input+(r+3)*k+c, aligned), inv) : zero;
            *(HVX_Vector *)(scratch+c) = x0;
            *(HVX_Vector *)(scratch+k+c) = x1;
            *(HVX_Vector *)(scratch+2*k+c) = x2;
            *(HVX_Vector *)(scratch+3*k+c) = x3;
            m0 = Q6_Vsf_vmax_VsfVsf(m0, Q6_V_vand_VV(x0, signmask));
            m1 = Q6_Vsf_vmax_VsfVsf(m1, Q6_V_vand_VV(x1, signmask));
            m2 = Q6_Vsf_vmax_VsfVsf(m2, Q6_V_vand_VV(x2, signmask));
            m3 = Q6_Vsf_vmax_VsfVsf(m3, Q6_V_vand_VV(x3, signmask));
        }
        // While quantizing these four VTCM rows, bring the next group into L2.
        // Stay within this worker's valid rows; no extra VTCM buffer is needed.
        const int next_rows = rows-r-4 < 4 ? rows-r-4 : 4;
        if (next_rows > 0)
            l2fetch(input+(r+4)*k, k*4, k < 512 ? k*4 : 2048, next_rows, 0);
        scales[r] = row_scale(m0);
        if (valid > 1) scales[r+1] = row_scale(m1);
        if (valid > 2) scales[r+2] = row_scale(m2);
        if (valid > 3) scales[r+3] = row_scale(m3);
        const HVX_Vector s0 = splat_float(1/scales[r]);
        const HVX_Vector s1 = splat_float(valid>1 ? 1/scales[r+1] : 1);
        const HVX_Vector s2 = splat_float(valid>2 ? 1/scales[r+2] : 1);
        const HVX_Vector s3 = splat_float(valid>3 ? 1/scales[r+3] : 1);
        for (int c = 0; c < k; c += 32) {
            HVX_Vector v0 = quantize_scaled(*(HVX_Vector *)(scratch+c), s0, offset);
            HVX_Vector v1 = quantize_scaled(*(HVX_Vector *)(scratch+k+c), s1, offset);
            HVX_Vector v2 = quantize_scaled(*(HVX_Vector *)(scratch+2*k+c), s2, offset);
            HVX_Vector v3 = quantize_scaled(*(HVX_Vector *)(scratch+3*k+c), s3, offset);
            HVX_Vector h0 = Q6_Vh_vpack_VwVw_sat(v1, v0);
            HVX_Vector h1 = Q6_Vh_vpack_VwVw_sat(v3, v2);
            *(HVX_Vector *)(packed+(c/32)*2048+r*32) = Q6_Vub_vpack_VhVh_sat(h1, h0);
        }
    }
}

static void prepare_tile(uint8_t *packed, float *scales, const float *input,
                         const float *inverse, float *scratch, int rows, int k) {
    if ((((uintptr_t)input | (uintptr_t)inverse) & 127) == 0)
        prepare_tile_impl(packed, scales, input, inverse, scratch, rows, k, 1);
    else
        prepare_tile_impl(packed, scales, input, inverse, scratch, rows, k, 0);
}

static void read_int32(uint8_t *planes, uint32_t *tables) {
    for (int h = 0; h < 2; ++h) for (int p = 0; p < 4; ++p) {
        Q6_bias_mxmem2_A(tables+p*64);
        if (p < 3)
            asm volatile("mxmem(%0,%1):after:retain:cm.ub = acc" :: "r"(planes+h*8192+p*2048), "r"(0) : "memory");
        else
            asm volatile("mxmem(%0,%1):after:cm.ub = acc" :: "r"(planes+h*8192+p*2048), "r"(0) : "memory");
    }
}

static inline void store_output_row(float *output, HVX_Vector accum, HVX_Vector correction,
                                     HVX_Vector weight_scale, float row_scale) {
    HVX_Vector x = Q6_Vw_vsub_VwVw(accum, correction);
    HVX_Vector value = Q6_Vsf_equals_Vw(x);
    value = qfloat_mul(qfloat_mul(value, weight_scale), splat_float(row_scale));
    *(unaligned_vector *)output = value;
}

// Four complete rows keep independent conversion/multiply chains in flight.
// The tail and unaligned direct callers retain the scalar-row store helper.
static inline __attribute__((always_inline)) void output_tile_impl(float *restrict output, const uint8_t *restrict planes,
                        const float *restrict row_scales, const float *restrict weight_scales,
                        const int32_t *restrict sums, int rows, int n, int n0, int row_first) {
    const int aligned = ((uintptr_t)output & 127) == 0;
    const int groups = (rows+3)/4;
    for (int i = 0; i < (row_first ? groups : 2); ++i) {
        for (int j = 0; j < (row_first ? 2 : groups); ++j) {
            const int h = row_first ? j : i;
            const int q = row_first ? i : j;
            const HVX_Vector correction = Q6_Vw_vasl_VwR(*(const unaligned_vector *)(sums+n0+h*32), 7);
            const HVX_Vector sw = *(const unaligned_vector *)(weight_scales+n0+h*32);
            const HVX_Vector *p = (const HVX_Vector *)(planes+h*8192+q*128);
            HVX_VectorPair b01 = Q6_W_vshuff_VVR(p[16], p[0], -1);
            HVX_VectorPair b23 = Q6_W_vshuff_VVR(p[48], p[32], -1);
            HVX_VectorPair w0 = Q6_W_vshuff_VVR(Q6_V_lo_W(b23), Q6_V_lo_W(b01), -2);
            HVX_VectorPair w1 = Q6_W_vshuff_VVR(Q6_V_hi_W(b23), Q6_V_hi_W(b01), -2);
            float *dst = output+(size_t)q*4*n+n0+h*32;
            if (aligned && q*4+3 < rows) {
                HVX_Vector x0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w0), correction);
                HVX_Vector x1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w0), correction);
                HVX_Vector x2 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w1), correction);
                HVX_Vector x3 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w1), correction);
                x0 = Q6_Vsf_equals_Vw(x0); x1 = Q6_Vsf_equals_Vw(x1);
                x2 = Q6_Vsf_equals_Vw(x2); x3 = Q6_Vsf_equals_Vw(x3);
                x0 = qfloat_mul(x0, sw); x1 = qfloat_mul(x1, sw);
                x2 = qfloat_mul(x2, sw); x3 = qfloat_mul(x3, sw);
                x0 = qfloat_mul(x0, splat_float(row_scales[q*4]));
                x1 = qfloat_mul(x1, splat_float(row_scales[q*4+1]));
                x2 = qfloat_mul(x2, splat_float(row_scales[q*4+2]));
                x3 = qfloat_mul(x3, splat_float(row_scales[q*4+3]));
                __builtin_nontemporal_store(x0, (HVX_Vector *)(dst));
                __builtin_nontemporal_store(x1, (HVX_Vector *)(dst+n));
                __builtin_nontemporal_store(x2, (HVX_Vector *)(dst+2*n));
                __builtin_nontemporal_store(x3, (HVX_Vector *)(dst+3*n));
            } else {
                store_output_row(dst, Q6_V_lo_W(w0), correction, sw, row_scales[q*4]);
                if (q*4+1 < rows) store_output_row(dst+n, Q6_V_hi_W(w0), correction, sw, row_scales[q*4+1]);
                if (q*4+2 < rows) store_output_row(dst+2*n, Q6_V_lo_W(w1), correction, sw, row_scales[q*4+2]);
                if (q*4+3 < rows) store_output_row(dst+3*n, Q6_V_hi_W(w1), correction, sw, row_scales[q*4+3]);
            }
        }
    }
}

static void output_tile(float *restrict output, const uint8_t *restrict planes,
                        const float *restrict row_scales, const float *restrict weight_scales,
                        const int32_t *restrict sums, int rows, int n, int n0) {
    // At 4 KiB-multiple row strides, finish adjacent halves together. Other
    // strides favor reusing each half's scale/correction across the row group.
    if (n%1024 == 0)
        output_tile_impl(output, planes, row_scales, weight_scales, sums, rows, n, n0, 1);
    else
        output_tile_impl(output, planes, row_scales, weight_scales, sums, rows, n, n0, 0);
}

// Compact each half tile's four planes into the same rows/columns occupied by
// its final FP32 values. Tail tiles export only valid rows, without padding the
// tensor or writing beyond its allocation. The CPU gathers a half before any
// in-place FP32 stores can overwrite these bytes.
static void export_plane_tile(float *output, const uint8_t *planes,
                               int rows, int n, int n0) {
    for (int h = 0; h < 2; ++h) {
        size_t flat = 0;
        for (int p = 0; p < 4; ++p) {
            const uint8_t *src = planes+h*8192+p*2048;
            size_t remaining = rows*32;
            while (remaining) {
                const int within = flat%128;
                size_t bytes = 128-within;
                if (bytes > remaining) bytes = remaining;
                uint8_t *dst = (uint8_t *)(output+(flat/128)*(size_t)n+n0+h*32)+within;
                if (bytes == 128) *(unaligned_vector *)dst = *(const unaligned_vector *)src;
                else memcpy(dst, src, bytes);
                src += bytes; flat += bytes; remaining -= bytes;
            }
        }
    }
}

typedef struct {
    worker_synctoken_t sync;
    unsigned int next, scratch_next;
    int jobs, phase, rows, k, n, n0, columns;
    uint8_t *activation, *weights, *planes;
    float *scales, *output;
    const float *input, *inverse, *sw;
    const int32_t *sums;
    const uint8_t *source;
    size_t bytes;
    int raw_output;
} linear_task;

static void copy_weight_range(uint8_t *restrict dst, const uint8_t *restrict src,
                               size_t bytes) {
    // Production RPCMEM tensors and packed weight offsets are 128-byte aligned.
    // Ranges consist of complete K32/N64 tiles (multiples of 2048 bytes).
    // Keep an unaligned fallback for direct requests from other callers.
    l2fetch(src,128,128,bytes<4096 ? bytes/128 : 32,0);
    if (((uintptr_t)src & 127) == 0) {
        for (size_t off = 0; off < bytes; off += 512) {
            if (off%4096 == 0 && off+4096 < bytes) {
                size_t remain = bytes-off-4096;
                l2fetch(src+off+4096,128,128,remain<4096 ? remain/128 : 32,0);
            }
            const HVX_Vector *in = (const HVX_Vector *)(src+off);
            HVX_Vector *out = (HVX_Vector *)(dst+off);
            const HVX_Vector v0=in[0], v1=in[1], v2=in[2], v3=in[3];
            out[0]=v0; out[1]=v1; out[2]=v2; out[3]=v3;
        }
    } else {
        for (size_t off = 0; off < bytes; off += 128) {
            if (off%4096 == 0 && off+4096 < bytes) {
                size_t remain = bytes-off-4096;
                l2fetch(src+off+4096,128,128,remain<4096 ? remain/128 : 32,0);
            }
            *(HVX_Vector *)(dst+off) = *(const unaligned_vector *)(src+off);
        }
    }
}

static void linear_worker(void *opaque, int worker_index) {
    (void)worker_index;
    linear_task *s = (linear_task *)opaque;
    const unsigned int scratch_slot = s->phase == 0 ? worker_pool_atomic_inc_return(&s->scratch_next)-1 : 0;
    for (;;) {
        const unsigned int task = worker_pool_atomic_inc_return(&s->next)-1;
        if (task >= s->jobs) break;
        if (s->phase == 0) {
            const int row = task*16;
            int rows = s->rows-row;
            if (rows > 16) rows = 16;
            uint8_t *tile = s->activation+(size_t)(row/64)*64*s->k;
            prepare_tile(tile+(row%64)*32, s->scales+row,
                         s->input+(size_t)row*s->k, s->inverse,
                         (float *)s->planes+(size_t)scratch_slot*4*s->k, rows, s->k);
            if (row+rows == s->rows && s->rows%64) {
                // The final job owns the padded rows. Four-row stores never
                // overlap any other job, including the final partial group.
                const HVX_Vector padding = Q6_V_vsplat_R(0x80808080);
                const int first_padding = ((s->rows%64)+3)/4*4;
                for (int c = 0; c < s->k/32; ++c)
                    for (int r = first_padding; r < 64; r += 4)
                        *(HVX_Vector *)(tile+c*2048+r*32) = padding;
            }
        } else if (s->phase == 1) {
            // Independent contiguous ranges, prefetching the next 4 KiB like W16.
            size_t begin = (size_t)task*65536;
            size_t end = begin+65536 < s->bytes ? begin+65536 : s->bytes;
            copy_weight_range(s->weights+begin,s->source+begin,end-begin);
        } else {
            int tile = task/s->columns, col = task%s->columns;
            int rows = s->rows-tile*64;
            if (rows > 64) rows = 64;
            if (s->raw_output)
                export_plane_tile(s->output+(size_t)tile*64*s->n, s->planes+(size_t)task*16384,
                                  rows, s->n, s->n0+col*64);
            else
                output_tile(s->output+(size_t)tile*64*s->n, s->planes+(size_t)task*16384,
                            s->scales+tile*64, s->sw, s->sums, rows, s->n, s->n0+col*64);
        }
    }
    worker_pool_synctoken_jobdone(&s->sync);
}

static void parallel_phase(linear_task *s, int phase, int jobs) {
    s->phase = phase; s->jobs = jobs; s->next = 0; s->scratch_next = 0;
    int workers = num_hvx128_contexts ? num_hvx128_contexts : 1;
    // Output planes are unused until activation preparation is complete.
    // Reuse that 1 MiB as private four-row scratch for each active worker.
    if (phase == 0 && workers > 1024*1024/(16*s->k)) workers = 1024*1024/(16*s->k);
    if (workers > jobs) workers = jobs;
    worker_pool_synctoken_init(&s->sync,workers);
    worker_pool_job_t job = {linear_worker,s};
    for (int i=0; i<workers; ++i) worker_pool_submit(NULL,job);
    worker_pool_synctoken_wait(&s->sync);
}

static int native_w8a8(float *output, const float *input, const uint8_t *constants,
                       int m, int k, int n, struct NativeW8A8Profile *profile,
                       struct W8A8CpuControl *control) {
    uint64_t phase_start = profile ? HAP_perf_get_qtimer_count() : 0;
    const int detailed = profile && profile->mode == 2;
    if (profile) {
        const uint32_t mode = profile->mode;
        memset(profile, 0, sizeof(*profile));
        profile->mode = mode;
    }
    if (!output || !input || !constants || m < 1 || m > 4096 || k < 32 || k > 6144 ||
        k%32 || n < 64 || n > 16384 || n%64) return -10;
    uint8_t *base = (uint8_t *)vtcm_manager_get_vtcm_base();
    // Known fused-DSP shapes use the offline minimum-weight-traffic table.
    // Other shapes and the opt-in raw-output strategy retain their allocation.
    // All paths reserve the same planes/activation-worker scratch area.
    const size_t vtcm_bytes = vtcm_manager_get_vtcm_size();
    const size_t plane_area_bytes = 1024*1024;
    const size_t weight_bytes = (size_t)k*64;
    const size_t metadata_bytes = 1024+4096*4;
    if (!base || vtcm_bytes < plane_area_bytes+metadata_bytes+2*weight_bytes) return -11;
    // Host resets progress and writes flags through the uncached mapping.
    // Discard the previous call's cached first line before reading flags or
    // updating diagnostics; an eviction must never republish stale progress.
    if (control)
        qurt_mem_cache_clean((qurt_addr_t)control, 128, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    // Keep all weights only when they fit alongside one activation tile.
    // Otherwise retain the optimized activation-first allocation and traversal.
    const size_t full_weight_bytes = (size_t)k*n;
    const int full_resident = control && (control->flags & W8A8_CONTROL_WEIGHT_FIRST) &&
        full_weight_bytes <= vtcm_bytes-plane_area_bytes-metadata_bytes-weight_bytes;
    int max_tiles;
    size_t weight_area_bytes;
    int table_plan = 0;
    if (!control && vtcm_bytes == 8*1024*1024) {
        for (size_t i = 0; i < sizeof(w8a8_allocation_table)/sizeof(w8a8_allocation_table[0]); ++i) {
            const struct w8a8_allocation_entry *entry = w8a8_allocation_table+i;
            if (entry->m == m && entry->k == k && entry->n == n) {
                max_tiles = entry->activation_tiles;
                weight_area_bytes = entry->weight_tiles*weight_bytes;
                table_plan = 1;
                break;
            }
        }
    }
    if (table_plan) {
        // Already selected above; no cross-invocation weight residency.
    } else if (full_resident) {
        weight_area_bytes = full_weight_bytes;
        const int capacity_tiles = (vtcm_bytes-plane_area_bytes-metadata_bytes-full_weight_bytes)/weight_bytes;
        max_tiles = control->resident_batch_rows/64;
        if (max_tiles < 1) max_tiles = 1;
        if (max_tiles > capacity_tiles) max_tiles = capacity_tiles;
        if (max_tiles > (m+63)/64) max_tiles = (m+63)/64;
    } else {
        size_t activation_budget = vtcm_bytes-plane_area_bytes-metadata_bytes-weight_bytes;
        if (activation_budget > 2*1024*1024) activation_budget = 2*1024*1024;
        max_tiles = activation_budget/weight_bytes;
        if (max_tiles > 64) max_tiles = 64;
        if (max_tiles > (m+63)/64) max_tiles = (m+63)/64;
        weight_area_bytes =
            ((vtcm_bytes-max_tiles*weight_bytes-plane_area_bytes-metadata_bytes)/weight_bytes)*weight_bytes;
    }
    const size_t activation_bytes = max_tiles*weight_bytes;
    const size_t required = activation_bytes+weight_area_bytes+plane_area_bytes+1024+4096*4;
    if (vtcm_bytes < required || max_tiles < 1) return -11;
    // Diagnostic mode 3 reuses the existing trailer without changing its ABI.
    // Low/high 16 bits contain A64/W_N64 capacities; bit 31 marks a table hit.
    // mac_ticks/readout_ticks hold load count/bytes in this mode only.
    if (profile && profile->mode == 3)
        profile->reserved = (uint32_t)max_tiles |
            ((uint32_t)(weight_area_bytes/weight_bytes) << 16) |
            (table_plan ? 0x80000000u : 0);
    uint8_t *weights = base+activation_bytes, *planes = weights+weight_area_bytes;
    uint32_t *tables = (uint32_t *)(planes+plane_area_bytes);
    const uint32_t shifts[4] = {0x6000, 0x4000, 0x2000, 0};
    memset(tables, 0, 1024);
    for (int p = 0; p < 4; ++p) for (int c = 0; c < 32; ++c) tables[p*64+c] = shifts[p];
    const struct W8A8Header *header = (const struct W8A8Header *)constants;
    const float *sw = (const float *)(constants+header->scale_offset);
    const int32_t *sums = (const int32_t *)(constants+header->sum_offset);
    const float *inverse = (const float *)(constants+header->inverse_offset);
    float *scales = (float *)(tables+256);
    linear_task state = {.k=k,.n=n,.activation=base,.weights=weights,.planes=planes,
                         .scales=scales,.inverse=inverse,.sw=sw,.sums=sums,
                         .raw_output=control != NULL};
    uint32_t produced = 0;
    if (control) {
        control->vtcm_bytes = vtcm_bytes;
        control->activation_rows = max_tiles*64;
        control->weight_columns = weight_area_bytes/weight_bytes;
        control->weight_loads = 0;
        control->weight_loaded_bytes = 0;
    }
    if (profile) profile->setup_ticks = HAP_perf_get_qtimer_count()-phase_start;
    // Residency is valid only inside this invocation. Activation preparation
    // uses planes as scratch and leaves the separate weight area untouched.
    int resident_n0 = -1, resident_columns = 0;
    int reverse = 0;
    for (int m0 = 0; m0 < m; m0 += max_tiles*64, reverse ^= 1) {
        int chunk_rows = m-m0 < max_tiles*64 ? m-m0 : max_tiles*64;
        int tiles = (chunk_rows+63)/64;
        state.rows = chunk_rows;
        state.input = input+(size_t)m0*k;
        state.output = output+(size_t)m0*n;
        if (profile) phase_start = HAP_perf_get_qtimer_count();
        parallel_phase(&state,0,(chunk_rows+15)/16);
        if (control) {
            memcpy(control->scales+m0, scales, chunk_rows*sizeof(float));
            qurt_mem_cache_clean((qurt_addr_t)(control->scales+m0), chunk_rows*sizeof(float),
                                 QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        }
        if (profile) profile->activation_ticks += HAP_perf_get_qtimer_count()-phase_start;
        int max_columns = weight_area_bytes/weight_bytes;
        const int output_columns = plane_area_bytes/(tiles*16384);
        if (!full_resident && !table_plan && max_columns > output_columns) max_columns = output_columns;
        // Turn around at each M chunk so its first columns are still resident.
        // Keep that boundary chunk's size even when a shorter final M chunk
        // allows more columns; the remaining chunks use the new capacity.
        for (int cursor = reverse ? n : 0; reverse ? cursor > 0 : cursor < n;) {
            int columns = (reverse ? cursor : n-cursor)/64;
            if (columns > max_columns) columns = max_columns;
            if (resident_columns && resident_columns <= max_columns &&
                (reverse ? resident_n0+resident_columns*64 == cursor
                         : resident_n0 == cursor)) columns = resident_columns;
            const int n0 = reverse ? cursor-columns*64 : cursor;
            state.source = constants+header->weight_offset+(size_t)n0*k;
            state.bytes = columns*weight_bytes;
            state.weights = weights;
            state.n0 = n0; state.columns = columns;
            if (resident_n0 != n0 || resident_columns != columns) {
                if (profile) phase_start = HAP_perf_get_qtimer_count();
                parallel_phase(&state,1,(state.bytes+65535)/65536);
                if (profile) profile->weight_ticks += HAP_perf_get_qtimer_count()-phase_start;
                if (profile && profile->mode == 3) {
                    ++profile->mac_ticks;
                    profile->readout_ticks += state.bytes;
                }
                if (control) {
                    ++control->weight_loads;
                    control->weight_loaded_bytes += state.bytes;
                }
                resident_n0 = n0;
                resident_columns = columns;
            }
            for (int col0 = 0; col0 < columns; col0 += output_columns) {
                const int output_batch_columns = columns-col0 < output_columns ? columns-col0 : output_columns;
                state.n0 = n0+col0*64;
                state.columns = output_batch_columns;
                state.weights = weights+(size_t)col0*weight_bytes;
                if (profile) phase_start = HAP_perf_get_qtimer_count();
                hmx_unit_acquire();
                for (int tile = 0; tile < tiles; ++tile) {
                    const uint8_t *activation = base+(size_t)tile*64*k;
                    for (int col = 0; col < output_batch_columns; ++col) {
                        const uint8_t *weight = state.weights+col*weight_bytes;
                        uint64_t tile_start = detailed ? HAP_perf_get_qtimer_count() : 0;
                        Q6_mxclracc();
                        // Compact CM and deep weights preserve the per-K32 bank walk.
                        for (int t = 0; t < k/32; ++t) {
                            asm volatile("{ activation.ub = mxmem(%0,%2):cm\nweight.b = mxmem(%1,%3):deep }" ::
                                "r"(activation+t*2048), "r"(weight+t*2048), "r"(2047), "r"(2047) : "memory");
                        }
                        if (detailed) {
                            asm volatile("barrier" ::: "memory");
                            const uint64_t now = HAP_perf_get_qtimer_count();
                            profile->mac_ticks += now-tile_start;
                            tile_start = now;
                        }
                        read_int32(planes+((size_t)tile*output_batch_columns+col)*16384,tables);
                        if (detailed) {
                            asm volatile("barrier" ::: "memory");
                            profile->readout_ticks += HAP_perf_get_qtimer_count()-tile_start;
                        }
                    }
                }
                asm volatile("barrier" ::: "memory");
                hmx_unit_release();
                if (profile) profile->hmx_ticks += HAP_perf_get_qtimer_count()-phase_start;
                if (profile) phase_start = HAP_perf_get_qtimer_count();
                parallel_phase(&state,2,tiles*output_batch_columns);
                if (profile) profile->output_ticks += HAP_perf_get_qtimer_count()-phase_start;
            }
            cursor = reverse ? n0 : n0+columns*64;
        }
        if (control) {
            // Publish a whole M chunk: one contiguous cache flush replaces
            // thousands of small per-row flushes. Earlier M chunks belong to
            // the CPU now and must never be flushed or written again here.
            qurt_mem_cache_clean((qurt_addr_t)state.output, (size_t)chunk_rows*n*sizeof(float),
                                 QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            if (produced >= W8A8_CPU_MAX_BATCHES) return -12;
            struct W8A8CpuBatch *batch = control->batches+produced;
            *batch = (struct W8A8CpuBatch){m0, chunk_rows, 0, n/64};
            qurt_mem_cache_clean((qurt_addr_t)batch, sizeof(*batch), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            asm volatile("barrier" ::: "memory");
            control->produced = ++produced;
            qurt_mem_cache_clean((qurt_addr_t)control, 128, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            asm volatile("barrier" ::: "memory");
        }
    }
    asm volatile("barrier" ::: "memory");
    return 0;
}

int hmx_mat_mul_native_w8a8_profiled(float *output, const float *input, const uint8_t *constants,
                                    int m, int k, int n, struct NativeW8A8Profile *profile) {
    return native_w8a8(output, input, constants, m, k, n, profile, NULL);
}
int hmx_mat_mul_native_w8a8_cpu(float *output, const float *input, const uint8_t *constants,
                               int m, int k, int n, struct W8A8CpuControl *control) {
    if (!control) return -12;
    return native_w8a8(output, input, constants, m, k, n, NULL, control);
}
#else
int hmx_mat_mul_native_w8a8_cpu(float *output, const float *input, const uint8_t *constants,
                               int m, int k, int n, struct W8A8CpuControl *control) {
    (void)output; (void)input; (void)constants; (void)m; (void)k; (void)n; (void)control;
    return -75;
}
int hmx_mat_mul_native_w8a8_profiled(float *output, const float *input, const uint8_t *constants,
                                    int m, int k, int n, struct NativeW8A8Profile *profile) {
    (void)output; (void)input; (void)constants; (void)m; (void)k; (void)n;
    (void)profile;
    return -75;
}
#endif

int hmx_mat_mul_native_w8a8(float *output, const float *input, const uint8_t *constants,
                           int m, int k, int n) {
    return hmx_mat_mul_native_w8a8_profiled(output, input, constants, m, k, n, NULL);
}
