#include <stdint.h>
#include <string.h>
#include <hmx_hexagon_protos.h>
#include <HAP_compute_res.h>
#include <qurt_hvx.h>
#include <HAP_perf.h>
#include "hmx_probe.h"
#include "dsp/power.h"
#include "dsp/hmx_mgr.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

typedef struct {
    worker_synctoken_t sync;
    int mode, a, b, err, context;
    uint32_t bias, bias_high;
    uint8_t *output;
    const uint8_t *activation, *weight;
    int kind, alimit, wlimit, convert, store, repeats;
} probe_state;
static AEEResult execute(probe_state *state);
int hmx_study_run(uint8_t *vtcm, int kind, int columns, int ktiles,
                  int epilogue, int seed, int iterations, uint8_t *output);

static void benchmark(probe_state *s, uint8_t *a, uint8_t *b, uint8_t *c, uint32_t *bias) {
    const int floating = s->kind < 6 || s->kind == 9;
    const int compact = s->kind == 10 || s->kind == 11;
    const int row_tiles = s->convert ? 2 : 1;
    memset(a, 0, 8192); memset(b, 0, 8192);
    for (int t=0; t<4; ++t) {
        if (compact) memset(a+t*2048,1,2048);
        for (int i=0; i<1024; ++i) {
            if (s->kind == 9 || s->kind == 2) ((__fp16 *)(a+t*2048))[i] = 1;
            else if (compact) continue;
            else a[t*2048+i*2+(floating ? 0 : 1)] = floating ? 0x78 : 1;
        }
        int wbytes = s->kind == 9 ? 2048 : s->kind == 7 || s->kind == 11 ? 512 : 1024;
        if (s->kind == 9) for (int i=0; i<1024; ++i) ((__fp16 *)(b+t*wbytes))[i] = 1;
        else memset(b+t*wbytes, s->kind == 7 || s->kind == 11 ? 0x11 : floating ? 0x78 : 1, wbytes);
    }
    for (int i=0; i<32; ++i) {bias[i] = floating ? 0x3c00 : compact ? 0x6000 : 0x4000; bias[i+32]=0;}
    if (floating || compact) Q6_bias_mxmem2_A(bias);
    else asm volatile("bias = mxmem(%0)\n" :: "r"(bias) : "memory");
    uint64_t q0=HAP_perf_get_qtimer_count(), p0=HAP_perf_get_pcycles();
    const int kt = s->alimit;
#define BENCH(ACT, WEI, WB, WL, ASUF) \
    for (int rep=0; rep<s->repeats+100; ++rep) { \
        if (rep==100) { asm volatile("barrier" ::: "memory"); q0=HAP_perf_get_qtimer_count(); p0=HAP_perf_get_pcycles(); } \
        for (int rtile=0;rtile<row_tiles;++rtile) { \
        if (floating) Q6_mxclracc_hf(); else Q6_mxclracc(); \
        for (int t=0; t<kt; ++t) { \
            asm volatile("{ activation." ACT " = mxmem(%0,%2)" ASUF "\nweight." WEI " = mxmem(%1,%3) }\n" \
                :: "r"(a+t*2048), "r"(b+t*(WB)), "r"(2047), "r"(WL) : "memory"); \
        } \
        if (floating) {Q6_cvt_hf_acc_R(2);Q6_mxmem_cvt_RR(c+rtile*2048,0);} \
        else if (compact) {Q6_cvt_ub_acc_R(0);Q6_mxmem_cvt_RR_cm(c,0);} \
        else if (s->store == 5) { \
            asm volatile("bias = mxmem(%0)\n" :: "r"(bias) : "memory"); \
            asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1\n" :: "r"(c),"r"(2047) : "memory"); \
            asm volatile("bias = mxmem(%0)\n" :: "r"(bias+64) : "memory"); \
            asm volatile("mxmem(%0,%1):after.uh = acc:2x1\n" :: "r"(c+2048),"r"(2047) : "memory"); \
        } else asm volatile("mxmem(%0,%1):after.uh = acc:2x1\n" :: "r"(c),"r"(2047) : "memory"); \
        } \
    }
    switch (s->kind) {
#if __HMX_ARCH__ >= 79
        case 0: BENCH("f8", "f8",1024,1023,""); break;
        case 2: BENCH("hf", "f8",1024,1023,""); break;
#endif
        case 6: BENCH("ub", "b",1024,1023,""); break;
        case 7: BENCH("ub", "n",512,511,""); break;
        case 9: BENCH("hf", "hf",2048,2047,""); break;
        case 10: BENCH("ub", "b",1024,1023,":cm"); break;
        case 11: BENCH("ub", "n",512,511,":cm"); break;
    }
#undef BENCH
    asm volatile("barrier" ::: "memory");
    uint64_t p1=HAP_perf_get_pcycles(), q1=HAP_perf_get_qtimer_count();
    memcpy(s->output,c,s->store==5 || row_tiles==2?4096:2048);
    uint64_t stats[4]={q1-q0,p1-p0,(uint64_t)s->repeats,(uint64_t)kt};
    memcpy(s->output+4096-sizeof(stats),stats,sizeof(stats));
}

int32_t qhblas_hvx_matrix_matrix_mpy_ab(int8_t *, int8_t *, int8_t *, uint32_t, uint32_t, uint32_t);

static void probe_worker(void *data, int index) {
    (void)index;
    probe_state *s = data;
    uint8_t *vtcm = vtcm_manager_get_vtcm_base();
    uint8_t *a = vtcm, *b = vtcm + 8192, *c = vtcm + 16384;
    uint32_t *bias = (uint32_t *)(vtcm + 24576);
    memset(vtcm, 0, 32768);
    memset(c, 0xa5, 4096);
    if (s->mode == 4 || s->mode == 5) {
        const int k = s->mode == 4 ? 64 : 31;
        const int n = s->mode == 4 ? 128 : 65;
        for (int r = 0; r < 3; ++r)
            for (int d = 0; d < k; ++d)
                ((int8_t *)a)[r*k+d] = (r*11+d*7)%27-13;
        for (int d = 0; d < k; ++d)
            for (int col = 0; col < n; ++col)
                ((int8_t *)b)[d*n+col] = (d*13+col*5)%19-9;
        s->err = qurt_hvx_lock(QURT_HVX_MODE_128B);
        if (!s->err) {
            s->err = qhblas_hvx_matrix_matrix_mpy_ab((int8_t *)a, (int8_t *)b, (int8_t *)c, 3, k, n);
            qurt_hvx_unlock();
            memcpy(s->output, c, 4096);
        }
        worker_pool_synctoken_jobdone(&s->sync);
        return;
    }
    s->err = HAP_compute_res_hmx_lock2(s->context, HAP_COMPUTE_RES_HMX_SHARED);
    if (!s->err) {
        if (s->mode == 9) {
            s->err=hmx_study_run(vtcm,s->kind,s->wlimit,s->alimit,s->convert,s->a,s->repeats,s->output);
            HAP_compute_res_hmx_unlock2(s->context, HAP_COMPUTE_RES_HMX_SHARED);
            worker_pool_synctoken_jobdone(&s->sync);
            return;
        } else if (s->mode == 8) {
            benchmark(s,a,b,c,bias);
            HAP_compute_res_hmx_unlock2(s->context, HAP_COMPUTE_RES_HMX_SHARED);
            worker_pool_synctoken_jobdone(&s->sync);
            return;
        } else if (s->mode == 7) {
#if __HMX_ARCH__ >= 79
            memcpy(a, s->activation, 8192);
            memcpy(b, s->weight, 8192);
            for (int i = 0; i < 32; ++i) {
                bias[i] = s->bias;
                bias[i+32] = s->bias_high;
            }
            Q6_bias_mxmem2_A(bias);
            if (s->kind < 6 || s->kind == 9 || s->kind == 12) Q6_mxclracc_hf();
            else Q6_mxclracc();
            uint8_t *a_base=a, *b_base=b;
            for (int i = 0; i < (s->repeats & 255); ++i) {
                if (s->repeats & 256) {
                    a=a_base+i*2048;
                    b=b_base+i*(s->kind==7 || s->kind==11?512:s->kind==9 || s->kind==3?2048:1024);
                }
                if (s->kind == 12) {
                    asm volatile("{ activation.f8 = mxmem(%0, %2):deep\n"
                                 "weight.f8 = mxmem(%1, %3) }\n"
                                 :: "r"(a+1), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 0) {
                    asm volatile("{ activation.f8 = mxmem(%0, %2):deep\n"
                                 "weight.f8 = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 1) {
                    asm volatile("{ activation.f8 = mxmem(%0, %2)\n"
                                 "weight.f8 = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 2) {
                    asm volatile("{ activation.hf = mxmem(%0, %2):deep\n"
                                 "weight.f8 = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 3) {
                    asm volatile("{ activation.f8 = mxmem(%0, %2):deep\n"
                                 "weight.hf = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 4) {
                    asm volatile("{ activation.f8 = mxmem(%0, %2):deep\n"
                                 "weight.f8 = mxmem(%1, %3):deep }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 5) {
                    asm volatile("{ activation.f8 = mxmem(%0, %2)\n"
                                 "weight.f8 = mxmem(%1, %3):deep }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 9) {
                    asm volatile("{ activation.hf = mxmem(%0, %2):deep\n"
                                 "weight.hf = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 6) {
                    asm volatile("{ activation.ub = mxmem(%0, %2)\n"
                                 "weight.b = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 7) {
                    asm volatile("{ activation.ub = mxmem(%0, %2)\n"
                                 "weight.n = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 8) {
                    asm volatile("{ activation.ub = mxmem(%0, %2)\n"
                                 "weight.n = mxmem(%1, %3):2x }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else if (s->kind == 10) {
                    asm volatile("{ activation.ub = mxmem(%0, %2):cm\n"
                                 "weight.b = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                } else {
                    asm volatile("{ activation.ub = mxmem(%0, %2):cm\n"
                                 "weight.n = mxmem(%1, %3) }\n"
                                 :: "r"(a), "r"(b), "r"(s->alimit), "r"(s->wlimit) : "memory");
                }
            }
            if (s->kind < 6 || s->kind == 9 || s->kind == 12) {
                Q6_cvt_hf_acc_R(s->convert);
                if (s->store) Q6_mxmem_cvt_RR_deep(c, 0);
                else Q6_mxmem_cvt_RR(c, 0);
            } else if (s->store >= 2) {
                if (s->store == 2) memset(bias, s->bias & 255, 2048);
                asm volatile("bias = mxmem(%0)\n" :: "r"(bias) : "memory");
                if(s->store == 5) {
                    /* Low bits at unity; retained accumulator at scale 2^-16 gives the signed high word. */
                    asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1\n" :: "r"(c), "r"(2047) : "memory");
                    for(int j=0;j<32;++j)bias[j]=0;
                    asm volatile("bias = mxmem(%0)\n" :: "r"(bias) : "memory");
                    asm volatile("mxmem(%0,%1):after.uh = acc:2x1\n" :: "r"(c+2048), "r"(2047) : "memory");
                } else if(s->store == 4)
                    asm volatile("mxmem(%0,%1):after.uh = acc:2x2\n" :: "r"(c), "r"(4095) : "memory");
                else asm volatile("mxmem(%0,%1):after.uh = acc:2x1\n" :: "r"(c), "r"(2047) : "memory");
            } else {
                Q6_cvt_ub_acc_R(s->convert);
                if (s->store) Q6_mxmem_cvt_RR_cm(c, 0);
                else Q6_mxmem_cvt_RR(c, 0);
            }
#endif
        } else if (s->mode == 0 || s->mode == 6) {
            for (int i = 0; i < 1024; ++i) {
                ((__fp16 *)a)[i] = (__fp16)1;
                ((__fp16 *)b)[i] = (__fp16)1;
            }
            for (int i = 0; i < 32; ++i) bias[i] = 0x00003c00;
            Q6_bias_mxmem2_A(bias);
            if (s->mode == 0) {
                asm volatile("{ activation.hf = mxmem(%0, %2):deep\n"
                         "weight.hf = mxmem(%1, %2) }\n"
                         :: "r"(a), "r"(b), "r"(2047) : "memory");
            } else {
#if __HMX_ARCH__ >= 79
                memset(a, s->a, 8192);
                memset(b, s->b, 8192);
                asm volatile("{ activation.f8 = mxmem(%0, %2):deep\n"
                             "weight.f8 = mxmem(%1, %2) }\n"
                             :: "r"(a), "r"(b), "r"(2047) : "memory");
#endif
            }
            Q6_cvt_hf_acc_R(2);
            Q6_mxmem_cvt_RR(c, 0);
        } else {
            memset(a, s->a, 8192);
            memset(b, s->b, 8192);
            for (int i = 0; i < 32; ++i) {
                bias[i] = s->bias;
                bias[i+32] = s->bias_high;
            }
            Q6_bias_mxmem2_A(bias);
            Q6_mxclracc();
            if (s->mode == 1) {
                asm volatile("{ activation.ub = mxmem(%0, %2):cm\n"
                             "weight.b = mxmem(%1, %2) }\n"
                             :: "r"(a), "r"(b), "r"(2047) : "memory");
            } else if (s->mode == 2) {
                asm volatile("{ activation.ub = mxmem(%0, %2):cm\n"
                             "weight.n = mxmem(%1, %3) }\n"
                             :: "r"(a), "r"(b), "r"(2047), "r"(1023) : "memory");
            } else {
                asm volatile("{ activation.ub = mxmem(%0, %2):cm\n"
                             "weight.n = mxmem(%1, %3):2x }\n"
                             :: "r"(a), "r"(b), "r"(2047), "r"(1023) : "memory");
            }
            Q6_cvt_ub_acc_R(2);
            Q6_mxmem_cvt_RR_cm(c, 0);
        }
        asm volatile("barrier" ::: "memory");
        memcpy(s->output, c, 4096);
        HAP_compute_res_hmx_unlock2(s->context, HAP_COMPUTE_RES_HMX_SHARED);
    }
    worker_pool_synctoken_jobdone(&s->sync);
}

AEEResult hmx_probe_bench(remote_handle64 handle, int32 kind, int32 k_tiles, int32 iterations,
                        uint8 *output, int outputLen) {
    (void)handle;
    int store=0;
    int row_tiles=1;
    if(kind==19){kind=9;row_tiles=2;}
    if(kind==16 || kind==17){kind-=10;store=5;}
#if __HMX_ARCH__ < 79
    if (kind == 0 || kind == 2) return AEE_EUNSUPPORTED;
#endif
    if ((kind!=0 && kind!=2 && kind!=6 && kind!=7 && kind!=9 && kind!=10 && kind!=11) || k_tiles<1 || k_tiles>4 ||
        iterations<100 || iterations>1000000 || outputLen!=4096) return AEE_EBADPARM;
    probe_state state={.mode=8,.kind=kind,.alimit=k_tiles,.repeats=iterations,.output=output,.store=store,.convert=row_tiles==2};
    return execute(&state);
}

AEEResult hmx_probe_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    static int dummy_handle;
    *handle = (remote_handle64)(uintptr_t)&dummy_handle;
    return AEE_SUCCESS;
}

AEEResult hmx_probe_close(remote_handle64 handle) {
    (void)handle;
    return AEE_SUCCESS;
}

static AEEResult execute(probe_state *state) {
    int mode = state->mode;
    power_setup();
    int err = vtcm_manager_setup_fraction(1, mode>=9?2:16);
    if (err) err = 0x10000 | (err & 0xffff);
    if (!err) err = vtcm_manager_get_vtcm_size() < (mode>=9?2097152:32768) ? AEE_ENOMEMORY : 0;
    int context = 0;
    compute_res_attr_t request;
    if (!err && mode != 4 && mode != 5) {
        err = HAP_compute_res_attr_init(&request);
        if (!err) err = HAP_compute_res_attr_set_hmx_param(&request, 1);
        if (!err) {
            context = HAP_compute_res_acquire(&request, 10000);
            if (!context) err = AEE_EFAILED;
        }
        if (err) err = 0x20000 | (err & 0xffff);
    }
    if (!err) {
        state->context = context;
        worker_pool_synctoken_init(&state->sync, 1);
        worker_pool_job_t job = {.fptr=probe_worker, .dptr=state};
        err = worker_pool_submit(NULL, job);
        if (!err) {
            worker_pool_synctoken_wait(&state->sync);
            err = state->err ? 0x30000 | (state->err & 0xffff) : 0;
        }
    }
    if (context) HAP_compute_res_release(context);
    vtcm_manager_reset();
    power_reset();
    return err;
}

AEEResult hmx_probe_study(remote_handle64 handle, int32 kind, int32 columns, int32 k_tiles,
                        int32 epilogue, int32 seed, int32 iterations, uint8 *output, int outputLen) {
    (void)handle;
    if(kind<0 || kind>4 || (columns!=32 && columns!=64) || k_tiles<1 || k_tiles>192 ||
       epilogue<0 || epilogue>3 || seed<0 || seed>31 || iterations<1 || iterations>100000 ||
       outputLen!=16416 || (epilogue>=2 && seed>=30)) return AEE_EBADPARM;
    probe_state state={.mode=9,.kind=kind,.wlimit=columns,.alimit=k_tiles,.convert=epilogue,
                       .a=seed,.repeats=iterations,.output=output};
    return execute(&state);
}

AEEResult hmx_probe_run(remote_handle64 handle, int32 mode, int32 activation_fill,
                       int32 weight_fill, uint32 bias_fill, uint32 bias_high, uint8 *output, int outputLen) {
    (void)handle;
    if (mode < 0 || mode > 6 || outputLen != 4096) return AEE_EBADPARM;
#if __HMX_ARCH__ < 79
    if (mode == 6) return AEE_EUNSUPPORTED;
#endif
    probe_state state = {.mode=mode, .a=activation_fill, .b=weight_fill,
                         .bias=bias_fill, .bias_high=bias_high, .output=output};
    return execute(&state);
}

AEEResult hmx_probe_tile(remote_handle64 handle, const uint8 *activation, int activationLen,
                       const uint8 *weight, int weightLen, int32 kind, int32 activation_limit,
                       int32 weight_limit, int32 convert, int32 store, int32 repeats,
                       uint32 bias_low, uint32 bias_high, uint8 *output, int outputLen) {
    (void)handle;
#if __HMX_ARCH__ < 79
    return AEE_EUNSUPPORTED;
#else
    if (activationLen != 8192 || weightLen != 8192 || outputLen != 4096 ||
        kind < 0 || kind > 12 || activation_limit < 127 || activation_limit > 8191 ||
        weight_limit < 127 || weight_limit > 8191 ||
        !((repeats>=1 && repeats<=8) || (repeats>=257 && repeats<=260)) ||
        convert < 0 || convert > 0xfffff || store < 0 || store > 5) return AEE_EBADPARM;
    probe_state state = {.mode=7, .activation=activation, .weight=weight, .output=output,
        .kind=kind, .alimit=activation_limit, .wlimit=weight_limit, .convert=convert,
        .store=store, .repeats=repeats, .bias=bias_low, .bias_high=bias_high};
    return execute(&state);
#endif
}
