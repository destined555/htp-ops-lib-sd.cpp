#include <stdint.h>
#include <string.h>
#include <hmx_hexagon_protos.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <qurt_hvx.h>
#include <HAP_perf.h>
#include "study_data.h"

/* Four retained nonsaturating CM reads expose exact accumulator byte planes.
 * Recipe: onnxsim/tinygrad c817073e, hand/hmx/hmx_qconv.h; verified here on v79. */
static void read_planes(uint8_t *scratch,uint32_t *table,int banks) {
    for(int h=0;h<banks;++h) for(int p=0;p<4;++p) {
        Q6_bias_mxmem2_A(table+(h*4+p)*64);
        if(p<3) asm volatile("mxmem(%0,%1):after:retain:cm.ub = acc" :: "r"(scratch+h*8192+p*2048),"r"(0):"memory");
        else asm volatile("mxmem(%0,%1):after:cm.ub = acc" :: "r"(scratch+h*8192+p*2048),"r"(0):"memory");
    }
}

static void reconstruct(uint8_t *scratch,uint8_t *result,int banks,int fp16,const int32_t *correction) {
    HVX_Vector magic=Q6_V_vsplat_R(0x4b400000), scale=Q6_V_vsplat_R(0x39800000); /* 1/4096 */
    for(int h=0;h<banks;++h) for(int q=0;q<16;++q) {
        const HVX_Vector *p=(const HVX_Vector *)(scratch+h*8192+q*128);
        HVX_VectorPair b01=Q6_W_vshuff_VVR(p[16],p[0],-1), b23=Q6_W_vshuff_VVR(p[48],p[32],-1);
        HVX_VectorPair w0=Q6_W_vshuff_VVR(Q6_V_lo_W(b23),Q6_V_lo_W(b01),-2);
        HVX_VectorPair w1=Q6_W_vshuff_VVR(Q6_V_hi_W(b23),Q6_V_hi_W(b01),-2);
        HVX_Vector v[4]={Q6_V_lo_W(w0),Q6_V_hi_W(w0),Q6_V_lo_W(w1),Q6_V_hi_W(w1)};
        if(!fp16) {
            HVX_Vector *dst=(HVX_Vector *)(result+h*8192+q*512);
            for(int i=0;i<4;++i) dst[i]=v[i];
        } else {
            HVX_Vector corr=*(const HVX_Vector *)(correction+h*32);
            for(int i=0;i<4;++i) {
                HVX_Vector x=Q6_Vw_vsub_VwVw(v[i],corr);
                /* This exact conversion requires corrected sums in [-2^22,2^22].
                 * The RPC rejects the extreme seeds for this epilogue. */
                v[i]=Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(Q6_Vw_vadd_VwVw(x,magic),magic),scale);
            }
            HVX_Vector *dst=(HVX_Vector *)(result+h*4096+q*256);
            dst[0]=Q6_Vhf_vcvt_VsfVsf(v[0],v[1]);
            dst[1]=Q6_Vhf_vcvt_VsfVsf(v[2],v[3]);
        }
    }
}

static void quantize_pack(const uint8_t *source,uint8_t *a,int kt) {
    HVX_Vector scale=Q6_V_vsplat_R(0x41800000),zp=Q6_V_vsplat_R(0x43000000),magic=Q6_V_vsplat_R(0x4b400000);
    for(int t=0;t<kt;++t) for(int q=0;q<16;++q) {
        const HVX_Vector *src=(const HVX_Vector *)(source+t*4096+q*256);
        HVX_VectorPair x0=Q6_Wsf_vcvt_Vhf(src[0]),x1=Q6_Wsf_vcvt_Vhf(src[1]);
        HVX_Vector v[4]={Q6_V_lo_W(x0),Q6_V_hi_W(x0),Q6_V_lo_W(x1),Q6_V_hi_W(x1)};
        for(int i=0;i<4;++i) {
            HVX_Vector x=Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v[i],scale),zp);
            v[i]=Q6_Vw_vsub_VwVw(Q6_Vsf_vadd_VsfVsf(x,magic),magic);
        }
        HVX_Vector h0=Q6_Vh_vpack_VwVw_sat(v[1],v[0]),h1=Q6_Vh_vpack_VwVw_sat(v[3],v[2]);
        *(HVX_Vector *)(a+t*2048+q*128)=Q6_Vub_vpack_VhVh_sat(h1,h0);
    }
}

int hmx_study_run(uint8_t *vtcm,int kind,int columns,int kt,int epi,int seed,int iterations,uint8_t *out) {
    int floating=kind==0 || kind>=3;
    int banks=columns/32, wb=floating?2048:kind==1?1024:512;
    uint8_t *a=vtcm, *w=vtcm+786432, *scratch=vtcm+1572864, *result=scratch+32768;
    uint32_t *table=(uint32_t *)(scratch+65536);
    int32_t *correction=(int32_t *)(scratch+69632);
    memset(a,0,786432); memset(w,0,786432); memset(scratch,0,73728);
    const uint8_t *source=vtcm;
    if(!floating && epi==3){a=vtcm+786432;w=vtcm+1179648;}
    for(int t=0;t<kt;++t) {
        for(int r=0;r<64;++r) for(int k=0;k<32;++k) {
            int av=study_a(r,t*32+k,seed);
            if(floating) ((__fp16 *)a)[(r/32)*kt*1024+t*1024+((r%32)&~1)*32+k*2+(r&1)]=(__fp16)((av-128)/16.0f);
            else {
                a[t*2048+r*32+k]=av;
                if(epi==3) ((__fp16 *)vtcm)[t*2048+(r/32)*1024+((r%32)&~1)*32+k*2+(r&1)]=(__fp16)((av-128)/16.0f);
            }
        }
        for(int h=0;h<banks;++h) for(int k=0;k<32;++k) for(int n=0;n<32;++n) {
            int v=study_w(t*32+k,h*32+n,seed,floating?2:kind);
            correction[h*32+n]+=128*v;
            uint8_t *tile=w+(t*banks+h)*wb;
            if(floating) ((__fp16 *)tile)[(k&~1)*32+n*2+(k&1)]=(__fp16)(v/256.0f);
            else if(kind==1) tile[(k/4)*128+n*4+k%4]=(uint8_t)v;
            else tile[(k/8)*128+n*4+k%4]|=(v&15)<<((k&4)?4:0);
        }
    }
    /* Integer readout treats exponent zero with an implicit leading one;
     * 0x0000, not IEEE-half 0x0200, gives /2^24 here. */
    const uint16_t scales[4]={0x6000,0x4000,0x2000,0x0000};
    for(int h=0;h<banks;++h) for(int p=0;p<4;++p) for(int n=0;n<32;++n) {
        table[(h*4+p)*64+n]=floating?0x3c00:scales[p];
        table[(h*4+p)*64+n+32]=!floating && epi==0?(uint32_t)-correction[h*32+n]:0;
    }
    int err=qurt_hvx_lock(QURT_HVX_MODE_128B);
    if(err) return err;
    uint64_t q0=0,p0=0;
    for(int rep=0;rep<iterations+100;++rep) {
        if(rep==100){asm volatile("barrier" ::: "memory");q0=HAP_perf_get_qtimer_count();p0=HAP_perf_get_pcycles();}
        if(kind>=3) {
            for(int r=0;r<2;++r) {
                Q6_bias_mxmem2_A(table); Q6_mxclracc_hf();
                for(int t=0;t<kt;) {
                    int chunk=kind==4?(kt-t<32/banks?kt-t:32/banks):1;
                    if(banks==2) asm volatile("{ activation.hf = mxmem(%0,%2):deep\nweight.hf = mxmem(%1,%3):deep }" ::
                        "r"(a+(r*kt+t)*2048),"r"(w+t*banks*2048),"r"(chunk*2048-1),"r"(chunk*banks*2048-1):"memory");
                    else asm volatile("{ activation.hf = mxmem(%0,%2):deep\nweight.hf = mxmem(%1,%3) }" ::
                        "r"(a+(r*kt+t)*2048),"r"(w+t*2048),"r"(chunk*2048-1),"r"(chunk*2048-1):"memory");
                    t+=chunk;
                }
                for(int h=0;h<banks;++h) {
                    Q6_cvt_hf_acc_R(2);Q6_mxmem_cvt_RR(result+(h*2+r)*2048,0);
                }
            }
        } else if(kind==0) {
            for(int r=0;r<2;++r) for(int h=0;h<banks;++h) {
                Q6_bias_mxmem2_A(table); Q6_mxclracc_hf();
                for(int t=0;t<kt;++t) asm volatile("{ activation.hf = mxmem(%0,%2):deep\nweight.hf = mxmem(%1,%3) }" ::
                    "r"(a+(r*kt+t)*2048),"r"(w+(t*banks+h)*2048),"r"(2047),"r"(2047):"memory");
                Q6_cvt_hf_acc_R(2);Q6_mxmem_cvt_RR(result+(h*2+r)*2048,0);
            }
        } else {
            if(epi==3) quantize_pack(source,a,kt);
            Q6_mxclracc();
            for(int t=0;t<kt;++t) {
                if(kind==1 && banks==2) asm volatile("{ activation.ub = mxmem(%0,%2):cm\nweight.b = mxmem(%1,%3):deep }" ::
                    "r"(a+t*2048),"r"(w+t*banks*wb),"r"(2047),"r"(banks*wb-1):"memory");
                else if(kind==2 && banks==2) asm volatile("{ activation.ub = mxmem(%0,%2):cm\nweight.n = mxmem(%1,%3):deep }" ::
                    "r"(a+t*2048),"r"(w+t*banks*wb),"r"(2047),"r"(banks*wb-1):"memory");
                else if(kind==1) asm volatile("{ activation.ub = mxmem(%0,%2):cm\nweight.b = mxmem(%1,%3) }" ::
                    "r"(a+t*2048),"r"(w+t*wb),"r"(2047),"r"(wb-1):"memory");
                else asm volatile("{ activation.ub = mxmem(%0,%2):cm\nweight.n = mxmem(%1,%3) }" ::
                    "r"(a+t*2048),"r"(w+t*wb),"r"(2047),"r"(wb-1):"memory");
            }
            if(epi==0) for(int h=0;h<banks;++h) {
                Q6_bias_mxmem2_A(table+h*256);
                asm volatile("mxmem(%0,%1):after:cm:sat.ub = acc" :: "r"(result+h*2048),"r"(0):"memory");
            } else {
                read_planes(scratch,table,banks);
                asm volatile("barrier" ::: "memory");
                reconstruct(scratch,result,banks,epi>=2,correction);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
    uint64_t p1=HAP_perf_get_pcycles(),q1=HAP_perf_get_qtimer_count();
    memcpy(out,result,16384);
    uint64_t stats[4]={q1-q0,p1-p0,(uint64_t)iterations,(uint64_t)kt};
    memcpy(out+16384,stats,sizeof(stats));
    qurt_hvx_unlock();
    return 0;
}
