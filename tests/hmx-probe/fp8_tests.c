#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hmx_probe.h"

static uint8_t a[8192], w[8192], out[4096];
static int kind, limit = 2047, convert = 2, store, repeats = 1;
static uint32_t bias = 0x3c00, bias_high;
static float half(uint16_t h);

static float fp8(uint8_t v) {
    if (v == 0x80) return NAN;
    int e = (v >> 3) & 15, m = v & 7;
    float x = e ? ldexpf(1 + m/8.0f, e-15) : ldexpf(m, -17);
    return v & 128 ? -x : x;
}

/* Byte offsets established by one-hot probes, for kind 0/1 and a 32-wide K tile. */
static int ai(int r, int k) { return 2*((r & ~1)*32 + k*2 + (r & 1)); }
static int wi(int k, int n) { return (k & ~3)*32 + n*4 + ((k & 1)*2) + ((k & 2)/2); }
static int ci(int r, int n) { return (r & ~1)*32 + n*2 + (r & 1); }

static int call(remote_handle64 handle) {
    return hmx_probe_tile(handle, a, sizeof(a), w, sizeof(w), kind, limit, limit,
                         convert, store, repeats, bias, bias_high, out, sizeof(out));
}

static uint32_t rng(uint32_t *seed) {
    *seed = *seed*1664525u + 1013904223u;
    return *seed;
}

static int validate(remote_handle64 handle) {
    int failed = 0, cases = 0;
    kind = 0; limit = 2047; convert = 2; store = 0; repeats = 1;
    for (int r = 0; r < 32; ++r) for (int k = 0; k < 32; ++k) {
        int n = (r*7 + k*11) & 31;
        memset(a, 0, sizeof(a)); memset(w, 0, sizeof(w));
        a[ai(r,k)] = w[wi(k,n)] = 0x78;
        int err = call(handle);
        if (err) { printf("onehot RPC=0x%x\n", err); return 1; }
        uint16_t *h = (uint16_t *)out;
        for (int i = 0; i < 1024; ++i)
            if (h[i] != (i == ci(r,n) ? 0x3c00 : 0)) { ++failed; break; }
        ++cases;
    }
    printf("FP8 onehot M/N/K packing: %s cases=%d failures=%d\n", failed ? "FAIL" : "PASS", cases, failed);
    int encoding_fail = 0;
    memset(w, 0, sizeof(w));
    for (int k = 0; k < 32; ++k) for (int n = 0; n < 32; ++n) w[wi(k,n)] = 0x78;
    for (int v = 0; v < 256; ++v) {
        memset(a, 0, sizeof(a));
        for (int r = 0; r < 32; ++r) for (int k = 0; k < 32; ++k) a[ai(r,k)] = v;
        int err = call(handle);
        if (err) { printf("encoding RPC=0x%x\n", err); return 1; }
        __fp16 ref = (__fp16)(32*fp8(v));
        uint16_t bits; memcpy(&bits, &ref, 2);
        uint16_t *h = (uint16_t *)out;
        for (int i = 0; i < 1024; ++i) {
            if (v == 0x80 ? !isnan(half(h[i])) : h[i] != bits) { ++encoding_fail; break; }
        }
    }
    printf("FP8 all 256 encodings (0x80=NaN): %s failures=%d\n", encoding_fail ? "FAIL" : "PASS", encoding_fail);
    failed += encoding_fail;
    for (int test = 0; test < 36; ++test) {
        uint8_t ar[1024], wr[1024];
        uint32_t seed = 1234u + test;
        kind = test/12; /* FP8/FP8, regular activation, FP16/FP8. */
        repeats = 1 + (test%3);
        store = (test/3)%2;
        bias = test%4 == 0 ? 0x3800 : test%4 == 1 ? 0x4000 : 0x3c00;
        if (test%4 == 2) bias |= 0x38000000; /* +0.5 output bias. */
        memset(a, 0, sizeof(a)); memset(w, 0, sizeof(w));
        for (int r = 0; r < 32; ++r) for (int k = 0; k < 32; ++k) {
            uint32_t x = rng(&seed);
            uint8_t v = 0x50 + ((x >> 16) % 48);
            if (x & 0x80000000u) v |= 128;
            ar[r*32+k] = v;
            if (kind == 2) {
                __fp16 value = (__fp16)fp8(v);
                memcpy(a + ai(r,k), &value, 2);
            } else a[ai(r,k)] = v;
        }
        for (int k = 0; k < 32; ++k) for (int n = 0; n < 32; ++n) {
            uint32_t x = rng(&seed);
            uint8_t v = 0x50 + ((x >> 16) % 48);
            if (x & 0x80000000u) v |= 128;
            wr[k*32+n] = w[wi(k,n)] = v;
        }
        int err = call(handle);
        if (err) { printf("random RPC=0x%x\n", err); return 1; }
        int mismatch = 0, ulp_mismatch = 0;
        float max_error = 0;
        uint16_t *h = (uint16_t *)out;
        for (int r = 0; r < 32; ++r) for (int n = 0; n < 32; ++n) {
            float sum = 0;
            for (int k = 0; k < 32; ++k) sum += fp8(ar[r*32+k])*fp8(wr[k*32+n]);
            __fp16 expected = (__fp16)(sum*repeats*half(bias & 0xffff) + half(bias >> 16));
            uint16_t bits; memcpy(&bits, &expected, 2);
            uint16_t got = h[ci(r,n)];
            if (got != bits) {
                ++ulp_mismatch;
                if ((bias >> 16) == 0 || (got > bits ? got-bits : bits-got) > 1) {
                    if (mismatch < 2) printf("mismatch test=%d r=%d n=%d expected=%04x got=%04x\n", test,r,n,bits,got);
                    ++mismatch;
                }
            }
            float error = fabsf(half(got) - (float)expected);
            if (error > max_error) max_error = error;
        }
        printf("FP8 seeded GEMM test=%d kind=%d repeats=%d store=%d bias=%08x %s failures=%d bit_differences=%d max_abs=%g\n",
               test,kind,repeats,store,bias,mismatch ? "FAIL" : "PASS",mismatch,ulp_mismatch,max_error);
        failed += mismatch != 0;
    }
    printf("FP8 validation: %s failures=%d\n", failed ? "FAIL" : "PASS", failed);
    return failed ? 1 : 0;
}

static float half(uint16_t h) {
    int e = (h >> 10) & 31;
    float v = e == 0 ? ldexpf(h & 1023, -24) :
        e == 31 ? ((h & 1023) ? NAN : INFINITY) : ldexpf(1 + (h & 1023)/1024.0f, e-15);
    return h & 0x8000 ? -v : v;
}

static int run(remote_handle64 handle, const char *name) {
    int err = call(handle);
    if (err) { printf("%s RPC error=0x%x\n", name, err); return err; }
    uint16_t *h = (uint16_t *)out;
    int nonzero = 0, changed = 0, sentinel = 0;
    for (int i = 0; i < 2048; ++i) {
        if (h[i] == 0xa5a5) { ++sentinel; continue; }
        ++changed;
        if (h[i] & 0x7fff) ++nonzero;
    }
    printf("%s kind=%d limit=%d cvt=%d store=%d repeat=%d bias=%08x/%08x written=%d nz=%d first=%04x(%g) hits:",
           name, kind, limit, convert, store, repeats, bias, bias_high, changed, nonzero, h[0], half(h[0]));
    int shown = 0;
    for (int i = 0; i < 2048 && shown < 12; ++i)
        if (h[i] != 0xa5a5 && (h[i] & 0x7fff)) { printf(" %d:%04x", i, h[i]); ++shown; }
    printf("\n");
    return 0;
}

int fp8_tests(remote_handle64 handle, const char *command) {
    if (!strcmp(command,"fp8-block")) {
        int failures=0;
        const int params[]={2046,2047};
        for(int select=0;select<2;++select)for(int slot=0;slot<2;++slot)
        for(int p=0;p<2;++p) {
            kind=select?12:0;limit=params[p];convert=2;store=0;repeats=1;bias=0x3c00;bias_high=0;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            for(int r=0;r<32;++r)for(int k=0;k<32;++k)a[ai(r,k)+slot]=0x78;
            memset(w,0x78,2048);
            if(call(handle))return 1;
            uint16_t *h=(uint16_t*)out;
            int bad=0;uint16_t ref=select==slot?0x5000:0;
            for(int i=0;i<1024;++i)bad+=h[i]!=ref;
            printf("FP8 block Rs.bit0=%d slot=%d Rt=%d single_packet K=32 expected=%04x first=%04x failures=%d\n",select,slot,limit,ref,h[0],bad);
            failures+=bad;
        }
        for(kind=0;kind<=12;kind+=12)for(int test=0;test<6;++test){
            limit=2047;convert=2;store=0;repeats=1;bias=0x3c00;bias_high=0;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            for(int r=0;r<32;++r)for(int k=0;k<32;++k)a[ai(r,k)+(kind==12)]=((r*7+k*11+test)%2?0xf8:0x78);
            for(int k=0;k<32;++k)for(int n=0;n<32;++n)w[wi(k,n)]=((k*13+n*5+test)%3?0x78:0xf8);
            /* Extra weights must not turn the spare activation byte into K=32..63. */
            memset(w+1024,0x78,1024);
            if(call(handle))return 1;
            int bad=0;uint16_t *h=(uint16_t*)out;
            for(int r=0;r<32;++r)for(int n=0;n<32;++n){
                int sum=0;for(int k=0;k<32;++k)sum+=((r*7+k*11+test)%2?-1:1)*((k*13+n*5+test)%3?1:-1);
                __fp16 ref=sum;uint16_t bits;memcpy(&bits,&ref,2);bad+=h[ci(r,n)]!=bits;
            }
            printf("FP8 full block 32x32x32 select=%d signed_matrix=%d failures=%d\n",kind==12,test,bad);failures+=bad;
        }
        return failures?1:0;
    }
    if (!strcmp(command,"fp8-tiled")) {
        int failures=0;
        for(kind=0;kind<=2;++kind)for(int kt=1;kt<=4;++kt){
            int m=5,depth=kt*32-1,ncols=17;
            limit=2047;convert=2;store=0;bias=0x3c00;bias_high=0;repeats=256+kt;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            for(int r=0;r<m;++r)for(int k=0;k<depth;++k){
                uint8_t v=0x50+(r*13+k*17)%48;if((r+k)&1)v|=128;
                int off=(k/32)*2048+ai(r,k%32);
                if(kind==2){__fp16 f=fp8(v);memcpy(a+off,&f,2);}else a[off]=v;
            }
            for(int k=0;k<depth;++k)for(int n=0;n<ncols;++n){uint8_t v=0x50+(k*11+n*5)%48;if((k+n)%3==0)v|=128;w[(k/32)*1024+wi(k%32,n)]=v;}
            if(call(handle))return 1;
            int bad=0;uint16_t *h=(uint16_t*)out;
            for(int r=0;r<32;++r)for(int n=0;n<32;++n){
                float sum=0;
                if(r<m && n<ncols)for(int k=0;k<depth;++k){uint8_t av=0x50+(r*13+k*17)%48,wv=0x50+(k*11+n*5)%48;if((r+k)&1)av|=128;if((k+n)%3==0)wv|=128;sum+=fp8(av)*fp8(wv);}
                __fp16 ref=sum;uint16_t bits;memcpy(&bits,&ref,2);bad+=bits!=h[ci(r,n)];
            }
            printf("FP8 padded GEMM kind=%d M=%d K=%d N=%d distinct_K_tiles=%d bit_failures=%d\n",kind,m,depth,ncols,kt,bad);failures+=bad;
        }
        return failures?1:0;
    }
    if (!strcmp(command, "fp8-validate")) return validate(handle);
    int err = 0;
    char name[64];
    if (!strcmp(command, "fp8-deep")) {
        for (kind = 4; kind < 6; ++kind) {
            for (limit = 1023; limit <= 4095; limit = limit*2+1) {
                memset(a, 0x78, sizeof(a)); memset(w, 0x78, sizeof(w));
                if ((err = run(handle, "deep-dimension"))) return err;
            }
            limit = 2047;
            for (int side = 0; side < 2; ++side)
                for (int i = 0; i < 12; ++i) {
                    memset(a, side ? 0x78 : 0, sizeof(a));
                    memset(w, side ? 0 : 0x78, sizeof(w));
                    (side ? w : a)[i] = 0x78;
                    snprintf(name, sizeof(name), "deep-%s%d", side ? "W" : "A", i);
                    if ((err = run(handle, name))) return err;
                }
            for (int ai = 0; ai < 16; ++ai)
                for (int wi = 0; wi < 8; ++wi) {
                    memset(a, 0, sizeof(a)); memset(w, 0, sizeof(w));
                    a[ai] = w[wi] = 0x78;
                    snprintf(name, sizeof(name), "deep-pair-A%d-W%d", ai, wi);
                    if ((err = run(handle, name))) return err;
                }
        }
        return 0;
    }
    const int values[] = {0,1,2,7,8,9,15,16,55,56,57,63,64,112,119,120,121,126,127,128,129,184,248,255};
    memset(w, 0x78, sizeof(w));
    for (unsigned i = 0; i < sizeof(values)/sizeof(values[0]); ++i) {
        memset(a, values[i], sizeof(a));
        snprintf(name, sizeof(name), "encoding-%02x", values[i]);
        if ((err = run(handle, name))) return err;
    }
    memset(a, 0x78, sizeof(a));
    for (limit = 511; limit <= 4095; limit = limit*2+1) {
        for (kind = 0; kind < 2; ++kind)
            if ((err = run(handle, "dimension"))) return err;
    }
    limit = 2047; kind = 0;
    for (convert = 0; convert < 4; ++convert)
        if ((err = run(handle, "conversion"))) return err;
    convert = 2;
    for (store = 0; store < 2; ++store)
        if ((err = run(handle, "store"))) return err;
    store = 0;
    for (repeats = 1; repeats < 4; ++repeats)
        if ((err = run(handle, "accumulate"))) return err;
    repeats = 1;
    const int offsets[] = {0,1,2,3,4,5,6,7,8,31,32,63,64,65,127,128,255,256,511,512,1023,1024,2047,2048};
    for (kind = 0; kind < 2; ++kind) {
    for (int side = 0; side < 2; ++side) {
        for (unsigned i = 0; i < sizeof(offsets)/sizeof(offsets[0]); ++i) {
            memset(a, side ? 0x78 : 0, sizeof(a));
            memset(w, side ? 0 : 0x78, sizeof(w));
            (side ? w : a)[offsets[i]] = 0x78;
            snprintf(name, sizeof(name), "%s-onehot-%d", side ? "W" : "A", offsets[i]);
            if ((err = run(handle, name))) return err;
        }
    }
    }
    kind = 0;
    for (int ai = 0; ai < 20; ai += 2) {
        for (int wi = 0; wi < 8; ++wi) {
            memset(a, 0, sizeof(a)); memset(w, 0, sizeof(w));
            a[ai] = w[wi] = 0x78;
            snprintf(name, sizeof(name), "pair-A%d-W%d", ai, wi);
            if ((err = run(handle, name))) return err;
        }
    }
    for (kind = 2; kind < 4; ++kind) {
        memset(a, 0x78, sizeof(a)); memset(w, 0x78, sizeof(w));
        uint16_t *hf = (uint16_t *)(kind == 2 ? a : w);
        for (int i = 0; i < 4096; ++i) hf[i] = 0x3c00;
        if ((err = run(handle, "mixed-fp16-fp8"))) return err;
    }
    return 0;
}
