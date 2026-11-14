#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "hmx_probe.h"
int fp8_tests(remote_handle64 handle, const char *command);
int integer_tests(remote_handle64 handle, const char *command);
int acceleration_study(remote_handle64 handle,const char *command);

static int bench_compact(remote_handle64 handle) {
    const int kinds[]={19,10,11};
    const char *names[]={"FP16-M64","UINT8xINT8-CM","UINT8xINT4-CM"};
    uint8_t out[4096];int failures=0;
    for(int kt=1;kt<=4;kt*=4)for(int trial=0;trial<9;++trial)for(int j=0;j<3;++j){
        int err=hmx_probe_bench(handle,kinds[j],kt,100000,out,sizeof(out));
        if(err){printf("COMPACT RPC kind=%d error=0x%x\n",kinds[j],err);return 1;}
        uint64_t stats[4];memcpy(stats,out+4096-sizeof(stats),sizeof(stats));
        double ns=(double)stats[0]*1000/19.2/stats[2];int bad=0;
        if(j==0){__fp16 ref=32*kt;uint16_t bits;memcpy(&bits,&ref,2);for(int i=0;i<2032;++i)bad+=((uint16_t*)out)[i]!=bits;}
        else for(int i=0;i<2048;++i)bad+=out[i]!=32*kt;
        printf("COMPACT type=%s M=64 K=%d N=32 trial=%d iterations=%llu ns=%.3f GOPS=%.3f cycles=%.3f dsp_MHz=%.1f failures=%d\n",
            names[j],kt*32,trial,(unsigned long long)stats[2],ns,2.0*64*32*kt*32/ns,(double)stats[1]/stats[2],(double)stats[1]*19.2/stats[0],bad);failures+=bad;
    }
    return failures?1:0;
}

static int bench(remote_handle64 handle) {
    const int kinds[]={9,0,2,6,7,16,17};
    const char *names[]={"FP16","FP8","FP16xFP8","UINT8xINT8","UINT8xINT4","UINT8xINT8-wide","UINT8xINT4-wide"};
    uint8_t output[4096];
    int failures=0;
    for(int kt=1;kt<=4;kt*=4)for(int trial=0;trial<7;++trial)for(int i=0;i<7;++i){
        struct timespec t0,t1;clock_gettime(CLOCK_MONOTONIC,&t0);
        int err=hmx_probe_bench(handle,kinds[i],kt,50000,output,sizeof(output));
        clock_gettime(CLOCK_MONOTONIC,&t1);
        if(err){printf("BENCH RPC kind=%d error=0x%x\n",kinds[i],err);return 1;}
        uint64_t stats[4];memcpy(stats,output+4096-sizeof(stats),sizeof(stats));
        double us=stats[0]/19.2, ns=us*1000/stats[2];
        double rpcus=(t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_nsec-t0.tv_nsec)/1e3;
        uint16_t expected=0;
        if(kinds[i]<6 || kinds[i]==9){__fp16 value=32*kt;memcpy(&expected,&value,2);}else expected=32*kt;
        int mismatches=0;uint16_t *h=(uint16_t*)output;
        for(int j=0;j<1024;++j){mismatches+=h[j]!=expected;if(kinds[i]>=16 && j<1008)mismatches+=h[j+1024]!=0;}
        failures+=mismatches!=0;
        printf("BENCH type=%s M=32 K=%d N=32 trial=%d iterations=%llu us=%.3f ns_per_gemm=%.3f GOPS=%.3f cycles=%.3f dsp_MHz=%.1f RPC_us=%.1f mismatches=%d\n",
            names[i],kt*32,trial,(unsigned long long)stats[2],us,ns,2.0*32*32*32*kt/ns,
            (double)stats[1]/stats[2],stats[1]/us,rpcus,mismatches);
    }
    return failures?1:0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s mode [activation_byte=1] [weight_byte=1] [bias_low=0] [bias_high=0]\n"
                "mode: 0=FP16, 1=weight.b, 2=weight.n, 3=weight.n:2x, 4=HVX INT8, 5=HVX INT8 tail, 6=FP8 HMX\n"
                "suites: fp8-validate, fp8-tiled, fp8-block, int-cm-block, int-validate, int-validate8, int-validate4, int-wide, int-tiled, bench, bench-compact, study-validate, study-bench, study-baseline\n"
                "exploratory: fp8-explore, fp8-deep, int-explore, int-layout, int-cvt, int-bias, int-pairs, int-cm, int-2x\n", argv[0]);
        return 2;
    }
    const int mode = atoi(argv[1]);
    const int a = argc > 2 ? atoi(argv[2]) : 1;
    const int b = argc > 3 ? atoi(argv[3]) : 1;
    const uint32_t bias = argc > 4 ? strtoul(argv[4], NULL, 0) : 0;
    const uint32_t bias_high = argc > 5 ? strtoul(argv[5], NULL, 0) : 0;
    struct remote_rpc_control_unsigned_module ctrl = {.domain = 3, .enable = 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ctrl, sizeof(ctrl));
    if (err) fprintf(stderr, "unsigned PD control: 0x%x\n", err);
    remote_handle64 handle = (remote_handle64)-1;
    if (!err) {
        err = hmx_probe_open(hmx_probe_URI "&_dom=cdsp", &handle);
        if (err) fprintf(stderr, "open: 0x%x\n", err);
    }
    uint8_t output[4096];
    memset(output, 0xa5, sizeof(output));
    if (!err) {
        if(!strncmp(argv[1],"study-",6)) {err=acceleration_study(handle,argv[1]);hmx_probe_close(handle);return err;}
        if(!strcmp(argv[1],"bench-compact")) {err=bench_compact(handle);hmx_probe_close(handle);return err;}
        if(!strcmp(argv[1],"bench")) {err=bench(handle);hmx_probe_close(handle);return err;}
        if (!strncmp(argv[1], "fp8-", 4) || !strncmp(argv[1], "int-", 4)) {
            err = argv[1][0] == 'i' ? integer_tests(handle, argv[1]) : fp8_tests(handle, argv[1]);
            hmx_probe_close(handle);
            return err;
        }
        err = hmx_probe_run(handle, mode, a, b, bias, bias_high, output, sizeof(output));
        if (err) fprintf(stderr, "run: 0x%x\n", err);
    }
    if (handle != (remote_handle64)-1) hmx_probe_close(handle);
    if (err) {
        fprintf(stderr, "HMX probe RPC failed: 0x%x\n", err);
        return 1;
    }
    if (mode == 0) {
        int mismatches = 0;
        for (int i = 0; i < 1024; ++i) {
            uint16_t value;
            memcpy(&value, output + 2*i, 2);
            mismatches += value != 0x5000; /* 32.0 in FP16. */
        }
        printf("FP16 all-ones 32x32x32: %s mismatches=%d\n",
               mismatches ? "FAIL" : "PASS", mismatches);
        printf("first FP16 bits:");
        for (int i = 0; i < 16; ++i) {
            uint16_t value;
            memcpy(&value, output + 2*i, 2);
            printf(" %04x", value);
        }
        printf("\n");
        return mismatches ? 1 : 0;
    }
    if (mode == 6) {
        printf("EXECUTED FP8 HMX a_byte=0x%02x w_byte=0x%02x; FP16 output bits:\n", a, b);
        for (int i = 0; i < 16; ++i) {
            uint16_t value;
            memcpy(&value, output + 2*i, 2);
            printf(" %04x", value);
        }
        printf("\nExecution alone does not validate FP8 GEMM numerics or layout.\n");
        return 0;
    }
    if (mode == 4 || mode == 5) {
        const int k = mode == 4 ? 64 : 31;
        const int n = mode == 4 ? 128 : 65;
        int mismatches = 0;
        for (int r = 0; r < 3; ++r) {
            for (int col = 0; col < n; ++col) {
                int sum = 0;
                for (int d = 0; d < k; ++d)
                    sum += ((r*11+d*7)%27-13)*((d*13+col*5)%19-9);
                int expected = sum >> 7;
                if (expected < -128) expected = -128;
                if (expected > 127) expected = 127;
                mismatches += (int8_t)output[r*n+col] != expected;
            }
        }
        printf("Official QHL HVX INT8 3x%dx%d: %s mismatches=%d\n",
               k, n, mismatches ? "FAIL" : "PASS", mismatches);
        return mismatches ? 1 : 0;
    }
    unsigned histogram[256] = {0};
    for (int i = 0; i < 4096; ++i) ++histogram[output[i]];
    printf("EXECUTED integer HMX mode=%d a_byte=%d w_byte=%d bias_low=0x%x bias_high=0x%x\n",
           mode, a, b, bias, bias_high);
    printf("raw output histogram:");
    for (int i = 0; i < 256; ++i) if (histogram[i]) printf(" %d:%u", i, histogram[i]);
    printf("\nfirst 128 bytes:");
    for (int i = 0; i < 128; ++i) printf(" %u", output[i]);
    printf("\nExecution alone does not validate quantized GEMM numerics or layout.\n");
    return 0;
}
