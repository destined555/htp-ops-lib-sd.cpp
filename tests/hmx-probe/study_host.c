#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "hmx_probe.h"
#include "study_data.h"

static int check(const uint8_t *out,int kind,int n,int kt,int epi,int seed) {
    int bad=0;float worst=0;
    for(int r=0;r<64;++r) for(int c=0;c<n;++c) {
        int32_t raw=0,signed_sum=0;
        for(int k=0;k<kt*32;++k) {
            int a=study_a(r,k,seed),w=study_w(k,c,seed,kind==0 || kind>=3?2:kind);
            raw+=a*w; signed_sum+=(a-128)*w;
        }
        int j=(c/32)*2048+r*32+c%32;
        if(kind==0 || kind>=3 || epi>=2) {
            int h=(c/32)*2048+(r&~1)*32+(c%32)*2+(r&1);
            __fp16 actual;memcpy(&actual,out+2*h,2);
            __fp16 expected=(float)signed_sum/4096;
            float error=fabsf((float)actual-(float)expected);
            float tolerance=kind==0 || kind>=3?0.003f+0.015f*fabsf((float)expected):0;
            int fail=!isfinite((float)actual) || error>tolerance;
            if(fail && bad<3)printf("DIFF r=%d c=%d got=%g expected=%g sum=%d\n",r,c,(double)actual,(double)expected,signed_sum);
            bad+=fail; if(error>worst)worst=error;
        } else {
            int32_t actual=out[j],expected=signed_sum<0?0:signed_sum>255?255:signed_sum;
            if(epi==1){memcpy(&actual,out+4*j,4);expected=raw;}
            if(actual!=expected && bad<3)printf("DIFF r=%d c=%d got=%d expected=%d\n",r,c,actual,expected);
            bad+=actual!=expected;
        }
    }
    printf("CHECK kind=%d M=64 K=%d N=%d epi=%d seed=%d failures=%d worst_fp16_error=%g\n",kind,kt*32,n,epi,seed,bad,(double)worst);
    return bad;
}

int acceleration_study(remote_handle64 handle,const char *command) {
    uint8_t out[16416];int failures=0;
    if(!strcmp(command,"study-validate")) {
        for(int n=32;n<=64;n*=2) for(int kind=1;kind<=2;++kind)
            for(int seed=0;seed<24;++seed) for(int kt=1;kt<=4;kt*=4) for(int epi=0;epi<=3;++epi) {
                int err=hmx_probe_study(handle,kind,n,kt,epi,seed,1,out,sizeof(out));
                if(err){printf("STUDY RPC error=0x%x\n",err);return 1;}
                failures+=check(out,kind,n,kt,epi,seed);
            }
        for(int kind=1;kind<=2;++kind)for(int seed=30;seed<=31;++seed)for(int kt=4;kt<=192;kt*=48) {
            int err=hmx_probe_study(handle,kind,64,kt,1,seed,1,out,sizeof(out));
            if(err){printf("STUDY RPC error=0x%x\n",err);return 1;}
            failures+=check(out,kind,64,kt,1,seed);
        }
        for(int kt=48;kt<=192;kt*=4)for(int kind=0;kind<=2;++kind)for(int epi=2;epi<=3;++epi) {
            if(kind==0 && epi==3)continue;
            int err=hmx_probe_study(handle,kind,64,kt,epi,7,1,out,sizeof(out));
            if(err)return 1;
            failures+=check(out,kind,64,kt,epi,7);
        }
    } else if(!strcmp(command,"study-baseline")) {
        const int sizes[]={1,4,48,192};
        for(int n=32;n<=64;n*=2)for(int si=0;si<4;++si)for(int trial=0;trial<7;++trial)
            for(int kind=3;kind<=4;++kind) {
                int kt=sizes[si],iterations=kt<=4?10000:1000;
                int err=hmx_probe_study(handle,kind,n,kt,2,7,iterations,out,sizeof(out));
                if(err){printf("STUDY RPC error=0x%x\n",err);return 1;}
                uint64_t stats[4];memcpy(stats,out+16384,sizeof(stats));
                double ns=(double)stats[0]*1000/19.2/stats[2];
                int bad=trial==0?check(out,kind,n,kt,2,7):0;failures+=bad;
                printf("STUDY kind=%d M=64 K=%d N=%d epi=2 trial=%d iterations=%llu ns=%.3f GOPS=%.3f cycles=%.3f dsp_MHz=%.1f failures=%d\n",
                    kind,kt*32,n,trial,(unsigned long long)stats[2],ns,2.0*64*n*kt*32/ns,
                    (double)stats[1]/stats[2],(double)stats[1]*19.2/stats[0],bad);
            }
    } else if(!strcmp(command,"study-bench")) {
        const int sizes[]={1,4,48,192};
        for(int n=32;n<=64;n*=2)for(int si=0;si<4;++si)for(int trial=0;trial<7;++trial)
            for(int epi=0;epi<=3;++epi)for(int kind=0;kind<=2;++kind) {
                if(kind==0 && epi!=2)continue;
                int kt=sizes[si],iterations=kt<=4?10000:1000;
                int err=hmx_probe_study(handle,kind,n,kt,epi,7,iterations,out,sizeof(out));
                if(err){printf("STUDY RPC error=0x%x\n",err);return 1;}
                uint64_t stats[4];memcpy(stats,out+16384,sizeof(stats));
                double ns=(double)stats[0]*1000/19.2/stats[2];
                int bad=trial==0?check(out,kind,n,kt,epi,7):0;failures+=bad;
                printf("STUDY kind=%d M=64 K=%d N=%d epi=%d trial=%d iterations=%llu ns=%.3f GOPS=%.3f cycles=%.3f dsp_MHz=%.1f failures=%d\n",
                    kind,kt*32,n,epi,trial,(unsigned long long)stats[2],ns,2.0*64*n*kt*32/ns,
                    (double)stats[1]/stats[2],(double)stats[1]*19.2/stats[0],bad);
            }
    } else return 2;
    printf("STUDY RESULT failures=%d\n",failures);
    return failures?1:0;
}
