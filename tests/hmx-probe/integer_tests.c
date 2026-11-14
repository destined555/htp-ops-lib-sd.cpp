#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "hmx_probe.h"

static uint8_t a[8192], w[8192], out[4096];
static int kind=6, al=2047, wl=2047, cvt=2, store, repeats=1;
static uint32_t blo, bhi;
static int ai(int r,int k){return 2*((r&~1)*32+k*2+(r&1))+1;}
static int wi(int k,int n){return (k/4)*128+n*4+k%4;}
static int w4i(int k,int n){return (k/8)*128+n*4+k%4;}
static void set4(int k,int n,int v){int off=w4i(k,n),shift=(k&4)?4:0;w[off]=(w[off]&~(15<<shift))|((v&15)<<shift);}
static int get4(int k,int n){int v=(w[w4i(k,n)]>>((k&4)?4:0))&15;return v>=8?v-16:v;}
static int ci(int r,int n){return (r&~1)*32+n*2+(r&1);}
static int signed4(int v){v&=15;return v>=8?v-16:v;}
static int decode2x(int v){return signed4(v)+16*signed4(v>>4);}

static int call(remote_handle64 h) {
    return hmx_probe_tile(h,a,sizeof(a),w,sizeof(w),kind,al,wl,cvt,store,repeats,blo,bhi,out,sizeof(out));
}

static int show(remote_handle64 h, const char *label) {
    int err=call(h);
    if(err) {printf("%s RPC=0x%x\n",label,err); return err;}
    unsigned hist[256]={0};
    for(int i=0;i<4096;++i)++hist[out[i]];
    printf("%s kind=%d A=%d W=%d cvt=%x store=%d bias=%08x/%08x first=%u hist:",label,kind,al,wl,cvt,store,blo,bhi,out[0]);
    for(int i=0;i<256;++i)if(hist[i])printf(" %d:%u",i,hist[i]);
    printf(" hits:");int shown=0;
    for(int i=0;i<4096 && shown<16;++i)if(out[i] && out[i]!=0xa5){printf(" %d:%u",i,out[i]);++shown;}
    printf("\n");return 0;
}

int integer_tests(remote_handle64 h,const char *command) {
    (void)command;
    if(!strcmp(command,"int-cm-block")) {
        int failures=0;
        for(kind=10;kind<=11;++kind) {
            al=2047;wl=kind==10?1023:511;store=1;cvt=0;blo=0x6000;bhi=0;repeats=1;
            memset(a,1,sizeof(a));memset(w,kind==10?1:0x11,sizeof(w));
            if(show(h,"cm-block-ones"))return 1;
            for(int side=0;side<2;++side)for(int off=0;off<8;++off){
                memset(a,side?1:0,sizeof(a));memset(w,side?0:kind==10?1:0x11,sizeof(w));
                (side?w:a)[off]=1;
                char label[64];snprintf(label,sizeof(label),"cm-block-%s%d",side?"W":"A",off);
                if(show(h,label))return 1;
            }
            int onehot_bad=0;
            for(int r=0;r<64;++r)for(int k=0;k<32;++k){
                int n=(r*7+k*11)&31;
                memset(a,0,sizeof(a));memset(w,0,sizeof(w));a[r*32+k]=1;
                if(kind==10)w[wi(k,n)]=1;else set4(k,n,1);
                if(call(h))return 1;
                for(int i=0;i<2048;++i)if(out[i]!=(i==r*32+n)){++onehot_bad;break;}
            }
            printf("CM full block 64x32x32 kind=%d onehot_cases=2048 failures=%d\n",kind,onehot_bad);failures+=onehot_bad;
            for(int trial=0;trial<24;++trial){
                uint32_t seed=1234u+trial;uint8_t ar[2048];int8_t wr[1024];
                memset(a,0,sizeof(a));memset(w,0,sizeof(w));
                for(int r=0;r<64;++r)for(int k=0;k<32;++k){seed=seed*1664525u+1013904223u;ar[r*32+k]=a[r*32+k]=(seed>>16)%4;}
                for(int k=0;k<32;++k)for(int n=0;n<32;++n){seed=seed*1664525u+1013904223u;int v=(seed>>16)%3-1;wr[k*32+n]=v;if(kind==10)w[wi(k,n)]=v;else set4(k,n,v);}
                if(call(h))return 1;
                int bad=0;
                for(int r=0;r<64;++r)for(int n=0;n<32;++n){int sum=0;for(int k=0;k<32;++k)sum+=ar[r*32+k]*wr[k*32+n];int ref=sum<0?0:sum>255?255:sum;bad+=out[r*32+n]!=ref;}
                printf("CM full block 64x32x32 kind=%d signed_random=%d failures=%d first=%d last=%d\n",kind,trial,bad,out[0],out[2047]);failures+=bad;
            }
        }
        return failures?1:0;
    }
    if(!strcmp(command,"int-cm-modern")) {
        kind=10;al=2047;wl=1023;store=1;
        memset(a,1,sizeof(a));memset(w,1,sizeof(w));
        const uint32_t biases[]={0x4c00,0x10000|0x4c00,0x10000|0x4800,0x10000|0x5000,0x4c01,0x4fff,0x4e00};
        for(unsigned b=0;b<sizeof(biases)/sizeof(*biases);++b)for(cvt=0;cvt<4;++cvt){
            blo=biases[b];bhi=0;if(show(h,"cm-modern"))return 1;
        }
        return 0;
    }
    if(!strcmp(command,"int-wide")) {
        int failures=0;
        for(kind=6;kind<=7;++kind)for(int trial=0;trial<16;++trial){
            store=5;blo=0x4000;bhi=0;al=2047;wl=kind==6?1023:511;repeats=1+trial%3;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            for(int r=0;r<32;++r)for(int k=0;k<32;++k)a[ai(r,k)]=(r*13+k*17+trial*7)%256;
            for(int k=0;k<32;++k)for(int n=0;n<32;++n){int value=kind==6?(k*11+n*5+trial*7)%256-128:(k*11+n*5+trial)%16-8;if(kind==6)w[wi(k,n)]=value;else set4(k,n,value);}
            if(call(h))return 1;
            int bad=0;
            for(int r=0;r<32;++r)for(int n=0;n<32;++n){
                int sum=0;for(int k=0;k<32;++k)sum+=a[ai(r,k)]*(kind==6?(int8_t)w[wi(k,n)]:get4(k,n));sum*=repeats;
                int pos=ci(r,n);uint16_t lo;int16_t hi;memcpy(&lo,out+2*pos,2);memcpy(&hi,out+2048+2*pos,2);
                int got=hi*65536+lo;
                if(got!=sum){if(bad<3)printf("wide bad kind=%d sum=%d lo=%d hi=%d got=%d\n",kind,sum,lo,hi,got);++bad;}
            }
            printf("wide integer kind=%d trial=%d repeats=%d failures=%d\n",kind,trial,repeats,bad);failures+=bad;
        }
        kind=6;store=5;blo=0x4000;wl=1023;repeats=1;
        const int targets[]={0,1,-1,32767,32768,32769,-32767,-32768,-32769,65535,65536,65537,-65535,-65536,-65537,131071,131072,1036320,-1036320};
        for(unsigned t=0;t<sizeof(targets)/sizeof(targets[0]);++t){
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            int rem=targets[t]<0?-targets[t]:targets[t],sign=targets[t]<0?-1:1,k=0;
            while(rem>=32385){a[ai(0,k)]=255;w[wi(k,0)]=sign*127;rem-=32385;++k;}
            if(rem>=127){a[ai(0,k)]=rem/127;w[wi(k,0)]=sign*127;rem%=127;++k;}
            if(rem){a[ai(0,k)]=rem;w[wi(k,0)]=sign;}
            if(call(h))return 1;
            uint16_t lo;int16_t hi;memcpy(&lo,out,2);memcpy(&hi,out+2048,2);
            int got=hi*65536+lo;
            printf("wide boundary target=%d got=%d %s\n",targets[t],got,got==targets[t]?"PASS":"FAIL");failures+=got!=targets[t];
        }
        return failures?1:0;
    }
    if(!strcmp(command,"int-tiled")) {
        int failures=0;
        for(kind=6;kind<=7;++kind)for(int kt=1;kt<=4;++kt){
            const int m=5,ncols=17,depth=kt*32-1;
            al=2047;wl=kind==6?1023:511;store=3;blo=0x4000;bhi=0;repeats=256+kt;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            for(int r=0;r<m;++r)for(int k=0;k<depth;++k)a[(k/32)*2048+ai(r,k%32)]=(r*13+k*17)%256;
            for(int k=0;k<depth;++k)for(int n=0;n<ncols;++n){
                int value=(k*11+n*5)%16-8;
                if(kind==6)w[(k/32)*1024+wi(k%32,n)]=value;
                else{int off=(k/32)*512+w4i(k%32,n),shift=(k&4)?4:0;w[off]|=(value&15)<<shift;}
            }
            if(call(h))return 1;
            int bad=0;uint16_t *o=(uint16_t*)out;
            for(int r=0;r<32;++r)for(int n=0;n<32;++n){
                int sum=0,colsum=0;
                if(r<m && n<ncols)for(int k=0;k<depth;++k){int v=(k*11+n*5)%16-8;sum+=((r*13+k*17)%256-128)*v;colsum+=v;}
                int got=(int16_t)o[ci(r,n)];
                if(r<m && n<ncols)got-=128*colsum;
                if(got!=sum)++bad;
            }
            printf("UINT8xINT%d signed activation zp=128 padded M=%d K=%d N=%d distinct_K_tiles=%d failures=%d\n",kind==6?8:4,m,depth,ncols,kt,bad);failures+=bad;
        }
        return failures?1:0;
    }
    if(!strcmp(command,"int-cm")) {
        store=4;blo=0x4000;bhi=0;al=2047;wl=4095;
        for(kind=10;kind<=11;++kind){
            wl=kind==10?4095:2047;
            memset(a,1,sizeof(a));memset(w,kind==10?1:0x11,sizeof(w));
            if(show(h,"cm-dimension"))return 1;
            for(int side=0;side<2;++side)for(int off=0;off<12;++off){
                memset(a,side?1:0,sizeof(a));memset(w,side?0:kind==10?1:0x11,sizeof(w));
                (side?w:a)[off]=1;
                char label[64];snprintf(label,sizeof(label),"cm-%s%d",side?"W":"A",off);
                if(show(h,label))return 1;
            }
        }
        return 0;
    }
    if(!strcmp(command,"int-validate") || !strcmp(command,"int-2x")) {
        int fails=0;store=3;blo=0x4000;bhi=0;al=2047;
        int first=!strcmp(command,"int-2x")?8:6,last=!strcmp(command,"int-2x")?8:7;
        for(kind=first;kind<=last;++kind){
            wl=kind==6?1023:511;repeats=1;
            int encbad=0;
            for(int v=0;v<256;++v){
                memset(a,0,sizeof(a));memset(w,0,sizeof(w));
                a[ai(0,0)]=1;w[0]=v;
                if(call(h))return 1;
                int expected=kind==7?signed4(v):kind==8?decode2x(v):(int8_t)v;
                int got=(int16_t)((uint16_t*)out)[0];
                if(got!=expected){if(encbad<3)printf("weight decode kind=%d raw=%02x expected=%d got=%d\n",kind,v,expected,got);++encbad;}
            }
            printf("integer weight decoding kind=%d tested=256 failures=%d\n",kind,encbad);fails+=encbad;
            for(int trial=0;trial<24;++trial){
                repeats=1+trial%3;memset(a,0,sizeof(a));memset(w,0,sizeof(w));
                for(int r=0;r<32;++r)for(int k=0;k<32;++k)a[ai(r,k)]=(r*13+k*17+trial*7)%256;
                for(int k=0;k<32;++k)for(int n=0;n<32;++n){
                    int value=(k*11+n*5+trial)%16-8;
                    if(kind==7)set4(k,n,value);else w[wi(k,n)]=value;
                }
                if(call(h))return 1;
                int bad=0;uint16_t *o=(uint16_t*)out;
                for(int r=0;r<32;++r)for(int n=0;n<32;++n){
                    int sum=0;for(int k=0;k<32;++k)sum+=a[ai(r,k)]*(kind==7?get4(k,n):kind==8?decode2x(w[wi(k,n)]):(int8_t)w[wi(k,n)]);
                    int expected=(int16_t)(sum*repeats);int got=(int16_t)o[ci(r,n)];
                    if(got!=expected){if(bad<2)printf("integer bad kind=%d r=%d n=%d expected=%d got=%d\n",kind,r,n,expected,got);++bad;}
                }
                printf("integer GEMM kind=%d trial=%d repeats=%d failures=%d\n",kind,trial,repeats,bad);fails+=bad;
            }
        }
        printf("Integer validation failures=%d\n",fails);return fails?1:0;
    }
    if(!strcmp(command,"int-bias")) {
        kind=6;al=2047;wl=1023;store=3;
        const uint32_t words[]={0,0x4000,0x40000000,0x40004000,0x2000,0x20002000,0x3c003c00,0x3c00,0x40404040,0x00800080};
        for(unsigned i=0;i<sizeof(words)/sizeof(words[0]);++i) {
            blo=words[i];bhi=0;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));a[ai(0,0)]=3;w[0]=2;
            if(show(h,"legacy-bias"))return 1;
            printf("legacy output first-half=%04x second-half=%04x\n",((uint16_t*)out)[0],((uint16_t*)out)[1]);
        }
        return 0;
    }
    if(!strcmp(command,"int-validate4")) {
        kind=7;al=2047;wl=511;store=2;blo=0x40;
        int fails=0;
        for(int r=0;r<32;++r)for(int k=0;k<32;++k){
            int n=(r*7+k*11)&31;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            a[ai(r,k)]=1;set4(k,n,1);
            if(call(h))return 1;
            uint16_t *o=(uint16_t*)out;
            for(int i=0;i<1024;++i)if(o[i]!=(i==ci(r,n)?1:0)){++fails;break;}
        }
        printf("INT4 onehot: cases=1024 failures=%d\n",fails);
        for(int trial=0;trial<16;++trial){
            repeats=1+trial%3;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            for(int r=0;r<32;++r)for(int k=0;k<32;++k)a[ai(r,k)]=(r*3+k*7+trial)%9;
            for(int k=0;k<32;++k)for(int n=0;n<32;++n)set4(k,n,(k*11+n*5+trial)%16-8);
            if(call(h))return 1;
            uint16_t *o=(uint16_t*)out;int bad=0;
            for(int r=0;r<32;++r)for(int n=0;n<32;++n){
                int sum=0;for(int k=0;k<32;++k)sum+=a[ai(r,k)]*get4(k,n);
                uint16_t expected=(uint16_t)((sum*repeats*17+8)>>4);
                if(o[ci(r,n)]!=expected){if(bad<2)printf("bad r=%d n=%d sum=%d expected=%04x got=%04x\n",r,n,sum,expected,o[ci(r,n)]);++bad;}
            }
            printf("INT4 random trial=%d repeats=%d failures=%d\n",trial,repeats,bad);fails+=bad;
        }
        return fails?1:0;
    }
    if(!strcmp(command,"int-cvt")) {
        blo=0x20000000;bhi=0x80000000;store=0;al=2047;wl=1023;
        memset(a,1,sizeof(a));memset(w,1,sizeof(w));
        for(cvt=0;cvt<4;++cvt){
            if(show(h,"normal-cvt"))return 1;
            memset(a,0,sizeof(a));if(show(h,"zero-cvt"))return 1;
            a[ai(0,0)]=1;w[0]=255;if(show(h,"negative-one-cvt"))return 1;
            memset(a,1,sizeof(a));memset(w,1,sizeof(w));
        }
        return 0;
    }
    if(!strcmp(command,"int-pairs")) {
        store=2;blo=0x40;al=2047;
        for(kind=6;kind<=8;++kind){
            wl=kind==6?1023:511;
            for(int k=0;k<32;++k)for(int pos=0;pos<8;++pos){
                memset(a,0,sizeof(a));memset(w,0,sizeof(w));a[ai(0,k)]=1;
                w[pos/2]=(pos&1)?0x10:1;
                if(call(h))return 1;
                uint16_t *o=(uint16_t*)out;
                if(o[0])printf("pair kind=%d k=%d byte=%d nibble=%d output=%04x\n",kind,k,pos/2,pos&1,o[0]);
            }
        }
        return 0;
    }
    if(!strcmp(command,"int-validate8")) {
        kind=6;al=2047;wl=1023;store=2;blo=0x40;
        int fails=0;
        for(int r=0;r<32;++r)for(int k=0;k<32;++k){
            int n=(r*7+k*11)&31;
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            a[ai(r,k)]=1;w[wi(k,n)]=1;
            if(call(h))return 1;
            uint16_t *o=(uint16_t*)out;
            for(int i=0;i<1024;++i)if(o[i]!=(i==ci(r,n)?1:0)){++fails;break;}
        }
        printf("INT8 community packing onehot: cases=1024 failures=%d\n",fails);
        for(int trial=0;trial<4;++trial){
            memset(a,0,sizeof(a));memset(w,0,sizeof(w));
            for(int r=0;r<32;++r)for(int k=0;k<32;++k)a[ai(r,k)]=(r*3+k*7+trial)%9;
            for(int k=0;k<32;++k)for(int n=0;n<32;++n)w[wi(k,n)]=(k*11+n*5+trial)%15-7;
            if(call(h))return 1;
            uint16_t *o=(uint16_t*)out;int bad=0;
            for(int r=0;r<32;++r)for(int n=0;n<32;++n){
                int sum=0;for(int k=0;k<32;++k)sum+=a[ai(r,k)]*(int8_t)w[wi(k,n)];
                uint16_t expected=(uint16_t)((sum*17+8)>>4);
                if(o[ci(r,n)]!=expected){if(bad<2)printf("bad r=%d n=%d sum=%d expected=%04x got=%04x\n",r,n,sum,expected,o[ci(r,n)]);++bad;}
            }
            printf("INT8 random trial=%d failures=%d\n",trial,bad);fails+=bad;
        }
        return fails?1:0;
    }
    if(!strcmp(command,"int-layout")) {
        blo=0x40;bhi=0;store=2;
        for(kind=6;kind<=8;++kind) {
            wl=kind==6?1023:511;
            memset(a,1,sizeof(a));memset(w,kind==6?1:0x11,sizeof(w));
            for(al=511;al<=4095;al=al*2+1)if((int)(call(h)))return 1;else show(h,"dimension");
            al=2047;
            const int ofs[]={0,1,2,3,4,7,8,15,16,31,32,63,64,65,127,128,255,256,511,512,1023,1024,2047,2048};
            for(int side=0;side<2;++side)for(unsigned i=0;i<sizeof(ofs)/sizeof(ofs[0]);++i){
                memset(a,side?1:0,sizeof(a));memset(w,side?0:(kind==6?1:0x11),sizeof(w));
                (side?w:a)[ofs[i]]=1;
                char label[64];snprintf(label,sizeof(label),"%s-onehot-%d",side?"W":"A",ofs[i]);
                if(show(h,label))return 1;
            }
            for(int av=0;av<5;++av)for(int wv=0;wv<5;++wv){
                const int vals[]={1,7,128,255,0};
                memset(a,vals[av],sizeof(a));memset(w,vals[wv],sizeof(w));
                char label[64];snprintf(label,sizeof(label),"encoding-%d-%d",vals[av],vals[wv]);
                if(show(h,label))return 1;
            }
        }
        return 0;
    }
    const uint32_t scales[]={0,1,2,0x100,0x10000,0x1000000,0x10000000,0x40000000,0x7fffffff,0x80000000,0xffffffff};
    int err;
    memset(a,1,sizeof(a));memset(w,1,sizeof(w));
    for(int hi=0;hi<2;++hi)for(unsigned i=0;i<sizeof(scales)/sizeof(scales[0]);++i){
        blo=hi ? 0 : scales[i];bhi=hi ? scales[i] : 0;
        if((err=show(h,"scale")))return err;
    }
    blo=0;bhi=0;
    for(cvt=0;cvt<64;++cvt)if((err=show(h,"cvt")))return err;
    return 0;
}
