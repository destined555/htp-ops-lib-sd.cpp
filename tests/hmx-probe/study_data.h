#ifndef HMX_STUDY_DATA_H
#define HMX_STUDY_DATA_H
#include <stdint.h>
static inline uint32_t study_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; return x ^ (x >> 16);
}
static inline int study_a(int r,int k,int seed) {
    if(seed==0) return 129;
    if(seed==30 || seed==31) return 255;
    return study_hash((uint32_t)(r*7919+k*13+seed*104729))&255;
}
static inline int study_w(int k,int n,int seed,int kind) {
    if(seed==0) return 1;
    if(seed==30) return kind==1?127:7;
    if(seed==31) return kind==1?-128:-8;
    if(seed>=12 && seed<24 && kind==1)
        return (int)(study_hash((uint32_t)(k*3571+n*23+seed*65537))&255)-128;
    return (int)(study_hash((uint32_t)(k*3571+n*23+seed*65537))&15)-8;
}
#endif
