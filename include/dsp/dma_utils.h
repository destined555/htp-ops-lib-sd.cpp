#pragma once

#include <hexagon_protos.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define DM0_STATUS_MASK  3
#define DM0_STATUS_IDLE  0
#define DM0_STATUS_RUN   1
#define DM0_STATUS_ERROR 2

#define DMA_DESC_DSTATE_PENDING 0
#define DMA_DESC_DSTATE_DONE    1

#define DMA_DESC_TYPE_1D 0
#define DMA_DESC_TYPE_2D 1

#define DMA_DESC_CACHEALLOC_MASK      3
#define DMA_DESC_CACHEALLOC_NONE      0
#define DMA_DESC_CACHEALLOC_WRITEONLY 1
#define DMA_DESC_CACHEALLOC_READONLY  2
#define DMA_DESC_CACHEALLOC_READWRITE 3

struct dma_desc_1d {
  uint32_t next;

  union {
    struct {
      unsigned length     : 24;
      unsigned type       : 2;
      unsigned dst_dlbc   : 1;
      unsigned src_dlbc   : 1;
      unsigned dst_bypass : 1;
      unsigned src_bypass : 1;
      unsigned ordered    : 1;
      unsigned dstate     : 1;
    } __attribute__((packed));

    uint32_t dstate_order_bypass_type_length;
  };

  uint32_t src;
  uint32_t dst;
} __attribute__((packed));

struct dma_desc_2d {
  uint32_t next;

  union {
    struct {
      unsigned length     : 24;
      unsigned type       : 2;
      unsigned dst_dlbc   : 1;
      unsigned src_dlbc   : 1;
      unsigned dst_bypass : 1;
      unsigned src_bypass : 1;
      unsigned ordered    : 1;
      unsigned dstate     : 1;
    } __attribute__((packed));

    uint32_t dstate_order_bypass_type_length;
  };

  uint32_t src;
  uint32_t dst;

  unsigned _pad1 : 24, cache_alloc : 2, _pad2 : 6;
  // assume little-endian
  uint16_t roi_width;
  uint16_t roi_height;
  uint16_t src_stride;
  uint16_t dst_stride;
  uint16_t src_width_offset;
  uint16_t dst_width_offset;
} __attribute__((packed));

typedef struct dma_desc_1d dma_desc_1d_t;
typedef struct dma_desc_2d dma_desc_2d_t;

enum dma_submit_result {
  DMA_SUBMIT_OK = 0,
  DMA_SUBMIT_BUSY,
  DMA_SUBMIT_HW_ERROR,
  DMA_SUBMIT_BAD_DESC,
};

#ifdef __cplusplus
static_assert(sizeof(dma_desc_2d_t) == 32);
#else
_Static_assert(sizeof(dma_desc_2d_t) == 32, "unexpected UserDMA 2D descriptor size");
#endif

static inline void dmstart(void *next) {
  asm volatile("release(%0):at" ::"r"(next));
  Q6_dmstart_A(next);
}

static inline void dmlink(void *cur, void *next) {
  asm volatile("release(%0):at" ::"r"(next));
  Q6_dmlink_AA(cur, next);
}

static inline uint32_t dmpoll() {
  return Q6_R_dmpoll();
}

static inline uint32_t dmwait() {
  return Q6_R_dmwait();
}

static inline bool dma_wait_for_idle() {
  uint32_t dm0_status = dmwait() & DM0_STATUS_MASK;

  return (dm0_status == DM0_STATUS_IDLE);
}

// NOTE(hzx): The submission will fail if there's current DMA request running
static inline int dma_submit_one(dma_desc_1d_t *desc) {
  if (!desc) {
    return -1;
  }

  uint32_t dm0_status = dmpoll() & DM0_STATUS_MASK;
  if (dm0_status != DM0_STATUS_IDLE) {
    return 1;  // TODO: define error code
  }

  dmstart(desc);
  dmpoll();
  return 0;
}

static inline int dma_desc_2d_init(dma_desc_2d_t *desc, dma_desc_2d_t *next, const void *src, void *dst,
                                   size_t roi_width, size_t roi_height, size_t src_stride, size_t dst_stride) {
  if (desc == NULL || src == NULL || dst == NULL || roi_width == 0 || roi_height == 0 ||
      roi_width > UINT16_MAX || roi_height > UINT16_MAX || src_stride > UINT16_MAX || dst_stride > UINT16_MAX ||
      src_stride < roi_width || dst_stride < roi_width || (uintptr_t) src > UINT32_MAX ||
      (uintptr_t) dst > UINT32_MAX || (next != NULL && (uintptr_t) next > UINT32_MAX)) {
    return DMA_SUBMIT_BAD_DESC;
  }

  memset(desc, 0, sizeof(*desc));
  desc->next       = (uint32_t) (uintptr_t) next;
  desc->length     = 0;
  desc->type       = DMA_DESC_TYPE_2D;
  desc->src_bypass = 1;
  desc->dst_bypass = 0;
  desc->ordered    = 1;
  desc->dstate     = DMA_DESC_DSTATE_PENDING;
  desc->src        = (uint32_t) (uintptr_t) src;
  desc->dst        = (uint32_t) (uintptr_t) dst;
  desc->cache_alloc = DMA_DESC_CACHEALLOC_NONE;
  desc->roi_width  = (uint16_t) roi_width;
  desc->roi_height = (uint16_t) roi_height;
  desc->src_stride = (uint16_t) src_stride;
  desc->dst_stride = (uint16_t) dst_stride;
  return DMA_SUBMIT_OK;
}

static inline int dma_chain_submit(dma_desc_2d_t *first) {
  if (first == NULL || ((uintptr_t) first & 63u) != 0 || first->type != DMA_DESC_TYPE_2D) {
    return DMA_SUBMIT_BAD_DESC;
  }
  const uint32_t status = dmpoll() & DM0_STATUS_MASK;
  if (status == DM0_STATUS_RUN) {
    return DMA_SUBMIT_BUSY;
  }
  if (status != DM0_STATUS_IDLE) {
    return DMA_SUBMIT_HW_ERROR;
  }
  dmstart(first);
  dmpoll();
  return DMA_SUBMIT_OK;
}

static inline int dma_chain_poll(const dma_desc_2d_t *descs, size_t count) {
  if (descs == NULL || count == 0) {
    return DMA_SUBMIT_BAD_DESC;
  }
  const uint32_t status = dmpoll() & DM0_STATUS_MASK;
  if (status == DM0_STATUS_ERROR) {
    return DMA_SUBMIT_HW_ERROR;
  }
  for (size_t i = 0; i < count; ++i) {
    if (descs[i].dstate != DMA_DESC_DSTATE_DONE) {
      return DMA_SUBMIT_BUSY;
    }
  }
  return DMA_SUBMIT_OK;
}

static inline int dma_chain_wait(const dma_desc_2d_t *descs, size_t count) {
  if (descs == NULL || count == 0) {
    return DMA_SUBMIT_BAD_DESC;
  }
  const uint32_t status = dmwait() & DM0_STATUS_MASK;
  if (status != DM0_STATUS_IDLE) {
    return DMA_SUBMIT_HW_ERROR;
  }
  for (size_t i = 0; i < count; ++i) {
    if (descs[i].dstate != DMA_DESC_DSTATE_DONE) {
      return DMA_SUBMIT_HW_ERROR;
    }
  }
  return DMA_SUBMIT_OK;
}
