#pragma once

#include <stddef.h>
#include <stdint.h>

struct MessageState {
  union {
    volatile uint8_t  v[8];
    volatile uint64_t d;
  };
} __attribute__((packed));

struct MessageHeader {
  struct MessageState state;
  uint32_t            checksum;
  int32_t             n_reqs;
  int32_t             req_offsets[0];  // n_reqs + 1 entries
} __attribute__((packed));

struct RequestHeader {
  int32_t state;
  int32_t type;
  uint8_t data[0];
} __attribute__((packed));

enum RequestType {
  REQUEST_TYPE_NO_OP = 0,
  REQUEST_TYPE_RPCMEM_MAP,
  REQUEST_TYPE_OP_COMPUTE,
  REQUEST_TYPE_COUNT,
};

struct RpcmemMapRequest {
  int32_t n_puts;
  int32_t n_gets;
  int32_t fds[0];
} __attribute__((packed));

struct OpComputeRequest {
  uint32_t op;
  uint8_t  payload[0];
} __attribute__((packed));

static inline int message_header_validate_layout(const struct MessageHeader *h, size_t capacity,
                                                 size_t *total_size) {
  if (h == NULL || total_size == NULL ||
      capacity < sizeof(struct MessageHeader) + sizeof(int32_t)) {
    return 0;
  }

  const int32_t n_reqs = h->n_reqs;
  if (n_reqs < 0) {
    return 0;
  }

  const size_t offset_count = (size_t) n_reqs + 1;
  const size_t max_offset_count =
    (capacity - sizeof(struct MessageHeader)) / sizeof(int32_t);
  if (offset_count > max_offset_count) {
    return 0;
  }

  const size_t header_size =
    sizeof(struct MessageHeader) + offset_count * sizeof(int32_t);
  const int32_t first_wire_offset = h->req_offsets[0];
  if (first_wire_offset < 0) {
    return 0;
  }

  size_t begin = (size_t) first_wire_offset;
  if (begin < header_size || begin > capacity) {
    return 0;
  }

  for (int32_t i = 0; i < n_reqs; ++i) {
    const int32_t end_wire_offset = h->req_offsets[i + 1];
    if (end_wire_offset < 0) {
      return 0;
    }
    const size_t end = (size_t) end_wire_offset;
    if (end < begin || end > capacity || end - begin < sizeof(struct RequestHeader)) {
      return 0;
    }
    begin = end;
  }

  *total_size = begin;
  return 1;
}

static inline size_t message_header_size(const struct MessageHeader *h) {
  return sizeof(struct MessageHeader) + (h->n_reqs + 1) * sizeof(int32_t);
}

static inline size_t message_total_size(const struct MessageHeader *h) {
  return h->req_offsets[h->n_reqs];
}

static inline size_t message_header_get_request_size(const struct MessageHeader *h, int req_idx) {
  return h->req_offsets[req_idx + 1] - h->req_offsets[req_idx];
}

static inline struct RequestHeader *message_header_get_request_ptr(struct MessageHeader *h, int req_idx) {
  return (struct RequestHeader *) ((uintptr_t) (h) + h->req_offsets[req_idx]);
}
