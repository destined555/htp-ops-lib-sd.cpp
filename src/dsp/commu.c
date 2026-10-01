// Communication & RPC related interfaces

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt_memory.h>
#include <qurt_signal.h>
#include <qurt_thread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/hmx_mgr.h"
#include "dsp/mmap_mgr.h"
#include "dsp/op_executor.h"
#include "dsp/ops.h"
#include "dsp/power.h"
#include "dsp/vtcm_mgr.h"
#include "htp_ops.h"  // QAIC auto-generated header for FastRPC
#include "message.h"

static int dummy_handle;  // served as the global handle

// The host owns one global session handle and serializes open/init/create/close calls.
// These lifecycle flags are not intended for concurrent lifecycle RPCs.
static bool backend_opened;
static bool backend_initialized;
static bool backend_mapping_only;

static AEEResult require_backend_initialized(const char *caller) {
  if (!backend_opened || !backend_initialized) {
    FARF(ERROR, "%s: backend is not initialized", caller);
    return AEE_EBADSTATE;
  }
  return AEE_SUCCESS;
}

struct MessageChannel {
  uint8_t      *msg;
  int           rpcmem_fd;
  size_t        max_msg_size;
  bool          msg_receiver_should_stop;
  qurt_signal_t msg_receiver_ready;
  qurt_thread_t msg_receiver_thread;
  uint64_t      session_token;
  volatile bool hmx_enable_requested;
  volatile bool hmx_disable_requested;
  volatile bool hmx_enabled;
  volatile int  hmx_control_result;
};

static struct MessageChannel global_msg_chan;

static int message_channel_request_hmx_enable(struct MessageChannel *chan) {
  if (chan == NULL) {
    return -1;
  }
  if (chan->hmx_enabled) {
    return 0;
  }
  chan->hmx_control_result = 0;
  chan->hmx_disable_requested = false;
  chan->hmx_enable_requested = true;
  if (chan->msg == NULL) {
    // The receiver will acquire the lock when the channel is created.
    return 0;
  }
  for (unsigned int i = 0; i < 1000000U; ++i) {
    if (!chan->hmx_enable_requested) {
      return chan->hmx_enabled && chan->hmx_control_result == 0 ? 0 : -1;
    }
    qurt_sleep(1);
  }
  return -1;
}

static int message_channel_request_hmx_disable(struct MessageChannel *chan) {
  if (chan == NULL) {
    return -1;
  }
  chan->hmx_enable_requested = false;
  chan->hmx_disable_requested = true;
  if (chan->msg == NULL) {
    chan->hmx_disable_requested = false;
    chan->hmx_enabled = false;
    return 0;
  }
  for (unsigned int i = 0; i < 1000000U; ++i) {
    if (!chan->hmx_disable_requested) {
      return chan->hmx_enabled ? -1 : 0;
    }
    qurt_sleep(1);
  }
  return -1;
}

static void msg_receiver_loop(void *param) {
  struct MessageChannel *chan = (struct MessageChannel *) param;
  qurt_signal_set(&(chan->msg_receiver_ready), 1);

  const int SLEEP_TIME_US = 1;

  while (1) {
    if (chan->msg_receiver_should_stop) {
      break;
    }

    // HMX locks belong to the thread that acquired them.  Keep one lock for
    // all operations on this channel and perform release/acquire hand-offs
    // only when the host switches the session lease.
    if (chan->hmx_disable_requested) {
      if (chan->hmx_enabled) {
        hmx_manager_disable_execution();
        chan->hmx_enabled = false;
      }
      chan->hmx_disable_requested = false;
    }
    if (chan->hmx_enable_requested && !chan->hmx_enabled) {
      hmx_manager_enable_execution();
      chan->hmx_control_result = hmx_manager_get_last_lock_result();
      chan->hmx_enabled = chan->hmx_control_result == 0;
      chan->hmx_enable_requested = false;
    }

    struct MessageHeader *msg_hdr = (struct MessageHeader *) chan->msg;
    if (msg_hdr == NULL) {
      qurt_sleep(SLEEP_TIME_US);  // wait until shared message buffer become available
    }

    // invalidate cache
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, chan->max_msg_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    // NOTE(hzx): do we need a barrier here?
    // asm volatile("barrier" ::: "memory");

    // FIXME: memory order may not be working
    // volatile _Atomic uint64_t *d_ptr = (volatile _Atomic uint64_t *) &(msg_hdr->state.d);
    // uint64_t d_val = atomic_load_explicit(d_ptr, memory_order_acquire);

    volatile uint64_t d_val, *d_ptr = &(msg_hdr->state.d);
    asm volatile("%0 = memd_aq(%1)" : "=r"(d_val) : "r"(d_ptr) : "memory");

    // TODO(hzx): use more proper message state
    uint8_t v0 = d_val & 0xff;
    uint8_t v1 = (d_val >> 8) & 0xff;
    if (v0 == 0 || v1 != 0) {
      qurt_sleep(SLEEP_TIME_US);
      continue;
    }

    // simple checksum
    if (0) {
      uint32_t  sum   = 0;
      uint32_t *begin = ((uint32_t *) msg_hdr);
      uint32_t *end   = begin + chan->max_msg_size / 4;
      for (uint32_t *p = begin; p < end; ++p) {
        sum += *p;
      }

      if (sum != 0) {
        __builtin_trap();  // die
      }
    }

    size_t message_bytes = sizeof(struct MessageHeader);
    if (!message_header_validate_layout(msg_hdr, chan->max_msg_size, &message_bytes)) {
      FARF(ERROR, "%s: invalid message layout: n_reqs=%d max=%u", __func__,
           msg_hdr->n_reqs, (unsigned int) chan->max_msg_size);
    } else {
      for (int i = 0; i < msg_hdr->n_reqs; ++i) {
        const size_t wire_req_size =
          (size_t) msg_hdr->req_offsets[i + 1] - (size_t) msg_hdr->req_offsets[i];
        const size_t req_data_size = wire_req_size - sizeof(struct RequestHeader);
        struct RequestHeader *req_hdr = message_header_get_request_ptr(msg_hdr, i);
        switch (req_hdr->type) {
          case REQUEST_TYPE_OP_COMPUTE:
            {
              // TODO(hzx): use separate thread (pool) to execute op
              struct OpComputeRequest *compute_req = (struct OpComputeRequest *) req_hdr->data;
              mmap_manager_set_context(chan->session_token);
              req_hdr->state = backend_mapping_only ? AEE_EBADSTATE :
                                                       execute_op_simple(compute_req, req_data_size);
            }
            break;
          case REQUEST_TYPE_RPCMEM_MAP:
            {
              if (req_data_size < sizeof(struct RpcmemMapRequest)) {
                req_hdr->state = -1;
                break;
              }
              mmap_manager_set_context(chan->session_token);
              const bool v2 = req_data_size >= sizeof(struct RpcmemMapRequestV2) &&
                  ((struct RpcmemMapRequestV2 *) req_hdr->data)->magic == RPCMEM_MAP_REQUEST_SESSION_MAGIC &&
                  ((struct RpcmemMapRequestV2 *) req_hdr->data)->version == RPCMEM_MAP_REQUEST_VERSION_SESSION;
              if (v2) {
                struct RpcmemMapRequestV2 *map_req = (struct RpcmemMapRequestV2 *) req_hdr->data;
                const size_t available = (req_data_size - sizeof(*map_req)) / sizeof(struct RpcmemMapEntryV2);
                if (map_req->header_size < sizeof(*map_req) || map_req->n_puts < 0 || map_req->n_gets < 0 ||
                    map_req->session_token != chan->session_token) {
                  req_hdr->state = -1;
                  break;
                }
                const size_t count = (size_t) map_req->n_puts + (size_t) map_req->n_gets;
                if (count > available) {
                  req_hdr->state = -1;
                  break;
                }
                for (int j = 0; j < map_req->n_puts; ++j) {
                  struct RpcmemMapEntryV2 *entry = &map_req->entries[j];
                  if (mmap_manager_put_map_view(entry->fd, entry->offset, entry->size) != 0) {
                    req_hdr->state = -1;
                    break;
                  }
                }
                if (req_hdr->state == -1) break;
                for (int j = 0; j < map_req->n_gets; ++j) {
                  struct RpcmemMapEntryV2 *entry = &map_req->entries[map_req->n_puts + j];
                  if (mmap_manager_get_map(entry->fd, entry->offset, entry->size) == NULL) {
                    req_hdr->state = -1;
                    break;
                  }
                }
              } else {
                struct RpcmemMapRequest *map_req = (struct RpcmemMapRequest *) req_hdr->data;
                const size_t available_fds =
                  (req_data_size - sizeof(struct RpcmemMapRequest)) / sizeof(int32_t);
                if (map_req->n_puts < 0 || map_req->n_gets < 0 ||
                    (size_t) map_req->n_puts > available_fds ||
                    (size_t) map_req->n_gets > available_fds - (size_t) map_req->n_puts) {
                  req_hdr->state = -1;
                  break;
                }
                for (int j = 0; j < map_req->n_puts; ++j) mmap_manager_put_map(map_req->fds[j]);
              }
#ifdef HTP_OPS_MMAP_TRACE
              FARF(ALWAYS, "HTP_MMAP_TRACE request token=%llu v2=%d",
                   (unsigned long long) chan->session_token, v2 ? 1 : 0);
#endif
              if (req_hdr->state != -1) req_hdr->state = 0;
            }
            break;
          default:
            break;
        }
      }
    }

    // first payload cache flush
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, message_bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    asm volatile("barrier" ::: "memory");

    // FIXME: memory order may not be working
    // atomic_uchar *v1_ptr = (atomic_uchar *) &(msg_hdr->state.v[1]);
    // atomic_store_explicit(v1_ptr, 1, memory_order_release);

    d_val = v0 | (1 << 8);
    asm volatile("memd_rl(%0):at = %1" ::"r"(d_ptr), "r"(d_val) : "memory");

    // flush cache
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, message_bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    // TODO(hzx): estimate host's job completion time and sleep
    qurt_sleep(SLEEP_TIME_US);
  }

}

// init an empty (semantically unintialized) message channel
void message_channel_init(struct MessageChannel *chan) {
  chan->msg          = NULL;
  chan->rpcmem_fd    = -1;
  chan->max_msg_size = 0;
  chan->session_token = 0;
  chan->hmx_enable_requested = false;
  chan->hmx_disable_requested = false;
  chan->hmx_enabled = false;
  chan->hmx_control_result = 0;

  chan->msg_receiver_should_stop = false;
}

bool message_channel_is_active(const struct MessageChannel *chan) {
  return chan->msg != NULL;
}

int message_channel_create(struct MessageChannel *chan, int rpcmem_fd, size_t max_msg_size) {
  uint8_t *p;
  int      err = HAP_mmap_get(rpcmem_fd, (void **) &p, NULL);
  if (err) {
    FARF(ALWAYS, "%s: HAP_mmap_get failed with %x", __func__, err);
    return -1;
  }

  // clear message state
  memset(p, 0, max_msg_size);

  chan->msg_receiver_should_stop = false;
  chan->msg          = p;
  chan->rpcmem_fd    = rpcmem_fd;
  chan->max_msg_size = max_msg_size;
  qurt_signal_init(&(chan->msg_receiver_ready));

  const size_t stack_size = 8192;
  void        *stack      = memalign(4096, stack_size);
  if (!stack) {
    FARF(ALWAYS, "%s: failed to allocate memory for thread stack", __func__);
    qurt_signal_destroy(&(chan->msg_receiver_ready));
    return -1;
  }

  // launch message receiver thread
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "hops-msg-recv");
  qurt_thread_attr_set_priority(&attr, 64);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, stack_size);
  qurt_thread_attr_set_autostack(&attr, QURT_THREAD_AUTOSTACK_ENABLED);

  err = qurt_thread_create(&(chan->msg_receiver_thread), &attr, msg_receiver_loop, chan);
  if (err) {
    FARF(ALWAYS, "%s: qurt_thread_create failed with 0x%x", __func__, err);
    qurt_signal_destroy(&(chan->msg_receiver_ready));
    return -1;
  }

  // wait until msg reciever thread is ready
  qurt_signal_wait_all(&(chan->msg_receiver_ready), 1);

  // Backend initialization normally happens before the channel exists.  In
  // that order the enable helper records a request for the receiver thread;
  // do not let the host submit its first operation until that request has
  // completed and the receiver owns the HMX lock.
  if (chan->hmx_enable_requested && message_channel_request_hmx_enable(chan) != 0) {
    FARF(ERROR, "%s: receiver failed to acquire pending HMX lease", __func__);
    chan->msg_receiver_should_stop = true;
    qurt_thread_join(chan->msg_receiver_thread, &err);
    qurt_signal_destroy(&(chan->msg_receiver_ready));
    HAP_mmap_put(chan->rpcmem_fd);
    message_channel_init(chan);
    return -1;
  }
  return 0;
}

int message_channel_destroy(struct MessageChannel *chan) {
  if (!message_channel_is_active(chan)) {
    return 0;
  }

  (void) message_channel_request_hmx_disable(chan);

  // signal message receiver thread to stop
  chan->msg_receiver_should_stop = true;

  int status;
  qurt_thread_join(chan->msg_receiver_thread, &status);
  qurt_signal_destroy(&(chan->msg_receiver_ready));
  HAP_mmap_put(chan->rpcmem_fd);

  message_channel_init(chan);
  return 0;
}

// FastRPC interface
AEEResult htp_ops_open(const char *uri, remote_handle64 *handle) {
  // We may keep this function simple and leave real initialization somewhere else

  if (backend_opened) {
    return AEE_EALREADY;
  }
  *handle = (remote_handle64) &dummy_handle;

  backend_opened      = true;
  backend_initialized = false;
  backend_mapping_only = false;
  message_channel_init(&global_msg_chan);
  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_close(remote_handle64 handle) {
  if (!backend_opened) {
    return AEE_EBADSTATE;
  }
  const bool was_mapping_only = backend_mapping_only;
  backend_initialized = false;

  mmap_manager_set_context(global_msg_chan.session_token);
  mmap_manager_release_all();
  AEEResult result = message_channel_destroy(&global_msg_chan) == 0 ? AEE_SUCCESS : AEE_EFAILED;

  if (!was_mapping_only) {
    if (vtcm_manager_reset() != 0) {
      return AEE_EFAILED;
    }
    if (hmx_manager_reset() != 0) {
      return AEE_EFAILED;
    }
    power_reset();
  }
  backend_mapping_only = false;
  backend_opened = false;

  return result;
}

// FastRPC interface
AEEResult htp_ops_init_backend(remote_handle64 handle) {
#ifdef HTP_OPS_QUIET_BENCH
  // FARF logging is asynchronous but still adds DSP/RPC pressure to short graphs.
  // Keep it disabled only in the explicitly requested benchmark build.
  HAP_setFARFRuntimeLoggingParams(0, NULL, 0);
#endif
  FARF(ALWAYS, "init_backend called");

  if (!backend_opened) {
    return AEE_EBADSTATE;
  }
  if (backend_initialized && !backend_mapping_only) {
    return AEE_SUCCESS;
  }
  // A mapping-only session keeps its channel and mappings alive, but must
  // acquire the compute resources before it can execute an operation.
  backend_initialized = false;

  power_setup();
  const int    vtcm_result = vtcm_manager_setup();
  void *const  vtcm_base   = vtcm_manager_get_vtcm_base();
  const size_t vtcm_size   = vtcm_manager_get_vtcm_size();
  if (vtcm_result != 0 || !vtcm_base || vtcm_size == 0) {
    FARF(ERROR, "VTCM manager setup failed");
    if (vtcm_manager_reset() != 0) {
      FARF(ERROR, "VTCM manager cleanup failed after setup error");
    }
    power_reset();
    return AEE_ENOMEMORY;
  }
  FARF(ALWAYS, "VTCM manager acquired %u bytes", (unsigned int) vtcm_size);
  if (hmx_manager_setup() != 0) {
    FARF(ERROR, "HMX manager setup failed");
    if (hmx_manager_reset() != 0) {
      FARF(ERROR, "HMX manager cleanup failed after setup error");
    }
    if (vtcm_manager_reset() != 0) {
      FARF(ERROR, "VTCM manager cleanup failed after HMX setup error");
    }
    power_reset();
    return AEE_EFAILED;
  }
  backend_initialized = true;
  backend_mapping_only = false;

  if (message_channel_request_hmx_enable(&global_msg_chan) != 0) {
    FARF(ERROR, "HMX receiver thread failed to acquire execution lock");
    (void) hmx_manager_reset();
    (void) vtcm_manager_reset();
    power_reset();
    backend_initialized = false;
    return AEE_EFAILED;
  }

  return AEE_SUCCESS;
}

AEEResult htp_ops_init_mapping_backend(remote_handle64 handle) {
  FARF(ALWAYS, "init_mapping_backend called");
  if (!backend_opened) {
    return AEE_EBADSTATE;
  }
  if (backend_initialized) {
    return AEE_SUCCESS;
  }
  backend_initialized = true;
  backend_mapping_only = true;
  return AEE_SUCCESS;
}

AEEResult htp_ops_release_backend(remote_handle64 handle) {
  FARF(ALWAYS, "release_backend called");
  if (!backend_opened) {
    return AEE_EBADSTATE;
  }
  if (!backend_initialized || backend_mapping_only) {
    backend_initialized = false;
    backend_mapping_only = true;
    return AEE_SUCCESS;
  }

  if (message_channel_request_hmx_disable(&global_msg_chan) != 0) {
    FARF(ERROR, "HMX receiver thread failed to release execution lock");
    return AEE_EFAILED;
  }

  if (hmx_manager_reset() != 0 || vtcm_manager_reset() != 0) {
    return AEE_EFAILED;
  }
  power_reset();
  backend_initialized = false;
  backend_mapping_only = true;
  return AEE_SUCCESS;
}

AEEResult htp_ops_init_backend_fraction(remote_handle64 handle, uint32 numerator, uint32 denominator) {
#ifdef HTP_OPS_QUIET_BENCH
  HAP_setFARFRuntimeLoggingParams(0, NULL, 0);
#endif
  FARF(ALWAYS, "init_backend_fraction called numerator=%u denominator=%u", numerator, denominator);

  if (!backend_opened) {
    return AEE_EBADSTATE;
  }
  if (backend_initialized) {
    return AEE_SUCCESS;
  }
  if (numerator == 0 || denominator == 0 || numerator > denominator) {
    return AEE_EBADPARM;
  }

  power_setup();
  const int vtcm_result = vtcm_manager_setup_fraction(numerator, denominator);
  void *const vtcm_base   = vtcm_manager_get_vtcm_base();
  const size_t vtcm_size  = vtcm_manager_get_vtcm_size();
  if (vtcm_result != 0 || !vtcm_base || vtcm_size == 0) {
    FARF(ERROR, "fractional VTCM manager setup failed");
    (void) vtcm_manager_reset();
    power_reset();
    return AEE_ENOMEMORY;
  }
  if (hmx_manager_setup() != 0) {
    FARF(ERROR, "HMX manager setup failed after fractional VTCM setup");
    (void) vtcm_manager_reset();
    power_reset();
    return AEE_EFAILED;
  }
  backend_initialized = true;
  if (message_channel_request_hmx_enable(&global_msg_chan) != 0) {
    FARF(ERROR, "HMX receiver thread failed to acquire execution lock");
    (void) hmx_manager_reset();
    (void) vtcm_manager_reset();
    power_reset();
    backend_initialized = false;
    return AEE_EFAILED;
  }
  return AEE_SUCCESS;
}

static AEEResult create_channel_internal(int32 fd, uint32 size, uint64_t session_token) {
  const AEEResult guard_result = require_backend_initialized(__func__);
  if (guard_result != AEE_SUCCESS) {
    return guard_result;
  }
  if (message_channel_is_active(&global_msg_chan)) {
    return AEE_EALREADY;
  }

  global_msg_chan.session_token = session_token;
  return message_channel_create(&global_msg_chan, fd, size);
}

// FastRPC interface
AEEResult htp_ops_create_channel(remote_handle64 handle, int32 fd, uint32 size) {
  return create_channel_internal(fd, size, 0);
}

AEEResult htp_ops_create_channel_with_token(remote_handle64 handle, uint64 session_token, int32 fd, uint32 size) {
  return create_channel_internal(fd, size, session_token);
}

// FastRPC interface
AEEResult htp_ops_destroy_channel(remote_handle64 handle) {
  return message_channel_destroy(&global_msg_chan);
}

// FastRPC interface
AEEResult htp_ops_rms_norm_f32(remote_handle64 handle, int32 fd0, int32 offset0, int32 fd1, int32 offset1, int32 ne0,
                               int32 ne1) {
  const AEEResult guard_result = require_backend_initialized(__func__);
  if (guard_result != AEE_SUCCESS) {
    return guard_result;
  }
  int64_t t0 = HAP_perf_get_qtimer_count();

  // TODO(hzx): maybe we should cache fd -> address mapping
  uint8_t *p0, *p1;
  p0 = p1 = NULL;

  int err = HAP_mmap_get(fd0, (void **) &p0, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  err = HAP_mmap_get(fd1, (void **) &p1, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  int64_t t1 = HAP_perf_get_qtimer_count();

  const float *input      = (float *) (p1 + offset1);
  size_t       input_size = ne0 * ne1 * sizeof(float);  // This can be inaccurate
  qurt_mem_cache_clean((qurt_addr_t) input, input_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

  float *output = (float *) (p0 + offset0);
  err           = hvx_rms_norm_f32(output, input, ne0, ne1);
  if (err) {
    FARF(ALWAYS, "%s: bad input or alignment", __func__);
    goto bail;
  }

  // TODO(hzx): we need a smarter way to do this
  size_t output_size = ne0 * ne1 * sizeof(float);  // This can be inaccurate
  err                = qurt_mem_cache_clean((qurt_addr_t) output, output_size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

  int64_t t2 = HAP_perf_get_qtimer_count();

bail:
  if (p0) {
    HAP_mmap_put(fd0);
  }
  if (p1) {
    HAP_mmap_put(fd1);
  }

  int64_t elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  FARF(ALWAYS, "rms_norm_f32 (ne0=%d, ne1=%d) took %ld us", ne0, ne1, elapsed);
  FARF(ALWAYS, "    core + cache inv+flush: %ld us", HAP_perf_qtimer_count_to_us(t2 - t1));
  return err;
}

// FastRPC interface
AEEResult htp_ops_mat_mul_permuted_w16a32(remote_handle64 handle, int32 output_fd, int32 output_offset,
                                          int32 activation_fd, int32 activation_offset, int32 weight_fd,
                                          int32 weight_offset, int32 m, int32 k, int32 n) {
  const AEEResult guard_result = require_backend_initialized(__func__);
  if (guard_result != AEE_SUCCESS) {
    return guard_result;
  }
  uint8_t *p0, *p1, *p2;
  p0 = p1 = p2 = NULL;

  int err = HAP_mmap_get(output_fd, (void **) &p0, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  err = HAP_mmap_get(activation_fd, (void **) &p1, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  err = HAP_mmap_get(weight_fd, (void **) &p2, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  float        *output     = (float *) (p0 + output_offset);
  const float  *activation = (const float *) (p1 + activation_offset);
  const __fp16 *weight     = (const __fp16 *) (p2 + weight_offset);

  size_t output_size     = m * n * sizeof(float);
  size_t activation_size = m * k * sizeof(float);
  size_t weight_size     = k * n * sizeof(__fp16);

  qurt_mem_cache_clean((qurt_addr_t) activation, activation_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t) weight, weight_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

  // static char print_buf[256];

  // const uint16_t *w = (const uint16_t *) weight;
  // sprintf(print_buf, "%s: weight digest %04x %04x %04x %04x | %04x %04x", __func__, w[0], w[1], w[2], w[3], w[64], w[65]);
  // FARF(ALWAYS, "%s", print_buf);

  // const float *a = activation;
  // sprintf(print_buf, "%s: activa digest %g %g %g %g", __func__, a[0], a[1], a[2], a[3]);
  // FARF(ALWAYS, "%s", print_buf);

  hmx_manager_enable_execution();
  err = hmx_mat_mul_permuted_w16a32(output, activation, weight, m, k, n);
  hmx_manager_disable_execution();

  // sprintf(print_buf, "%s: output digest %g %g %g %g", __func__, output[0], output[1], output[2], output[3]);
  // FARF(ALWAYS, "%s", print_buf);

  qurt_mem_cache_clean((qurt_addr_t) output, output_size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

bail:
  if (p0) {
    HAP_mmap_put(output_fd);
  }
  if (p1) {
    HAP_mmap_put(activation_fd);
  }
  if (p2) {
    HAP_mmap_put(weight_fd);
  }
  return err;
}

static uint64_t htp_ops_debug_hash_bytes(const void *data, size_t size) {
  const uint8_t *bytes = (const uint8_t *) data;
  uint64_t hash = UINT64_C(14695981039346656037);
  if (bytes == NULL) {
    return 0;
  }
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

// Diagnostic-only direct FastRPC entry point. It deliberately runs the same
// hmx_mat_mul_permuted_w16a32 kernel as the production entry point and returns
// hashes computed in DSP address space before the output cache flush.
AEEResult htp_ops_mat_mul_permuted_w16a32_trace(remote_handle64 handle, int32 output_fd, int32 output_offset,
                                                int32 activation_fd, int32 activation_offset, int32 weight_fd,
                                                int32 weight_offset, int32 m, int32 k, int32 n,
                                                uint64_t *activation_hash, uint64_t *output_hash,
                                                uint64_t *output_hash_after_remap) {
  const AEEResult guard_result = require_backend_initialized(__func__);
  if (guard_result != AEE_SUCCESS) {
    return guard_result;
  }
  if (activation_hash != NULL) {
    *activation_hash = 0;
  }
  if (output_hash != NULL) {
    *output_hash = 0;
  }
  if (output_hash_after_remap != NULL) {
    *output_hash_after_remap = 0;
  }

  uint8_t *p0 = NULL, *p1 = NULL, *p2 = NULL;
  int err = HAP_mmap_get(output_fd, (void **) &p0, NULL);
  if (err) {
    goto bail;
  }
  err = HAP_mmap_get(activation_fd, (void **) &p1, NULL);
  if (err) {
    goto bail;
  }
  err = HAP_mmap_get(weight_fd, (void **) &p2, NULL);
  if (err) {
    goto bail;
  }

  float *output = (float *) (p0 + output_offset);
  const float *activation = (const float *) (p1 + activation_offset);
  const __fp16 *weight = (const __fp16 *) (p2 + weight_offset);
  size_t output_size = (size_t) m * (size_t) n * sizeof(float);
  size_t activation_size = (size_t) m * (size_t) k * sizeof(float);
  size_t weight_size = (size_t) k * (size_t) n * sizeof(__fp16);

  qurt_mem_cache_clean((qurt_addr_t) activation, activation_size,
                       QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t) weight, weight_size,
                       QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  if (activation_hash != NULL) {
    *activation_hash = htp_ops_debug_hash_bytes(activation, activation_size);
  }

  hmx_manager_enable_execution();
  err = hmx_mat_mul_permuted_w16a32(output, activation, weight, m, k, n);
  hmx_manager_disable_execution();
  if (output_hash != NULL) {
    *output_hash = htp_ops_debug_hash_bytes(output, output_size);
  }
  qurt_mem_cache_clean((qurt_addr_t) output, output_size,
                       QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

  // Reacquire the output mapping after the flush. This diagnostic distinguishes
  // a stale DSP cache alias from bytes that are not visible through a fresh HAP
  // mapping of the same RPCMEM allocation.
  HAP_mmap_put(output_fd);
  p0 = NULL;
  err = HAP_mmap_get(output_fd, (void **) &p0, NULL);
  if (err) {
    goto bail;
  }
  output = (float *) (p0 + output_offset);
  qurt_mem_cache_clean((qurt_addr_t) output, output_size,
                       QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  if (output_hash_after_remap != NULL) {
    *output_hash_after_remap = htp_ops_debug_hash_bytes(output, output_size);
  }

bail:
  if (p0) {
    HAP_mmap_put(output_fd);
  }
  if (p1) {
    HAP_mmap_put(activation_fd);
  }
  if (p2) {
    HAP_mmap_put(weight_fd);
  }
  return err;
}

void internal_op_tests();

// FastRPC interface
AEEResult htp_ops_test_ops(remote_handle64 handle, int32_t *userdma_scope) {
  const AEEResult guard_result = require_backend_initialized(__func__);
  if (guard_result != AEE_SUCCESS) {
    return guard_result;
  }
  if (userdma_scope == NULL) {
    return AEE_EBADPARM;
  }
  FARF(ALWAYS, "Op Tests!");

  internal_op_tests();
  *userdma_scope = block_sparse_halo_probe_userdma_scope();

  return 0;
}
