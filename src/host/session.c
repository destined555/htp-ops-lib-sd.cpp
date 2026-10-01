#include <AEEStdErr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp_capabilities_utils.h"
#include "htp_ops.h"
#include "host/session.h"

#if !defined(_WIN32)
#pragma weak remote_session_control
#endif

struct htp_ops_session {
  remote_handle64 handle;
  uint32_t session_id;
  /* Host-side reserved-session ordinal.  The DSP compares the V2 RPCMEM map
   * request token against the channel token, and the host backend derives that
   * token from its own session index (session 0 is the default session).
   * Keep this ordinal in the same creation order so both sides agree. */
  uint32_t session_index;
  uint32_t effective_domain_id;
  int base_domain_id;
  int reserved;
  int backend_initialized;
  int channel_initialized;
};

static htp_ops_session *default_session = NULL;
static htp_ops_session *active_session = NULL;
static remote_handle64 session_handle = -1;  /* legacy compatibility alias */

#define HTP_OPS_MAX_RESERVED_SESSIONS 32
static int reserved_index_in_use[HTP_OPS_MAX_RESERVED_SESSIONS];

static uint32_t alloc_reserved_index(void) {
  for (uint32_t i = 1; i < HTP_OPS_MAX_RESERVED_SESSIONS; ++i) {
    if (!reserved_index_in_use[i]) {
      reserved_index_in_use[i] = 1;
      return i;
    }
  }
  return 0;
}

static void free_reserved_index(uint32_t index) {
  if (index > 0 && index < HTP_OPS_MAX_RESERVED_SESSIONS) {
    reserved_index_in_use[index] = 0;
  }
}

static uint64_t session_channel_token(const htp_ops_session *session) {
  if (session == NULL || !session->reserved) {
    return 0;
  }
  return ((uint64_t) session->session_index << 32) | session->effective_domain_id;
}

static int enable_unsigned_module(uint32_t domain_id) {
  if (&remote_session_control == NULL) {
    return AEE_EUNSUPPORTED;
  }
  struct remote_rpc_control_unsigned_module ctrl;
  ctrl.domain = (int) domain_id;
  ctrl.enable = 1;
  return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ctrl, sizeof(ctrl));
}

static int open_domain_uri(htp_ops_session *session, const char *module_uri) {
  domain *my_domain = get_domain((int) session->effective_domain_id);
  if (!my_domain) {
    return AEE_EBADPARM;
  }

  const char *uri = module_uri != NULL && module_uri[0] != '\0' ? module_uri : htp_ops_URI;
  int uri_domain_len = (int) strlen(uri) + MAX_DOMAIN_URI_SIZE;
  char *uri_domain = (char *) malloc((size_t) uri_domain_len);
  if (!uri_domain) {
    return AEE_ENOMEMORY;
  }

  int written = snprintf(uri_domain, (size_t) uri_domain_len, "%s%s", uri, my_domain->uri);
  if (written < 0 || written >= uri_domain_len) {
    free(uri_domain);
    return AEE_EFAILED;
  }

  int err = htp_ops_open(uri_domain, &session->handle);
  free(uri_domain);
  return err;
}

static int open_reserved_session(htp_ops_session *session, const char *module_uri) {
  if (&remote_session_control == NULL) {
    return AEE_EUNSUPPORTED;
  }

  const char *uri = module_uri != NULL && module_uri[0] != '\0' ? module_uri : htp_ops_URI;
  struct remote_rpc_reserve_new_session reserve = {};
  /* FastRPC names the compute domain "cdsp" on the Android targets used by
   * this backend. The numeric base domain remains configurable. */
  const char *domain_name = "cdsp";
  reserve.domain_name_len = (uint32_t) strlen(domain_name);
  reserve.domain_name = (char *) domain_name;
  reserve.session_name = (char *) "sd.cpp-htp";
  reserve.session_name_len = (uint32_t) strlen(reserve.session_name);

  int err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, &reserve, sizeof(reserve));
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "htp_ops: FASTRPC_RESERVE_NEW_SESSION failed err=0x%x (%d)\n", err, err);
    return err;
  }
  session->session_id = reserve.session_id;
  session->effective_domain_id = reserve.effective_domain_id;

  int unsigned_err = enable_unsigned_module(session->effective_domain_id);
  if (unsigned_err != AEE_SUCCESS) {
    fprintf(stderr, "htp_ops: DSPRPC_CONTROL_UNSIGNED_MODULE failed domain=%u err=0x%x (%d)\n",
            session->effective_domain_id, unsigned_err, unsigned_err);
    return unsigned_err;
  }

  char session_uri[MAX_DOMAIN_URI_SIZE + 256] = {};
  struct remote_rpc_get_uri get_uri = {};
  get_uri.session_id = session->session_id;
  get_uri.domain_name = (char *) domain_name;
  get_uri.domain_name_len = (uint32_t) strlen(domain_name);
  get_uri.module_uri = (char *) uri;
  get_uri.module_uri_len = (uint32_t) strlen(uri);
  get_uri.uri = session_uri;
  get_uri.uri_len = sizeof(session_uri);

  err = remote_session_control(FASTRPC_GET_URI, &get_uri, sizeof(get_uri));
  if (err == AEE_SUCCESS) {
    return htp_ops_open(session_uri, &session->handle);
  }

  fprintf(stderr, "htp_ops: FASTRPC_GET_URI failed session_id=%u err=0x%x (%d); trying domain URI fallback\n",
          session->session_id, err, err);

  /* Older libcdsprpc versions can reserve a session but lack GET_URI. The
   * effective-domain URI remains a useful compatibility fallback. */
  return open_domain_uri(session, uri);
}

int htp_ops_session_open(htp_ops_session **out, int base_domain, const char *name, int reserve_new) {
  if (out == NULL) {
    return -1;
  }
  *out = NULL;

  htp_ops_session *session = (htp_ops_session *) calloc(1, sizeof(*session));
  if (!session) {
    return -1;
  }
  session->handle = (remote_handle64) -1;
  session->base_domain_id = base_domain;
  session->effective_domain_id = (uint32_t) base_domain;
  session->reserved = reserve_new != 0;

  int err = reserve_new ? open_reserved_session(session, name) : open_domain_uri(session, name);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "htp_ops: session open failed reserve=%d base_domain=%d err=0x%x (%d)\n",
            reserve_new, base_domain, err, err);
    free(session);
    return -1;
  }

  if (session->reserved) {
    session->session_index = alloc_reserved_index();
    if (session->session_index == 0) {
      fprintf(stderr, "htp_ops: no free reserved session index (max %d)\n",
              HTP_OPS_MAX_RESERVED_SESSIONS - 1);
      htp_ops_close(session->handle);
      free(session);
      return -1;
    }
  }

  struct remote_rpc_control_latency lat_ctrl;
  lat_ctrl.enable = RPC_PM_QOS;
  lat_ctrl.latency = 50;
  err = remote_handle64_control(session->handle, DSPRPC_CONTROL_LATENCY, &lat_ctrl, sizeof(lat_ctrl));
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "htp_ops: DSPRPC_CONTROL_LATENCY failed session_id=%u err=0x%x (%d)\n",
            session->session_id, err, err);
    htp_ops_close(session->handle);
    free_reserved_index(session->session_index);
    free(session);
    return -1;
  }

  fprintf(stderr, "HTP session opened: id=%u index=%u domain=%u reserved=%d handle=0x%llx\n",
          session->session_id, session->session_index, session->effective_domain_id,
          session->reserved, (unsigned long long) session->handle);
  *out = session;
  return 0;
}

htp_ops_session *htp_ops_default_session(void) {
  return default_session;
}

int htp_ops_session_init_backend(htp_ops_session *session) {
  if (session == NULL || session->handle == (remote_handle64) -1) {
    return -1;
  }
  int err;
  if (session->reserved && getenv("SD_HTP_MULTI_SESSION") != NULL &&
      getenv("SD_HTP_MULTI_SESSION")[0] != '0') {
    err = htp_ops_init_mapping_backend(session->handle);
  } else {
    err = htp_ops_init_backend(session->handle);
  }
  if (err == AEE_SUCCESS) {
    session->backend_initialized = 1;
  }
  return err;
}

int htp_ops_session_acquire_backend(htp_ops_session *session) {
  if (session == NULL || session->handle == (remote_handle64) -1) {
    return -1;
  }
  const int err = htp_ops_init_backend(session->handle);
  if (err == AEE_SUCCESS) {
    session->backend_initialized = 1;
  }
  return err;
}

int htp_ops_session_release_backend(htp_ops_session *session) {
  if (session == NULL || session->handle == (remote_handle64) -1) {
    return -1;
  }
  const int err = htp_ops_release_backend(session->handle);
  if (err == AEE_SUCCESS) {
    session->backend_initialized = 0;
  }
  return err;
}

int htp_ops_session_create_channel(htp_ops_session *session, int fd, unsigned int max_msg_size) {
  if (session == NULL || session->handle == (remote_handle64) -1) {
    return -1;
  }
  int err = htp_ops_create_channel(session->handle, fd, max_msg_size);
  if (err == AEE_SUCCESS) {
    session->channel_initialized = 1;
  }
  return err;
}

int htp_ops_session_create_channel_with_token(htp_ops_session *session, int fd, unsigned int max_msg_size) {
  if (session == NULL || session->handle == (remote_handle64) -1) {
    return -1;
  }
  const uint64_t token = session_channel_token(session);
  int err = htp_ops_create_channel_with_token(session->handle, token, fd, max_msg_size);
  if (err == AEE_SUCCESS) session->channel_initialized = 1;
  return err;
}

remote_handle64 htp_ops_session_handle(const htp_ops_session *session) {
  return session != NULL ? session->handle : (remote_handle64) -1;
}

int htp_ops_session_destroy_channel(htp_ops_session *session) {
  if (session == NULL || session->handle == (remote_handle64) -1) {
    return -1;
  }
  const int err = htp_ops_destroy_channel(session->handle);
  if (err == AEE_SUCCESS) {
    session->channel_initialized = 0;
  }
  return err;
}

int htp_ops_set_active_session(htp_ops_session *session) {
  if (session == NULL || session->handle == (remote_handle64) -1) {
    return -1;
  }
  active_session = session;
  return 0;
}

int htp_ops_set_default_session(void) {
  if (default_session == NULL || default_session->handle == (remote_handle64) -1) {
    return -1;
  }
  active_session = default_session;
  return 0;
}

uint32_t htp_ops_session_id(const htp_ops_session *session) {
  return session != NULL ? session->session_id : 0;
}

uint32_t htp_ops_session_domain(const htp_ops_session *session) {
  return session != NULL ? session->effective_domain_id : 0;
}

int htp_ops_session_is_reserved(const htp_ops_session *session) {
  return session != NULL && session->reserved;
}

int htp_ops_session_is_initialized(const htp_ops_session *session) {
  return session != NULL && session->backend_initialized;
}

int htp_ops_session_has_channel(const htp_ops_session *session) {
  return session != NULL && session->channel_initialized;
}

void htp_ops_session_close(htp_ops_session *session) {
  if (session == NULL) {
    return;
  }
  if (session->handle != (remote_handle64) -1) {
    if (session->channel_initialized) {
      (void) htp_ops_destroy_channel(session->handle);
    }
    htp_ops_close(session->handle);
  }
  if (session->reserved) {
    free_reserved_index(session->session_index);
  }
  if (active_session == session) {
    active_session = default_session == session ? NULL : default_session;
  }
  free(session);
}

remote_handle64 get_global_handle() {
  if (active_session != NULL) {
    return active_session->handle;
  }
  return default_session != NULL ? default_session->handle : session_handle;
}

int open_dsp_session(int domain_id, int unsigned_pd_enabled) {
  if (default_session != NULL) {
    return 0;
  }
  if (unsigned_pd_enabled) {
    int unsigned_err = enable_unsigned_module((uint32_t) domain_id);
    if (unsigned_err != AEE_SUCCESS) {
      return -1;
    }
  }
  int err = htp_ops_session_open(&default_session, domain_id, NULL, 0);
  if (err == 0) {
    session_handle = default_session->handle;
    active_session = default_session;
  }
  return err;
}

void close_dsp_session() {
  htp_ops_session_close(default_session);
  default_session = NULL;
  active_session = NULL;
  session_handle = (remote_handle64) -1;
}

void init_htp_backend() {
  if (default_session != NULL) {
    (void) htp_ops_session_init_backend(default_session);
  }
}

int release_htp_backend() {
  return htp_ops_session_release_backend(default_session);
}

int create_htp_message_channel(int fd, unsigned int max_msg_size) {
  return htp_ops_session_create_channel(default_session, fd, max_msg_size);
}

int destroy_htp_message_channel() {
  return htp_ops_session_destroy_channel(default_session);
}
