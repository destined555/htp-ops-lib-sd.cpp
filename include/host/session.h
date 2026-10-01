#pragma once

#include <remote.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A FastRPC handle is tied to one session/domain. Keep that ownership
 * explicit while retaining the legacy global wrappers below. */
typedef struct htp_ops_session htp_ops_session;

int htp_ops_session_open(htp_ops_session **out, int base_domain, const char *name, int reserve_new);
/* Returns the process-wide default session opened by open_dsp_session().
 * This is an accessor for the new explicit-session path; callers must not
 * mutate or close the returned object directly. */
htp_ops_session *htp_ops_default_session(void);
int htp_ops_session_init_backend(htp_ops_session *session);
int htp_ops_session_acquire_backend(htp_ops_session *session);
int htp_ops_session_release_backend(htp_ops_session *session);
int htp_ops_session_create_channel(htp_ops_session *session, int fd, unsigned int max_msg_size);
int htp_ops_session_create_channel_with_token(htp_ops_session *session, int fd, unsigned int max_msg_size);
int htp_ops_session_destroy_channel(htp_ops_session *session);
remote_handle64 htp_ops_session_handle(const htp_ops_session *session);
/* Select which session backs the legacy exported op wrappers. */
int htp_ops_set_active_session(htp_ops_session *session);
int htp_ops_set_default_session(void);
uint32_t htp_ops_session_id(const htp_ops_session *session);
uint32_t htp_ops_session_domain(const htp_ops_session *session);
int htp_ops_session_is_reserved(const htp_ops_session *session);
int htp_ops_session_is_initialized(const htp_ops_session *session);
int htp_ops_session_has_channel(const htp_ops_session *session);
void htp_ops_session_close(htp_ops_session *session);

#ifdef __cplusplus
}
#endif

int open_dsp_session(int domain_id, int unsigned_pd_enabled);
void close_dsp_session();

remote_handle64 get_global_handle();

void init_htp_backend();
int release_htp_backend();
int create_htp_message_channel(int fd, unsigned int max_msg_size);
int destroy_htp_message_channel();
