#include "dsp/hmx_mgr.h"
#include "dsp/worker_pool.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <stddef.h>

static int hmx_mgr_ctx_id;
static int hmx_mgr_spin_lock;
static int hmx_mgr_last_lock_result;

worker_pool_context_t hmx_worker_pool_ctx;

int hmx_manager_setup() {
  // NOTE(hzx): HMX should be already powered up in power_setup()

  if (hmx_mgr_ctx_id && hmx_worker_pool_ctx) {
    return 0;
  }

  if (hmx_worker_pool_ctx) {
    worker_pool_deinit(&hmx_worker_pool_ctx);
  }
  if (hmx_mgr_ctx_id) {
    const int err = HAP_compute_res_release(hmx_mgr_ctx_id);
    if (err) {
      FARF(ERROR, "%s: failed to release incomplete context %d, return code 0x%x", __func__, hmx_mgr_ctx_id, err);
      return -1;
    }
    hmx_mgr_ctx_id = 0;
  }
  hmx_mgr_spin_lock = 0;
  hmx_mgr_last_lock_result = 0;

  compute_res_attr_t req;
  int                err = HAP_compute_res_attr_init(&req);
  if (err) {
    FARF(ERROR, "%s: HAP_compute_res_attr_init failed with return code 0x%x", __func__, err);
    return -1;
  }
  err = HAP_compute_res_attr_set_hmx_param(&req, 1);
  if (err) {
    FARF(ERROR, "%s: failed to configure HMX request, return code 0x%x", __func__, err);
    return -1;
  }

  const int new_ctx_id = HAP_compute_res_acquire(&req, 10000);  // 10ms timeout
  if (new_ctx_id == 0) {
    FARF(ERROR, "%s: HAP_compute_res_acquire failed", __func__);
    return -1;
  }

  worker_pool_context_t new_worker_pool = NULL;
  err = worker_pool_init_ex(&new_worker_pool, 8192, 1, 1);
  if (err || !new_worker_pool) {
    FARF(ERROR, "%s: HMX worker pool init failed with return code 0x%x", __func__, err);
    if (new_worker_pool) {
      worker_pool_deinit(&new_worker_pool);
    }
    const int release_err = HAP_compute_res_release(new_ctx_id);
    if (release_err) {
      FARF(ERROR, "%s: failed to roll back context %d, return code 0x%x", __func__, new_ctx_id, release_err);
      hmx_mgr_ctx_id = new_ctx_id;
    }
    return -1;
  }

  hmx_mgr_ctx_id     = new_ctx_id;
  hmx_worker_pool_ctx = new_worker_pool;
  return 0;
}

int hmx_manager_reset() {
  if (hmx_worker_pool_ctx) {
    worker_pool_deinit(&hmx_worker_pool_ctx);
  }
  hmx_mgr_spin_lock = 0;

  if (hmx_mgr_ctx_id) {
    const int err = HAP_compute_res_release(hmx_mgr_ctx_id);
    if (err) {
      FARF(ERROR, "%s: failed to release context %d, return code 0x%x", __func__, hmx_mgr_ctx_id, err);
      return -1;
    }
    hmx_mgr_ctx_id = 0;
  }
  return 0;
}

void hmx_manager_enable_execution() {
  if (!hmx_mgr_ctx_id || !hmx_worker_pool_ctx) {
    hmx_mgr_last_lock_result = -1;
    return;
  }

  // enable current thread to timeshare HMX unit
  int err = HAP_compute_res_hmx_lock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
  hmx_mgr_last_lock_result = err;
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_hmx_lock2 failed with return code 0x%x", err);
  }
}

int hmx_manager_get_last_lock_result() {
  return hmx_mgr_last_lock_result;
}

void hmx_manager_disable_execution() {
  if (!hmx_mgr_ctx_id || !hmx_worker_pool_ctx) {
    return;
  }

  HAP_compute_res_hmx_unlock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
}

int hmx_manager_get_context_id() {
  return hmx_mgr_ctx_id;
}

void * hmx_manager_get_worker_pool() {
  return (void *) hmx_worker_pool_ctx;
}

void hmx_unit_acquire() {
  int *lock_ptr = &hmx_mgr_spin_lock;
  asm volatile(
    "1:  r0 = memw_locked(%0)     \n"
    "    p0 = cmp.eq(r0, #0)      \n"
    "    if (!p0) jump 2f         \n"
    "    memw_locked(%0, p0) = %0 \n"
    "    if (p0) jump 3f          \n"
    "2:  pause(#8)                \n"
    "    jump 1b                  \n"
    "3:"
    : "+r"(lock_ptr)::"p0", "r0");
}

void hmx_unit_release() {
  *(volatile int *) &hmx_mgr_spin_lock = 0;
}
