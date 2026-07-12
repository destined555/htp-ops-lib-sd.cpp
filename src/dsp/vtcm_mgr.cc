#include "dsp/vtcm_mgr.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>

#include <cstring>

namespace vtcm_manager {

uint8_t *vtcm_base      = nullptr;
size_t   vtcm_size      = 0;
int      vtcm_mgr_ctx_id = 0;

}  // namespace vtcm_manager

extern "C" {

int vtcm_manager_setup() {
  using namespace vtcm_manager;

  int err;

  if (vtcm_mgr_ctx_id && vtcm_base && vtcm_size > 0) {
    return 0;
  }

  if (vtcm_mgr_ctx_id) {
    err = HAP_compute_res_release(vtcm_mgr_ctx_id);
    if (err) {
      FARF(ERROR, "%s: failed to release existing context %d, return code 0x%x", __func__, vtcm_mgr_ctx_id, err);
      return -1;
    }
    vtcm_mgr_ctx_id = 0;
  }
  vtcm_base = nullptr;
  vtcm_size = 0;

  unsigned int            avail_size, total_size;
  compute_res_vtcm_page_t avail_pages, total_pages;

  err = HAP_compute_res_query_VTCM(0, &total_size, &total_pages, &avail_size, &avail_pages);
  if (err) {
    FARF(ERROR, "%s: HAP_compute_res_query_VTCM failed with return code 0x%x", __func__, err);
    return -1;
  }
  if (avail_size == 0) {
    FARF(ERROR, "%s: no VTCM is currently available", __func__);
    return -1;
  }
  FARF(ALWAYS, "available VTCM size: %u KiB, total VTCM size: %u KiB", avail_size / 1024, total_size / 1024);

  compute_res_attr_t req;
  err = HAP_compute_res_attr_init(&req);
  if (err) {
    FARF(ERROR, "%s: HAP_compute_res_attr_init failed with return code 0x%x", __func__, err);
    return -1;
  }

  err = HAP_compute_res_attr_set_vtcm_param(&req, avail_size, 1);
  if (err) {
    FARF(ERROR, "%s: failed to request %u VTCM bytes in one page, return code 0x%x", __func__, avail_size, err);
    return -1;
  }
  FARF(ALWAYS, "%s: requesting %u VTCM bytes in one page", __func__, avail_size);

  const int new_ctx_id = HAP_compute_res_acquire(&req, 10000);  // timeout 10ms
  if (new_ctx_id == 0) {
    FARF(ERROR, "%s: HAP_compute_res_acquire failed for %u VTCM bytes", __func__, avail_size);
    return -1;
  }

  uint8_t *const acquired_base = (uint8_t *) HAP_compute_res_attr_get_vtcm_ptr(&req);
  if (!acquired_base) {
    FARF(ERROR, "%s: acquired context %d but VTCM pointer is null", __func__, new_ctx_id);
    err = HAP_compute_res_release(new_ctx_id);
    if (err) {
      FARF(ERROR, "%s: failed to release context %d, return code 0x%x", __func__, new_ctx_id, err);
      vtcm_mgr_ctx_id = new_ctx_id;
    }
    return -1;
  }

  vtcm_mgr_ctx_id = new_ctx_id;
  vtcm_base       = acquired_base;
  vtcm_size       = avail_size;
  memset(vtcm_base, 0, vtcm_size);
  FARF(ALWAYS, "%s: acquired %u VTCM bytes", __func__, (unsigned int) vtcm_size);
  return 0;
}

int vtcm_manager_reset() {
  using namespace vtcm_manager;

  if (vtcm_mgr_ctx_id) {
    const int err = HAP_compute_res_release(vtcm_mgr_ctx_id);
    if (err) {
      FARF(ERROR, "%s: failed to release context %d, return code 0x%x", __func__, vtcm_mgr_ctx_id, err);
      return -1;
    }
  }
  vtcm_mgr_ctx_id = 0;
  vtcm_base       = nullptr;
  vtcm_size       = 0;
  return 0;
}

void *vtcm_manager_get_vtcm_base() {
  return vtcm_manager::vtcm_base;
}

size_t vtcm_manager_get_vtcm_size() {
  return vtcm_manager::vtcm_size;
}
}
