#include "dsp/mmap_mgr.h"

#ifdef HTP_OPS_MMAP_TRACE
#include <HAP_farf.h>
#endif
#include <HAP_mem.h>

#include <unordered_map>

namespace mmap_manager {

struct MapKey {
  uint64_t token;
  int fd;
  uint64_t offset;
  uint64_t size;
  bool operator==(const MapKey &other) const {
    return token == other.token && fd == other.fd && offset == other.offset && size == other.size;
  }
};

struct MapKeyHash {
  size_t operator()(const MapKey &key) const {
    size_t h = static_cast<size_t>(key.token ^ (key.token >> 32));
    h ^= static_cast<size_t>(key.fd) * 0x9e3779b1u;
    h ^= static_cast<size_t>(key.offset ^ (key.offset >> 32)) * 0x85ebca6bu;
    h ^= static_cast<size_t>(key.size ^ (key.size >> 32)) * 0xc2b2ae35u;
    return h;
  }
};

MapKey make_map_key(uint64_t session_token, int fd, uint64_t offset, uint64_t size) {
  // Domain 3 is the original one-session ABI: HAP_mmap_get() maps an fd, not
  // an individual tensor view.  Keeping its cache fd-keyed preserves the
  // baseline hot path while nonzero session tokens retain independent views.
  if (session_token == 0) {
    return MapKey{0, fd, 0, 0};
  }
  return MapKey{session_token, fd, offset, size};
}

struct MapValue { void *addr; uint32_t refs; };
std::unordered_map<MapKey, MapValue, MapKeyHash> mapping;
uint64_t current_token;

}  // namespace mmap_manager

extern "C" {

void mmap_manager_set_context(uint64_t session_token) {
  mmap_manager::current_token = session_token;
}

void *mmap_manager_get_map(int fd, uint64_t offset, uint64_t size) {
  const mmap_manager::MapKey key = mmap_manager::make_map_key(
      mmap_manager::current_token, fd, offset, size);
  auto it = mmap_manager::mapping.find(key);
  if (it != mmap_manager::mapping.end()) {
#ifdef HTP_OPS_MMAP_TRACE
    FARF(ALWAYS, "HTP_MMAP_TRACE get token=%llu fd=%d off=%llu size=%llu hit=1 refs=%u addr=%p",
         (unsigned long long)key.token, fd, (unsigned long long)offset,
         (unsigned long long)size, (unsigned)it->second.refs, it->second.addr);
#endif
    return it->second.addr;
  }
  void *p = nullptr;
  const int err = HAP_mmap_get(fd, &p, nullptr);
  if (err != 0) {
#ifdef HTP_OPS_MMAP_TRACE
    FARF(ALWAYS, "HTP_MMAP_TRACE get token=%llu fd=%d off=%llu size=%llu hit=0 err=0x%x",
         (unsigned long long)key.token, fd, (unsigned long long)offset,
         (unsigned long long)size, err);
#endif
    return nullptr;
  }
  mmap_manager::mapping.emplace(key, mmap_manager::MapValue{p, 1});
#ifdef HTP_OPS_MMAP_TRACE
  FARF(ALWAYS, "HTP_MMAP_TRACE get token=%llu fd=%d off=%llu size=%llu hit=0 refs=1 addr=%p",
       (unsigned long long)key.token, fd, (unsigned long long)offset,
       (unsigned long long)size, p);
#endif
  return p;
}

int mmap_manager_put_map(int fd) {
  for (auto it = mmap_manager::mapping.begin(); it != mmap_manager::mapping.end(); ++it) {
    if (it->first.token != mmap_manager::current_token || it->first.fd != fd) continue;
    if (it->second.refs > 1) {
      --it->second.refs;
      return 0;
    }
    const int ret = HAP_mmap_put(fd);
    mmap_manager::mapping.erase(it);
    return ret;
  }
  return 0;
}

int mmap_manager_put_map_view(int fd, uint64_t offset, uint64_t size) {
  if (mmap_manager::current_token == 0) {
    return mmap_manager_put_map(fd);
  }
  const mmap_manager::MapKey key = mmap_manager::make_map_key(
      mmap_manager::current_token, fd, offset, size);
  auto it = mmap_manager::mapping.find(key);
  if (it == mmap_manager::mapping.end() && offset == 0 && size != 0) {
    int released = 0;
    for (auto cur = mmap_manager::mapping.begin(); cur != mmap_manager::mapping.end();) {
      if (cur->first.token == mmap_manager::current_token && cur->first.fd == fd &&
          cur->first.offset + cur->first.size <= size) {
        HAP_mmap_put(fd);
        cur = mmap_manager::mapping.erase(cur);
        released = 1;
      } else {
        ++cur;
      }
    }
    return released ? 0 : -1;
  }
  if (it == mmap_manager::mapping.end()) return -1;
  if (it->second.refs > 1) {
    --it->second.refs;
    return 0;
  }
  const int ret = HAP_mmap_put(fd);
  mmap_manager::mapping.erase(it);
  return ret;
}

void mmap_manager_release_all() {
  for (auto it = mmap_manager::mapping.begin(); it != mmap_manager::mapping.end();) {
    if (it->first.token == mmap_manager::current_token) {
      HAP_mmap_put(it->first.fd);
      it = mmap_manager::mapping.erase(it);
    } else {
      ++it;
    }
  }
}

}
