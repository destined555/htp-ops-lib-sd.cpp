#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mmap_manager_set_context(uint64_t session_token);
void *mmap_manager_get_map(int fd, uint64_t offset, uint64_t size);
int mmap_manager_put_map(int fd);
int mmap_manager_put_map_view(int fd, uint64_t offset, uint64_t size);
void mmap_manager_release_all();

#ifdef __cplusplus
}
#endif
