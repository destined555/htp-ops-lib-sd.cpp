#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int hmx_manager_setup();
int hmx_manager_reset();

void hmx_manager_enable_execution(); // enable HMX execution for current thread
void hmx_manager_disable_execution(); // disable HMX execution for current thread

void hmx_unit_acquire();
void hmx_unit_release();

#ifdef __cplusplus
}
#endif
