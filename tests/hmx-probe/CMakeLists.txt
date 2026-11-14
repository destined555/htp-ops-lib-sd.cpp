include_directories(${CMAKE_CURRENT_BINARY_DIR})
if(OS_TYPE MATCHES "HLOS")
    add_executable(hmx_probe_test host.c fp8_tests.c integer_tests.c study_host.c ${CMAKE_CURRENT_BINARY_DIR}/hmx_probe_stub.c)
    build_idl(hmx_probe.idl hmx_probe_test)
    set_common_compile_and_link_options(hmx_probe_test)
    choose_dsprpc(${DSP_TYPE} probe_dsprpc)
    link_custom_library(hmx_probe_test ${probe_dsprpc})
    target_link_libraries(hmx_probe_test log dl)
    copy_binaries(hmx_probe_test)
else()
    add_library(hmx_probe_skel SHARED dsp.c study_dsp.c ${CMAKE_CURRENT_BINARY_DIR}/hmx_probe_skel.c
        ${HEXAGON_SDK_ROOT}/libs/qhl_hvx/src/qhblas_hvx/qhblas_hvx_matrix_matrix_mpy_ab.c
        ../../src/dsp/power.c ../../src/dsp/hmx_mgr.c
        ../../src/dsp/worker_pool.c ../../src/dsp/vtcm_mgr.cc)
    target_include_directories(hmx_probe_skel BEFORE PRIVATE
        ${HEXAGON_SDK_ROOT}/libs/qhl_hvx/inc/internal)
    build_idl(hmx_probe.idl hmx_probe_skel)
    target_link_libraries(hmx_probe_skel
        ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1
        ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1)
    copy_binaries(hmx_probe_skel)
endif()
