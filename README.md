# Custom Op library for Qualcomm's Hexagon Tensor Processor

This is the code repository for the paper [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324). It contains an operator library supporting LLM inference on Qualcomm Hexagon NPU, which needs to be used with the [llama.cpp main repository](https://github.com/haozixu/llama.cpp-npu). This project is primarily a research prototype and is not recommended for production environments.

**Hardware requirements**: Android phones with Qualcomm Snapdragon 8 Gen 2 or higher SoC, specifically requiring Hexagon DSP version 73 or above. Note that this implementation relies on **FP16 HMX**, which may not be available on some mid-to-low-end devices.

**Software requirements**: Hexagon SDK 6.7.0.0 (Hexagon Tools 21.0.04, Android NDK r27d).

## Compilation

For this workspace's SDK 6.7 / v75 configuration, run `bash ../quant/build-sdk67.sh` from this directory. The script builds the host library, DSP skeleton, sd.cpp, matching runtimes, and isolated HMX/HVX diagnostic. See `../quant/sdk67-matmul-report.md` for phone verification and quantized-compute limitations.

The isolated native v79 study in `tests/hmx-probe` now validates direct UINT8 x INT8, UINT8 x INT4 and FP8 HMX multiplication on Snapdragon 8 Elite. See `../quant/hmx-precision-study.md` for tested layouts, output conversion, correctness results and microbenchmarks. Run `bash ../quant/run-hmx-study.sh` to reproduce in this workspace. These diagnostic paths are not connected to production GEMM or attention dispatch.

Native UINT8 x INT8 linear opcode 15 in `src/dsp/ops/linear_w8a8.c` is enabled on v75 and later. Both Snapdragon 8 Gen 3 and 8 Elite use HVX QFloat add/multiply for activation quantization and output scaling, with integer round-to-nearest-even. The kernel inherits the DSP target ISA; it no longer forces this source to v79. Input/output tensors and scale metadata remain FP32. The sd.cpp dense DiT integration replaces selected FP16 weight storage with INT8 constants in the existing shared RPCMEM chunks; it has no per-call host weight copy. See `../quant/w8a8-v75-qfloat-speed.md` for the v75 operator study and numerical validation. INT4/FP8 remain diagnostic paths, and attention QK/AV products are unchanged.

The current W8A8 kernel uses worker-pool weight copies, shape-dependent VTCM
chunks and parallel quantization/output conversion. Its benchmark directly
uses production FP16 dispatch; see `../quant/w8a8-optimization.md` for results.

1. First, ensure the Hexagon SDK environment is set up. Run the following command in the root directory of the Hexagon SDK:

```sh
source setup_sdk_env.source
```

2. Execute the following **two** commands in the root directory of this project:

```sh
build_cmake android
build_cmake hexagon DSP_ARCH=v73
```

Here, `DSP_ARCH` specifies the target Hexagon NPU architecture version. We recommend using `v73` by default for better compatibility. (The NPU architecture version on Snapdragon 8 Gen 2 is v73; you can modify this option according to your target hardware.)

After compilation, you should see two directories: `android_ReleaseG_aarch64` and `hexagon_ReleaseG_toolv21_v73` (the actual names may vary depending on the compilation mode, specific toolchain, and target architecture version). Note the following two products:

- `android_ReleaseG_aarch64/libhtp_ops.so`
- `hexagon_ReleaseG_toolv21_v73/libhtp_ops_skel.so`

These two dynamic link libraries will be used later; please refer to the instructions in the main repository. In FastRPC terminology, they are the Stub (`libhtp_ops.so`) and Skeleton (`libhtp_ops_skel.so`) respectively. You can use `ldd` to distinguish between the two shared objects: `libhtp_ops.so` targets the AArch64 architecture and runs on the CPU; `libhtp_ops_skel.so` targets the Q6DSP architecture and runs on the Hexagon NPU (cDSP).

## Known Issues

Compiling for DSP Arch v79 targets with higher versions of the Hexagon SDK may cause errors in some floating-point calculations. When the target hardware is DSP Arch v79, we recommend compiling for v73/v75 and disabling `vgather`-based exp calculations (set `const bool enable_vgather_exp` to `false` in `simple_flash_attn_f16_core` in `src/dsp/ops/flash_attn.c`).

## About FP16 HMX

Our current implementation heavily relies on FP16 HMX for dequantize GEMM and FlashAttention. The relevant instructions were derived from the qhl_hmx sample in Hexagon SDK 5.x. SDK 6.7 includes the official FP16 library and example under `libs/qhl_hmx` and `examples/qhl_hmx` again. Integer HMX intrinsics in the compiler headers do not imply that this library already implements quantized GEMM.

For DSP versions v73 and above, the following inline assembly snippet is used to load tile data (FP16 Croutons) from TCM and accumulate the tile-level inner product results into the internal accumulator. The HMX activation loading instruction and weight loading instruction must be in the same instruction packet.

```C
static HMX_INLINE_ALWAYS void hmx_load_tiles_fp16(const __fp16 *row_tiles, const __fp16 *col_tiles, size_t n_tiles) {
  size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  asm volatile(
    "{ activation.hf = mxmem(%0, %1):deep\n"
    "weight.hf = mxmem(%2, %3) }\n" ::"r"(row_tiles),
    "r"(limit), "r"(col_tiles), "r"(limit)
    : "memory");
}
```

The following inline assembly snippet is used to output the content of the internal accumulator to TCM. Judging from the name, HMX contains a converter component, but its conversion capabilities are not yet clear.

```C
static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
  asm volatile(
    "cvt.hf = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(2),
    "r"(out), "r"(0)
    : "memory");
}
```
The input and output tiles of FP16 HMX here follow the FP16 Crouton layout, which is shown in Figure 4 of our paper. You can also refer to the `test_mat_mul_rpc` function in `src/host/test.c` for details.

The HMX component also supports per-channel scale & bias for the output tile. The size of this data region is 256 bytes.

```C
static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
  asm volatile("bias = mxmem2(%0)" ::"r"(scales));
}
```

**Note that the above information is specific to a particular sequence of HMX instructions. There may be other FP16 HMX instructions that require different data layouts and have functional differences.** There are numerous instruction variants of HMX, and it is difficult for us to associate them with all existing layouts. You can check QNN's `include/QNN/HTP/core/memory_layout.h` and `include/QNN/HTP/core/tile_extract.h` for some information about HMX layouts. If you discover other usable HMX instructions, please feel free to share them with us.

## Citation

If you find our work helpful, please cite us.

```bibtex
@article{hao2025scaling,
  title={Scaling LLM Test-Time Compute with Mobile NPU on Smartphones},
  author={Zixu Hao and Jianyu Wei and Tuowei Wang and Minxing Huang and Huiqiang Jiang and Shiqi Jiang and Ting Cao and Ju Ren},
  journal={arXiv preprint arXiv:2509.23324},
  year={2025}
}
```
