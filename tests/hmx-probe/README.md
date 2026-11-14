# Isolated HMX diagnostic

Enable `HTP_BUILD_HMX_PROBE=ON`. This builds a separate FastRPC interface and test executable;
it does not change the production operator ABI or dispatch.
Native v79 is required for the FP8 suites and the general `tile` experiments.

Validated commands:

- `0`: FP16 all-ones baseline.
- `int-validate8`, `int-validate4`: 1024 one-hot layout cases each, followed by matrix comparisons.
- `int-validate`: UINT8 activation values across 0..255, signed INT8/INT4 decoding and matrix comparisons.
- `int-wide`: INT32 reconstruction, full-range INT8 weights, 32 matrix cases and 19 boundary cases.
- `int-tiled`: signed activation zero-point correction, padding and distinct K tiles.
- `fp8-validate`: all 256 byte encodings, 1024 one-hot cases, 36 matrix cases with scales/bias.
- `fp8-tiled`: padding and distinct K tiles, including FP16 x FP8.
- `fp8-block`: low/high byte-plane selection and full 32 x 32 x 32 FP8 block comparisons.
- `int-cm-block`: compact 64 x 32 x 32 INT8/INT4 blocks, 2048 one-hot cases each and 24 signed random matrices each; UINT8 saturated output.
- `bench`: FP16/FP8/integer tile kernels; reports qtimer, pcycles and RPC wall time separately.
- `bench-compact`: compares FP16 M64 with compact INT8/INT4 M64, saturated UINT8 output.
- `study-validate`: 786 signed/full-range/zero-point/FP16-conversion comparisons, including CM N64 `weight:deep` and K=6144.
- `study-bench`: M64, N32/64, K32/128/1536/6144; UINT8, reconstructed INT32, FP16, and input-quantization-inclusive timings.
- `study-baseline`: FP16 N64 `weight:deep` and batched K loads; checks that the FP16 comparison is not limited by a scalar tile loop.

FP8 cases without output bias require bit-exact FP16 results. Nonzero output bias allows at most
one FP16 ULP, with the number of bit differences printed explicitly.
Integer expected values come from scalar integer arithmetic, independently of HMX output.
The benchmark's two-word INT32 output excludes the final host/HVX reconstruction operation.

Exploratory commands (`int-2x`, `int-cm`, `int-cvt`, etc.) are not passing GEMM recipes.
In particular, `weight.n:2x` must not be interpreted as a drop-in faster `weight.n`.

This workspace's complete measured recipes and limitations are in
[hmx-precision-study.md](../../../hmx-precision-study.md).
The compact/deep acceleration measurements and official/community references are in
[hmx-quantization-acceleration.md](../../../hmx-quantization-acceleration.md).
Run `bash ../../../run-hmx-study.sh` from this directory for container builds and serial-specific phone testing.
Pass suite names to run a subset, for example `bash ../../../run-hmx-study.sh study-validate study-bench study-baseline`.
