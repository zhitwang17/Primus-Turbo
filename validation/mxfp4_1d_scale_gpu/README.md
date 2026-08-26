# MXFP4 1D Scale GPU Validation

This directory contains an op-level validation harness for the experimental
`Float4QuantConfig.weight_quant_mode` values:

- `2d_direct` (production-compatible default)
- `1d_direct`
- `1d_qdq` (`q(dq(q(w, axis=-1)), axis=-2)`)

The harness intentionally validates numerical behavior and operator cost only;
it does not run training or make convergence claims.

Run on a gfx950 GPU with the GEAK ROCm/FlyDSL image:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home:/home -v /mnt:/mnt \
  -e HIP_VISIBLE_DEVICES=1 \
  -e LD_LIBRARY_PATH=/home/zhitwang/primus-turbo/primus_turbo/lib:/opt/rocm/lib \
  geak/rocm-pytorch-flydsl:0.2.4-rocm7.2.2 \
  python /home/zhitwang/primus-turbo-branches/mxfp4-1d-scale-gpu-validation/validation/mxfp4_1d_scale_gpu/run_validation.py \
    all --output-dir /home/zhitwang/primus-turbo-branches/mxfp4-1d-scale-gpu-validation/validation/mxfp4_1d_scale_gpu/results
```

The default extension path points at the compatible prebuilt extension in
`/home/zhitwang/primus-turbo`. Override it with `--extension` when using a new
build.
