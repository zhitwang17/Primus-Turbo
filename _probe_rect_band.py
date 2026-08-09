"""Sweep the L2 band (group_m, group_n) for the rectangular TN macro tile."""
import torch

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as G
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.pytorch.core.low_precision import float8_e4m3

FP8 = float8_e4m3
SHAPES = [("s2", 3072, 12288, 16384), ("s3", 12288, 3072, 16384), ("s1", 3072, 3072, 16384)]
BANDS = [(4, 2), (2, 2), (8, 2), (4, 1), (4, 4), (2, 4), (8, 1), (2, 8), (1, 2)]
REPS = 5


def fp8(shape):
    ref = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    fmax = torch.finfo(FP8).max
    s = (ref.abs().amax().clamp(min=1e-8) / fmax).to(torch.float32)
    return (ref.float() / s).clamp(-fmax, fmax).to(FP8), s.reshape(1)


def time_ms(fn, iters=30):
    for _ in range(12):
        fn()
    torch.cuda.synchronize()
    ts = []
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


ops = {}
for name, Mp, Np, Kp in SHAPES:
    a, sa = fp8((Kp, Mp))
    b, sb = fp8((Kp, Np))
    ops[name] = (a, sa, b, sb)

res = {}
for rep in range(REPS):
    for band in BANDS:
        G._TN_WAVE4_CACHE.clear()
        G._tn_wave4_band = lambda M, N, geom, _b=band: _b
        for name, Mp, Np, Kp in SHAPES:
            a, sa, b, sb = ops[name]
            fn = lambda: gemm_fp8_tensorwise_flydsl_kernel(
                a, sa, b, sb, trans_a=True, trans_b=False, out_dtype=torch.bfloat16
            )
            res.setdefault((band, name), []).append(time_ms(fn))
    print(f"rep {rep}", flush=True)

print(f"\n{'band':>8} " + " ".join(f"{n:>16}" for n, *_ in SHAPES))
base = {n: sorted(res[(BANDS[0], n)])[REPS // 2] for n, *_ in SHAPES}
for band in BANDS:
    row = f"{str(band):>8} "
    for name, *_ in SHAPES:
        v = sorted(res[(band, name)])[REPS // 2]
        row += f"  {v:8.4f} {100*(v/base[name]-1):+6.2f}%"
    print(row, flush=True)
