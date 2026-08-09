"""Time forced TN macro-tile geometries on the six scored kernel shapes (alternating arms)."""
import sys
import torch

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as G
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.pytorch.core.low_precision import float8_e4m3

FP8 = float8_e4m3

SHAPES = [  # (M', N', K')
    ("s0", 3072, 9216, 16384),
    ("s1", 3072, 3072, 16384),
    ("s2", 3072, 12288, 16384),
    ("s3", 12288, 3072, 16384),
    ("s4", 3072, 21504, 32768),
    ("s5", 15360, 3072, 32768),
]

ARMS = {
    "SQ": (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 3), (1, 128, 3))),
    "R2": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 2))),
    "R2d": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 3))),
    "R3d": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 64, 3), (1, 64, 3))),
}
REPS = 3


def fp8(shape, scale=0.5):
    ref = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale
    amax = ref.abs().amax().clamp(min=1e-8)
    fmax = torch.finfo(FP8).max
    s = (amax / fmax).to(torch.float32)
    return (ref.to(torch.float32) / s).clamp(-fmax, fmax).to(FP8), s.reshape(1), ref


def snr(out, ref):
    out, ref = out.float(), ref.float()
    n = (out - ref).pow(2).mean()
    return 99.0 if n == 0 else float(10 * torch.log10(ref.pow(2).mean() / n))


def time_ms(fn, iters=30):
    for _ in range(15):
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
    a, sa, ar = fp8((Kp, Mp))
    b, sb, br = fp8((Kp, Np))
    ops[name] = (a, sa, b, sb, (ar.t() @ br).to(torch.bfloat16))

res = {}
snrs = {}
for rep in range(REPS):
    for arm, geom in ARMS.items():
        G._TN_WAVE4_CACHE.clear()
        G._TN4_WS_CACHE.clear()
        G._tn4_geom = lambda M, N, k, n, _g=geom: _g
        for name, Mp, Np, Kp in SHAPES:
            a, sa, b, sb, ref = ops[name]
            fn = lambda: gemm_fp8_tensorwise_flydsl_kernel(
                a, sa, b, sb, trans_a=True, trans_b=False, out_dtype=torch.bfloat16
            )
            try:
                if rep == 0:
                    snrs[(arm, name)] = snr(fn(), ref)
                res.setdefault((arm, name), []).append(time_ms(fn))
            except Exception as exc:
                print(f"{arm} {name} EXC {str(exc)[:120]}", flush=True)
    print(f"rep {rep} done", flush=True)

print(f"\n{'shape':>6} " + " ".join(f"{a:>18}" for a in ARMS))
for name, Mp, Np, Kp in SHAPES:
    row = f"{name:>6} "
    base = None
    for arm in ARMS:
        v = res.get((arm, name))
        if not v:
            row += f"{'-':>19}"
            continue
        v = sorted(v)[len(v) // 2]
        if base is None:
            base = v
        row += f"  {v:8.4f} {100*(v/base-1):+6.1f}% {snrs.get((arm,name),0):4.1f}"
    print(row, flush=True)
