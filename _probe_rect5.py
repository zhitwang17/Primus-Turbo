"""Bisect the npool==5 + nbuf>2 LDS corruption in the dense TN 4-wave kernel."""
import sys
import torch

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as G
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.pytorch.core.low_precision import float8_e4m3

FP8 = float8_e4m3


def fp8(shape, scale=0.5):
    ref = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale
    amax = ref.abs().amax().clamp(min=1e-8)
    fmax = torch.finfo(FP8).max
    s = (amax / fmax).to(torch.float32)
    q = (ref.to(torch.float32) / s).clamp(-fmax, fmax).to(FP8)
    return q, s.reshape(1), ref


def snr(out, ref):
    out, ref = out.float(), ref.float()
    n = (out - ref).pow(2).mean()
    return 99.0 if n == 0 else float(10 * torch.log10(ref.pow(2).mean() / n))


VARIANTS = {
    # name: (geom, nosplit)
    "G1_rect_all2": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 2))),
    "G2_rect_p4deep": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 3))),
    "G3_rect_p64deep": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 128, 2), (1, 64, 3))),
    "G4_sq5_lastdeep": (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 2), (1, 64, 2), (1, 64, 3))),
    "G5_sq5_all2": (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 2), (1, 64, 2), (1, 64, 2))),
    "G6_sq4_onedeep": (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 2), (1, 128, 3))),
    "G7_sq4_bothdeep": (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 3), (1, 128, 3))),
    "G9_rect6_lastdeep": (
        384,
        192,
        ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 64, 2), (1, 64, 3)),
    ),
}

FORCE_FULL_DRAIN = "--fulldrain" in sys.argv
FORCE_ELGK0 = "--elgk0" in sys.argv

if FORCE_ELGK0:
    G._TN4_ELGK = 0

if FORCE_FULL_DRAIN:
    _orig_asm = G._dense_tn_wave4_asm

    def _asm(geom, cbsz, blgp):
        key = (geom, cbsz, blgp)
        G._TN4_ASM_CACHE.pop(key, None)
        a, c, t = _orig_asm(geom, cbsz, blgp)
        lines = []
        for ln in a.split("\n"):
            if ln.startswith("s_waitcnt vmcnt(") and "lgkmcnt" in ln:
                ln = "s_waitcnt vmcnt(0) lgkmcnt(0)"
            lines.append(ln)
        out = ("\n".join(lines), c, t)
        G._TN4_ASM_CACHE[key] = out
        return out

    G._dense_tn_wave4_asm = _asm

G._dense_tn_split = lambda tiles, k_iters, ncu: None

MP, NP, KP = 1536, 1536, 1536  # M', N', K' -- divisible by 384/192 and 256, 12 k-iters


def run(name, geom):
    G._TN_WAVE4_CACHE.clear()
    G._TN4_WS_CACHE.clear()
    G._TN_AUTOTUNE_CACHE.clear()
    G._tn4_geom = lambda M, N, k_iters, ncu, _g=geom: _g
    a, sa, ar = fp8((KP, MP))
    b, sb, br = fp8((KP, NP))
    ref = (ar.t() @ br).to(torch.bfloat16)
    try:
        out = gemm_fp8_tensorwise_flydsl_kernel(
            a, sa, b, sb, trans_a=True, trans_b=False, out_dtype=torch.bfloat16
        )
        torch.cuda.synchronize()
        print(f"{name:22s} lds={sum(p[1]*128*p[2] for p in geom[2]):7d} SNR={snr(out, ref):6.2f}", flush=True)
    except Exception as exc:
        print(f"{name:22s} EXC {type(exc).__name__}: {str(exc)[:200]}", flush=True)


print(f"fulldrain={FORCE_FULL_DRAIN} elgk0={FORCE_ELGK0}", flush=True)
for n, g in VARIANTS.items():
    run(n, g)
