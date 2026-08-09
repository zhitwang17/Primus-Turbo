"""Dump the emitted TN4 phase body for a broken geometry and A/B a few candidate fixes."""
import re
import sys
import torch

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as G
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.pytorch.core.low_precision import float8_e4m3

FP8 = float8_e4m3
G._dense_tn_split = lambda tiles, k_iters, ncu: None

BAD = (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 3)))
GOOD = (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 3), (1, 128, 3)))

_orig_asm = G._dense_tn_wave4_asm.__wrapped__ if hasattr(G._dense_tn_wave4_asm, "__wrapped__") else G._dense_tn_wave4_asm


def raw_asm(geom):
    G._TN4_ASM_CACHE.clear()
    return _orig_asm(geom, 0, 0)[0]


def show(geom, tag):
    a = raw_asm(geom)
    blocks = a.split("s_barrier")
    print(f"\n##### {tag} : {len(blocks)} barrier-separated blocks")
    # first in-loop phase = block index 2 (0 = prologue reads, 1 = prime wait)
    b = blocks[2]
    lines = [l for l in b.split("\n") if l.strip()]
    print(f"  phase0 lines={len(lines)}")
    for l in lines:
        if l.startswith("s_add_u32 m0") or l.startswith("buffer_load"):
            print("   " + l)
    print("  --- ds_read offsets by operand pair ---")
    off = {}
    for l in lines:
        m = re.match(r"ds_read_b64_tr_b8 v\[(\d+):\d+\], \$(\d+) offset:(\d+)", l)
        if m:
            off.setdefault(int(m.group(1)), []).append((m.group(2), int(m.group(3))))
    for v in sorted(off):
        print(f"   v{v}: {off[v]}")
    print("  tail:", [l for l in lines if l.startswith("s_waitcnt")])


show(BAD, "BAD rect 384x192 p4=(1,128,3)")
show(GOOD, "GOOD square")

if "--fix" not in sys.argv:
    sys.exit(0)

# ---- candidate fixes, judged by SNR on a single-workgroup tile ----
ONE = torch.tensor([1.0], device="cuda", dtype=torch.float32)
MP, NP, KP = 384, 192, 1536


def patch(fn):
    def _a(geom, cbsz, blgp):
        G._TN4_ASM_CACHE.pop((geom, cbsz, blgp), None)
        a, c, t = _orig_asm(geom, cbsz, blgp)
        out = (fn(a), c, t)
        G._TN4_ASM_CACHE[(geom, cbsz, blgp)] = out
        return out

    G._dense_tn_wave4_asm = _a


def f_ident(a):
    return a


def f_lgkm0(a):
    return re.sub(r"lgkmcnt\(\d+\)", "lgkmcnt(0)", a)


def f_vm0(a):
    return re.sub(r"s_waitcnt vmcnt\(\d+\) lgkmcnt", "s_waitcnt vmcnt(0) lgkmcnt", a)


def f_both(a):
    return f_lgkm0(f_vm0(a))


def f_g2s_last(a):
    """Move every m0/buffer_load pair to the end of its phase block."""
    out, pend = [], []
    lines = a.split("\n")
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("s_add_u32 m0") and i + 1 < len(lines) and lines[i + 1].startswith("buffer_load"):
            pend += [l, lines[i + 1]]
            i += 2
            continue
        if l.startswith("s_waitcnt vmcnt"):
            out += pend
            pend = []
        out.append(l)
        i += 1
    return "\n".join(out + pend)


def f_g2s_last_vm0(a):
    return f_vm0(f_g2s_last(a))


def snr(out, ref):
    out, ref = out.float(), ref.float()
    n = (out - ref).pow(2).mean()
    return 99.0 if n == 0 else float(10 * torch.log10(ref.pow(2).mean() / n))


def fp8(shape):
    ref = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    fmax = torch.finfo(FP8).max
    s = (ref.abs().amax().clamp(min=1e-8) / fmax).to(torch.float32)
    return (ref.float() / s).clamp(-fmax, fmax).to(FP8), s.reshape(1), ref


a, sa, ar = fp8((KP, MP))
b, sb, br = fp8((KP, NP))
ref = (ar.t() @ br).to(torch.bfloat16)

for nm, fn in [
    ("ident", f_ident),
    ("lgkm0", f_lgkm0),
    ("vmcnt0", f_vm0),
    ("both", f_both),
    ("g2s_last", f_g2s_last),
    ("g2s_last+vm0", f_g2s_last_vm0),
]:
    patch(fn)
    G._TN_WAVE4_CACHE.clear()
    G._TN4_WS_CACHE.clear()
    G._tn4_geom = lambda M, N, k, n: BAD
    try:
        o = gemm_fp8_tensorwise_flydsl_kernel(a, sa, b, sb, trans_a=True, trans_b=False, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print(f"FIX {nm:14s} SNR={snr(o, ref):6.2f}", flush=True)
    except Exception as exc:
        print(f"FIX {nm:14s} EXC {str(exc)[:150]}", flush=True)
