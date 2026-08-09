"""K-block / column fingerprint of one TN macro-tile geometry on a single-workgroup tile.

A = ones, B = indicator of one K-block -> every output element must come out at exactly
BLOCK_K. Anything else names the K-block that got dropped or double-counted, and the column
range it happened in names the LDS pool.
"""
import torch

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as G
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.pytorch.core.low_precision import float8_e4m3

FP8 = float8_e4m3
G._dense_tn_split = lambda tiles, k_iters, ncu: None

VARIANTS = {
    "R2_all2": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 2))),
    "R2_p4deep": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 3))),
    "SQ_bothdeep": (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 3), (1, 128, 3))),
}
ONE = torch.tensor([1.0], device="cuda", dtype=torch.float32)


def gemm(a, b):
    return gemm_fp8_tensorwise_flydsl_kernel(
        a, ONE, b, ONE, trans_a=True, trans_b=False, out_dtype=torch.bfloat16
    )


for name, geom in VARIANTS.items():
    BM, BN = geom[0], geom[1]
    KP, NKB = 1536, 12
    G._TN_WAVE4_CACHE.clear()
    G._TN4_WS_CACHE.clear()
    G._tn4_geom = lambda M, N, k, n, _g=geom: _g
    a = torch.ones((KP, BM), device="cuda", dtype=torch.bfloat16).to(FP8)
    print(f"\n== {name}  tile {BM}x{BN} K={KP}")
    cols = [c for c in range(0, BN, 32)]
    for kb in range(NKB):
        b = torch.zeros((KP, BN), device="cuda", dtype=torch.bfloat16)
        b[kb * 128 : (kb + 1) * 128] = 1.0
        out = gemm(a, b.to(FP8))
        torch.cuda.synchronize()
        r0 = [int(out[0, c].item()) for c in cols]
        r1 = [int(out[BM - 1, c].item()) for c in cols]
        print(f"  kb={kb:2d} row0 {r0}  row{BM-1} {r1}", flush=True)
