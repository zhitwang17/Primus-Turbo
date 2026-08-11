"""Drift-immune race of wgrad tile-blocking configs on the DEPLOY arm (analysis only, not scored).

usage: python _probe_wgcfg.py <proj> <cfg;cfg;...> [reps]

All candidate configs are timed in ONE steady-state process, round-robin in palindrome order
(A B C C B A) so every arm sees the same mean position in the drift curve -- the sequential
sweep in _probe_cfg_sweep.py has a ~3% position drift that hides 1%-level differences.
The reference arm (G=4) runs the shipped large-M config and is the drift monitor.
"""
import statistics
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm import mxfp4_grouped_kernel as mk

DEV = "cuda"
KP = 2944
M_TOTAL = 131072
SHAPES = {"gate_up": 5760, "down": 2944}
G_DEP, G_REF = 32, 4
BASE = mk._GMXFP4_WGRAD_CFG


def f4(*s):
    return torch.randint(0, 256, s, dtype=torch.uint8, device=DEV)


def sc(*s):
    return torch.full(s, 127, dtype=torch.uint8, device=DEV)


def offs(E):
    return torch.arange(0, E + 1, dtype=torch.int64, device=DEV) * (M_TOTAL // E)


def main():
    proj = sys.argv[1]
    cfgs = [tuple(int(x) for x in c.split(",")) for c in sys.argv[2].split(";")]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    N = SHAPES[proj]
    bf = torch.bfloat16
    l4, ls4 = f4(KP, M_TOTAL // 2), sc(KP, M_TOTAL // 32)
    r4, rs4 = f4(N, M_TOTAL // 2), sc(N, M_TOTAL // 32)
    o_dep, o_ref = offs(G_DEP), offs(G_REF)

    def call(cfg, o, g):
        mk._GMXFP4_WGRAD_CFG = cfg
        mk._select_gmxfp4_wgrad_cfg = lambda _mt, _g, _c=cfg: _c
        return mk.grouped_gemm_mxfp4_variable_k_flydsl_kernel(
            l4, ls4, r4, rs4, o, KP, N, g, out_dtype=bf
        )

    arms = {}
    for cfg in cfgs:
        arms[f"dep{cfg}"] = lambda cfg=cfg: call(cfg, o_dep, G_DEP)
    arms["ref"] = lambda: call(BASE, o_ref, G_REF)

    for _ in range(6):
        for f in arms.values():
            f()
    torch.cuda.synchronize()
    seq = list(arms.items())
    seq = seq + seq[::-1]
    acc = {k: [] for k in arms}
    for _ in range(reps):
        for k, f in seq:
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            f()
            e1.record()
            torch.cuda.synchronize()
            acc[k].append(e0.elapsed_time(e1))
    base = statistics.median(acc[f"dep{cfgs[0]}"])
    for k in arms:
        v = acc[k]
        print(
            f"{proj:8s} {k:18s} med {statistics.median(v):7.4f}ms  min {min(v):7.4f}  "
            f"p25 {statistics.quantiles(v, n=4)[0]:7.4f}  vs_first {base / statistics.median(v) - 1:+.2%}"
        )


if __name__ == "__main__":
    main()
