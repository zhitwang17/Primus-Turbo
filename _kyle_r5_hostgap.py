"""Split the scored op time into host prologue vs GPU work.

Times N back-to-back calls of the same quant op inside ONE event window.  The
host prologue of call k+1 overlaps the GPU work of call k, so
    T(N) = host_prologue + N * gpu_per_op
and a linear fit over N gives both terms.
"""
import statistics

import torch

import primus_turbo.pytorch  # noqa: F401

dev = "cuda"
F8 = torch.float8_e4m3fn
HIN = 2880
G = 32
M_TOTAL = 131072


def hip_quant(x):
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x.contiguous(), F8, None, 128)


def time_n(x, n, reps=30):
    def run():
        out = []
        for _ in range(n):
            out.append(hip_quant(x))
        return out

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(True)
        e1 = torch.cuda.Event(True)
        e0.record()
        keep = run()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)  # us
        del keep
    return statistics.median(ts)


torch.manual_seed(1234)
tensors = {
    "q_a": torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device=dev),
    "q_b": torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device=dev),
}

for name, x in tensors.items():
    t = {n: time_n(x, n) for n in (1, 2, 3)}
    gpu = (t[3] - t[1]) / 2.0
    host = t[1] - gpu
    print(
        f"{name}: T1={t[1]:8.2f}us T2={t[2]:8.2f}us T3={t[3]:8.2f}us "
        f"-> gpu/op={gpu:7.2f}us host_prologue={host:6.2f}us ({100.0*host/t[1]:.2f}% of scored op)",
        flush=True,
    )
