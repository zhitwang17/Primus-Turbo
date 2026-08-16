# SPDX-License-Identifier: Apache-2.0
"""Price the part of a scored quant op that is NOT kernel time.

The scored bench brackets each op with `e0.record(); fn(); e1.record()` and then
synchronises, so everything the host does before the FIRST kernel reaches the GPU
is inside the measured window while the GPU sits idle.  Running the identical op
on a tiny tensor leaves only that fixed cost, and a CPU-side clock around the
same call separates the python/ATen part from the launch part.
"""
import statistics
import time

import torch

import primus_turbo.pytorch  # noqa: F401

F8 = torch.float8_e4m3fn
HIN, G, M_TOTAL = 2880, 32, 131072
dev = "cuda"


def q(x):
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, F8, None, 128)


def ev(fn, reps=40):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    return statistics.median(ts)


def cpu(fn, reps=40):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(ts)


cases = {
    "tiny  [8,2880]": torch.randn(8, HIN, dtype=torch.bfloat16, device=dev),
    "small [4096,2880]": torch.randn(4096, HIN, dtype=torch.bfloat16, device=dev),
    "q_b   [32,2880,2880]": torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device=dev),
    "q_a   [131072,2880]": torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device=dev),
}


def empty():
    pass


print(f"event-timed empty region: {ev(empty):8.2f} us  (event pair floor)")
for name, x in cases.items():
    e = ev(lambda x=x: q(x))
    c = cpu(lambda x=x: q(x))
    print(f"{name:24s} event={e:9.2f} us  cpu_enqueue={c:8.2f} us")
