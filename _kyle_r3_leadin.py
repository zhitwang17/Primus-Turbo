"""Round-3 probe: how much of the event-measured op time is host lead-in?

`_time_once` in the scored bench records e0, then the host builds the op
(allocations + 3 launches) while the GPU is idle.  Batching N calls under one
event pair lets the host run ahead, so t1 - tN isolates that lead-in.
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
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
        x.contiguous(), F8, None, 128)


def time_once(fn):
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1)


torch.manual_seed(1234)
a = torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device=dev)
b = torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device=dev)

# floor: a minimal real dispatch under the same event pair
z = torch.zeros(1, device=dev)
for _ in range(20):
    time_once(lambda: z.zero_())
print("floor zero_():", round(statistics.median([time_once(lambda: z.zero_()) for _ in range(40)]), 4), "ms")


def batched(x, n):
    def f():
        for _ in range(n):
            hip_quant(x)
    return f


for name, x in (("q_a", a), ("q_b", b)):
    for n in (1, 1, 4, 4):
        for _ in range(6):
            batched(x, n)()
        torch.cuda.synchronize()
        ts = [time_once(batched(x, n)) / n for _ in range(20)]
        print(f"{name} n={n}: per_op_med={statistics.median(ts):.4f}ms "
              f"min={min(ts):.4f}ms", flush=True)
