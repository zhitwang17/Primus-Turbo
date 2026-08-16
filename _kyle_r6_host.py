# SPDX-License-Identifier: Apache-2.0
"""How much of the scored op time is GPU-idle host latency?

The bench times ONE op call between two events on an otherwise idle stream, so
everything the host does before the first kernel is dispatched is charged to the
score. This probe separates that from the kernel time:

  t_solo   one call between two events, stream idle beforehand  (== the bench)
  t_amort  N calls back to back between two events, / N         (host runs ahead)
  exposed  t_solo - t_amort                                     (the host prologue)

and then prices the individual host steps the op takes.
"""
import statistics
import time

import torch

import primus_turbo.pytorch  # noqa: F401

G, M_TOTAL, HIN, KP = 32, 131072, 2880, 2944
F8 = torch.float8_e4m3fn
dev = "cuda"


def q(x):
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, F8, None, 128)


def ev(fn, n=1):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / n


def med(f, reps=30):
    return statistics.median([f() for _ in range(reps)])


tensors = [
    ("q_a", torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device=dev)),
    ("q_b", torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device=dev)),
]
for _ in range(8):
    for _, x in tensors:
        q(x)
torch.cuda.synchronize()

print("op      t_solo    t_amort   exposed")
for name, x in tensors:
    solo = med(lambda: ev(lambda: q(x)))
    amort = med(lambda: ev(lambda: q(x), 8), reps=10)
    print("%-6s %8.2f %8.2f %8.2f us" % (name, solo, amort, solo - amort))

# Price the host steps the op itself performs, on a busy stream so the timings
# are pure CPU work and not a sync in disguise.
x = tensors[1][1]
out_shape = (G, HIN, KP)
opts = torch.empty(0, dtype=F8, device=dev)


def cpu(fn, reps=200):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(t)


print()
print("host step costs (CPU wall, us)")
print("  full op call            %7.2f" % cpu(lambda: q(x)))
print("  empty(out_shape) fp8    %7.2f" % cpu(lambda: torch.empty(out_shape, dtype=F8, device=dev)))
print("  empty(2051) fp32        %7.2f" % cpu(lambda: torch.empty(2051, dtype=torch.float32, device=dev)))
print("  empty(0) fp32           %7.2f" % cpu(lambda: torch.empty(0, dtype=torch.float32, device=dev)))
print("  x.contiguous() (no-op)  %7.2f" % cpu(lambda: x.contiguous()))
print("  torch.ops noop (numel)  %7.2f" % cpu(lambda: x.numel()))
torch.cuda.synchronize()
