"""Kernel-trace driver: three scored ops in the scored bench's rhythm.

Each call is followed by a sync, exactly like the bench's _time_once, so the
gaps in the trace are the host prologue the GPU really idles through.
"""
import torch

import primus_turbo.pytorch  # noqa: F401

F8 = torch.float8_e4m3fn
HIN, G, M = 2880, 32, 131072


def hip(x):
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, F8, None, 128)


torch.manual_seed(1234)
a = torch.randn(M, HIN, dtype=torch.bfloat16, device="cuda")
b = torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device="cuda")
go = torch.randn(M, HIN, dtype=torch.bfloat16, device="cuda")
for x in (a, b, go):
    for _ in range(3):
        hip(x)
torch.cuda.synchronize()
for x in (a, b, go):
    for _ in range(3):
        y = hip(x)
        torch.cuda.synchronize()
        del y
torch.cuda.synchronize()
