# SPDX-License-Identifier: Apache-2.0
"""Drive the quant ops in the BENCH regime: one call between two events with a
sync around it, so the GPU is idle when the call starts. Under a kernel trace
the idle gaps inside the op become visible (host prologue, output allocation).
"""
import torch

import primus_turbo.pytorch  # noqa: F401

G, M_TOTAL, HIN = 32, 131072, 2880
F8 = torch.float8_e4m3fn
dev = "cuda"


def q(x):
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, F8, None, 128)


a = torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device=dev)
b = torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device=dev)
for _ in range(6):
    q(a)
    q(b)
torch.cuda.synchronize()

for x in (b, b, b, a, a, a):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    q(x)
    e1.record()
    torch.cuda.synchronize()
print("done")
