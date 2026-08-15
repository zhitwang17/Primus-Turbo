# SPDX-License-Identifier: Apache-2.0
"""Kernel-trace driver for the gpt-oss down-projection tensorwise quant ops.

Runs the three scored shapes a few times so rocprofv3 --kernel-trace can show the
per-kernel split (amax pass / finalise / quant+pad) and the stream gaps between
them. Deliberately few iterations: rocprofv3 serialises dispatches.
"""
import torch
import primus_turbo.pytorch as turbo  # noqa: F401  (loads the .so)

G, M_TOTAL, HIN = 32, 131072, 2880
F8 = torch.float8_e4m3fn
dev = "cuda"


def q(x):
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, F8, None, 128)


a = torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device=dev)
b = torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device=dev)
go = torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device=dev)
for _ in range(2):
    q(a), q(b), q(go)
torch.cuda.synchronize()
for _ in range(4):
    q(a), q(b), q(go)
torch.cuda.synchronize()
print("done")
