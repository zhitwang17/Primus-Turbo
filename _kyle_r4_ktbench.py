# SPDX-License-Identifier: Apache-2.0
"""Replicate the scored bench's timing rhythm for a kernel trace.

_bench_gptoss_down_quant.py times each op as HIP -> sync -> torch-ref -> sync,
repeated per shape. The torch ref moves several GB (it materialises an fp32 copy),
so the cache state the HIP kernels see there is nothing like a back-to-back loop
over the HIP op alone. This driver reproduces that rhythm so the trace shows the
per-kernel times under the conditions that are actually scored.

  MODE=alt   HIP and torch-ref interleaved (the scored rhythm)
  MODE=solo  HIP op only, back-to-back  (the _kyle_qkt.py rhythm)
"""
import os

import torch

import primus_turbo.pytorch  # noqa: F401

G, M_TOTAL, HIN = 32, 131072, 2880
F8 = torch.float8_e4m3fn
QMAX = float(torch.finfo(F8).max)
KP = (HIN + 127) // 128 * 128
MODE = os.environ.get("MODE", "alt")
REPS = int(os.environ.get("REPS", "4"))


def hip_q(x):
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x.contiguous(), F8, None, 128)


def ref_q(x):
    amax = x.abs().max().float()
    scale = QMAX / torch.clamp(amax, min=1e-12)
    y = (x.float() * scale).clamp(-QMAX, QMAX).to(F8)
    pad = KP - x.shape[-1]
    if pad > 0:
        y = torch.nn.functional.pad(y, (0, pad))
    return y, (1.0 / scale)


torch.manual_seed(1234)
a = torch.randn(M_TOTAL, HIN, dtype=torch.bfloat16, device="cuda")
b = torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device="cuda")

for name, x in (("q_a", a), ("q_b", b)):
    for _ in range(2):
        hip_q(x)
        if MODE == "alt":
            ref_q(x)
    torch.cuda.synchronize()
    for _ in range(REPS):
        hip_q(x)
        torch.cuda.synchronize()
        if MODE == "alt":
            ref_q(x)
            torch.cuda.synchronize()
print("done", MODE)
