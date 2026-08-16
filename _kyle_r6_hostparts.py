"""Price the host-side pieces the quant op pays before its first kernel can start."""
import time

import torch

import primus_turbo.pytorch  # noqa: F401

F8 = torch.float8_e4m3fn
dev = "cuda"
HIN, G, M = 2880, 32, 131072
KP = 2944


def bench(fn, n=400):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


b = torch.randn(G, HIN, HIN, dtype=torch.bfloat16, device=dev)
buf = torch.empty(2051, dtype=torch.float, device=dev)

print(f"empty(())            {bench(lambda: torch.empty((), dtype=torch.float, device=dev)):7.3f}us")
print(f"empty(2048)          {bench(lambda: torch.empty(2048, dtype=torch.float, device=dev)):7.3f}us")
print(f"empty(2051)          {bench(lambda: torch.empty(2051, dtype=torch.float, device=dev)):7.3f}us")
print(f"4x empty(())         {bench(lambda: [torch.empty((), dtype=torch.float, device=dev) for _ in range(4)]):7.3f}us")
print(f"empty(out q_b)       {bench(lambda: torch.empty((G, HIN, KP), dtype=F8, device=dev)):7.3f}us")
print(f"empty(out q_a)       {bench(lambda: torch.empty((M, KP), dtype=F8, device=dev)):7.3f}us")
print(f"as_strided view      {bench(lambda: buf.as_strided((), (), 2050)):7.3f}us")
print(f"narrow+view          {bench(lambda: buf.narrow(0, 2050, 1).view(())):7.3f}us")
print(f"b.contiguous()       {bench(lambda: b.contiguous()):7.3f}us")
print(f"full quant op host   {bench(lambda: torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(b, F8, None, 128), n=60):7.3f}us")
