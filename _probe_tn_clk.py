"""wall / sclk / power for the two binding TN shapes, palindrome-ordered fly/sm/sm/fly."""
import re
import statistics
import subprocess
import sys
import threading
import time

import torch

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.pytorch.core.low_precision import float8_e4m3

FP8 = float8_e4m3
DRIVE_S = 8.0
SHAPES = [("s2", 3072, 12288, 16384), ("s3", 12288, 3072, 16384), ("s1", 3072, 3072, 16384)]


def smi():
    out = subprocess.run(
        ["rocm-smi", "-d", "1", "--showgpuclocks", "--showpower", "--csv"],
        capture_output=True,
        text=True,
    ).stdout
    clk = re.search(r"(\d+)\s*Mhz", out) or re.search(r"sclk.*?(\d+)", out)
    pw = re.search(r"(\d+\.\d+)", out.split("\n")[-2] if len(out.split("\n")) > 2 else "")
    mhz = int(clk.group(1)) if clk else 0
    watt = 0.0
    for tok in out.replace(",", " ").split():
        try:
            v = float(tok)
        except ValueError:
            continue
        if 200.0 < v < 1500.0:
            watt = v
    return mhz, watt


def drive(fn, secs):
    samples = []
    stop = threading.Event()

    def sampler():
        time.sleep(1.5)
        while not stop.is_set():
            samples.append(smi())
            time.sleep(0.4)

    t = threading.Thread(target=sampler)
    t.start()
    t0 = time.time()
    n = 0
    while time.time() - t0 < secs:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        n += 20
    stop.set()
    t.join()
    dt = (time.time() - t0) / n * 1e3
    if not samples:
        return dt, 0, 0.0
    return dt, statistics.median(s[0] for s in samples), statistics.median(s[1] for s in samples)


def fp8(shape):
    ref = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    fmax = torch.finfo(FP8).max
    s = (ref.abs().amax().clamp(min=1e-8) / fmax).to(torch.float32)
    return (ref.float() / s).clamp(-fmax, fmax).to(FP8), s.reshape(1)


for name, Mp, Np, Kp in SHAPES:
    a, sa = fp8((Kp, Mp))
    b, sb = fp8((Kp, Np))
    sm_a, sm_b = a.t().contiguous(), b.t().contiguous().t()

    def fly():
        return gemm_fp8_tensorwise_flydsl_kernel(
            a, sa, b, sb, trans_a=True, trans_b=False, out_dtype=torch.bfloat16
        )

    def sm():
        return torch._scaled_mm(sm_a, sm_b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

    fly(), sm()
    torch.cuda.synchronize()
    r = {}
    for tag, fn in (("fly", fly), ("sm", sm), ("sm", sm), ("fly", fly)):
        d, mhz, w = drive(fn, DRIVE_S)
        r.setdefault(tag, []).append((d, mhz, w))
    for tag in ("fly", "sm"):
        v = r[tag]
        print(
            f"{name} {tag:3s} ms={statistics.mean(x[0] for x in v):.4f} "
            f"sclk={statistics.mean(x[1] for x in v):.0f}MHz "
            f"power={statistics.mean(x[2] for x in v):.0f}W",
            flush=True,
        )
