"""Round-3 probe: where does the tensorwise quant op time actually go?

Splits the event-measured op time into (a) GPU kernel time, (b) inter-kernel
gaps, (c) host lead-in before the first dispatch.

  python _kyle_r3_qgap.py events
  rocprofv3 --kernel-trace --output-format csv -d /tmp/rpq -o q -- \
      python _kyle_r3_qgap.py trace
  python _kyle_r3_qgap.py parse /tmp/rpq/q_kernel_trace.csv
"""
import statistics
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "events"

if MODE == "parse":
    import csv
    from collections import defaultdict

    rows = []
    with open(sys.argv[2]) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    key_s = "Start_Timestamp"
    key_e = "End_Timestamp"
    key_n = "Kernel_Name"
    rows.sort(key=lambda r: int(r[key_s]))
    agg = defaultdict(list)
    for r in rows:
        agg[r[key_n][:78]].append(int(r[key_e]) - int(r[key_s]))
    print("=== per-kernel (ns) ===")
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print(f"{len(v):5d} calls  med={statistics.median(v)/1000.0:9.2f}us "
              f"min={min(v)/1000.0:9.2f}us  tot={sum(v)/1e6:8.3f}ms  {k}")
    print("=== consecutive timeline (last 24 dispatches) ===")
    prev_e = None
    for r in rows[-24:]:
        s, e = int(r[key_s]), int(r[key_e])
        gap = (s - prev_e) / 1000.0 if prev_e is not None else 0.0
        prev_e = e
        print(f"gap={gap:8.2f}us dur={(e-s)/1000.0:9.2f}us {r[key_n][:64]}")
    # gap histogram between quant-chain kernels only
    gaps = []
    prev_e = None
    for r in rows:
        s, e = int(r[key_s]), int(r[key_e])
        if prev_e is not None:
            gaps.append((s - prev_e) / 1000.0)
        prev_e = e
    gaps = [g for g in gaps if 0.0 <= g < 50.0]
    if gaps:
        print(f"=== gaps: n={len(gaps)} med={statistics.median(gaps):.2f}us "
              f"mean={sum(gaps)/len(gaps):.2f}us min={min(gaps):.2f}us "
              f"max={max(gaps):.2f}us")
    sys.exit(0)

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
ops = [("q_a", a), ("q_b", b)]

if MODE == "trace":
    for _, x in ops:
        for _ in range(3):
            hip_quant(x)
    torch.cuda.synchronize()
    for _, x in ops:
        for _ in range(4):
            hip_quant(x)
    torch.cuda.synchronize()
    sys.exit(0)

for name, x in ops:
    for _ in range(12):
        hip_quant(x)
    torch.cuda.synchronize()
    ev = [time_once(lambda: hip_quant(x)) for _ in range(40)]
    # host-side enqueue cost of the op (async) and total wall
    enq, wall = [], []
    for _ in range(40):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = hip_quant(x)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        del y
        enq.append((t1 - t0) * 1e3)
        wall.append((t2 - t0) * 1e3)
    print(f"{name}: event_med={statistics.median(ev):.4f}ms "
          f"host_enqueue_med={statistics.median(enq):.4f}ms "
          f"wall_med={statistics.median(wall):.4f}ms", flush=True)
