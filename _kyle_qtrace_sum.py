# SPDX-License-Identifier: Apache-2.0
"""Summarise a rocprofv3 kernel_trace.csv per (kernel, grid) and report the
inter-kernel gaps on the stream.

Grouping by grid size as well as by name is mandatory here: the three scored
quant ops share the same kernels at two different shapes, and averaging over
both invents time that neither shape spends (see campaign memory round 4).

  python3 _kyle_qtrace_sum.py <kernel_trace.csv>
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append(r)


def col(r, *names):
    for n in names:
        for k in r:
            if k.strip().lower() == n:
                return r[k]
    raise KeyError(names)


recs = []
for r in rows:
    name = col(r, "kernel_name", "name").strip()
    g = int(float(col(r, "grid_size_x", "grid_size", "workgroup_size_x")))
    vgpr = col(r, "vgpr_count")
    scr = col(r, "scratch_size")
    t0 = int(float(col(r, "start_timestamp", "start")))
    t1 = int(float(col(r, "end_timestamp", "end")))
    recs.append((t0, t1, name, g, vgpr, scr))
recs.sort()

agg = defaultdict(list)
for t0, t1, name, g, vgpr, scr in recs:
    agg[(name[:48], g, vgpr, scr)].append((t1 - t0) / 1000.0)

print(f"{'kernel':50s} {'grid':>9s} {'vgpr':>5s} {'scr':>4s} {'n':>4s} {'med_us':>9s} {'tot_us':>9s}")
for (name, g, vgpr, scr), v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
    v.sort()
    print(f"{name:50s} {g:9d} {vgpr:>5s} {scr:>4s} {len(v):4d} "
          f"{v[len(v)//2]:9.2f} {sum(v):9.1f}")

gaps = defaultdict(list)
for i in range(1, len(recs)):
    prev, cur = recs[i - 1], recs[i]
    gaps[(prev[2][:24], cur[2][:24])].append((cur[0] - prev[1]) / 1000.0)
print("\ninter-kernel gaps (us, median):")
for k, v in sorted(gaps.items(), key=lambda kv: -len(kv[1])):
    v.sort()
    print(f"  {k[0]:26s} -> {k[1]:26s} n={len(v):3d} med={v[len(v)//2]:7.2f}")
