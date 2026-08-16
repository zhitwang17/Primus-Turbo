# SPDX-License-Identifier: Apache-2.0
"""Group a rocprofv3 kernel trace by (kernel, grid) so the three quant shapes split apart.

Also reports the stream gap between consecutive dispatches, which is what a
launch-count change is supposed to move.
"""
import csv
import glob
import statistics
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kt3"
paths = glob.glob(f"{root}/**/*kernel_trace.csv", recursive=True)
if not paths:
    sys.exit(f"no kernel_trace.csv under {root}")

rows = []
for p in paths:
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)


def col(r, *names):
    for n in names:
        if n in r:
            return r[n]
    raise KeyError(names)


recs = []
for r in rows:
    name = col(r, "Kernel_Name", "Name").split("(")[0]
    grid = col(r, "Grid_Size", "Workgroup_Size_X")
    beg = int(col(r, "Start_Timestamp", "Begin_Timestamp"))
    end = int(col(r, "End_Timestamp"))
    recs.append((beg, end, name, grid))
recs.sort()

groups = {}
for i, (beg, end, name, grid) in enumerate(recs):
    key = (name[-70:], grid)
    gap = (beg - recs[i - 1][1]) / 1000.0 if i else 0.0
    groups.setdefault(key, []).append(((end - beg) / 1000.0, gap))

print(f"{'kernel':<72}{'grid':>10}{'n':>4}{'avg_us':>10}{'med_us':>10}{'gap_us':>9}")
tot = 0.0
for (name, grid), vals in sorted(groups.items(), key=lambda kv: -sum(v[0] for v in kv[1])):
    dur = [v[0] for v in vals]
    gaps = [v[1] for v in vals[1:]] or [0.0]
    tot += sum(dur)
    print(
        f"{name:<72}{grid:>10}{len(dur):>4}{statistics.mean(dur):>10.2f}"
        f"{statistics.median(dur):>10.2f}{statistics.median(gaps):>9.2f}"
    )
print(f"total kernel time {tot:.1f} us over {len(recs)} dispatches")
