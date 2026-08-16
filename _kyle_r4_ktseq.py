# SPDX-License-Identifier: Apache-2.0
"""Print a rocprofv3 kernel trace as an ordered dispatch sequence.

The three quant ops all launch the same three kernels, and the amax pass uses the
same grid for every shape, so grouping by (kernel, grid) merges them. The driver
issues the ops in a fixed order, so the raw sequence separates them.
"""
import csv
import glob
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kt4"
rows = []
for p in glob.glob(f"{root}/**/*kernel_trace.csv", recursive=True):
    rows += list(csv.DictReader(open(p, newline="")))
if not rows:
    sys.exit(f"no kernel_trace.csv under {root}")


def col(r, *names):
    for n in names:
        if n in r:
            return r[n]
    return "?"


if rows:
    print("columns:", ",".join(rows[0].keys()))


recs = sorted(
    (
        int(col(r, "Start_Timestamp", "Begin_Timestamp")),
        int(col(r, "End_Timestamp")),
        col(r, "Kernel_Name", "Name").split("(")[0],
        col(r, "Grid_Size_X"),
        col(r, "Workgroup_Size_X"),
        col(r, "VGPR_Count"),
        col(r, "Accum_VGPR_Count"),
        col(r, "SGPR_Count"),
        col(r, "Scratch_Size"),
    )
    for r in rows
)
hdr = f"{'#':>3} {'kernel':<40}{'gridX':>10}{'wg':>5}{'vgpr':>6}{'agpr':>5}{'sgpr':>5}{'scr':>5}{'us':>9}{'gap':>7}"
print(hdr)
prev = None
tot = 0.0
for i, (b, e, n, g, wg, v, av, s, scr) in enumerate(recs):
    gap = (b - prev) / 1000.0 if prev else 0.0
    prev = e
    tot += (e - b) / 1000.0
    print(
        f"{i:>3} {n[-40:]:<40}{g:>10}{wg:>5}{v:>6}{av:>5}{s:>5}{scr:>5}"
        f"{(e - b) / 1000.0:>9.2f}{gap:>7.2f}"
    )
print(f"total {tot:.1f} us over {len(recs)} dispatches")
