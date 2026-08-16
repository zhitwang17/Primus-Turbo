# SPDX-License-Identifier: Apache-2.0
"""Timeline view of a rocprofv3 --kernel-trace csv.

Prints every dispatch in stream order with the idle gap before it, so the
host-side prologue of each quant op (GPU idle between the previous op's last
kernel and this op's first kernel) is visible next to the kernel times.
"""
import csv
import glob
import sys

path = glob.glob(sys.argv[1] + "/**/*kernel_trace.csv", recursive=True)[0]
rows = sorted(csv.DictReader(open(path)), key=lambda r: int(r["Start_Timestamp"]))
rows = [r for r in rows if "at::native" not in r["Kernel_Name"]]

prev_end = None
for r in rows:
    s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
    gap = "" if prev_end is None else "%8.2f" % ((s - prev_end) / 1e3)
    name = r["Kernel_Name"].split("(")[0].replace("void primus_turbo::", "")[:46]
    print("gap %8s us   dur %8.2f us   %s" % (gap, (e - s) / 1e3, name))
    prev_end = e
