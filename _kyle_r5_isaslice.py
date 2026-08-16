# SPDX-License-Identifier: Apache-2.0
"""Print an instruction slice of one kernel out of a device ISA dump.

Usage: python _kyle_r5_isaslice.py <dump.s> <name-substring> <first> <last>
"""
import sys

path, key = sys.argv[1], sys.argv[2]
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = int(sys.argv[4]) if len(sys.argv) > 4 else 120

lines = open(path, errors="ignore").read().splitlines()
starts = [i for i, l in enumerate(lines) if key in l and l.split(";")[0].rstrip().endswith(":")]
if not starts:
    sys.exit(f"{key} not found")
st = starts[0]
en = st
while "s_endpgm" not in lines[en]:
    en += 1
body = [
    l.strip()
    for l in lines[st : en + 1]
    if l.strip() and not l.strip().startswith(";") and not l.strip().startswith(".")
]
print(f"=== {key}: {len(body)} instructions ===")
for i, l in enumerate(body):
    if lo <= i <= hi:
        print(f"{i:4d} {l}")
