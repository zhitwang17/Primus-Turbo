# SPDX-License-Identifier: Apache-2.0
"""Print the prologue and the hot loop of one kernel out of a device ISA dump.

Usage: python _kyle_r4_isa.py <dump.s> <name-substring> [head] [tail]
"""
import sys

path = sys.argv[1]
key = sys.argv[2]
head = int(sys.argv[3]) if len(sys.argv) > 3 else 60
tail = int(sys.argv[4]) if len(sys.argv) > 4 else 0

lines = open(path, errors="ignore").read().splitlines()
start = None
for i, ln in enumerate(lines):
    if key in ln and ln.rstrip().endswith(":"):
        start = i
        break
if start is None:
    for i, ln in enumerate(lines):
        if key in ln:
            start = i
            break
if start is None:
    sys.exit(f"{key} not found in {path}")

end = start
while end + 1 < len(lines) and "s_endpgm" not in lines[end]:
    end += 1

body = [ln for ln in lines[start : end + 1] if ln.strip() and not ln.strip().startswith(";")]
print(f"=== {key}: {len(body)} instruction lines ===")
for ln in body[:head]:
    print(ln)
if tail and len(body) > head:
    print(f"...  ({len(body) - head - tail} lines skipped)  ...")
    for ln in body[-tail:]:
        print(ln)
