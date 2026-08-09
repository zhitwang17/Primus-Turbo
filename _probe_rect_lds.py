"""Recover the LDS pool base addresses the compiler actually materialised."""
import glob
import os
import re
import sys

import torch

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as G
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel

G._dense_tn_split = lambda tiles, k_iters, ncu: None
GEOMS = {
    "BAD": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 3))),
    "GOOD": (256, 256, ((0, 128, 2), (0, 128, 2), (1, 128, 3), (1, 128, 3))),
}
tag = sys.argv[1]
geom = GEOMS[tag]
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = f"/tmp/isa_{tag}"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.system(f"rm -rf /tmp/isa_{tag}")
G._tn4_geom = lambda M, N, k, n: geom
ONE = torch.tensor([1.0], device="cuda", dtype=torch.float32)
x = torch.ones((1536, geom[0]), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
y = torch.ones((1536, geom[1]), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
gemm_fp8_tensorwise_flydsl_kernel(x, ONE, y, ONE, trans_a=True, trans_b=False, out_dtype=torch.bfloat16)
torch.cuda.synchronize()

expect = []
off = 0
for i, (side, w, nb) in enumerate(geom[2]):
    for b in range(nb):
        expect.append((f"p{i}b{b}", off))
        off += w * 128
print(f"{tag} expected field offsets: {expect}  total={off}")

f = f"/tmp/isa_{tag}/kernel_dense_tn_wave4_0/21_final_isa.s"
txt = open(f).read()
lines = txt.split("\n")
# defs of every SGPR that feeds an m0 add
bases = sorted({m.group(1) for m in re.finditer(r"s_add_u32 m0, (s\d+),", txt)})
print("m0 base sgprs:", bases)
val = {}
for i, l in enumerate(lines):
    m = re.search(r"v_readfirstlane_b32 (s\d+), (v\d+)", l)
    if m and m.group(1) in bases:
        src = m.group(2)
        for j in range(i - 1, max(0, i - 60), -1):
            d = re.search(rf"{src}, (0x[0-9a-f]+|\d+), (v\d+)\s*$", lines[j])
            if re.search(rf"^\s*v_\w+ {src},", lines[j]):
                val[m.group(1)] = lines[j].strip()
                break
for b in bases:
    print(f"  {b}: {val.get(b, '?')}")
print("\n-- all v_add/v_mov with large hex constants --")
seen = set()
for l in lines:
    m = re.search(r"v_(add_u32|mov_b32|lshl_add_u32|or_b32)\w*\s+(v\d+), (0x[0-9a-f]+)", l)
    if m and int(m.group(3), 16) >= 4096:
        s = l.strip()
        if s not in seen:
            seen.add(s)
            print("  " + s)
