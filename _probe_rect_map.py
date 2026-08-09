"""Map every (pool, buffer) of a TN4 geometry to the LDS address the compiler gave it."""
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
    "R3D": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 64, 3), (1, 64, 3))),
    "G9": (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 64, 2), (1, 64, 3))),
}
tag = sys.argv[1]
geom = GEOMS[tag]
pools = G._tn4_pools(geom)

from flydsl.expr.struct import _storage_layout  # noqa: E402

os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = f"/tmp/map_{tag}"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.system(f"rm -rf /tmp/map_{tag}")
G._tn4_geom = lambda M, N, k, n: geom
ONE = torch.tensor([1.0], device="cuda", dtype=torch.float32)
x = torch.ones((1536, geom[0]), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
y = torch.ones((1536, geom[1]), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
gemm_fp8_tensorwise_flydsl_kernel(x, ONE, y, ONE, trans_a=True, trans_b=False, out_dtype=torch.bfloat16)
torch.cuda.synchronize()

# expected issue order of (pool, buf) per phase, from the emitter
ap = [i for i, p in enumerate(pools) if p.side == 0]
bp = [i for i, p in enumerate(pools) if p.side == 1]
order = [p for _ in range(pools[0].steps) for p in ap] + [p for p in bp for _ in range(pools[p].steps)]

txt = open(f"/tmp/map_{tag}/kernel_dense_tn_wave4_0/21_final_isa.s").read()
lines = txt.split("\n")
sg = {}
for i, l in enumerate(lines):
    m = re.search(r"v_readfirstlane_b32 (s\d+), (v\d+)", l)
    if not m:
        continue
    src = m.group(2)
    for j in range(i - 1, max(0, i - 200), -1):
        d = re.match(rf"\s*v_(or|add_u32|lshlrev)\w*(_e32)? {src}, (0x[0-9a-f]+|\d+), v\d+\s*$", lines[j])
        if d:
            c = d.group(3)
            sg[m.group(1)] = 0 if d.group(1) == "lshlrev" else int(c, 16) if c.startswith("0x") else int(c)
            break

m0seq = [m.group(1) for m in re.finditer(r"s_add_u32 m0, (s\d+),", txt)]
print(f"{tag}: {len(m0seq)} m0 sites; issue order per phase = {order}")
print("declared layout:", _storage_layout(None) if False else "")
addr = {}
for ph in range(6):
    seg = m0seq[ph * len(order) : (ph + 1) * len(order)]
    if len(seg) < len(order):
        break
    for pos, p in enumerate(order):
        b = ph % pools[p].nbuf
        a = sg.get(seg[pos])
        if a is None:
            continue
        prev = addr.setdefault((p, b), a)
        if prev != a:
            print(f"  !! (p{p},b{b}) inconsistent {prev} vs {a}")
for p in range(len(pools)):
    row = [addr.get((p, b)) for b in range(pools[p].nbuf)]
    ok = all(r is not None for r in row) and all(
        row[b] - row[0] == b * pools[p].buf for b in range(len(row))
    )
    print(f"  pool{p} w={pools[p].width} nbuf={pools[p].nbuf} addrs={row} contiguous={ok}")
