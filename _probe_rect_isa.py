"""Emitted-asm and final-ISA view of the broken deep-pool geometry."""
import glob
import os
import re

import torch

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as G
from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel

G._dense_tn_split = lambda tiles, k_iters, ncu: None
BAD = (384, 192, ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 3)))

a = G._dense_tn_wave4_asm(BAD, 0, 0)[0]
blocks = a.split("s_barrier")
print(f"blocks={len(blocks)}")
for bi, b in enumerate(blocks[1:8], start=1):
    m0 = [l for l in b.split("\n") if l.startswith("s_add_u32 m0")]
    ds = sorted(
        {
            (int(m.group(1)), int(m.group(2)))
            for m in re.finditer(r"ds_read_b64_tr_b8 v\[(\d+):\d+\], \$\d+ offset:(\d+)", b)
            if int(m.group(1)) >= 120
        }
    )
    print(f"-- block {bi}: m0 bases {[l.split(',')[1].strip() for l in m0]}")
    print(f"   pool4 frag offsets {ds}")

os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = "/tmp/badisa"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.system("rm -rf /tmp/badisa")
G._TN_WAVE4_CACHE.clear()
G._tn4_geom = lambda M, N, k, n: BAD
ONE = torch.tensor([1.0], device="cuda", dtype=torch.float32)
x = torch.ones((1536, 384), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
y = torch.ones((1536, 192), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
gemm_fp8_tensorwise_flydsl_kernel(x, ONE, y, ONE, trans_a=True, trans_b=False, out_dtype=torch.bfloat16)
torch.cuda.synchronize()

for f in sorted(glob.glob("/tmp/badisa/**/*.s", recursive=True)):
    txt = open(f).read()
    if "ds_read_b64_tr_b8" not in txt:
        continue
    print(f"\n=== {f}")
    for k in (".vgpr_count", ".agpr_count", "spill_count", "group_segment_fixed_size", ".sgpr_count"):
        for l in txt.split("\n"):
            if k in l:
                print("  " + l.strip())
                break
    body = txt.split("\n")
    m0s = [l.strip() for l in body if re.search(r"s_add_u32 m0", l)]
    print(f"  distinct m0 forms ({len(m0s)}):")
    seen = []
    for l in m0s:
        if l not in seen:
            seen.append(l)
    for l in seen[:40]:
        print("    " + l)
    lit = sorted({int(m.group(1)) for m in re.finditer(r"s_mov_b32 s\d+, 0x([0-9a-f]+)", txt) or []} if False else set())
    # LDS base constants materialised for the pools
    cst = sorted({int(m.group(1), 16) for m in re.finditer(r"v_mov_b32_e32 v\d+, 0x([0-9a-f]+)", txt)})
    print("  hex v_mov constants:", [c for c in cst if c % 1024 == 0][:30])
    cst2 = sorted({int(m.group(1), 16) for m in re.finditer(r"s_mov_b32 s\d+, 0x([0-9a-f]+)", txt)})
    print("  hex s_mov constants:", [c for c in cst2 if c % 1024 == 0][:30])
    break
