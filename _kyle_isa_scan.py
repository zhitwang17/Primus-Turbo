"""Scan a device .s dump for the quant pad kernels and report the ISA signature."""
import re
import sys

src = open(sys.argv[1]).read()
blocks = re.split(r"\n(?=[A-Za-z_][A-Za-z0-9_]*:\s*;+\s*@)", src)
pats = {
    "cvt_pk_fp8": r"v_cvt_pk_fp8_f32",
    "cvt_pk_bf8": r"v_cvt_pk_bf8_f32",
    "cvt_scalar": r"v_cvt_(?!pk)[a-z0-9_]*fp8",
    "ld_x4_nt": r"global_load_dwordx4[^\n]*nt",
    "ld_x4": r"global_load_dwordx4",
    "ld_x2": r"global_load_dwordx2",
    "st_x2_nt": r"global_store_dwordx2[^\n]*nt",
    "st_x2": r"global_store_dwordx2",
    "st_x1": r"global_store_dword ",
    "mul_hi": r"_mul_hi",
    "rcp": r"v_rcp",
}
for b in blocks:
    m = re.match(r"([A-Za-z0-9_]+):", b)
    if not m:
        continue
    name = m.group(1)
    if "pad_row" not in name and "pad_kernel" not in name:
        continue
    counts = {k: len(re.findall(p, b)) for k, p in pats.items()}
    counts = {k: v for k, v in counts.items() if v}
    vg = re.search(r"vgpr_count:\s*(\d+)", b)
    sg = re.search(r"sgpr_count:\s*(\d+)", b)
    print(name[:100])
    print("   ", counts, "vgpr", vg.group(1) if vg else "?", "sgpr", sg.group(1) if sg else "?")
