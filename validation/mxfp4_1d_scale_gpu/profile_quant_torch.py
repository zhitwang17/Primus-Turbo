#!/usr/bin/env python3
"""Fallback profiler for environments where rocprofv3 cannot finalize."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_validation import DEFAULT_EXTENSION, DEFAULT_REPO, MODES, _bootstrap, _load_apis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--shape", default="32,5760,2880")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    args = parser.parse_args()

    torch = _bootstrap(args.repo, args.extension)
    apis = _load_apis()
    shape = tuple(int(value) for value in args.shape.split(","))
    torch.manual_seed(503)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def run_once():
        return apis["quantize_weight"](
            weight,
            apis["float4"],
            apis["ScalingGranularity"].MX_BLOCKWISE,
            block_size=32,
            weight_quant_mode=args.mode,
            use_preshuffle=False,
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(args.iterations):
            run_once()
        torch.cuda.synchronize()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / f"quant_{args.mode}_torch_profiler.txt"
    trace_path = args.output_dir / f"quant_{args.mode}_torch_trace.json"
    table_path.write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40) + "\n")
    profiler.export_chrome_trace(str(trace_path))
    print(table_path.read_text())
    print(f"Wrote {table_path}")
    print(f"Wrote {trace_path}")


if __name__ == "__main__":
    main()
