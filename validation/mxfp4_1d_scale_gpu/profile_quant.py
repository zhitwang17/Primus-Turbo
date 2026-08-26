#!/usr/bin/env python3
"""Small deterministic target for rocprofv3 kernel/launch tracing."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_validation import DEFAULT_EXTENSION, DEFAULT_REPO, MODES, _bootstrap, _load_apis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--shape", default="32,5760,2880")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
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
    for _ in range(args.iterations):
        run_once()
    torch.cuda.synchronize()
    print(f"mode={args.mode} shape={shape} iterations={args.iterations}")


if __name__ == "__main__":
    main()
