###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU training proxy for static randomized RHT in the MXFP4 MLP.

This benchmark compares the legacy fixed H16x2 transform (``rht_seed=0``)
with one or more static randomized-RHT campaign seeds.  It exercises the
public :func:`primus_turbo.pytorch.ops.mlp_fp4` API with a real forward,
backward, and optimizer update on every step.  Every campaign starts from the
same weights and optimizer state and consumes the same synthetic data stream.

The workload is deliberately small enough to run on one gfx950.  It is a
representative training proxy, not a complete GPT-OSS training or convergence
run.  Its loss trajectory is useful for detecting regressions and comparing
seeds, but it must not be presented as evidence of full-model convergence.

Example::

    HIP_VISIBLE_DEVICES=4 python benchmark/randomized_rht_e2e.py \
        --warmup 5 --steps 30 --seeds 1 42 20260918 \
        --json randomized-rht-e2e.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
from primus_turbo.pytorch.ops import mlp_fp4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed and static randomized RHT with a single-gfx950 "
            "MXFP4 MLP training proxy (not full GPT-OSS E2E training)."
        )
    )
    parser.add_argument("--m", type=int, default=512, help="Tokens/rows per training step.")
    parser.add_argument("--k", type=int, default=1024, help="Input and output hidden size.")
    parser.add_argument("--intermediate", type=int, default=1024, help="SwiGLU intermediate size I.")
    parser.add_argument(
        "--preheat",
        type=int,
        default=100,
        help="Untimed fixed-RHT steps used once to stabilize GPU clocks before all campaigns.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Untimed compile/warmup updates per campaign.")
    parser.add_argument("--steps", type=int, default=20, help="Measured training updates per campaign.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[1, 42, 20260918],
        help=(
            "Randomized-RHT uint32 seeds, separated by spaces or commas. "
            "Seed 0 is the fixed baseline and is inserted first only when omitted; "
            "specify it explicitly to rotate campaign order across repeated runs."
        ),
    )
    parser.add_argument("--device", default="cuda:0", help="Logical gfx950 device (default: cuda:0).")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--input-std", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=0.5)
    parser.add_argument("--init-seed", type=int, default=20260918)
    parser.add_argument("--data-seed", type=int, default=314159)
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="Write the full result as JSON. Use '-' to print JSON to stdout.",
    )
    args = parser.parse_args()

    if args.preheat < 0 or args.warmup < 0:
        parser.error("--preheat and --warmup must be non-negative")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if min(args.m, args.k, args.intermediate) <= 0:
        parser.error("--m, --k, and --intermediate must be positive")
    # Keep these checks in sync with dense_glu_epi_quant_supported(), the
    # dGLU counterpart, and the MXFP4 quantizer.  Catching unsupported shapes
    # here produces a useful CLI error instead of a late assertion in FlyDSL.
    if args.m % 256:
        parser.error(
            f"--m must be a multiple of 256 for the fused dense GLU/dGLU "
            f"quantization epilogues; got M={args.m}"
        )
    if args.intermediate % 64:
        parser.error(
            f"--intermediate must be a multiple of 64 for the fused dense GLU "
            f"quantization epilogue; got I={args.intermediate}"
        )
    if args.k % 32:
        parser.error(f"--k must be a multiple of the MXFP4 block size (32); got K={args.k}")

    padded_k_tiles = (args.k + 255) // 256
    if padded_k_tiles < 4:
        parser.error(
            f"--k is too small for the fused dense GLU kernel: ceil(K/256) must "
            f"be at least 4; got K={args.k}, ceil(K/256)={padded_k_tiles}"
        )
    if padded_k_tiles % 2 and (args.k + 127) // 128 != 2 * padded_k_tiles - 1:
        parser.error(
            f"--k={args.k} is not covered by the fused dense GLU K-loop: with the odd "
            f"ceil(K/256)={padded_k_tiles}, K must end in the first 128 values of its "
            "final 256-wide tile"
        )
    if args.lr <= 0 or args.init_std <= 0 or args.input_std <= 0:
        parser.error("--lr, --init-std, and --input-std must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")

    try:
        args.seeds = _normalize_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _normalize_seeds(values: Iterable[Any]) -> list[int]:
    """Parse comma/space-separated seeds and ensure one fixed campaign is present."""
    seeds = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            seed = int(item, 0)
            if not 0 <= seed <= 0xFFFFFFFF:
                raise ValueError(f"RHT seed {seed} is outside the uint32 range")
            if seed not in seeds:
                seeds.append(seed)
    return seeds if 0 in seeds else [0, *seeds]


def _check_device(device: torch.device) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is unavailable; this proxy requires one gfx950 GPU")
    if device.type != "cuda":
        raise ValueError(f"--device must name a CUDA/ROCm device, got {device}")

    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 5):
        raise RuntimeError(
            f"the fused randomized-RHT path requires gfx950, got "
            f"{props.name} (capability {props.major}.{props.minor})"
        )
    gcn_arch = getattr(props, "gcnArchName", "gfx950")
    return {
        "name": props.name or gcn_arch,
        "logical_device": str(device),
        "capability": f"{props.major}.{props.minor}",
        "gcn_arch": gcn_arch,
        "total_memory_bytes": props.total_memory,
    }


def _make_initial_weights(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the one canonical initialization shared by all campaigns."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.init_seed)
    w1 = torch.randn((2 * args.intermediate, args.k), generator=generator, dtype=torch.float32).mul_(
        args.init_std
    )
    w2 = torch.randn((args.k, args.intermediate), generator=generator, dtype=torch.float32).mul_(
        args.init_std
    )
    return w1.to(device=device, dtype=dtype), w2.to(device=device, dtype=dtype)


def _make_optimizer(args: argparse.Namespace, params: list[torch.Tensor]) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            foreach=False,
        )
    return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, foreach=False)


def _reset_after_warmup(
    params: list[torch.Tensor],
    initial: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> None:
    """Undo warmup updates while retaining compiled kernels and allocated state."""
    with torch.no_grad():
        for param, value in zip(params, initial):
            param.copy_(value)
        optimizer.zero_grad(set_to_none=True)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    value.zero_()
                elif isinstance(value, (int, float)):
                    state[key] = type(value)(0)


def _tensor_l2(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    squares = [tensor.float().square().sum() for tensor in tensors]
    return torch.stack(squares).sum().sqrt()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _one_step(
    *,
    args: argparse.Namespace,
    config: Float4QuantConfig,
    dtype: torch.dtype,
    device: torch.device,
    data_generator: torch.Generator,
    w1: torch.Tensor,
    w2: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    measure: bool,
) -> dict[str, float] | None:
    """Run one genuine FP4 MLP training step; batch creation is not timed."""
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn((args.m, args.k), device=device, dtype=dtype, generator=data_generator).mul_(
        args.input_std
    )
    x.requires_grad_(True)
    # A deterministic nonlinear autoencoding target keeps data generation cheap.
    # Detaching is important: the only gradient path is through the FP4 MLP.
    target = torch.tanh(x.detach().float().mul(args.target_scale)).to(dtype)

    start = torch.cuda.Event(enable_timing=True) if measure else None
    end = torch.cuda.Event(enable_timing=True) if measure else None
    if start is not None:
        start.record()

    out = mlp_fp4(
        x,
        w1,
        w2,
        out_dtype=dtype,
        config=config,
        activation="silu",
    )
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    optimizer.step()

    if end is not None:
        end.record()
    if not measure:
        return None

    # These reductions are enqueued after the end event, so metric collection is
    # excluded from step_ms. optimizer.step() does not clear parameter gradients.
    grad_norm = _tensor_l2(param.grad for param in (w1, w2) if param.grad is not None)
    input_grad_norm = _tensor_l2([x.grad])
    assert start is not None and end is not None
    end.synchronize()
    step_ms = start.elapsed_time(end)
    metrics = {
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "input_grad_norm": float(input_grad_norm),
        "step_ms": step_ms,
        "tokens_per_second": args.m * 1000.0 / step_ms,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite training metric: {metrics}")
    return metrics


def _prepare_campaign(
    args: argparse.Namespace,
    *,
    rht_seed: int,
    initial: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    w1 = initial[0].detach().clone().requires_grad_(True)
    w2 = initial[1].detach().clone().requires_grad_(True)
    params = [w1, w2]
    optimizer = _make_optimizer(args, params)
    # Keep gradient SR disabled: its backend-global launch counter cannot be
    # rewound between in-process campaigns, so enabling it would give each RHT
    # seed a different rounding-noise sequence and invalidate the paired A/B.
    config = Float4QuantConfig(use_gradient_sr=False, rht_seed=rht_seed)
    data_generator = torch.Generator(device=device)

    data_generator.manual_seed(args.data_seed)
    for _ in range(args.warmup):
        _one_step(
            args=args,
            config=config,
            dtype=dtype,
            device=device,
            data_generator=data_generator,
            w1=w1,
            w2=w2,
            optimizer=optimizer,
            measure=False,
        )
    torch.cuda.synchronize(device)

    # Warmup may compile multiple seed-specialized kernels and allocate optimizer
    # state, but it must not change the measured training trajectory.
    _reset_after_warmup(params, initial, optimizer)
    data_generator.manual_seed(args.data_seed)
    torch.cuda.synchronize(device)

    return {
        "rht_seed": rht_seed,
        "config": config,
        "w1": w1,
        "w2": w2,
        "optimizer": optimizer,
        "data_generator": data_generator,
        "per_step": [],
    }


def _summarize_campaign(
    args: argparse.Namespace,
    state: dict[str, Any],
    initial: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    rht_seed = state["rht_seed"]
    config = state["config"]
    w1, w2 = state["w1"], state["w2"]
    per_step = state["per_step"]

    times = [item["step_ms"] for item in per_step]
    losses = [item["loss"] for item in per_step]
    grad_norms = [item["grad_norm"] for item in per_step]
    input_grad_norms = [item["input_grad_norm"] for item in per_step]
    update_norm = float(_tensor_l2((w1 - initial[0], w2 - initial[1])).detach())
    if not math.isfinite(update_norm) or update_norm == 0.0:
        raise RuntimeError(f"invalid weight update norm for RHT seed {rht_seed}: {update_norm}")

    total_ms = sum(times)
    return {
        "rht_seed": rht_seed,
        "rht_mask": f"0x{config.rht_mask:08x}",
        "transform": "fixed_h16x2" if rht_seed == 0 else "randomized_static_h16x2",
        "loss_first": losses[0],
        "loss_final": losses[-1],
        "loss_mean": statistics.fmean(losses),
        "grad_norm_mean": statistics.fmean(grad_norms),
        "input_grad_norm_mean": statistics.fmean(input_grad_norms),
        "weight_update_norm": update_norm,
        "step_ms_mean": statistics.fmean(times),
        "step_ms_p50": _percentile(times, 0.50),
        "step_ms_p95": _percentile(times, 0.95),
        "tokens_per_second": args.m * args.steps * 1000.0 / total_ms,
        "per_step": per_step,
    }


def _run_campaigns_interleaved(
    args: argparse.Namespace,
    *,
    initial: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run paired campaigns round-robin, rotating order at every logical step."""
    states = []
    for seed in args.seeds:
        print(f"Preparing RHT seed {seed} ...", flush=True)
        states.append(
            _prepare_campaign(
                args,
                rht_seed=seed,
                initial=initial,
                dtype=dtype,
                device=device,
            )
        )

    count = len(states)
    for step in range(args.steps):
        # Rotation removes a stable first/last position bias while every
        # campaign still consumes the identical logical data sequence.
        order = [*range(step % count, count), *range(0, step % count)]
        for index in order:
            state = states[index]
            metrics = _one_step(
                args=args,
                config=state["config"],
                dtype=dtype,
                device=device,
                data_generator=state["data_generator"],
                w1=state["w1"],
                w2=state["w2"],
                optimizer=state["optimizer"],
                measure=True,
            )
            assert metrics is not None
            metrics["step"] = step
            state["per_step"].append(metrics)

    return [_summarize_campaign(args, state, initial) for state in states]


def _preheat_device(
    args: argparse.Namespace,
    *,
    initial: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Compile the fixed path and stabilize GPU clocks before ordered A/B runs."""
    if args.preheat == 0:
        return
    w1 = initial[0].detach().clone().requires_grad_(True)
    w2 = initial[1].detach().clone().requires_grad_(True)
    optimizer = _make_optimizer(args, [w1, w2])
    config = Float4QuantConfig(use_gradient_sr=False, rht_seed=0)
    data_generator = torch.Generator(device=device).manual_seed(args.data_seed ^ 0xA5A55A5A)
    for _ in range(args.preheat):
        _one_step(
            args=args,
            config=config,
            dtype=dtype,
            device=device,
            data_generator=data_generator,
            w1=w1,
            w2=w2,
            optimizer=optimizer,
            measure=False,
        )
    torch.cuda.synchronize(device)


def _relative_percent(value: float, reference: float) -> float | None:
    return None if reference == 0.0 else 100.0 * (value - reference) / reference


def _compare_to_fixed(campaigns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed = next(campaign for campaign in campaigns if campaign["rht_seed"] == 0)
    fixed_losses = [item["loss"] for item in fixed["per_step"]]
    comparisons = []
    for campaign in campaigns:
        if campaign["rht_seed"] == 0:
            continue
        losses = [item["loss"] for item in campaign["per_step"]]
        curve_rmse = math.sqrt(statistics.fmean((a - b) ** 2 for a, b in zip(losses, fixed_losses)))
        comparisons.append(
            {
                "rht_seed": campaign["rht_seed"],
                "rht_mask": campaign["rht_mask"],
                "loss_curve_rmse_vs_fixed": curve_rmse,
                "final_loss_delta_percent": _relative_percent(campaign["loss_final"], fixed["loss_final"]),
                "step_ms_delta_percent": _relative_percent(campaign["step_ms_mean"], fixed["step_ms_mean"]),
                "throughput_delta_percent": _relative_percent(
                    campaign["tokens_per_second"], fixed["tokens_per_second"]
                ),
            }
        )
    return comparisons


def _print_summary(campaigns: list[dict[str, Any]]) -> None:
    header = (
        f"{'seed':>10}  {'mask':>10}  {'loss(first)':>12}  {'loss(final)':>12}  "
        f"{'grad_norm':>12}  {'step_ms':>10}  {'tokens/s':>12}"
    )
    print("\nRandomized-RHT MXFP4 training proxy (not full GPT-OSS E2E)")
    print(header)
    print("-" * len(header))
    for item in campaigns:
        print(
            f"{item['rht_seed']:>10d}  {item['rht_mask']:>10}  "
            f"{item['loss_first']:>12.6g}  {item['loss_final']:>12.6g}  "
            f"{item['grad_norm_mean']:>12.6g}  {item['step_ms_mean']:>10.3f}  "
            f"{item['tokens_per_second']:>12.1f}"
        )


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    # Both fused activation-quantization epilogues used by mlp_fp4 are
    # BF16-accumulator-only.  Plain FP4 GEMM supports FP16, but this fused MLP does not.
    dtype = torch.bfloat16
    gpu = _check_device(device)
    initial = _make_initial_weights(args, device, dtype)

    print(
        "Running representative single-GPU training proxy; this is not a full "
        "GPT-OSS E2E/convergence experiment."
    )
    print(
        f"GPU={gpu['name']} shape=(M={args.m}, K={args.k}, I={args.intermediate}) "
        f"dtype=bf16 preheat={args.preheat} warmup={args.warmup} "
        f"steps={args.steps} seeds={args.seeds}"
    )

    _preheat_device(args, initial=initial, dtype=dtype, device=device)

    campaigns = _run_campaigns_interleaved(
        args,
        initial=initial,
        dtype=dtype,
        device=device,
    )
    comparisons = _compare_to_fixed(campaigns)
    _print_summary(campaigns)

    result = {
        "schema_version": 1,
        "study": "Primus-Turbo static randomized-RHT single-GPU training proxy",
        "scope": (
            "Representative synthetic MXFP4 dense-MLP training proxy only; "
            "not complete GPT-OSS E2E training or convergence evidence."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "torch": torch.__version__,
            "rocm": torch.version.hip,
        },
        "gpu": gpu,
        "config": {
            "m": args.m,
            "k": args.k,
            "intermediate": args.intermediate,
            "dtype": "bf16",
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "preheat_steps": args.preheat,
            "warmup": args.warmup,
            "measured_steps": args.steps,
            "rht_seeds": args.seeds,
            "gradient_stochastic_rounding": False,
            "init_seed": args.init_seed,
            "data_seed": args.data_seed,
            "init_std": args.init_std,
            "input_std": args.input_std,
            "target": f"tanh(input * {args.target_scale})",
            "timing_scope": (
                "CUDA events around mlp_fp4 forward, FP32 MSE, backward, and optimizer.step; "
                "batch creation and metric reductions excluded; campaigns run round-robin "
                "with per-step rotation"
            ),
        },
        "campaigns": campaigns,
        "comparisons_to_fixed": comparisons,
    }

    if args.json_path is not None:
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if str(args.json_path) == "-":
            print(payload, end="")
        else:
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(payload, encoding="utf-8")
            print(f"Wrote JSON results to {args.json_path}")


if __name__ == "__main__":
    main()
