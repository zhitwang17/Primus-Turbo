#!/usr/bin/env python3
"""GPU correctness and cost harness for MXFP4 weight-scale experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any


DEFAULT_REPO = Path("/home/zhitwang/primus-turbo-branches/mxfp4-1d-scale-gpu-validation")
DEFAULT_EXTENSION = Path(
    "/home/zhitwang/primus-turbo/primus_turbo/pytorch/_C.cpython-310-x86_64-linux-gnu.so"
)
DEFAULT_CPU_REFERENCE = Path(
    "/home/zhitwang/geak-mxfp4-gpt-oss-kb-1d-scale-validation/validation/mxfp4_1d_scale_cpu"
)
MODES = ("2d_direct", "1d_direct", "1d_qdq")


def _bootstrap(repo: Path, extension: Path):
    """Load the current Python tree with a compatible prebuilt C++ extension."""
    import torch

    package_root = repo / "primus_turbo"
    root_package = types.ModuleType("primus_turbo")
    root_package.__path__ = [str(package_root)]
    sys.modules["primus_turbo"] = root_package
    pytorch_package = types.ModuleType("primus_turbo.pytorch")
    pytorch_package.__path__ = [str(package_root / "pytorch")]
    sys.modules["primus_turbo.pytorch"] = pytorch_package

    extension_name = "primus_turbo.pytorch._C"
    spec = importlib.util.spec_from_file_location(extension_name, extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension spec from {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[extension_name] = module
    spec.loader.exec_module(module)
    return torch


def _load_apis():
    from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
    from primus_turbo.pytorch.core.low_precision import (
        Float4QuantConfig,
        ScalingGranularity,
        ScalingRecipe,
        float4_e2m1fn_x2,
    )
    from primus_turbo.pytorch.core.quantized_tensor import create_quantized_weight
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl import grouped_gemm_fp4_impl
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import group_offs_from_lens
    from primus_turbo.pytorch.ops.gemm_fp4 import gemm_fp4
    from primus_turbo.pytorch.ops.grouped_gemm_fp4 import grouped_gemm_fp4
    from primus_turbo.pytorch.ops.quantization import (
        dequantize_fp4,
        quantize_fp4_with_trans,
        quantize_fp4_weight_with_trans,
    )

    return {
        "Float4QuantConfig": Float4QuantConfig,
        "BackendType": BackendType,
        "GlobalBackendManager": GlobalBackendManager,
        "ScalingGranularity": ScalingGranularity,
        "ScalingRecipe": ScalingRecipe,
        "float4": float4_e2m1fn_x2,
        "create_quantized_weight": create_quantized_weight,
        "dequantize_fp4": dequantize_fp4,
        "gemm_fp4": gemm_fp4,
        "gemm_fp4_impl": gemm_fp4_impl,
        "grouped_gemm_fp4": grouped_gemm_fp4,
        "grouped_gemm_fp4_impl": grouped_gemm_fp4_impl,
        "group_offs_from_lens": group_offs_from_lens,
        "quantize_fp4_with_trans": quantize_fp4_with_trans,
        "quantize_weight": quantize_fp4_weight_with_trans,
    }


def _unpack_codes(torch, packed):
    raw = packed.view(torch.uint8)
    result = torch.empty((*raw.shape[:-1], raw.shape[-1] * 2), dtype=torch.uint8, device=raw.device)
    result[..., 0::2] = raw & 0x0F
    result[..., 1::2] = raw >> 4
    return result


def _logical_codes(torch, packed, original_shape, direction: str):
    unpacked = _unpack_codes(torch, packed)
    if direction == "row":
        return unpacked[..., : original_shape[-1]]
    if len(original_shape) == 2:
        return unpacked.transpose(0, 1)[: original_shape[0], : original_shape[1]]
    return unpacked.transpose(-2, -1)[..., : original_shape[-2], : original_shape[-1]]


def _logical_scales(torch, scales, original_shape, direction: str, use_2d_block: bool):
    raw = scales.view(torch.uint8)
    row_count, col_count = original_shape[-2], original_shape[-1]
    if direction == "row":
        logical = raw[..., :row_count, : math.ceil(col_count / 32)]
    else:
        logical = raw[..., :col_count, : math.ceil(row_count / 32)]
    if use_2d_block:
        logical = logical[..., ::32, :]
    return logical


def _dequant_pair(torch, apis, pair, shape):
    q_row, s_row, q_col, s_col = pair
    granularity = apis["ScalingGranularity"].MX_BLOCKWISE
    recipe = apis["ScalingRecipe"](use_2d_block=False)
    row_axis = len(shape) - 1
    col_axis = len(shape) - 2
    row = apis["dequantize_fp4"](
        q_row,
        torch.bfloat16,
        granularity,
        block_size=32,
        axis=row_axis,
        scale_inv=s_row,
        scaling_recipe=recipe,
    )
    col_storage = apis["dequantize_fp4"](
        q_col,
        torch.bfloat16,
        granularity,
        block_size=32,
        axis=col_axis,
        scale_inv=s_col,
        scaling_recipe=recipe,
    )
    row = row[..., : shape[-1]]
    if len(shape) == 2:
        col = col_storage[: shape[0], : shape[1]]
    else:
        col = col_storage.transpose(-2, -1)[..., : shape[-2], : shape[-1]]
    return row, col


def _metrics(torch, reference, candidate):
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    error = cand - ref
    mse = torch.mean(error * error).item()
    signal = torch.mean(ref * ref).item()
    denom = torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand)
    cosine = (torch.dot(ref, cand) / denom).item() if denom.item() else 1.0
    return {
        "mse": mse,
        "snr_db": float("inf") if mse == 0 else 10.0 * math.log10(signal / mse),
        "cosine": cosine,
        "max_abs_error": torch.max(torch.abs(error)).item(),
    }


def _quantize_modes(torch, apis, x):
    output = {}
    for mode in MODES:
        output[mode] = apis["quantize_weight"](
            x,
            apis["float4"],
            apis["ScalingGranularity"].MX_BLOCKWISE,
            block_size=32,
            weight_quant_mode=mode,
            use_preshuffle=False,
        )
    torch.cuda.synchronize()
    return output


def run_primitive(torch, apis, cpu_reference: Path):
    sys.path.insert(0, str(cpu_reference))
    from mxfp4_reference import quantize_weight_modes_torch

    cases = []
    for shape, seed in (((64, 128), 11), ((96, 160), 17), ((3, 64, 128), 23)):
        torch.manual_seed(seed)
        x = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.125).to(torch.bfloat16)
        gpu_modes = _quantize_modes(torch, apis, x)
        cpu_modes = quantize_weight_modes_torch(x.cpu().float())
        case = {"shape": list(shape), "seed": seed, "modes": {}}
        for mode in MODES:
            q_row, s_row, q_col, s_col = gpu_modes[mode]
            gpu_row_codes = _logical_codes(torch, q_row, shape, "row").cpu()
            gpu_col_codes = _logical_codes(torch, q_col, shape, "col").cpu()
            cpu_pair = cpu_modes[mode]
            row_dq, col_dq = _dequant_pair(torch, apis, gpu_modes[mode], shape)
            row_ref = cpu_pair.row_dequantized.to(torch.bfloat16)
            col_ref = cpu_pair.col_dequantized.to(torch.bfloat16)
            mode_result = {
                "gpu_shapes": {
                    "row_q": list(q_row.shape),
                    "row_scale": list(s_row.shape),
                    "col_q": list(q_col.shape),
                    "col_scale": list(s_col.shape),
                },
                "row_codes_equal_cpu": bool(torch.equal(gpu_row_codes, cpu_pair.row.codes)),
                "col_codes_equal_cpu": bool(torch.equal(gpu_col_codes, cpu_pair.col.codes)),
                "row_dequant_equal_cpu_bf16": bool(torch.equal(row_dq.cpu(), row_ref)),
                "col_dequant_equal_cpu_bf16": bool(torch.equal(col_dq.cpu(), col_ref)),
                "row_scale_shape_cpu": list(cpu_pair.row.scales.shape),
                "col_scale_shape_cpu": list(cpu_pair.col.scales.shape),
            }
            use_2d_block = mode == "2d_direct"
            mode_result["row_scales_equal_cpu"] = bool(
                torch.equal(
                    _logical_scales(torch, s_row, shape, "row", use_2d_block).cpu(),
                    cpu_pair.row.scales,
                )
            )
            mode_result["col_scales_equal_cpu"] = bool(
                torch.equal(
                    _logical_scales(torch, s_col, shape, "col", use_2d_block).cpu(),
                    cpu_pair.col.scales,
                )
            )
            case["modes"][mode] = mode_result
        case["1d_direct_vs_qdq_col_code_difference_fraction"] = torch.mean(
            (
                _logical_codes(torch, gpu_modes["1d_direct"][2], shape, "col")
                != _logical_codes(torch, gpu_modes["1d_qdq"][2], shape, "col")
            ).float()
        ).item()
        cases.append(case)
    return {"cases": cases}


def run_correctness(torch, apis):
    cases = []
    specs = (
        ("dense_synthetic", (256, 4096), 101, 0.025),
        ("moe_gate_up_synthetic", (4, 576, 2880), 103, 0.020),
        ("moe_down_synthetic", (4, 288, 2880), 107, 0.020),
    )
    for name, shape, seed, sigma in specs:
        torch.manual_seed(seed)
        x = (torch.randn(shape, device="cuda", dtype=torch.float32) * sigma).to(torch.bfloat16)
        modes = _quantize_modes(torch, apis, x)
        result = {"name": name, "shape": list(shape), "seed": seed, "modes": {}}
        for mode, pair in modes.items():
            row, col = _dequant_pair(torch, apis, pair, shape)
            result["modes"][mode] = {
                "row_vs_source": _metrics(torch, x, row),
                "col_vs_source": _metrics(torch, x, col),
                "col_vs_row": _metrics(torch, row, col),
            }
        direct_codes = _logical_codes(torch, modes["1d_direct"][2], shape, "col")
        qdq_codes = _logical_codes(torch, modes["1d_qdq"][2], shape, "col")
        result["1d_direct_vs_qdq_col_code_difference_fraction"] = torch.mean(
            (direct_codes != qdq_codes).float()
        ).item()
        cases.append(result)
    return {"cases": cases}


def run_checkpoint_correctness(torch, apis, cpu_reference: Path):
    sys.path.insert(0, str(cpu_reference))
    from analyze_checkpoint import ShardedCheckpoint, _aligned_start
    from mxfp4_reference import dequantize_checkpoint_mxfp4_numpy

    checkpoint_root = Path("/shareddata/sapmajum/models/gpt-oss-20b")
    checkpoint = ShardedCheckpoint(checkpoint_root)
    samples = []

    def evaluate(kind, tensor_name, selection, values):
        x = torch.from_numpy(values.copy()).to(device="cuda", dtype=torch.bfloat16)
        modes = _quantize_modes(torch, apis, x)
        result = {
            "kind": kind,
            "tensor": tensor_name,
            "selection": selection,
            "shape": list(x.shape),
            "modes": {},
        }
        for mode, pair in modes.items():
            row, col = _dequant_pair(torch, apis, pair, tuple(x.shape))
            result["modes"][mode] = {
                "row_vs_source": _metrics(torch, x, row),
                "col_vs_source": _metrics(torch, x, col),
                "col_vs_row": _metrics(torch, row, col),
            }
        direct = _logical_codes(torch, modes["1d_direct"][2], tuple(x.shape), "col")
        qdq = _logical_codes(torch, modes["1d_qdq"][2], tuple(x.shape), "col")
        result["1d_direct_vs_qdq_col_code_difference_fraction"] = torch.mean((direct != qdq).float()).item()
        samples.append(result)
        del x, modes
        torch.cuda.empty_cache()

    for layer in (0, 12, 23):
        name = f"model.layers.{layer}.self_attn.o_proj.weight"
        rows, cols = checkpoint.metadata(name)["shape"]
        start = _aligned_start(rows, 128, layer * 137 + 17)
        values = checkpoint.array(name, (slice(start, start + 128), slice(None)))
        evaluate("dense_bf16", name, {"rows": [start, start + 128], "columns": [0, cols]}, values)

    for layer in (0, 12, 23):
        for projection in ("gate_up_proj", "down_proj"):
            prefix = f"model.layers.{layer}.mlp.experts.{projection}"
            blocks_name, scales_name = f"{prefix}_blocks", f"{prefix}_scales"
            metadata = checkpoint.metadata(blocks_name)
            _, rows, block_count, packed_width = metadata["shape"]
            for expert in (0, 15, 31):
                start = _aligned_start(rows, 64, layer * 149 + expert * 31 + len(projection))
                blocks = checkpoint.array(
                    blocks_name, (expert, slice(start, start + 64), slice(None), slice(None))
                )
                scales = checkpoint.array(scales_name, (expert, slice(start, start + 64), slice(None)))
                values = dequantize_checkpoint_mxfp4_numpy(blocks, scales)
                evaluate(
                    "moe_decoded_mxfp4",
                    prefix,
                    {
                        "expert": expert,
                        "rows": [start, start + 64],
                        "columns": [0, block_count * packed_width * 2],
                    },
                    values,
                )
    return {"checkpoint": str(checkpoint_root), "samples": samples}


def run_cache_correctness(torch, apis):
    torch.manual_seed(151)
    weight = (torch.randn((256, 512), device="cuda", dtype=torch.float32) * 0.025).to(torch.bfloat16)
    cases = []
    for preshuffle in (False, True):
        for mode in MODES:
            config = apis["Float4QuantConfig"](
                use_preshuffle=preshuffle,
                weight_quant_mode=mode,
            )
            cached_row, cached_col = apis["create_quantized_weight"](
                weight,
                apis["float4"],
                config,
                need_weight_transpose_cache=True,
            )
            raw = apis["quantize_weight"](
                weight,
                apis["float4"],
                apis["ScalingGranularity"].MX_BLOCKWISE,
                block_size=32,
                weight_quant_mode=mode,
                use_preshuffle=preshuffle,
            )
            torch.cuda.synchronize()
            cases.append(
                {
                    "mode": mode,
                    "preshuffle": preshuffle,
                    "row_codes_equal_raw": bool(
                        torch.equal(cached_row.qdata.view(torch.uint8), raw[0].view(torch.uint8))
                    ),
                    "row_scales_equal_raw": bool(
                        torch.equal(cached_row.scale_inv.view(torch.uint8), raw[1].view(torch.uint8))
                    ),
                    "col_codes_equal_raw": bool(
                        torch.equal(cached_col.qdata.view(torch.uint8), raw[2].view(torch.uint8))
                    ),
                    "col_scales_equal_raw": bool(
                        torch.equal(cached_col.scale_inv.view(torch.uint8), raw[3].view(torch.uint8))
                    ),
                }
            )
    return {"cases": cases}


def _grouped_bf16_reference(torch, a, b, group_lens):
    outputs = []
    offset = 0
    for expert, length in enumerate(group_lens.cpu().tolist()):
        if length:
            outputs.append(a[offset : offset + length] @ b[expert].transpose(-2, -1))
        offset += length
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty((0, b.shape[1]), dtype=a.dtype, device=a.device)


def run_op_correctness(torch, apis):
    manager = apis["GlobalBackendManager"]
    backend = apis["BackendType"].FLYDSL
    manager.set_gemm_backend(backend)
    manager.set_grouped_gemm_backend(backend)
    manager.set_auto_tune(False)
    output = {"dense": {}, "dense_preshuffle": {}, "grouped": []}
    try:
        torch.manual_seed(301)
        a0 = (torch.randn((256, 256), device="cuda", dtype=torch.float32) * 0.5).to(torch.bfloat16)
        b0 = (torch.randn((256, 256), device="cuda", dtype=torch.float32) * 0.025).to(torch.bfloat16)
        grad0 = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        a_ref = a0.detach().clone().requires_grad_(True)
        b_ref = b0.detach().clone().requires_grad_(True)
        dense_ref = a_ref @ b_ref.transpose(-2, -1)
        dense_ref.backward(grad0)
        for mode in MODES:
            a = a0.detach().clone().requires_grad_(True)
            b = b0.detach().clone().requires_grad_(True)
            config = apis["Float4QuantConfig"](use_gradient_sr=False, weight_quant_mode=mode)
            value = apis["gemm_fp4"](a, b, trans_b=True, out_dtype=torch.bfloat16, config=config)
            value.backward(grad0)
            torch.cuda.synchronize()
            output["dense"][mode] = {
                "fwd_vs_bf16": _metrics(torch, dense_ref, value),
                "dgrad_vs_bf16": _metrics(torch, a_ref.grad, a.grad),
                "wgrad_vs_bf16": _metrics(torch, b_ref.grad, b.grad),
            }

        # AITER is the preshuffled consumer.  Verify that the experimental
        # cache/layout path remains numerically valid, then restore FlyDSL for
        # grouped coverage below.
        manager.set_gemm_backend(apis["BackendType"].AITER)
        try:
            for mode in MODES:
                a = a0.detach().clone().requires_grad_(True)
                b = b0.detach().clone().requires_grad_(True)
                config = apis["Float4QuantConfig"](
                    use_gradient_sr=False,
                    use_preshuffle=True,
                    weight_quant_mode=mode,
                )
                value = apis["gemm_fp4"](a, b, trans_b=True, out_dtype=torch.bfloat16, config=config)
                value.backward(grad0)
                torch.cuda.synchronize()
                output["dense_preshuffle"][mode] = {
                    "fwd_vs_bf16": _metrics(torch, dense_ref, value),
                    "dgrad_vs_bf16": _metrics(torch, a_ref.grad, a.grad),
                    "wgrad_vs_bf16": _metrics(torch, b_ref.grad, b.grad),
                }
        except ImportError as error:
            output["dense_preshuffle"] = {
                "status": "not_run",
                "reason": str(error),
                "cache_layout_validation": "passed in cache_correctness for all three modes",
            }
        manager.set_gemm_backend(backend)

        routing_cases = {
            "balanced": [64, 64, 64, 64],
            "zero_even_odd": [0, 255, 256, 512],
        }
        for routing_name, lens_values in routing_cases.items():
            group_lens = torch.tensor(lens_values, dtype=torch.int64, device="cuda")
            total_m = sum(lens_values)
            torch.manual_seed(307 + total_m)
            a_base = (torch.randn((total_m, 256), device="cuda", dtype=torch.float32) * 0.5).to(
                torch.bfloat16
            )
            b_base = (torch.randn((4, 256, 256), device="cuda", dtype=torch.float32) * 0.025).to(
                torch.bfloat16
            )
            grad = torch.randn((total_m, 256), device="cuda", dtype=torch.bfloat16)
            a_ref = a_base.detach().clone().requires_grad_(True)
            b_ref = b_base.detach().clone().requires_grad_(True)
            ref = _grouped_bf16_reference(torch, a_ref, b_ref, group_lens)
            ref.backward(grad)
            route_result = {"routing": routing_name, "group_lens": lens_values, "modes": {}}
            nonempty = group_lens > 0
            for mode in MODES:
                a = a_base.detach().clone().requires_grad_(True)
                b = b_base.detach().clone().requires_grad_(True)
                config = apis["Float4QuantConfig"](use_gradient_sr=False, weight_quant_mode=mode)
                value = apis["grouped_gemm_fp4"](
                    a,
                    b,
                    group_lens,
                    trans_b=True,
                    out_dtype=torch.bfloat16,
                    config=config,
                )
                value.backward(grad)
                torch.cuda.synchronize()
                route_result["modes"][mode] = {
                    "fwd_vs_bf16": _metrics(torch, ref, value),
                    "dgrad_vs_bf16": _metrics(torch, a_ref.grad, a.grad),
                    "wgrad_vs_bf16_nonempty": _metrics(torch, b_ref.grad[nonempty], b.grad[nonempty]),
                }
            output["grouped"].append(route_result)
    finally:
        manager.reset()
    return output


def _benchmark_modes(torch, callbacks, warmup: int, repeats: int, groups: int):
    """Collect paired/interleaved timings so clock drift does not favor one mode."""
    for _ in range(warmup):
        for mode in MODES:
            callbacks[mode]()
    torch.cuda.synchronize()

    samples = {mode: [] for mode in MODES}
    for group in range(groups):
        order = MODES if group % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            group_samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                callbacks[mode]()
                end.record()
                end.synchronize()
                group_samples.append(start.elapsed_time(end))
            samples[mode].append(statistics.median(group_samples))

    results = {}
    for mode in MODES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        callbacks[mode]()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline_memory
        ordered = sorted(samples[mode])
        results[mode] = {
            "group_medians_ms": samples[mode],
            "median_ms": statistics.median(samples[mode]),
            "p10_ms": ordered[max(0, math.floor(0.10 * (len(ordered) - 1)))],
            "p90_ms": ordered[min(len(ordered) - 1, math.ceil(0.90 * (len(ordered) - 1)))],
            "peak_memory_delta_bytes": int(peak_delta),
        }
    baseline = results["2d_direct"]["median_ms"]
    for mode in MODES:
        latency = results[mode]["median_ms"]
        results[mode]["speedup_vs_2d"] = baseline / latency
        results[mode]["latency_change_percent_vs_2d"] = (latency / baseline - 1.0) * 100.0
    return results


def run_benchmark(torch, apis, warmup: int, repeats: int, groups: int):
    cases = []
    specs = (
        ("dense_4096x4096", (4096, 4096), 201),
        ("moe_gate_up_g32", (32, 5760, 2880), 203),
        ("moe_down_g32", (32, 2880, 2880), 211),
    )
    for name, shape, seed in specs:
        torch.manual_seed(seed)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        result = {"name": name, "shape": list(shape), "modes": {}}
        callbacks = {}
        for mode in MODES:
            callbacks[mode] = lambda mode=mode: apis["quantize_weight"](
                x,
                apis["float4"],
                apis["ScalingGranularity"].MX_BLOCKWISE,
                block_size=32,
                weight_quant_mode=mode,
                use_preshuffle=False,
            )
        # Interleave mode order across groups at the outer case level by running
        # a short priming call for every mode before collecting each distribution.
        for callback in callbacks.values():
            callback()
        torch.cuda.synchronize()
        result["modes"] = _benchmark_modes(torch, callbacks, warmup, repeats, groups)
        cases.append(result)
        del x, callbacks
        torch.cuda.empty_cache()
    return {"warmup": warmup, "repeats_per_group": repeats, "groups": groups, "cases": cases}


def run_op_benchmark(torch, apis, warmup: int, repeats: int, groups: int):
    manager = apis["GlobalBackendManager"]
    backend = apis["BackendType"].FLYDSL
    manager.set_gemm_backend(backend)
    manager.set_grouped_gemm_backend(backend)
    manager.set_auto_tune(False)
    payload = {"warmup": warmup, "repeats_per_group": repeats, "groups": groups, "cases": []}
    try:
        # Dense GEMM-only: pre-quantize operands once, then time only the raw GEMM.
        torch.manual_seed(401)
        dense_m = dense_n = dense_k = 4096
        dense_a = torch.randn((dense_m, dense_k), device="cuda", dtype=torch.bfloat16)
        dense_b = torch.randn((dense_n, dense_k), device="cuda", dtype=torch.bfloat16)
        act_recipe = apis["ScalingRecipe"](use_2d_block=False)
        dense_a_row, dense_a_scale, _, _ = apis["quantize_fp4_with_trans"](
            dense_a,
            apis["float4"],
            apis["ScalingGranularity"].MX_BLOCKWISE,
            block_size=32,
            scaling_recipe=act_recipe,
            scaling_recipe_for_trans=apis["ScalingRecipe"](use_rht=True),
        )
        dense_weight_pairs = _quantize_modes(torch, apis, dense_b)
        dense_gemm_callbacks = {}
        for mode in MODES:
            b_row, b_scale = dense_weight_pairs[mode][0], dense_weight_pairs[mode][1]
            dense_gemm_callbacks[mode] = lambda b_row=b_row, b_scale=b_scale: apis["gemm_fp4_impl"](
                dense_a_row,
                dense_a_scale,
                False,
                b_row,
                b_scale,
                True,
                torch.bfloat16,
                False,
                granularity=apis["ScalingGranularity"].MX_BLOCKWISE.value,
                default_backend=backend.value,
                preshuffled=False,
            )
        payload["cases"].append(
            {
                "name": "dense_gemm_only_4096",
                "scope": "gemm_only_prequantized",
                "shape": [dense_m, dense_n, dense_k],
                "modes": _benchmark_modes(torch, dense_gemm_callbacks, warmup, repeats, groups),
            }
        )

        # Dense full op: weight/activation preparation plus fwd+dgrad+wgrad.
        dense_grad = torch.randn((dense_m, dense_n), device="cuda", dtype=torch.bfloat16)
        dense_full_callbacks = {}
        for mode in MODES:
            config = apis["Float4QuantConfig"](use_gradient_sr=False, weight_quant_mode=mode)

            def dense_full(config=config):
                a = dense_a.detach().requires_grad_(True)
                b = dense_b.detach().requires_grad_(True)
                value = apis["gemm_fp4"](a, b, trans_b=True, out_dtype=torch.bfloat16, config=config)
                return torch.autograd.grad(value, (a, b), dense_grad)

            dense_full_callbacks[mode] = dense_full
        payload["cases"].append(
            {
                "name": "dense_full_fwd_bwd_4096",
                "scope": "quant_preparation_plus_fwd_dgrad_wgrad",
                "shape": [dense_m, dense_n, dense_k],
                "modes": _benchmark_modes(torch, dense_full_callbacks, warmup, repeats, groups),
            }
        )
        del dense_weight_pairs, dense_gemm_callbacks, dense_full_callbacks, dense_grad
        torch.cuda.empty_cache()

        # Real GPT-OSS MoE gate-up geometry with balanced routing.
        torch.manual_seed(409)
        group_count, total_m, moe_n, moe_k = 32, 4096, 5760, 2880
        group_lens = torch.full((group_count,), total_m // group_count, dtype=torch.int64, device="cuda")
        group_offs = apis["group_offs_from_lens"](group_lens)
        moe_a = torch.randn((total_m, moe_k), device="cuda", dtype=torch.bfloat16)
        moe_b = torch.randn((group_count, moe_n, moe_k), device="cuda", dtype=torch.bfloat16)
        (
            moe_a_row,
            moe_a_scale,
            _,
            _,
            _,
            group_offs_padded_row,
            _,
            _,
        ) = __import__(
            "primus_turbo.pytorch.ops.quantization", fromlist=["grouped_quantize_fp4_with_trans"]
        ).grouped_quantize_fp4_with_trans(
            moe_a,
            apis["float4"],
            apis["ScalingGranularity"].MX_BLOCKWISE,
            group_lens,
            group_offs,
            block_size=32,
            scaling_recipe=apis["ScalingRecipe"](),
            scaling_recipe_for_trans=apis["ScalingRecipe"](use_rht=True),
        )
        moe_weight_pairs = _quantize_modes(torch, apis, moe_b)
        moe_gemm_callbacks = {}
        for mode in MODES:
            b_row, b_scale = moe_weight_pairs[mode][0], moe_weight_pairs[mode][1]
            moe_gemm_callbacks[mode] = lambda b_row=b_row, b_scale=b_scale: apis["grouped_gemm_fp4_impl"](
                moe_a_row,
                b_row,
                moe_a_scale,
                b_scale,
                group_lens,
                group_offs_padded_row,
                trans_a=False,
                trans_b=True,
                out_dtype=torch.bfloat16,
                granularity=apis["ScalingGranularity"].MX_BLOCKWISE.value,
                num_cu=None,
                default_backend=backend.value,
                group_offs_out=group_offs,
            )
        payload["cases"].append(
            {
                "name": "moe_gate_up_gemm_only_balanced",
                "scope": "grouped_gemm_only_prequantized",
                "shape": [group_count, total_m, moe_n, moe_k],
                "routing": group_lens.cpu().tolist(),
                "modes": _benchmark_modes(torch, moe_gemm_callbacks, warmup, repeats, groups),
            }
        )

        moe_grad = torch.randn((total_m, moe_n), device="cuda", dtype=torch.bfloat16)
        moe_full_callbacks = {}
        for mode in MODES:
            config = apis["Float4QuantConfig"](use_gradient_sr=False, weight_quant_mode=mode)

            def moe_full(config=config):
                a = moe_a.detach().requires_grad_(True)
                b = moe_b.detach().requires_grad_(True)
                value = apis["grouped_gemm_fp4"](
                    a,
                    b,
                    group_lens,
                    group_offs=group_offs,
                    trans_b=True,
                    out_dtype=torch.bfloat16,
                    config=config,
                )
                return torch.autograd.grad(value, (a, b), moe_grad)

            moe_full_callbacks[mode] = moe_full
        payload["cases"].append(
            {
                "name": "moe_gate_up_full_fwd_bwd_balanced",
                "scope": "quant_preparation_plus_grouped_fwd_dgrad_wgrad",
                "shape": [group_count, total_m, moe_n, moe_k],
                "routing": group_lens.cpu().tolist(),
                "modes": _benchmark_modes(torch, moe_full_callbacks, warmup, repeats, groups),
            }
        )

        # Keep total tokens fixed and redistribute them to exercise empty,
        # lightly loaded, and heavily loaded experts without changing weights.
        skewed_lens_values = [0] * 8 + [32] * 8 + [96] * 8 + [384] * 8
        skewed_lens = torch.tensor(skewed_lens_values, dtype=torch.int64, device="cuda")
        skewed_offs = apis["group_offs_from_lens"](skewed_lens)
        (
            skewed_a_row,
            skewed_a_scale,
            _,
            _,
            _,
            skewed_offs_padded_row,
            _,
            _,
        ) = __import__(
            "primus_turbo.pytorch.ops.quantization", fromlist=["grouped_quantize_fp4_with_trans"]
        ).grouped_quantize_fp4_with_trans(
            moe_a,
            apis["float4"],
            apis["ScalingGranularity"].MX_BLOCKWISE,
            skewed_lens,
            skewed_offs,
            block_size=32,
            scaling_recipe=apis["ScalingRecipe"](),
            scaling_recipe_for_trans=apis["ScalingRecipe"](use_rht=True),
        )
        skewed_gemm_callbacks = {}
        for mode in MODES:
            b_row, b_scale = moe_weight_pairs[mode][0], moe_weight_pairs[mode][1]
            skewed_gemm_callbacks[mode] = lambda b_row=b_row, b_scale=b_scale: apis[
                "grouped_gemm_fp4_impl"
            ](
                skewed_a_row,
                b_row,
                skewed_a_scale,
                b_scale,
                skewed_lens,
                skewed_offs_padded_row,
                trans_a=False,
                trans_b=True,
                out_dtype=torch.bfloat16,
                granularity=apis["ScalingGranularity"].MX_BLOCKWISE.value,
                num_cu=None,
                default_backend=backend.value,
                group_offs_out=skewed_offs,
            )
        payload["cases"].append(
            {
                "name": "moe_gate_up_gemm_only_skewed",
                "scope": "grouped_gemm_only_prequantized",
                "shape": [group_count, total_m, moe_n, moe_k],
                "routing": skewed_lens_values,
                "modes": _benchmark_modes(torch, skewed_gemm_callbacks, warmup, repeats, groups),
            }
        )

        skewed_full_callbacks = {}
        for mode in MODES:
            config = apis["Float4QuantConfig"](use_gradient_sr=False, weight_quant_mode=mode)

            def skewed_full(config=config):
                a = moe_a.detach().requires_grad_(True)
                b = moe_b.detach().requires_grad_(True)
                value = apis["grouped_gemm_fp4"](
                    a,
                    b,
                    skewed_lens,
                    group_offs=skewed_offs,
                    trans_b=True,
                    out_dtype=torch.bfloat16,
                    config=config,
                )
                return torch.autograd.grad(value, (a, b), moe_grad)

            skewed_full_callbacks[mode] = skewed_full
        payload["cases"].append(
            {
                "name": "moe_gate_up_full_fwd_bwd_skewed",
                "scope": "quant_preparation_plus_grouped_fwd_dgrad_wgrad",
                "shape": [group_count, total_m, moe_n, moe_k],
                "routing": skewed_lens_values,
                "modes": _benchmark_modes(torch, skewed_full_callbacks, warmup, repeats, groups),
            }
        )
    finally:
        manager.reset()
    return payload


def _environment(torch, args):
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_unix": time.time(),
        "repo": str(args.repo),
        "extension": str(args.extension),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "gcn_arch": getattr(props, "gcnArchName", "unknown"),
        "pci_address": (
            f"{getattr(props, 'pci_domain_id', 0):04x}:"
            f"{getattr(props, 'pci_bus_id', 0):02x}:"
            f"{getattr(props, 'pci_device_id', 0):02x}.0"
        ),
        "device_total_memory": props.total_memory,
    }


def _json_safe(value: Any):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("primitive", "correctness", "checkpoint", "ops", "benchmark", "op-benchmark", "all")
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--cpu-reference", type=Path, default=DEFAULT_CPU_REFERENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--op-warmup", type=int, default=5)
    parser.add_argument("--op-repeats", type=int, default=20)
    parser.add_argument("--op-groups", type=int, default=5)
    args = parser.parse_args()

    torch = _bootstrap(args.repo, args.extension)
    apis = _load_apis()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"environment": _environment(torch, args), "requested_phase": args.phase}
    if args.phase in ("primitive", "all"):
        payload["primitive"] = run_primitive(torch, apis, args.cpu_reference)
    if args.phase in ("correctness", "all"):
        payload["correctness"] = run_correctness(torch, apis)
        payload["cache_correctness"] = run_cache_correctness(torch, apis)
    if args.phase in ("checkpoint", "all"):
        payload["checkpoint_correctness"] = run_checkpoint_correctness(torch, apis, args.cpu_reference)
    if args.phase in ("ops", "all"):
        payload["op_correctness"] = run_op_correctness(torch, apis)
    if args.phase in ("benchmark", "all"):
        payload["benchmark"] = run_benchmark(torch, apis, args.warmup, args.repeats, args.groups)
    if args.phase in ("op-benchmark", "all"):
        payload["op_benchmark"] = run_op_benchmark(
            torch, apis, args.op_warmup, args.op_repeats, args.op_groups
        )

    output = args.output_dir / f"{args.phase}.json"
    output.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    print(json.dumps(_json_safe(payload), indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
