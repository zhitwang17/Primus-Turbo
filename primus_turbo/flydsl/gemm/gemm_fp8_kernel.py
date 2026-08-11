###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL),
#   file kernels/gemm/fp8_gemm_8wave.py.
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Primus-Turbo dense FP8 GEMM kernel (FlyDSL): NT, NN and TN layouts, 256x256 tiles,
mfma_f32_16x16x128_f8f6f4, per-tensor scale, bf16/fp16 out, arbitrary K (TT unsupported).
NT/NN and the TN fallback are 8-wave; TN's primary path is the 4-wave whole-loop."""

import functools

import torch

# isort: off
# Primitives are vendored in flydsl/utils/gemm_helper.py (no 3rdparty/FlyDSL
# submodule; flydsl, the compiler, is the only FlyDSL dep) and imported as module
# globals (@flyc.kernel needs its dependencies as globals).
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCPerTensor,
    _readfirstlane_i32,
    _robust_time,
    asm_mma_do,
    block_mn,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_fp8_buffer_tensor_rebased,
    make_row_band_resource,
    make_value_attrs,
    mask_a_tail,
    wait_barrier,
    xcd_remap_pid,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# isort: on

# `nt` aux bit: C is write-once, so caching it evicts the A/B band the L2 swizzle keeps.
_CSTORE_AUX = 2

_PICK_RAMP_ITERS = 200


@functools.lru_cache(maxsize=256)
def _compile_dense_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 1,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    nt_vmcnt: int = 3,  # end-of-iter s_waitcnt vmcnt(N): N=3 → det=0 (gfx950 G2S buffer_load_lds/ds_read LDS hazard), <=1.1% cost; N>=4 races, N<3 costlier; -1 disables
    num_xcd: int = 8,  # XCD-aware PID remap: cluster same-XCD WGs into contiguous logical tiles for per-XCD L2 reuse (gfx950 MI355X = 8 XCD); 1 disables
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor out dtype: True -> fp16, else bf16
):
    """Build & cache the (K, BLOCK_M, BLOCK_N, GROUP_M)-specialised NT launch.

    GROUP_M is the super-block tile-id swizzle width for L2 reuse (WGs advance
    block_m first within each GROUP_M x n_blocks band; 1 = row-major). The main
    K-loop barriers are all load-bearing (each guards a compiler-reorder race).
    """
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert GROUP_M >= 1

    # Odd-K native K-tail: ceil(K/128) iters, the last of length K_TAIL (0 =
    # exact multiple). The tail's invalid K-columns are zeroed on A in Epilog 2
    # via mask_a_tail; G2S tail over-reads clamp to 0 via the buffer SRD bound.
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # NT semantics: A is [M, K] row-major K-contig.
        #               B_T is [N, K] row-major K-contig (= B^T storage of [K, N]).
        # Output       C is [M, N] row-major bf16.
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # Super-block tile swizzle for L2 reuse; group_size_m clamps the last
        # band so any GROUP_M >= 1 is correct (arith.select = integer min).
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

        # i64 input re-base: fold the per-tile row base (m_row*K, n_row*K) into the
        # SRD base; A/B_T K-contiguous (foldable), k*BLOCK_K small int32 -> no cap.
        a_base = arith.index_cast(T.index, block_m * BLOCK_M) * arith.index(K)
        b_base = arith.index_cast(T.index, block_n * BLOCK_N) * arith.index(K)
        a_nrec = (
            arith.index_cast(T.index, c_m) - arith.index_cast(T.index, block_m * BLOCK_M)
        ) * arith.index(K)
        b_nrec = (
            arith.index_cast(T.index, c_n) - arith.index_cast(T.index, block_n * BLOCK_N)
        ) * arith.index(K)
        A0_gl_offset = 0
        A1_gl_offset = LDS_BLOCK_M * K
        B0_gl_offset = 0
        B1_gl_offset = LDS_BLOCK_N * K

        gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
        gB = make_fp8_buffer_tensor_rebased(B_T, F8_IR_t, b_base, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(
            A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty, store_aux=_CSTORE_AUX
        )

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude: k=0 → cur, k=1 → next (a_next1 lazily on first main iter).
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Main K-loop. Each iter: s2r {a0,b0,b1,a1} → 4 mma (c00→c01→c10→c11)
        # interleaved with k+1 (a_next1) and k+2 (a_cur0, b_cur0, b_cur1) prefetches.
        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            if nt_vmcnt >= 0:
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string=f"s_waitcnt vmcnt({nt_vmcnt})",
                    constraints="",
                    has_side_effects=True,
                )  # end-of-iter G2S drain (race fix)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1 (k = K_ITERS - 2). The a_g2s.load(a_next1, A1 + (k+1)*BLOCK_K)
        # line is the c10/c11 stale-a1 pipeline fix -- without it epilog-2's
        # a1_frag would read older K-iter data and the bottom half of every
        # output tile loses the final K-tile contribution.
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 (k = K_ITERS - 1) -- the K-tail block. Mask the A operand
        # so invalid K-columns (>= K_TAIL) contribute 0. No-op when K_TAIL==0.
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale + store.
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nt(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nt


# ──────────────────────────────────────────────────────────────────────

_DPP_QUAD_SWAP1 = 0xB1  # quad_perm:[1,0,3,2] -- exchange with the neighbouring lane
_PERM_LO_PAIR = 0x05040100  # {own low half, right neighbour's low half}
_PERM_HI_PAIR = 0x03020706  # {left neighbour's high half, own high half}


class StoreCPerTensorPairN(StoreCPerTensor):
    """StoreCPerTensor folding a row's two n-fragments into one dword store: a fragment
    row spans 16 lanes, so a scalar 2 B store leaves as a 32 B request and the pair
    doubles it. Needs an even c_cols -- the pair is written as a unit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.trans and not self.pack_cols and self.n_tiles_b % 2 == 0
        assert self.out_ty is fx.BFloat16, "the packed convert is bf16-only"
        lane16 = self.lane_id % 16
        hi = (lane16 % 2) > 0
        # even lane e holds columns (e, e+1); odd lane o holds (16 + o - 1, 16 + o).
        self.pair_col = arith.select(hi, lane16 + 15, lane16)
        self.pair_sel = arith.select(hi, fx.Int32(_PERM_HI_PAIR), fx.Int32(_PERM_LO_PAIR))

    def store(self, c_frag, base_row, base_col, col_frags=()):
        """Row band store; see StoreCPerTensor.store for the col_frags contract."""
        scale = self._scale()
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        quads = [(0, c_frag)] + [(int(d), f) for d, f in col_frags]
        col0 = base_col + self.pair_col
        zero = arith._to_raw(fx.Int32(0))
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            vecs = [
                [
                    (Vec(f[self.c_idx_fn(ti, tj)]) * scale) if self.scaled else Vec(f[self.c_idx_fn(ti, tj)])
                    for tj in range_constexpr(self.n_tiles_b)
                ]
                for _, f in quads
            ]
            for i in range_constexpr(4):
                row_off = ((row_local + i) * self.c_cols + col0) * 2  # i32-small within band
                for q in range_constexpr(len(quads)):
                    for p in range_constexpr(self.n_tiles_b // 2):
                        dcol = quads[q][0] + p * 32
                        lo, hi = vecs[q][2 * p][i], vecs[q][2 * p + 1][i]
                        if self.elem_fn is not None:
                            lo, hi = self.elem_fn(lo), self.elem_fn(hi)
                        pk = rocdl.cvt_pk_bf16_f32(lo, hi)
                        sw = rocdl.update_dpp(pk.type, zero, pk, _DPP_QUAD_SWAP1, 0xF, 0xF, True)
                        pair_ok = None if self.col_safe else (col0 + dcol + 1) < self.c_cols
                        _buffer_ops.buffer_store(
                            rocdl.perm_b32(sw, pk, self.pair_sel),
                            rsrc,
                            row_off if dcol == 0 else row_off + dcol * 2,
                            mask=pair_ok,
                            cache_modifier=self.store_aux,
                            offset_is_bytes=True,
                        )


@functools.lru_cache(maxsize=128)
def _compile_dense_nn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D band (width group_n), as in NT
    num_xcd: int = 8,  # XCD-aware PID remap for per-XCD L2 reuse (MI355X = 8 XCD); 1 disables. See xcd_remap_pid.
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    # Issue ds_read_tr8_b64 as inline asm so the backend skips the auto vmcnt(0)
    # drain; vmcnt_hint supplies the LDS sync. Requires agpr_alloc > 0.
    b_inline_asm_load: bool = False,
    vmcnt_hint: int = 2,
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor out dtype: True -> fp16, else bf16
    i64_traverse: bool = False,  # B[K,N] traversal via per-load i64 SRD re-base (lifts k*n < 2^32 cap)
    pair_n: bool = False,  # fold the n-fragment pair into one dword store (needs even N)
    col_safe: bool = False,  # N % BLOCK_N == 0: drop the epilogue's per-store column clamp
):
    """NN-layout fp8 dense kernel. A [M, K], B [K, N], C [M, N].

    ``agpr_alloc`` / ``waves_per_eu`` mirror the NT kernel's knobs; see
    ``make_value_attrs`` for ``agpr_alloc`` encoding (N>0 = exact N AGPRs,
    -N = up to N, 0 = compiler default)."""
    if b_inline_asm_load and agpr_alloc == 0:
        raise ValueError(
            "b_inline_asm_load=True requires agpr_alloc > 0 (a compiler-decided "
            "AGPR count conflicts with the inline-asm operand constraints); "
            "pin AGPR to a nonzero value such as 32."
        )
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # Odd-K native K-tail: ceil iters; final iter masked on A (see NT note).
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K  # same byte count as NT, different layout

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Materialize thread_idx.x before S2RLoaderTr lazily uses it inside
        # range_constexpr loops, so the ds_read_tr8_b64 load order is correct.
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        block_m, block_n = block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # i64 input re-base. A[M,K]: fold row base (m_row*K) into SRD. B[K,N]: the
        # k*BLOCK_K*c_n contraction is i64 per load (cn_i), capped at 4GB by num_records.
        m_row = block_m * BLOCK_M
        cn_i = arith.index_cast(T.index, c_n)
        a_base = arith.index_cast(T.index, m_row) * arith.index(K)
        a_nrec = (arith.index_cast(T.index, c_m) - arith.index_cast(T.index, m_row)) * arith.index(K)
        b_base = arith.index_cast(T.index, block_n * BLOCK_N)
        b_nrec = arith.index(K) * cn_i - b_base
        A0_gl_offset = 0
        A1_gl_offset = LDS_BLOCK_M * K
        B0_gl_offset = 0
        B1_gl_offset = LDS_BLOCK_N

        gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
        gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        _nnwz = True  # wave bank-swizzle B; write and read sides must match
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS, wswz=_nnwz)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            # E5M2 / hybrid: rebuild the MFMA atom with per-operand fp8 fmt
            # (cbsz->srcA, blgp->srcB). Same instruction family / frag layout
            # as the default e4m3 atom, so loaders are unchanged.
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        # B[K,N] is the contraction-traversal operand: in i64 mode re-base its SRD
        # per load (k_offset folds into the i64 base) instead of a 32-bit soffset.
        b_rebase = (B, F8_IR_t, b_base, b_nrec) if i64_traverse else None
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, rebase=b_rebase)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, 32, inline_asm=b_inline_asm_load, vmcnt_hint=vmcnt_hint, wswz=_nnwz
        )
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        _store_cls = StoreCPerTensorPairN if pair_n else StoreCPerTensor
        store_c = _store_cls(
            A_scale,
            B_scale,
            C,
            c_m,
            c_n,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            _out_ty,
            col_safe=col_safe,
            store_aux=_CSTORE_AUX,
        )

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
        b_g2s.load(b_cur0, B0_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + arith.index(1 * BLOCK_K) * cn_i)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + arith.index(1 * BLOCK_K) * cn_i)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Main loop. Emits 7 barriers per K-iter (before/after each MFMA);
        # all are load-bearing — dropping any risks a compiler-reorder race.
        # vmcnt=-1: the trailing wait_barrier already drains g2s (the epilogue keeps its own).
        for k in range_constexpr(K_ITERS - 2):
            # drain=False: the a0 load drains these reads before c00 consumes b0.
            b0_frag = b_s2r.load(b_cur0, vmcnt=-1, drain=False)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1, vmcnt=-1)
            b_g2s.load(b_cur0, B0_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1.
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        wave_n_offset = _readfirstlane_i32(wave_n * (N_TILES_B * 16))
        wave_m_offset = _readfirstlane_i32(wave_m * (N_TILES_A * 16))
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        # Epilog 2 -- K-tail block. Mask A so K-cols >= K_TAIL contribute 0.
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Issue each group at its own last mfma: the exposed drain is the burst's tail.
        store_c.store(c00_frag, base_row, base_col)

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c01_frag, base_row, base_col + LDS_BLOCK_N)

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nn


@functools.lru_cache(maxsize=128)
def _compile_dense_tn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    waves_per_eu: int = 2,
    vmcnt_hint: int = 3,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D band (width group_n)
    num_xcd: int = 8,  # XCD-aware PID remap for per-XCD L2 reuse (MI355X = 8 XCD); 1 disables. See xcd_remap_pid.
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor out dtype: True -> fp16, else bf16
    i64_traverse: bool = False,  # A[K,M] & B[K,N] traversal via per-load i64 SRD re-base (lifts cap)
):
    """TN-layout fp8 dense kernel: A [K, M], B [K, N], C [M, N] = A^T @ B.
    Both A and B are K-row strided, so both go through the wave-coop
    ds_read_b64_tr_b8 transpose load (the mfma A and B operand register byte
    layouts are identical, so the same S2RLoaderTr feeds both operands).
    Inline-asm tr8 on both operands + asm-inplace MFMA (=a,v,v,0; D aliases C in
    AGPR -> accumulators spill-free, no per-K-iter A-side vmcnt(0) drain)."""
    _a_inline = True
    _b_inline = True
    _asm_mma_mode = "2"  # asm-inplace MFMA (accum in AGPR)
    _inplace = True
    agpr_alloc = 128
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # Odd-K native K-tail: ceil iters. No A-mask needed here -- TN's A [K,M]
    # and B [K,N] are K-row-major, so the tail's invalid K-rows are fully out
    # of bounds and clamp to 0 via the buffer SRD num_records bound.
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    # TN A path uses the wave-coop tr8 transpose load, whose K_log spans
    # [0, 128) and needs 2 G2S rounds = a 16K LDS slot. For BM=128 (natural
    # N_LDS_STEPS_A=1, 8K slot) force 2 rounds / 16K slot to match the K=128
    # transpose-load expectation.
    N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)  # ≥ 2 for tr8 K=128
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    # Bank-spread LDS chunk stride: 1056 (=1024+32) un-aligns the per-wave chunk
    # base across LDS banks to remove the transpose-read bank conflict; the G2S
    # writer and S2R reader must use the same value.
    _LDS_CS = 1056
    # a_lds_size: N rounds × 8 waves × chunk_stride. Pad to stride.
    a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
    b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    def _tn_block_mn(pid, num_pid_m, n_blocks, GM, GN):
        """Tile-id -> (block_m, block_n), resolved at trace time. GN==0: 1D
        GROUP_M super-row swizzle (block_m inner). GN>0: 2D band — N split into
        width-GN bands with GROUP_M inside each, keeping both A and B slabs
        L2-resident. Always a bijection."""
        if GN > 0:
            band_tiles = num_pid_m * GN
            band = pid // band_tiles
            pid_in_band = pid % band_tiles
            band_n0 = band * GN
            rem_n = n_blocks - band_n0
            band_w = arith.select(rem_n < GN, rem_n, fx.Int32(GN))
            nig = GM * band_w
            gid = pid_in_band // nig
            pig = pid_in_band % nig
            fpm = gid * GM
            rem_m = num_pid_m - fpm
            gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
            return fpm + (pig % gsm), band_n0 + (pig // gsm)
        nig = GM * n_blocks
        gid = pid // nig
        pig = pid % nig
        fpm = gid * GM
        rem_m = num_pid_m - fpm
        gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
        return fpm + (pig % gsm), pig // gsm

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4

        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        # Swizzle via plain-Python helper (NOT a kernel `if`: @flyc.kernel
        # wraps each if-branch in its own fn so vars defined inside aren't
        # visible after — see prelude note). Helper builds the expr graph
        # for one Python-selected path (1D GROUP_M or 2D band).
        block_m, block_n = _tn_block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # i64 input re-base. A[K,M]/B[K,N] K-row-major: fold column base into SRD; the
        # k*BLOCK_K*c_{m,n} traversal is i64 per load (int32 wraps > 2^31), capped at 4GB.
        cm_i = arith.index_cast(T.index, c_m)
        cn_i = arith.index_cast(T.index, c_n)
        a_base = arith.index_cast(T.index, block_m) * arith.index(BLOCK_M)
        b_base = arith.index_cast(T.index, block_n) * arith.index(BLOCK_N)
        a_nrec = arith.index(K) * cm_i - a_base
        b_nrec = arith.index(K) * cn_i - b_base
        A0_gl_offset = 0
        A1_gl_offset = LDS_BLOCK_M
        B0_gl_offset = 0
        B1_gl_offset = LDS_BLOCK_N

        gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
        gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # Both A+B use NN-style K-strided global swizzle.
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if _inplace:
            _mm = _asm_mma_mode
            mfma._do_mma = lambda _a, _b, _c, _m=_mm: asm_mma_do(_a, _b, _c, mode=_m, cbsz=cbsz, blgp=blgp)

        # TN: both A[K,M] and B[K,N] are contraction-traversal operands -> re-base
        # both SRDs per load in i64 mode (each k_offset folds into its i64 base).
        a_rebase = (A, F8_IR_t, a_base, a_nrec) if i64_traverse else None
        b_rebase = (B, F8_IR_t, b_base, b_nrec) if i64_traverse else None
        a_g2s = G2SLoader(
            a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS, rebase=a_rebase
        )
        b_g2s = G2SLoader(
            b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS, rebase=b_rebase
        )
        a_s2r = S2RLoaderTr(
            wave_m,
            N_TILES_A,
            LDS_BLOCK_M // 2,
            inline_asm=_a_inline,
            vmcnt_hint=vmcnt_hint,
            chunk_stride=_LDS_CS,
        )
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, 32, inline_asm=_b_inline, vmcnt_hint=vmcnt_hint, chunk_stride=_LDS_CS
        )
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(
            A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty, store_aux=_CSTORE_AUX
        )

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
        b_g2s.load(b_cur0, B0_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur0, A0_gl_offset + arith.index(0 * BLOCK_K) * cm_i)
        b_g2s.load(b_cur1, B1_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur1, A1_gl_offset + arith.index(0 * BLOCK_K) * cm_i)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + arith.index(1 * BLOCK_K) * cn_i)
        a_g2s.load(a_next0, A0_gl_offset + arith.index(1 * BLOCK_K) * cm_i)
        b_g2s.load(b_next1, B1_gl_offset + arith.index(1 * BLOCK_K) * cn_i)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Steady loop: per-iter A-half-0/A-half-1 × {b0,b1} MMA interleaved
        # with the next-tile G2S prefetch and one s_barrier per MMA quadrant.
        # All 7 barriers are load-bearing (dropping any races at the
        # MFMA-reorder level under some GROUP_M; gated by long det runs).
        for k in range_constexpr(K_ITERS - 2):
            # b0 drain=False: the b0 reads are covered by the immediately-
            # following a0 load's lgkmcnt(0) before c00 consumes b0, so the
            # b0 loader's own trailing drain is redundant. (b1 keeps its
            # drain — c01 consumes b1 with no covering drain between.)
            b0_frag = b_s2r.load(b_cur0, drain=False)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + arith.index((k + 1) * BLOCK_K) * cm_i)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + arith.index((k + 2) * BLOCK_K) * cm_i)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b_g2s.load(b_cur1, B1_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1.
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + arith.index((k + 1) * BLOCK_K) * cm_i)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2.
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)
        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_tn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_tn


# NN 4-wave kernel removed (consistently slower than 8w; HW caps 1
# wave/SIMD for this layout).


# TN 4-wave (occ=1): one tile per WG, whole K loop in one bare-asm region, accumulators in AGPR.
_TN4_BLOCK = 256  # square tile; ds_read_b64_tr_b8 transposes wrong at a 128-wide operand
_TN4_BLOCK_K = 128
_TN4_LDS_BLOCK = _TN4_BLOCK // 2  # rows/cols one LDS pool holds
_TN4_TILES = _TN4_BLOCK // 64  # 16x16 mfma tiles a wave owns per operand half
_TN4_STEPS = (_TN4_LDS_BLOCK * _TN4_BLOCK_K) // (256 * 16)  # dwordx4 G2S steps per buffer
_TN4_WAVES = 4
_TN4_CS = 1024  # per-wave LDS chunk stride, shared by the G2S writer and the S2R reader
_TN4_BUF = (_TN4_LDS_BLOCK * _TN4_BLOCK_K) // 1024 * _TN4_CS  # bytes per LDS buffer
_TN4_RS = (_TN4_LDS_BLOCK // 16) * _TN4_CS  # LDS delta between a tile's two transpose rows
_TN4_NBUF = (2, 2, 3, 3)  # per-pool buffering: A halves 2-deep, B halves 3-deep
_TN4_PHASES = 6  # lcm(_TN4_NBUF) = K-blocks one pass of the main loop consumes
_TN4_PIN = 8  # first VGPR of the pinned operand-fragment window
# Phase drain leaves this many ds_reads in flight; their buffers are a global trip away.
_TN4_ELGK = 12
_TN4_ASM_CACHE: dict = {}


def _dense_tn_slice_div(x, s):
    """``x // s`` for a compile-time split-K slice factor: one shift, or one fixed-point
    reciprocal multiply. Exact for every dividend the slice bounds reach (see _TN4_RCP_MAX)."""
    if s & (s - 1) == 0:
        return fx.Int32(x >> (s.bit_length() - 1))
    return fx.Int32(fx.Int32(x * (-(-(1 << 16) // s))) >> 16)


def _dense_tn_wave4_asm(cbsz, blgp):
    """Bare-asm K body for one output tile: the four mfma quadrants, the ds_read refills that
    feed them and a later K-block's global->LDS writes, rotated over the _TN4_NBUF buffers.
    Trip count and fused tail passes are runtime SGPRs. Returns (asm, constraints, type)."""
    if (cbsz, blgp) in _TN4_ASM_CACHE:
        return _TN4_ASM_CACHE[(cbsz, blgp)]
    nt, ns, npool = _TN4_TILES, _TN4_STEPS, len(_TN4_NBUF)
    nq = nt * nt  # accumulators per quadrant
    nacc = 4 * nq
    ntmp = npool * nt  # live operand fragments
    mods = f" cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
    assert (max(_TN4_NBUF) - 1) * _TN4_BUF + _TN4_RS < 65536, "buffer delta overflows ds offset"

    # Outputs: accumulators, fragments, counter, per-pool soffset; unwritten "=&s" = regalloc hazard.
    o_cnt = nacc + ntmp
    o_wsoff = [o_cnt + 1 + p for p in range(npool)]
    _at = o_cnt + 1 + npool

    def take(n):
        nonlocal _at
        _at += n
        return list(range(_at - n, _at))

    i_base = [take(2 * nt) for _ in range(npool)]
    i_gbase = [take(nb) for nb in _TN4_NBUF]
    i_gla, i_glb = take(ns), take(ns)
    i_rsa, i_rsb = take(1)[0], take(1)[0]
    i_ka, i_kb = take(1)[0], take(1)[0]
    i_nval, i_tail = take(1)[0], take(1)[0]
    i_soff0 = take(npool)
    i_gl = (i_gla, i_gla, i_glb, i_glb)
    i_rsrc = (i_rsa, i_rsa, i_rsb, i_rsb)
    i_kstep = (i_ka, i_ka, i_kb, i_kb)

    def ds_line(rbuf, tt):
        # The buffer delta rides the ds_read immediate, so one address pair covers the pool.
        p, ti = divmod(tt - nacc, nt)
        bo = rbuf[p] * _TN4_BUF
        v = _TN4_PIN + (tt - nacc) * 8
        ptr = (i_base[p][2 * ti], i_base[p][2 * ti + 1])
        return "\n".join(
            f"ds_read_b64_tr_b8 v[{v + 2 * j}:{v + 2 * j + 1}], "
            f"${ptr[j % 2]} offset:{bo + (j // 2) * _TN4_RS}"
            for j in range(4)
        )

    def emit_g2s(wbuf):
        # A pools step-interleaved to share the 128B line; B pools last, per the partial drain.
        order = [(p, st) for st in range(ns) for p in (0, 1)]
        order += [(p, st) for p in (2, 3) for st in range(ns)]
        return [
            f"s_add_u32 m0, ${i_gbase[p][wbuf[p]]}, {st * _TN4_WAVES * _TN4_CS}\n"
            f"buffer_load_dwordx4 ${i_gl[p][st]}, ${i_rsrc[p]}, ${o_wsoff[p]} offen lds"
            for p, st in order
        ]

    def mfma_seq():
        # srcA pool outer (this mfma is srcA-movement sensitive); 2x4 diagonal spreads refills.
        bm, bn = 2, 4
        nib, ncb = nt // bm, 2 * nt // bn
        seq = []
        for d in range(nib + ncb - 1):
            for iib in range(nib):
                if not 0 <= d - iib < ncb:
                    continue
                for di in range(bm):
                    for ah in range(2):
                        for dj in range(bn):
                            col, ii = (d - iib) * bn + dj, iib * bm + di
                            q = (ah * 2 + col // nt) * nq + ii * nt + col % nt
                            at = nacc + ah * nt + ii
                            bt = nacc + (2 + col // nt) * nt + col % nt
                            seq.append(
                                (f"v_mfma_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}{mods}", at, bt)
                            )
        return seq

    def emit_phase(rbuf, wbuf):
        # Refill a fragment right after its last consumer; global writes take the free slots.
        g2sl, mlist = emit_g2s(wbuf), mfma_seq()
        last = {}
        for mi, (_m, at, bt) in enumerate(mlist):
            last[at] = last[bt] = mi
        busy = {mi for mi, (_m, at, bt) in enumerate(mlist) if last[at] == mi or last[bt] == mi}
        free = [mi for mi in range(len(mlist)) if mi not in busy]
        gap = max(len(free) // len(g2sl), 1)
        gslot = {fi: k // gap for k, fi in enumerate(free) if k % gap == 0 and k // gap < len(g2sl)}
        out, gi, refilled = [], 0, set()
        for mi, (ml, at, bt) in enumerate(mlist):
            out.append(ml)
            for rt in (at, bt):
                if last[rt] == mi and rt not in refilled:
                    out.append(ds_line(rbuf, rt))
                    refilled.add(rt)
            if mi in gslot and gi < len(g2sl):
                out.append(g2sl[gi])
                gi += 1
        out += g2sl[gi:]
        out += [ds_line(rbuf, tt) for tt in range(nacc, nacc + ntmp) if tt not in refilled]
        return out

    # Partial drain: the emit order above leaves exactly the 3-buffered B writes in flight.
    n_out = sum(ns for p in range(npool) if _TN4_NBUF[p] == 3)
    drain = f"s_waitcnt vmcnt({n_out}) lgkmcnt({_TN4_ELGK})\ns_barrier"

    def phase_block(ph):
        blk = emit_phase([(ph + 1) % nb for nb in _TN4_NBUF], [ph % nb for nb in _TN4_NBUF])
        blk.append(drain)
        return blk + [f"s_add_u32 ${o_wsoff[p]}, ${o_wsoff[p]}, ${i_kstep[p]}" for p in range(npool)]

    L = [f"s_mov_b32 ${o_cnt}, 0"]
    L += [f"s_mov_b32 ${o_wsoff[p]}, ${i_soff0[p]}" for p in range(npool)]
    L += [ds_line([0] * npool, tt) for tt in range(nacc, nacc + ntmp)]
    # Deeper primes wait here to overlap the ds_read issue; the barrier still guards buf0.
    L += [f"s_waitcnt vmcnt({n_out}) lgkmcnt(0)", "s_barrier", "1:"]
    for ph in range(_TN4_PHASES):
        L += phase_block(ph)
    L += [
        f"s_add_u32 ${o_cnt}, ${o_cnt}, {_TN4_PHASES}",
        f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}",
        "s_cbranch_scc1 1b",
        # A partial drain needs a next phase, which no longer exists past the exit.
        f"s_cmp_eq_u32 ${i_tail}, 0",
        "s_cbranch_scc1 3f",
        "s_waitcnt vmcnt(0) lgkmcnt(0)",
        "s_barrier",
        "3:",
    ]
    for j in range(_TN4_PHASES - 1):  # gated single-K-block passes reusing the loop block
        L += [f"s_cmp_le_u32 ${i_tail}, {j}", f"s_cbranch_scc1 {j + 4}f"]
        L += phase_block(j) + [f"{j + 4}:"]
    L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")

    cons = ",".join(
        ["=a"] * nacc
        + [f"=&{{v[{_TN4_PIN + f * 8}:{_TN4_PIN + f * 8 + 7}]}}" for f in range(ntmp)]
        + ["=&s"] * (1 + npool)
        + ["v"] * (2 * nt * npool)
        + ["s"] * sum(_TN4_NBUF)
        + ["v"] * (2 * ns)
        + ["s"] * (6 + npool)
        + [str(q) for q in range(nacc)]
    )
    st = (
        "!llvm.struct<("
        + ", ".join(["vector<4xf32>"] * nacc + ["vector<8xi32>"] * ntmp + ["i32"] * (1 + npool))
        + ")>"
    )
    _TN4_ASM_CACHE[(cbsz, blgp)] = ("\n".join(L), cons, st)
    return _TN4_ASM_CACHE[(cbsz, blgp)]


# Must stay top-level: nested in @flyc.kernel its asm cache would look like global drift.
def _dense_tn_wave4_tile(
    d,
    *,
    M,
    N,
    K,
    K_ITERS,
    NBM,
    NBN,
    group_m,
    group_n,
    split,
    store_aux,
    lds,
    A,
    B,
    C,
    WS,
    A_scale,
    B_scale,
    gl_off_a,
    gl_off_b,
    wave_id,
    wave_m,
    wave_n,
    cbsz,
    blgp,
    out_ty,
    col_safe,
):
    """Emit one dispatch id's output tile. ``split`` is None or the (lo, n, s) split-K window,
    which always sits at the grid tail: ids from lo up carry a (tile, slice) pair, slice 0
    writing C and slice j>0 writing band j-1 of WS."""
    row_shift, store_base = None, None
    k0 = fx.Int32(0)
    ki = fx.Int32(K_ITERS)
    if split is None:
        t = d
    else:
        lo, _, s = split
        rel = fx.Int32(d) - fx.Int32(lo)
        pre = rel < fx.Int32(0)
        q = _dense_tn_slice_div(rel, s)
        t = _readfirstlane_i32(arith.select(pre, d, fx.Int32(lo) + q))
        # sid < 0 = whole tile: the ids below the window take the whole K range.
        sid = fx.Int32(_readfirstlane_i32(arith.select(pre, fx.Int32(-1), rel - q * fx.Int32(s))))
        whole = sid < fx.Int32(0)
        sc = fx.Int32(arith.select(whole, fx.Int32(0), sid))
        nxt = sc + fx.Int32(1)
        kb1 = fx.Int32(
            arith.select(
                nxt < fx.Int32(s), _dense_tn_slice_div(fx.Int32(K_ITERS) * nxt, s), fx.Int32(K_ITERS)
            )
        )
        k0 = fx.Int32(arith.select(whole, fx.Int32(0), _dense_tn_slice_div(fx.Int32(K_ITERS) * sc, s)))
        ki = fx.Int32(_readfirstlane_i32(arith.select(whole, fx.Int32(K_ITERS), kb1 - k0)))
        part = sid > fx.Int32(0)
        row_shift = _readfirstlane_i32(arith.select(part, (sid - fx.Int32(1)) * fx.Int32(M), fx.Int32(0)))
        store_base = arith.select(part, _buffer_ops.extract_base_index(WS), _buffer_ops.extract_base_index(C))

    block_m, block_n = block_mn(t, fx.Int32(NBM), fx.Int32(NBN), group_m, group_n)
    bm_off = _readfirstlane_i32(block_m) * fx.Int32(_TN4_BLOCK)
    bn_off = _readfirstlane_i32(block_n) * fx.Int32(_TN4_BLOCK)
    # Main loop takes the largest multiple of _TN4_PHASES; the remainder is the in-asm tail.
    n6 = (ki // _TN4_PHASES) * _TN4_PHASES
    nval = _readfirstlane_i32(n6)
    tail = _readfirstlane_i32(ki - n6)

    # A [K,M] / B [K,N] stride K: fold slice row + tile column into the SRD, num_records clamps.
    F8_IR_t = fx.Float8E4M3FN.ir_type
    k_row = arith.index_cast(T.index, k0) * arith.index(_TN4_BLOCK_K)
    rows = arith.index(K) - k_row
    a_base = k_row * arith.index(M) + arith.index_cast(T.index, bm_off)
    a_nrec = arith.maxsi(rows * arith.index(M) - arith.index_cast(T.index, bm_off), arith.index(0))
    b_base = k_row * arith.index(N) + arith.index_cast(T.index, bn_off)
    b_nrec = arith.maxsi(rows * arith.index(N) - arith.index_cast(T.index, bn_off), arith.index(0))
    gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
    gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)

    mfma = Mfma16x16x128(_TN4_TILES, _TN4_TILES)
    _s2r = functools.partial(
        S2RLoaderTr,
        n_tiles=_TN4_TILES,
        tile_stride=_TN4_TILES * 16,
        n_waves=_TN4_WAVES,
        chunk_stride=_TN4_CS,
        width=_TN4_LDS_BLOCK,
        wswz=True,  # wave bank-swizzle (matches gl_off_a/b in the kernel body)
    )
    a_s2r, b_s2r = _s2r(wave_m), _s2r(wave_n)
    a_g2s = G2SLoader(
        fx.logical_divide(gA, fx.make_layout(1, 1)),
        gl_off_a,
        _TN4_STEPS,
        F8_IR_t,
        wave_id,
        chunk_stride=_TN4_CS,
    )
    b_g2s = G2SLoader(
        fx.logical_divide(gB, fx.make_layout(1, 1)),
        gl_off_b,
        _TN4_STEPS,
        F8_IR_t,
        wave_id,
        chunk_stride=_TN4_CS,
    )
    c_rows = fx.Int32(M) if row_shift is None else fx.Int32(M) + row_shift
    store_c = StoreCPerTensor(
        A_scale,
        B_scale,
        C,
        c_rows,
        fx.Int32(N),
        mfma.idx,
        _TN4_TILES,
        _TN4_TILES,
        out_ty,
        col_safe=col_safe,
        store_aux=store_aux,
        c_base=store_base,
    )

    a_k = arith.index(_TN4_BLOCK_K) * arith.index(M)
    b_k = arith.index(_TN4_BLOCK_K) * arith.index(N)
    half = _TN4_LDS_BLOCK
    a_g2s.load(lds.A_lds_cur_0, 0 * a_k)
    b_g2s.load(lds.B_lds_cur_0, 0 * b_k)
    b_g2s.load(lds.B_lds_cur_1, half + 0 * b_k)
    a_g2s.load(lds.A_lds_cur_1, half + 0 * a_k)
    a_g2s.load(lds.A_lds_next_0, 1 * a_k)
    b_g2s.load(lds.B_lds_next_0, 1 * b_k)
    b_g2s.load(lds.B_lds_next_1, half + 1 * b_k)
    a_g2s.load(lds.A_lds_next_1, half + 1 * a_k)
    b_g2s.load(lds.B_lds_extra_1, half + 2 * b_k)
    b_g2s.load(lds.B_lds_extra_0, 2 * b_k)
    # Covers the buf0 primes only; the deeper ones are waited on inside the asm.
    wait_barrier((sum(_TN4_NBUF) - len(_TN4_NBUF)) * _TN4_STEPS)

    pools = [
        ((lds.A_lds_cur_0, lds.A_lds_next_0), a_s2r),
        ((lds.A_lds_cur_1, lds.A_lds_next_1), a_s2r),
        ((lds.B_lds_cur_0, lds.B_lds_next_0, lds.B_lds_extra_0), b_s2r),
        ((lds.B_lds_cur_1, lds.B_lds_next_1, lds.B_lds_extra_1), b_s2r),
    ]
    # A pool's buffers are read the same way, so only buffer 0 needs live address VGPRs.
    ins = [v for bufs, s2r in pools for pair in s2r.base_addr(bufs[0]) for v in pair]
    ins += [
        rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(_TN4_CS))
        for bufs, _s2r in pools
        for buf in bufs
    ]
    ins += [fx.Int32(gl_off_a[st]) for st in range_constexpr(_TN4_STEPS)]
    ins += [fx.Int32(gl_off_b[st]) for st in range_constexpr(_TN4_STEPS)]
    kstep_a = rocdl.readfirstlane(T.i32, fx.Int32(_TN4_BLOCK_K) * fx.Int32(M))
    kstep_b = rocdl.readfirstlane(T.i32, fx.Int32(_TN4_BLOCK_K) * fx.Int32(N))
    ins += [
        _buffer_ops.create_buffer_resource(
            A, max_size=False, num_records_bytes=a_nrec, base_byte_offset=a_base
        ),
        _buffer_ops.create_buffer_resource(
            B, max_size=False, num_records_bytes=b_nrec, base_byte_offset=b_base
        ),
        kstep_a,
        kstep_b,
        nval,
        tail,
    ]
    # soff0[p] = the global offset of the first in-loop write, targeting K-block _TN4_NBUF[p].
    ins += [
        rocdl.readfirstlane(T.i32, fx.Int32(_TN4_NBUF[0]) * kstep_a),
        rocdl.readfirstlane(T.i32, fx.Int32(half) + fx.Int32(_TN4_NBUF[1]) * kstep_a),
        rocdl.readfirstlane(T.i32, fx.Int32(_TN4_NBUF[2]) * kstep_b),
        rocdl.readfirstlane(T.i32, fx.Int32(half) + fx.Int32(_TN4_NBUF[3]) * kstep_b),
    ]
    nq = _TN4_TILES * _TN4_TILES
    ins += [mfma.zero_value] * (4 * nq)

    asm, cons, st = _dense_tn_wave4_asm(cbsz, blgp)
    r = _llvm.inline_asm(ir.Type.parse(st), [arith._to_raw(v) for v in ins], asm, cons, has_side_effects=True)
    res = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(4 * nq)]

    base_row = bm_off + wave_m * fx.Int32(_TN4_TILES * 16)
    if row_shift is not None:
        base_row = base_row + row_shift
    base_col = bn_off + wave_n * fx.Int32(_TN4_TILES * 16)
    for qi in range_constexpr(4):
        store_c.store(res[qi * nq : (qi + 1) * nq], base_row + (qi // 2) * half, base_col + (qi % 2) * half)


_TN4_SPLIT_S = (2, 3, 4)  # slice factors; an odd one is fine, the slices stay co-resident
_TN4_RCP_MAX = 1 << 15  # exactness bound on _dense_tn_slice_div's dividend
_TN4_RED_WPT = 4  # reduce workgroups per window tile
_TN4_RED_VEC = 8  # out_ty elements (128b) each reduce lane moves per pass


def _dense_tn_split(tiles, k_iters, ncu):
    """Split-K window ``(lo, n, s)`` for one dense TN grid, or None. With one uniform K the
    makespan quantizes only on the last partial round, so slice its ``rem`` tiles s ways;
    keep the shortest s that still fits the window inside one round."""
    rem = tiles % ncu
    if rem == 0:
        return None
    best, wn, wd = 1, 1, 1
    for s in _TN4_SPLIT_S:
        if k_iters < _TN4_PHASES * s or k_iters * (s - 1) >= _TN4_RCP_MAX:
            continue  # every slice must keep a whole main-loop pass, and stay exactly divisible
        rounds = ceildiv(rem * s, ncu) if tiles <= ncu else (1 if s * rem <= ncu else s)
        if rounds * wd < wn * s:
            best, wn, wd = s, rounds, s
    return None if best == 1 else (tiles - rem, rem, best)


_NUM_CUS = 0


def _dense_num_cus():
    """Device CU count, memoised: the property query is otherwise on the dispatch path."""
    global _NUM_CUS
    if not _NUM_CUS:
        _NUM_CUS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NUM_CUS


def _compile_dense_tn_wave4(
    M: int,
    N: int,
    K: int,
    group_m: int,
    group_n: int,
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,
):
    """4-wave (occ=1) dense TN C[M,N] = A[K,M]^T @ B[K,N] over 256x256 tiles, one tile per
    workgroup. Returns (launch, split-K scratch band count)."""
    NBM, NBN = ceildiv(M, _TN4_BLOCK), ceildiv(N, _TN4_BLOCK)
    TILES = NBM * NBN
    K_ITERS = ceildiv(K, _TN4_BLOCK_K)
    assert K_ITERS >= _TN4_PHASES, "4-wave dense TN needs a K of at least one main-loop pass"
    split = _dense_tn_split(TILES, K_ITERS, _dense_num_cus())
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    _bands = 0 if split is None else split[2] - 1
    _GRID = TILES + (0 if split is None else split[1] * _bands)
    # The store hint also lands on the reduce's input, so carry it only while bands are few.
    _STORE_AUX = _CSTORE_AUX if 2 * (_GRID - TILES) <= _GRID else 0
    _RED_ROWS = _TN4_BLOCK // _TN4_RED_WPT
    _RED_LPR = _TN4_BLOCK // _TN4_RED_VEC  # lanes spanning one tile row
    _RED_RPP = 256 // _RED_LPR  # rows one 256-thread pass covers
    _RED_GRID = 1 if split is None else split[1] * _TN4_RED_WPT
    _RED_LO, _SLICES = (0, 1) if split is None else (split[0], split[2])

    # Field ORDER must stay POOL-MAJOR: one register set + ds_read immediate reaches a pool.
    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        B_lds_extra_0: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]
        B_lds_extra_1: fx.Array[fx.Float8E4M3FN, _TN4_BUF, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_dense_tn_wave4(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        WS: fx.Tensor,
    ):
        _ = str(fx.thread_idx.x)
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        _dense_tn_wave4_tile(
            fx.block_idx.x,
            M=M,
            N=N,
            K=K,
            K_ITERS=K_ITERS,
            NBM=NBM,
            NBN=NBN,
            group_m=group_m,
            group_n=group_n,
            split=split,
            store_aux=_STORE_AUX,
            lds=fx.SharedAllocator().allocate(SharedStorage).peek(),
            A=A,
            B=B,
            C=C,
            WS=WS,
            A_scale=A_scale,
            B_scale=B_scale,
            gl_off_a=compute_global_swizzle_nn(
                lane_id, wave_id, M, _TN4_STEPS, width=_TN4_LDS_BLOCK, wswz=True
            ),
            gl_off_b=compute_global_swizzle_nn(
                lane_id, wave_id, N, _TN4_STEPS, width=_TN4_LDS_BLOCK, wswz=True
            ),
            wave_id=wave_id,
            wave_m=wave_id // 2,
            wave_n=wave_id % 2,
            cbsz=cbsz,
            blgp=blgp,
            out_ty=_out_ty,
            col_safe=N % _TN4_BLOCK == 0,
        )

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_dense_tn_wave4_reduce(C: fx.Tensor, WS: fx.Tensor):
        """Fold the split-K scratch bands back into C. Only the window's tiles are touched
        -- the rest of a band is never read and needs no zeroing pass. Slots are summed in
        a fixed 0..S-2 order in fp32, keeping the store bit-reproducible."""
        _ = str(fx.thread_idx.x)
        _ir_ty = _out_ty.ir_type
        f32v = fx.T.VectorType.get([_TN4_RED_VEC], fx.T.f32())
        outv = fx.T.VectorType.get([_TN4_RED_VEC], _ir_ty)
        c_base = _buffer_ops.extract_base_index(C)
        ws_base = _buffer_ops.extract_base_index(WS)
        tid = fx.thread_idx.x
        col_l = (tid % fx.Int32(_RED_LPR)) * fx.Int32(_TN4_RED_VEC)
        row_l = tid // fx.Int32(_RED_LPR)
        slot = _readfirstlane_i32(fx.block_idx.x // fx.Int32(_TN4_RED_WPT))
        sub = _readfirstlane_i32(fx.block_idx.x % fx.Int32(_TN4_RED_WPT))
        bm, bn = block_mn(fx.Int32(_RED_LO) + slot, fx.Int32(NBM), fx.Int32(NBN), group_m, group_n)
        bm_off = _readfirstlane_i32(bm) * fx.Int32(_TN4_BLOCK)
        col = _readfirstlane_i32(bn) * fx.Int32(_TN4_BLOCK) + col_l
        # Rows past M fall outside the band SRD (dropped); columns would wrap, hence col_ok.
        col_ok = col < fx.Int32(N)
        rs_c = make_row_band_resource(c_base, bm_off, M, N, 2)
        rs_w = [
            make_row_band_resource(ws_base, bm_off + fx.Int32((j - 1) * M), j * M, N, 2)
            for j in range_constexpr(1, _SLICES)
        ]
        off0 = (sub * fx.Int32(_RED_ROWS) + row_l) * fx.Int32(N) + col
        for p in range_constexpr(_RED_ROWS // _RED_RPP):
            off = off0 + fx.Int32(p * _RED_RPP * N)
            acc = arith.extf(
                f32v,
                _buffer_ops.buffer_load(rs_c, off, vec_width=_TN4_RED_VEC, dtype=_ir_ty, mask=col_ok),
            )
            for j in range_constexpr(1, _SLICES):
                acc = arith.addf(
                    acc,
                    arith.extf(
                        f32v,
                        _buffer_ops.buffer_load(
                            rs_w[j - 1], off, vec_width=_TN4_RED_VEC, dtype=_ir_ty, mask=col_ok
                        ),
                    ),
                )
            _buffer_ops.buffer_store(arith.trunc_f(outv, acc), rs_c, off, mask=col_ok)

    _ATTRS = make_value_attrs(1, 0, "256,256")

    @flyc.jit
    def launch_dense_tn_wave4(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        WS: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel_dense_tn_wave4(A, B, C, A_scale, B_scale, WS, value_attrs=_ATTRS).launch(
            grid=(_GRID, 1, 1), block=(256, 1, 1), stream=stream
        )
        if const_expr(split is not None):
            # Same stream: the reduce sees every slice partial.
            kernel_dense_tn_wave4_reduce(C, WS).launch(
                grid=(_RED_GRID, 1, 1), block=(256, 1, 1), stream=stream
            )

    return launch_dense_tn_wave4, _bands


_COMPILED_DENSE_CACHE: dict = {}

# fx.Stream packs a plain int, so no torch.cuda.Stream wrapper is built per call.
_raw_stream = torch._C._cuda_getCurrentRawStream


def _static_layout(args):
    """Wrap the tensor arguments as static-layout memrefs for a one-time compile: a bare
    torch.Tensor compiles layout-dynamic and re-reads shape/stride per launch, while the
    compiled object is already one per operand geometry."""
    return tuple(flyc.from_torch_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)


def _get_compiled_dense(launch, args):
    """Cache compiled launcher by (shape, stride, dtype, int-arg) tuple. Strides are in the
    key because the compile pins the operand layout; the trailing queue handle is not, as
    it selects where a launch goes and keying on it would recompile per stream."""
    key_parts = [id(launch)]
    for a in args[:-1]:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.stride(), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_DENSE_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *_static_layout(args))
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


def _pick_dense_candidate(cands, args):
    """Fastest of ``cands`` = [[launch, cfg, compiled], ...], sampled twice with the second
    pass reversed and kept at its min, behind a throwaway pass: the leading candidates sit
    closer than one sample's spread, so otherwise clock drift and warm-up do the ranking."""
    for _ in range(_PICK_RAMP_ITERS):
        cands[0][2](*args)
    torch.cuda.synchronize()
    order = list(range(len(cands)))
    ts = [float("inf")] * len(cands)
    for i in order + order[::-1]:
        ts[i] = min(ts[i], _robust_time(cands[i][2], args, warmup=2, reps=2, iters=40))
    return cands[min(order, key=ts.__getitem__)]


def _run_dense(entry, args):
    """Mode-split steady-state launch. entry = [raw @flyc.jit launch, cfg, compiled].
    Eager: run the one-time flyc.compile'd object (skips @flyc.jit's per-call drift-
    check + arg-hash, and the per-call arg-key rebuild). Capture: run the raw closure
    (a flyc.compile'd object regresses under CUDA-graph capture)."""
    if torch.cuda.is_current_stream_capturing():
        entry[0](*args)
    else:
        if entry[2] is None:
            entry[2] = flyc.compile(entry[0], *_static_layout(args))
        entry[2](*args)


def _dense_operand(t: torch.Tensor) -> torch.Tensor:
    # The kernels name their operand element type (F8_IR_t), so no i8 view is needed.
    return t if t.is_contiguous() else t.contiguous()


def _scalar_scale(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tensorwise scalar -> length-1 fp32 buffer (no broadcast): the kernel applies the
    single value per-tensor, so only an fp32/device cast is needed. A conforming buffer is
    returned as is, since .to()/.reshape() are no-ops on it but still cost two dispatches."""
    if scale.dtype is torch.float32 and scale.shape == (1,) and scale.device == device:
        return scale
    assert scale.numel() == 1, f"per-tensor expects scalar, got {scale.shape}"
    return scale.to(dtype=torch.float32, device=device).reshape(1)


# Wide and narrow grids want opposite bands over B's N-stripe, so both are raced.
_NN_CANDIDATES = [
    (256, 4, 0, 8, 32),
    (256, 4, 0, 1, 32),
    (256, 2, 0, 2, 32),
    (256, 1, 0, 4, 32),
    (128, 4, 0, 8, 48),
    (256, 4, 4, 8, 32),
    (256, 4, 2, 8, 32),
    (256, 8, 4, 8, 32),
]
_NN_AUTOTUNE_CACHE: dict = {}


def _autotune_nn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False, i64_traverse=False):
    """First-call bench of the NN candidates, best (launch, cfg) cached by (M,N,K); each is
    finite-checked before it is timed (see _pick_dense_candidate). ``i64_traverse`` re-bases
    B's SRD per load, lifting the k*n < 2^32 cap."""
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16, i64_traverse)
    if key in _NN_AUTOTUNE_CACHE:
        return _NN_AUTOTUNE_CACHE[key]
    out_view = args[2]
    cands = []
    for bm, gm, gn, xcd, ag in _NN_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is
        # bounded by c_m (StoreCPerTensor clamp) and the global SRD (HW OOB
        # clamp on the A G2S load), so no even-tiling filter is needed.
        try:
            # inline-asm ds_read_b64_tr_b8 on by default (drops the per-K-iter
            # compiler-auto vmcnt(0) drains).
            launch = _compile_dense_nn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                group_n=gn,
                num_xcd=xcd,
                agpr_alloc=ag,
                b_inline_asm_load=True,
                vmcnt_hint=2,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
                i64_traverse=i64_traverse,
                pair_n=N % 2 == 0 and not out_fp16,
                col_safe=N % 256 == 0,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            cands.append([launch, (bm, gm, gn, xcd, ag), c])  # c: compiled, reused eager
        except Exception:
            continue
    if not cands:
        raise RuntimeError(f"NN autotune found no working cfg for ({M},{N},{K})")
    best = _pick_dense_candidate(cands, args)
    _NN_AUTOTUNE_CACHE[key] = best
    return best


# NT per-shape autotune candidates (BLOCK_M, GROUP_M, num_xcd, AGPR). GROUP_M
# and num_xcd are fixed at the analytic L2 optimum; only BLOCK_M and AGPR are
# benched (occupancy/compute effects the hot-cache bench measures reliably).
_NT_CANDIDATES = [
    (256, 4, 8, 64),
    (256, 4, 8, 32),
    (128, 4, 8, 48),
    (128, 4, 8, 32),
]
_NT_AUTOTUNE_CACHE: dict = {}


def _autotune_nt_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench of the NT candidates, best (launch, cfg) cached by (M,N,K); each is
    finite-checked before it is timed (see _pick_dense_candidate)."""
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NT_AUTOTUNE_CACHE:
        return _NT_AUTOTUNE_CACHE[key]
    out_view = args[2]
    cands = []
    for bm, gm, xcd, ag in _NT_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is
        # bounded by c_m (StoreCPerTensor clamp) and the global SRD (HW OOB
        # clamp on the A G2S load), so no even-tiling filter is needed.
        try:
            launch = _compile_dense_nt(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                agpr_alloc=ag,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            cands.append([launch, (bm, gm, xcd, ag), c])  # c: compiled, reused eager
        except Exception:
            continue
    if not cands:
        raise RuntimeError(f"NT autotune found no working cfg for ({M},{N},{K})")
    best = _pick_dense_candidate(cands, args)
    _NT_AUTOTUNE_CACHE[key] = best
    return best


# TN dispatch: the 4-wave whole-loop above, with the 8-wave kernel for shapes it cannot take.

_TN_WAVE4_CACHE: dict = {}
_TN4_WS_CACHE: dict = {}
# TN whole-loop band: tiles visited before moving on, sized to keep operand slabs L2-resident.
_TN_WAVE4_BAND = (4, 2)
_TN_WAVE4_TALL_N = 8  # N width once N is the grid's short axis (see _tn_wave4_band)


def _tn_wave4_supported(N: int, K: int, i64_traverse: bool) -> bool:
    """The whole-loop reaches each operand through one per-tile buffer SRD and its split-K
    reduce moves 128-bit vectors, so spans needing the per-load i64 re-base, vector-unaligned
    output widths, and a K too short for one main-loop pass all go to the 8-wave kernel."""
    return (not i64_traverse) and N % _TN4_RED_VEC == 0 and ceildiv(K, _TN4_BLOCK_K) >= _TN4_PHASES


def _tn_wave4_band(M, N):
    """(group_m, group_n) for one TN output shape. On a grid taller than it is wide the N
    extent is a handful of tiles, so widen the band until one A slab serves every N stripe
    while the stripes still fit an XCD's L2 slice."""
    group_m, group_n = _TN_WAVE4_BAND
    if ceildiv(N, _TN4_BLOCK) < ceildiv(M, _TN4_BLOCK):
        group_n = _TN_WAVE4_TALL_N
    return group_m, min(group_n, ceildiv(N, _TN4_BLOCK))


def _tn_wave4_workspace(M, N, bands, device, dtype, out):
    """Scratch for the split-K slice partials: ``bands`` bands of M rows at C's row pitch, so
    a slice store only swaps the band SRD's base. Kept per (shape, device) because a fixed
    buffer is what CUDA-graph capture needs; no window -> pass C and allocate nothing."""
    if bands == 0:
        return out
    key = (device.index, dtype, bands * M, N)
    ws = _TN4_WS_CACHE.get(key)
    if ws is None:
        ws = torch.empty((bands * M, N), device=device, dtype=dtype)
        _TN4_WS_CACHE[key] = ws
    return ws


def _tn_wave4_dispatch(M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """Compile (or cache-hit) the 4-wave whole-loop launch for one TN problem, and the
    band count its split-K window needs. Returns ``(entry, bands)``."""
    key = (M, N, K, cbsz, blgp, out_fp16)
    hit = _TN_WAVE4_CACHE.get(key)
    if hit is None:
        group_m, group_n = _tn_wave4_band(M, N)
        launch, bands = _compile_dense_tn_wave4(M, N, K, group_m, group_n, cbsz, blgp, out_fp16)
        hit = ([launch, (_TN4_BLOCK, group_m, group_n, 1), None], bands)
        _TN_WAVE4_CACHE[key] = hit
    return hit


_TN_AUTOTUNE_CACHE: dict = {}


def _autotune_tn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False, i64_traverse=False):
    """First-call bench TN candidates, cache best (launch, cfg) by (M,N,K).

    1D GROUP_M=4 with num_xcd 8 vs 1 (XCD-aware PID remap); large
    (HBM-streaming) shapes expose the per-XCD L2 reuse on the hot bench,
    L2-resident shapes pick num_xcd=1. ``i64_traverse`` re-bases A's and B's
    SRDs per load (lifts the k*m / k*n < 2^32 cap; threaded to _compile_dense_tn).
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16, i64_traverse)
    if key in _TN_AUTOTUNE_CACHE:
        return _TN_AUTOTUNE_CACHE[key]
    # BLOCK_M fixed at 256: halving it to fill the grid halves an already feed-bound tile.
    bm = 256
    out_view = args[2]
    cands = []
    for xcd in (8, 1):
        try:
            launch = _compile_dense_tn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=4,
                vmcnt_hint=3,
                group_n=0,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
                i64_traverse=i64_traverse,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            cands.append([launch, (bm, 4, 0, xcd), c])  # c: compiled, reused eager
        except Exception:
            continue
    if not cands:
        raise RuntimeError(f"TN autotune found no working cfg for ({M},{N},{K})")
    best = _pick_dense_candidate(cands, args)
    _TN_AUTOTUNE_CACHE[key] = best
    return best


def gemm_fp8_tensorwise_flydsl_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Dense FP8 GEMM, per-tensor scaling. Inputs E4M3/E5M2/hybrid, out bf16/fp16,
    arbitrary K (native K-tail). Dispatch by (trans_a, trans_b): NT (F,T), NN
    (F,F, dgrad), TN (T,F) run native; TT (T,T) unsupported. trans_c=True returns
    out.t().contiguous()."""
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"FlyDSL wrapper emits bf16 or fp16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2
    # Element-count threshold past which a contraction-traversal operand's 32-bit
    # soffset wraps (fp8 = 1 byte/elem). At/above it the kernel re-bases the SRD per
    # load in i64; below it the cheaper fixed-base + 32-bit soffset path is used.
    cap = 2**32
    # Per-operand fp8 format -> MFMA cbsz(srcA)/blgp(srcB): 0=E4M3, 1=E5M2.
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    # fp16 vs bf16 output dtype for StoreCPerTensor (both from the f32 accumulator).
    out_fp16 = out_dtype == torch.float16

    if trans_a and (not trans_b):
        # TN native: A [K, M], B [K, N]. Math C = A^T @ B.
        K_a, M = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"TN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        device = a.device
        a_scale_v = _scalar_scale(a_scale_inv, device)
        b_scale_v = _scalar_scale(b_scale_inv, device)
        out = a.new_empty((M, N), dtype=out_dtype)
        # TN both operands traverse K: span k*m / k*n past 2^32 fp8 needs the
        # per-load i64 SRD re-base (else the 32-bit soffset wraps).
        i64_tr = (K * M >= cap) or (K * N >= cap)
        if _tn_wave4_supported(N, K, i64_tr):
            entry, bands = _tn_wave4_dispatch(M, N, K, cbsz, blgp, out_fp16)
            wargs = (
                _dense_operand(a),
                _dense_operand(b),
                out,
                a_scale_v,
                b_scale_v,
                _tn_wave4_workspace(M, N, bands, device, out_dtype, out),
                _raw_stream(device.index),
            )
            _run_dense(entry, wargs)
        else:
            args = (
                _dense_operand(a),
                _dense_operand(b),
                out,
                a_scale_v,
                b_scale_v,
                M,
                N,
                _raw_stream(device.index),
            )
            _run_dense(_autotune_tn_dispatch(args, M, N, K, cbsz, blgp, out_fp16, i64_tr), args)
        if trans_c:
            return out.t().contiguous()
        return out

    # Dispatch by layout.
    if (not trans_a) and (not trans_b):
        # NN native: A [M, K], B [K, N].
        M, K_a = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"NN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        device = a.device
        a_scale_v = _scalar_scale(a_scale_inv, device)
        b_scale_v = _scalar_scale(b_scale_inv, device)
        out = a.new_empty((M, N), dtype=out_dtype)
        # NN: per-shape runtime autotune over the candidate tiles, caches by
        # (M,N,K). Build args before autotune (it benches against them).
        args = (
            _dense_operand(a),
            _dense_operand(b),
            out,
            a_scale_v,
            b_scale_v,
            M,
            N,
            _raw_stream(device.index),
        )
        # NN: only B[K,N] traverses K; k*n past 2^32 fp8 needs the i64 re-base.
        i64_tr = K * N >= cap
        _run_dense(_autotune_nn_dispatch(args, M, N, K, cbsz, blgp, out_fp16, i64_tr), args)
    elif (not trans_a) and trans_b:
        # NT native: A [M, K], B [N, K] (B^T storage of [K, N]).
        M, K_a = a.shape
        N, K_b = b.shape
        assert K_a == K_b, f"NT K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        device = a.device
        a_scale_v = _scalar_scale(a_scale_inv, device)
        b_scale_v = _scalar_scale(b_scale_inv, device)
        out = a.new_empty((M, N), dtype=out_dtype)
        # NT: per-shape runtime autotune over the 8w/v3 candidate tiles, caches
        # by (M,N,K). Build args before autotune (it benches against them).
        args = (
            _dense_operand(a),
            _dense_operand(b),
            out,
            a_scale_v,
            b_scale_v,
            M,
            N,
            _raw_stream(device.index),
        )
        _run_dense(_autotune_nt_dispatch(args, M, N, K, cbsz, blgp, out_fp16), args)
    else:
        raise NotImplementedError(
            f"FlyDSL fp8 GEMM does not support the TT layout (trans_a={trans_a}, trans_b={trans_b})."
        )
    if trans_c:
        return out.t().contiguous()
    return out
