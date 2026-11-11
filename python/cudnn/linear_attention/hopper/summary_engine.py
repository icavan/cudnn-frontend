# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct KDA span summaries on SM90, independent of ordinary apply kernels.

One warp owns a state/transition row throughout a sequence. A 512-thread CTA
shares each 16-token tile's keys and decay factors across 16 state rows, keeping
the direct FP32 recurrence's operation order. Transition rows start from
identity and never read V/dO or the caller's boundary state. Subtracting
ordinary forwards F(I,V)-F(0,V) instead amplifies BF16 operand rounding.
The optional Tensor Core plan uses a separate mixed-precision BT16 recurrence
and caller-owned coefficients; the FP32 plan remains the default.
"""

import math
import threading
from pathlib import Path

import cudnn
from cudnn.engines.base import BaseEngine, CompiledPlan, bind_ports
from cudnn.frost import buffers
from cudnn.frost.device import build_device, current_device, device_context

from ..graph_analyzer import analyze, to_buffer_dtype

_LOCK = threading.Lock()
_LIBRARIES = {}
_TC_LIBRARIES = {}


def _library(device):
    from ..cake import compiler

    with _LOCK:
        if device not in _LIBRARIES:
            # Override CAKE's fast-math default: state decay stays FP32, with
            # subnormals preserved. Source/options/NVRTC version key its cubin.
            _LIBRARIES[device] = compiler.KernelLibrary(Path(__file__).with_name("cuda_kernels") / "kda_summary_sm90.cu", "sm_90", device, fast_math=False)
        return _LIBRARIES[device]


def check_contract(facts, node, name="kda_summary_hopper_cuda"):
    """Reject unsupported features before compilation or any output write."""

    def decline(reason):
        raise NotImplementedError(f"{name}: {reason}")

    if facts is None or facts.op != "KDA_SUMMARY":
        decline("requires one KDA_SUMMARY or KDA_SUMMARY_BWD node")
    if facts.invalid:
        decline(facts.invalid)
    if not facts.thd_layout or not facts.uniform_io or facts.io_dtype != cudnn.data_type.BFLOAT16:
        decline("requires BF16 THD inputs")
    if facts.d_qk != 128 or facts.d_v != 128 or not (facts.h_q == facts.h_k == facts.h_v == facts.h_o):
        decline("requires K=V=128 and equal head counts")
    if facts.h_o <= 0 or facts.n_seq < 0 or facts.total_t < 0 or facts.h_o * facts.n_seq * 16 >= 2**31:
        decline("invalid dimensions or CUDA grid exceeds signed-int32 range")
    if not facts.gates_at_ho or facts.gate_channels != 128:
        decline("requires g [T,H,128] and beta [T,H]")
    if facts.use_qk_l2norm or facts.safe_gate or facts.use_beta_sigmoid or facts.allow_neg_eigval or facts.beta_guard or facts.gate_domain != "log":
        decline("requires caller-normalized k/q, log-space g and activated beta")
    if facts.checkpoint_every_n_tokens or facts.output_context or facts.input_context or facts.has_state_indices or facts.overwrite_initial_state:
        decline("checkpoints, context and indexed/in-place state are unsupported")
    inputs = {"q", "k", "g", "beta", "dO", "cu_seqlens", "d_final_state"} if facts.is_bwd else {"k", "v", "g", "beta", "cu_seqlens", "initial_state"}
    outputs = {"d_initial_state", "transition"} if facts.is_bwd else {"final_state", "transition"}
    if set(node.inputs) - inputs or set(node.outputs) - outputs:
        decline("unsupported input/output ports")
    t, h, n = facts.total_t, facts.h_o, facts.n_seq
    shapes = {name: (t, h, 128) for name in ("q", "k", "v", "g", "dO")}
    shapes.update(beta=(t, h), cu_seqlens=(n + 1,))
    shapes.update({name: (n, h, 128, 128) for name in ("initial_state", "d_final_state", "final_state", "d_initial_state", "transition")})
    for name, tensor in {**node.inputs, **node.outputs}.items():
        dtype = tensor.get_data_type()
        wants = (
            (cudnn.data_type.INT32, cudnn.data_type.INT64)
            if name == "cu_seqlens"
            else ((cudnn.data_type.BFLOAT16,) if name in ("q", "k", "v", "dO") else (cudnn.data_type.FLOAT,))
        )
        if dtype not in wants:
            decline(f"{name} requires {wants}, got {dtype}")
        if tuple(tensor.dim) != shapes[name]:
            decline(f"{name} must have shape {shapes[name]}")
        stride = tensor.stride
        if stride and (any(s < 0 for s in stride) or (tensor.dim[-1] > 1 and stride[-1] != 1)):
            decline(f"{name} requires nonnegative strides and unit innermost stride")


class KdaSummaryHopperPlan(CompiledPlan):
    """One device's compiled recurrence over caller-owned inputs and outputs.

    Geometry is fixed by the graph; addresses and outer strides are rebound
    per execute. Kernel parameters are private to each invocation. The plan
    keeps no device scratch, and launches only after every binding validates.
    """

    takes_variant_pack = True
    plan_name = "kda_summary_hopper_cuda"
    rows_per_cta = 16
    threads = 512
    kernel_prefix = "cudnn_kda_summary_"

    def _load_library(self):
        return _library(self.device)

    def __init__(self, graph, facts):
        node = next(iter(graph.nodes))
        self.device = current_device()
        self.is_bwd = facts.is_bwd
        self.heads = facts.h_o
        self.seqs = facts.n_seq
        self.scale = float(facts.scale) if facts.scale is not None else 128**-0.5
        self.geometry = {name: (tuple(t.dim), to_buffer_dtype(t.get_data_type())) for name, t in {**node.inputs, **node.outputs}.items()}
        self.outputs = set(node.outputs)
        self.cu64 = facts.cu_dtype == cudnn.data_type.INT64
        self.ports = None
        self.library = self._load_library()
        self.function = self.library.function(self.kernel_prefix + ("bwd" if self.is_bwd else "fwd"))

    def get_workspace_size(self):
        return 0

    def execute(self, graph, variant_pack, ctx):
        from ..cake import compiler

        if variant_pack.device != self.device:
            raise ValueError(f"{self.plan_name}: build one plan per device (expected cuda:{self.device})")
        if ctx.context is not None:
            raise ValueError(f"{self.plan_name}: no context buffer is supported")
        if self.ports is None:
            (slots,) = bind_ports(graph, variant_pack).values()
            self.ports = tuple(zip(list(slots.inputs) + list(slots.outputs), list(slots.inputs.values()) + list(slots.outputs.values())))
        views = variant_pack.operands([i for _, i in self.ports])
        x, spans = {}, []
        for (name, slot), value in zip(self.ports, views):
            shape, dtype = self.geometry[name]
            if tuple(value.shape) != shape or buffers.dtype_name(value) != dtype:
                raise ValueError(f"{self.plan_name}: {name} requires shape {shape} and dtype {dtype}")
            strides = tuple(value.stride())
            if any(s < 0 or s >= 2**63 for s in strides) or (shape[-1] > 1 and strides[-1] != 1):
                raise ValueError(f"{self.plan_name}: {name} requires nonnegative strides and unit innermost stride")
            itemsize = buffers.DTYPE_ITEMSIZE[dtype]
            span = (1 + sum((d - 1) * s for d, s in zip(shape, strides))) * itemsize if math.prod(shape) else 0
            if span >= 2**63:
                raise ValueError(f"{self.plan_name}: {name} storage span exceeds signed-int64 addressing")
            ptr = value.data_ptr()
            if (span and not ptr) or ptr % itemsize:
                raise ValueError(f"{self.plan_name}: {name} is null or misaligned")
            capacity = variant_pack.observed_bytes(slot)
            if capacity >= 0 and capacity < span:
                raise ValueError(f"{self.plan_name}: {name} storage is too small")
            device = tuple(variant_pack.observed_device(slot))
            if device not in ((-1, -1), (2, self.device)):
                raise ValueError(f"{self.plan_name}: {name} must be on cuda:{self.device}")
            if name in self.outputs:
                extent = 1
                for stride, dim in sorted((s, d) for s, d in zip(strides, shape) if d > 1):
                    if stride < extent:
                        raise ValueError(f"{self.plan_name}: {name} has overlapping output rows")
                    extent = stride * dim
            spans.append((name, ptr, ptr + span))
            x[name] = (ptr, (0,) * (4 - len(strides)) + strides)
        for i, (name, start, end) in enumerate(spans):
            for other, lo, hi in spans[:i]:
                if start < end and lo < hi and start < hi and lo < end and (name in self.outputs or other in self.outputs):
                    raise ValueError(f"{self.plan_name}: {name} overlaps {other}")
        params = compiler.Params()
        names = (
            "q",
            "k",
            "dO" if self.is_bwd else "v",
            "g",
            "beta",
            "cu_seqlens",
            "d_final_state" if self.is_bwd else "initial_state",
            "d_initial_state" if self.is_bwd else "final_state",
            "transition",
        )
        for name in names:
            ptr, strides = x.get(name, (0, (0, 0, 0, 0)))
            params.ptr(ptr)
            for stride in strides:
                params.i64(stride)
        params.i32(self.heads).i32(int(self.cu64)).f32(self.scale)
        self._launch(params, x, spans, variant_pack, ctx)

    def _launch(self, params, x, spans, variant_pack, ctx):
        from ..cake import compiler

        groups = (256 if "transition" in x else 128) // self.rows_per_cta
        if self.seqs:
            with device_context(self.device):
                compiler.launch(self.function, (self.seqs * self.heads * groups, 1, 1), (self.threads, 1, 1), 0, int(ctx.stream or 0), params, self.plan_name)


class KdaSummaryHopperCudaEngine(BaseEngine):
    """SM90 forward/reverse summary provider for the existing KDA graph nodes."""

    name = "kda_summary_hopper_cuda"
    behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)
    plan_class = KdaSummaryHopperPlan

    def check_support(self, graph):
        facts = graph._facts_for(analyze)
        check_contract(facts, next(iter(graph.nodes), None), self.name)
        if buffers.current_sm() != 90:
            raise NotImplementedError(f"{self.name}: requires SM90")
        try:
            from ..cake import compiler

            compiler.cuda_include_dirs()
        except Exception as exc:
            raise NotImplementedError(f"{self.name}: requires NVRTC and CUDA headers: {exc}") from exc

    def build_plan(self, graph, plan, ctx=None):
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return self.plan_class(graph, graph._facts_for(analyze))


class KdaSummaryHopperTensorCorePlan(KdaSummaryHopperPlan):
    """BT16 direct-M BF16 MMA summaries with FP32 state carry.

    The binding, stream and empty-span contracts match the FP32 plan. A parallel
    coefficient pass precedes the recurrence on the caller's stream. Workspace
    is caller-owned, 32-byte aligned and disjoint from every nonempty operand.
    For T tokens, N sequences and H heads, it uses (T//16+N)*H tiles of 10400
    forward or 15456 reverse bytes; empty inputs require no workspace. Device
    offsets may change within those declared capacities under graph replay.
    Projection/update operands are rounded to BF16. A device-side per-tile
    range check uses the direct FP32 recurrence for unsafe gate prefixes.
    This changes numerical results; select this plan explicitly when the
    measured mixed-precision error meets the caller's accuracy requirements.
    """

    plan_name = "kda_summary_hopper_tc"
    rows_per_cta = 16
    threads = 128
    kernel_prefix = "cudnn_kda_summary_tc_"

    def __init__(self, graph, facts):
        super().__init__(graph, facts)
        self.chunks = facts.total_t // 16 + facts.n_seq
        self.workspace_bytes = self.chunks * self.heads * (15456 if self.is_bwd else 10400) if facts.total_t and self.seqs else 0
        self.prepare = self.library.function(self.kernel_prefix + ("prepare_bwd" if self.is_bwd else "prepare_fwd"))

    def get_workspace_size(self):
        return self.workspace_bytes

    def _launch(self, params, x, spans, variant_pack, ctx):
        from cudnn.frost.workspace import Workspace
        from ..cake import compiler

        ptr = 0
        if self.workspace_bytes:
            Workspace.over(variant_pack, self.workspace_bytes, self.plan_name, align=32)
            # VariantPack records the scratch extent, but not its device. For
            # typed buffers also reject host/foreign-device storage before the
            # first producer launch. Bare addresses retain the graph API's
            # caller-owned device/capacity contract.
            if type(ctx.workspace) is not int:
                device = getattr(ctx.workspace, "__dlpack_device__", None)
                if device is not None and tuple(device()) != (2, self.device):
                    raise ValueError(f"{self.plan_name}: workspace must be on cuda:{self.device}")
                Workspace(ctx.workspace, self.workspace_bytes, self.plan_name, align=32, device=self.device)
            ptr = variant_pack.workspace
            for name, start, end in spans:
                if start < end and ptr < end and start < ptr + self.workspace_bytes:
                    raise ValueError(f"{self.plan_name}: workspace overlaps {name}")
        params.ptr(ptr).i64(self.chunks).i32(self.seqs)
        if self.seqs:
            groups = (256 if "transition" in x else 128) // self.rows_per_cta
            with device_context(self.device):
                if self.workspace_bytes:
                    compiler.launch(self.prepare, (self.chunks * self.heads, 1, 1), (128, 1, 1), 0, int(ctx.stream or 0), params, self.plan_name)
                compiler.launch(self.function, (self.seqs * self.heads * groups, 1, 1), (self.threads, 1, 1), 0, int(ctx.stream or 0), params, self.plan_name)

    def _load_library(self):
        from ..cake import compiler

        with _LOCK:
            if self.device not in _TC_LIBRARIES:
                _TC_LIBRARIES[self.device] = compiler.KernelLibrary(
                    Path(__file__).with_name("cuda_kernels") / "kda_summary_tc_sm90.cu", "sm_90", self.device, fast_math=False
                )
            return _TC_LIBRARIES[self.device]


class KdaSummaryHopperTensorCoreEngine(KdaSummaryHopperCudaEngine):
    """Optional BF16 Tensor Core provider; the FP32 summary retains precedence."""

    name = "kda_summary_hopper_tc"
    plan_class = KdaSummaryHopperTensorCorePlan
    numerical_notes = (cudnn.numerical_note.TENSOR_CORE, cudnn.numerical_note.DOWN_CONVERT_INPUTS)

    def check_support(self, graph):
        super().check_support(graph)
        from ..cake import compiler

        facts = graph._facts_for(analyze)
        chunks = facts.total_t // 16 + facts.n_seq
        if chunks * facts.h_o >= 2**31 or chunks * facts.h_o * 15456 >= 2**63:
            raise NotImplementedError(f"{self.name}: coefficient grid or workspace exceeds the addressing range")
        includes = compiler.cuda_include_dirs()
        if not all(any((path / header).is_file() for path in includes) for header in ("mma.h", "crt/mma.h")):
            raise NotImplementedError(f"{self.name}: requires CUDA MMA headers; set CUDA_HOME to a complete CUDA toolkit")
