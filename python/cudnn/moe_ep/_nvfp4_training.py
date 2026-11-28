# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-capacity Blackwell NVFP4 training with deterministic 256-route K splits."""

from __future__ import annotations

from dataclasses import fields

from cuda.bindings import driver as cuda
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ._config import _required_padded_rows, MoeEpFc1WeightLayout
from ._math import positive_fp32
from ._nvfp4 import TileGemm
from ._types import (
    MoeEpNvfp4BackwardWeights,
    MoeEpNvfp4ForwardWeights,
    MoeEpNativeWeightStorageMode,
    MoeEpTrainingBackwardOutputs,
    MoeEpTrainingForwardOutputs,
    MoeEpTrainingWgradOperands,
    MoeFormat,
)
from ._validation import _validate_named_buffers, validate_readonly_against_outputs, validate_training_input, validate_training_non_aliasing

TILE = 256
_FORWARD_SCALES = ("input_norm", "fc1_norm_const", "fc31_alpha", "fc2_alpha")
_ROUTE_ROW_FIELDS = ("route_slots", "fc1_preact", "fc1_a", "fc1_sfa", "saved_a", "saved_sfa")


def route_table(ids, first, experts, rows):
    """Map padded expert-local rows to original rank/token/slot, or -1 for padding."""
    keys, order = torch.sort(ids.flatten(), stable=True)
    boundaries = torch.searchsorted(keys, torch.arange(first, first + experts + 1, device=ids.device, dtype=ids.dtype))
    counts = boundaries.diff()
    ends = (((counts + TILE - 1) // TILE) * TILE).cumsum(0)
    row = torch.arange(rows, device=ids.device)
    expert = torch.searchsorted(ends, row, right=True).clamp_max(experts - 1)
    local = row - F.pad(ends, (1, 0))[expert]
    slot = order[(boundaries[expert] + local).clamp_max(order.numel() - 1)]
    return torch.where(local < counts[expert], slot, -1), counts, ends


class Nvfp4Training:
    def __init__(self, resolved, device):
        from cudnn.frost.buffers import cutedsl_requirement_error

        error = cutedsl_requirement_error("Blackwell NVFP4 MoE training")
        if error:
            raise RuntimeError(error)
        config = resolved.public_config
        self.path = config.data_path
        wgrad_norms = (
            positive_fp32(self.path.dgu_norm * self.path.x_wgrad_norm, "FC1 wgrad divisor"),
            positive_fp32(self.path.dy_norm * self.path.a_wgrad_norm, "FC2 wgrad divisor"),
        )
        self.encoding_norms = (self.path.x_wgrad_norm, self.path.dgu_norm, self.path.a_wgrad_norm, self.path.dy_norm)
        self.topology = resolved.topology
        self.group = config.parallel.ep_group
        self.resolved = resolved
        self.device = device
        model = config.model
        self.h, self.i, self.k = h, i, k = model.hidden_size, model.intermediate_size, model.top_k
        self.max_tokens = t = config.parallel.max_tokens_per_rank
        if t is None or t <= 0:
            raise ValueError("NVFP4 preparation requires a positive max_tokens_per_rank")
        e = self.topology.experts_per_rank
        if self.topology.ep_size not in (1, 8):
            raise ValueError("NVFP4 training supports EP1 or one EP8 NVLink domain")
        if h % 32 or i % 32:
            raise ValueError("NVFP4 tensor-core operands require H and I divisible by 32")
        if self.path.fc1_weight_layout is not MoeEpFc1WeightLayout.GATE_THEN_UP:
            raise ValueError("NVFP4 training expects logical gate-then-up weights (from_trtllm converts the 16-row physical interleave)")
        if self.path.output_format is not MoeFormat.BF16 or self.path.combine_format is not MoeFormat.BF16 or not self.path.apply_topk_in_fc1:
            raise ValueError("NVFP4 training requires BF16 output/combine and pre-FC2 routing")
        if config.training_weight_storage_mode is not MoeEpNativeWeightStorageMode.CONTIGUOUS or config.parallel.drop_on_overflow:
            raise ValueError("NVFP4 training requires contiguous experts and dense routing")
        raw = t * k * self.topology.ep_size
        needed = _required_padded_rows(raw, experts_per_rank=e, padding_block=TILE)
        self.rows = p = resolved.receive_capacity.physical_recv_pool_rows
        if p < needed or p % TILE:
            raise ValueError(f"NVFP4 physical_recv_pool_rows must cover {needed} rows in multiples of {TILE}")
        tiles = p // TILE
        u8, sf, f32 = torch.uint8, torch.float8_e4m3fn, torch.float32
        self.requirements = {}

        def spec(name, shape, dtype):
            stride = torch.empty(shape, device="meta").stride()
            self.requirements[name] = (shape, stride, dtype, 16)

        for name, dim in (("fc1_a", h), ("saved_a", i)):
            spec(name, (p, dim // 2), u8)
        for name, dim in (("fc1_sfa", h), ("saved_sfa", i)):
            spec(name, (p, dim // 16), sf)
        for name, dim in (("fc1_wgrad_a", h), ("fc1_b", 2 * i), ("fc2_a", i), ("fc2_b", h)):
            spec(name, (tiles, dim, TILE // 2), u8)
        for name, dim in (("fc1_wgrad_sfa", h), ("fc1_sfb", 2 * i), ("fc2_sfa", i), ("fc2_sfb", h)):
            spec(name, (tiles, dim, TILE // 16), sf)
        spec("activation_norms", (e, 2), f32)
        spec("expert_offsets", (e,), torch.int64)
        spec("valid_route_counts", (e,), torch.int64)
        spec("wgrad_expert_offsets", (e,), torch.int64)
        spec("wgrad_valid_route_counts", (e,), torch.int64)
        spec("route_slots", (p,), torch.int64)
        spec("fc1_preact", (p, 2 * i), f32)
        spec("output", (t, h), torch.bfloat16)
        spec("grad_activation", (t, h), torch.bfloat16)
        spec("dprob", (t, k), f32)
        self.buffers = {
            name: torch.empty(self.requirements[name][0], device=device, dtype=self.requirements[name][2]) for name in ("output", "grad_activation")
        }
        comm_sms = 4 if self.topology.ep_size > 1 else 0
        self.gemms = {
            shape: TileGemm(*shape, device, communication_sms=comm_sms) for shape in sorted({(TILE, 2 * i, h), (TILE, h, i), (TILE, i, h), (TILE, h, 2 * i)})
        }

        direct_shapes = {(i, h), (h, 2 * i)}
        if h % 128 or i % 128:
            direct_shapes.update(((2 * i, h), (h, i)))
        for n, k_dim in sorted(direct_shapes):
            self.gemms[TILE, n, k_dim].prepare_dgrad(e, n, k_dim, device)

        from ._nvfp4_wgrad import GroupedWgrad

        self.group_ends = torch.zeros(e, dtype=torch.int32, device=device)
        self.grouped = [GroupedWgrad(m, n, p, e, self.group_ends, device, t * k // e, norm) for (m, n), norm in zip(((2 * i, h), (h, i)), wgrad_norms)]
        from ._nvfp4_pack import PackTile, RepackSaved, SwiGLUBackward, SwiGLUForward

        self.repack_x = RepackSaved(h, 0)
        self.repack_a = RepackSaved(i, 1)
        self.pack_x = PackTile(h, torch.bfloat16, False, False, forward=True)
        self.pack_a = PackTile(i, torch.float32, False, False, forward=True)
        self.activation_fwd = SwiGLUForward(i, self.path.gate_up_clamp, device)
        self.activation_bwd = SwiGLUBackward(i, self.path.gate_up_clamp, device)
        self.tile_starts = torch.arange(0, self.rows, TILE, device=device)
        self.tile_experts = torch.empty_like(self.tile_starts)

        self.pack_feature = {key: PackTile(*key, False, False) for key in {(h, torch.bfloat16), (2 * i, torch.float32)}}
        self.pack_token = {key: PackTile(*key[:2], True, key[2]) for key in {(h, torch.bfloat16, False), (2 * i, torch.float32, False)}}
        self.norm = {n: torch.tensor([n], dtype=torch.float32, device=device) for n in self.encoding_norms}
        self.feature_scratch = {
            dim: (torch.empty((TILE, dim // 2), dtype=torch.uint8, device=device), torch.empty((TILE, dim // 16), dtype=torch.float8_e4m3fn, device=device))
            for dim in (h, 2 * i)
        }
        self.weight_stream = torch.cuda.Stream(device=device)
        self.weight_join = torch.cuda.Event()
        self.weight_done = torch.cuda.Event(enable_timing=True)
        self.dx_ready = torch.cuda.Event(enable_timing=True)
        from ._nvfp4_peer import TileExchange

        self.exchange = TileExchange(self)
        from ._nvfp4_forward import ExpertForward

        self.grouped_forward = ExpertForward(self) if h % 128 == 0 and i % 128 == 0 else None
        return_rows = 2 * TILE
        self.dx = torch.empty((return_rows, h), device=device, dtype=torch.float32)
        self.dp = torch.empty((return_rows, 1), device=device, dtype=torch.float32)

        # Process full windows and the final live tail with device-side
        # loop bounds. Both the graph size and the window scratch are bounded.
        from ._nvfp4_schedule import WindowRoutes, RouteConditions, WindowBindings

        self.conditions = RouteConditions(device)
        # Use longer transfer/MMA groups once balanced routing averages four
        # or more 256-row tiles per local expert. Skew still uses live bounds.
        large_batch = t * k // e >= 1024 and tiles >= 64
        from ._nvfp4_batch import BackwardWindow

        self.bounded_groups = large_batch and i % 256 == 0
        # A long EP8 batch amortizes direct peer stores from the FC1 epilogue.
        routes_per_rank = t * k
        # Top-6 V4 has fewer routes than top-8 at the same context length.
        # At 512+ tiles, a dense expert (16+ tiles on average) amortizes
        # the 64-tile direct-store group and avoids the intermediate FP32 dX.
        long_return = routes_per_rank >= 262144 or (routes_per_rank >= 512 * TILE and routes_per_rank // e >= 16 * TILE)
        fc1_op = self.gemms[TILE, h, 2 * i].op
        self.direct_return = (
            self.bounded_groups
            and self.topology.ep_size == 8
            and h >= 1024
            and h % 128 == 0
            and long_return
            and fc1_op.cluster_shape_mn == (1, 1)
            and h % fc1_op.mma_tiler_mn[1] == 0
        )
        self.group_tiles = 64 if self.direct_return else 32
        if self.bounded_groups:
            # Expert padding adds roughly half a tile per local expert under
            # balanced routing. Only the captured group count uses this estimate.
            expected_tiles = t * k // TILE + e // 2
            group = self.group_tiles
            self.window = min(512, max(group, ((expected_tiles + group - 1) // group) * group))
            self.window_widths = (self.window, group) if self.window > group else (group,)
            batch_sizes = dict.fromkeys(self.window_widths, group)
            fallback = 1
            self.bounds = torch.empty(2, dtype=torch.int64, device=device)
            self.run_base, self.live_tiles = self.bounds.unbind()
        else:
            self.window = 64 if large_batch else 32 if tiles >= 32 else min(16, tiles)
            self.window_widths = (self.window, *(n for n in (32, 16, 4, 1) if n < self.window))
            batch_sizes = {n: min(n // 2, 16 if large_batch else 8) for n in self.window_widths if n % 2 == 0}
            fallback = max(n for n in self.window_widths if n % 2)
            self.window_ids = torch.arange(self.window, device=device)
            self.window_index = torch.empty_like(self.window_ids)
            self.run_base = torch.empty((), dtype=torch.int64, device=device)
            self.live_tiles = torch.empty_like(self.run_base)
        self.window_experts = torch.empty(self.window, dtype=torch.int64, device=device)
        self.window_inputs = {}
        self.window_outputs = {}
        for name in ("route_slots", "fc1_preact"):
            shape, _, dtype, _ = self.requirements[name]
            rows = TILE * (self.window if name == "route_slots" else fallback)
            self.window_inputs[name] = torch.empty((rows, *shape[1:]), device=device, dtype=dtype)
        for name in ("fc1_b", "fc1_sfb", "fc2_b", "fc2_sfb"):
            shape, _, dtype, _ = self.requirements[name]
            self.window_outputs[name] = torch.empty((fallback, *shape[1:]), device=device, dtype=dtype)

        self.read_window = WindowBindings(self.window_inputs, TILE, device, write=False)
        self.write_window = WindowBindings(self.window_outputs, 1, device, write=True)
        batches = {n: BackwardWindow(self, n) for n in sorted(set(batch_sizes.values()))}
        self.batches = {width: batches[n] for width, n in batch_sizes.items()}
        if self.bounded_groups:
            self.routes = WindowRoutes(self.read_window)
        self.dgrad_divisors = torch.ones(2, dtype=torch.float32, device=device)
        self.remaining_tiles = torch.empty_like(self.live_tiles)
        self.more_tiles = torch.empty((), dtype=torch.bool, device=device)
        # Eager uses the same device-side route count without a D2H read. The
        # two prepared graphs bind caller-owned records through a pointer table.
        eager_banks = tuple(self.gemms[TILE, n, kk].fallback_weights.view(torch.float4_e2m1fn_x2).permute(1, 2, 0) for n, kk in ((h, 2 * i), (i, h)))
        self.eager_graphs = {}
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        for overlap in (False, True):
            graph = torch.cuda.CUDAGraph()
            # torch.cuda.graph.__enter__ synchronizes the whole device. Explicit
            # capture preserves setup ordering without waiting for unrelated work.
            with torch.cuda.stream(capture_stream):
                graph.capture_begin()
                try:
                    self._live_windows(eager_banks, overlap)
                finally:
                    graph.capture_end()
            self.eager_graphs[overlap] = graph

    def close(self):
        with torch.cuda.device(self.device):
            self.eager_graphs.clear()
            self.conditions.close()
            self.exchange.close()

    def _live_windows(self, banks, overlap):
        record = MoeEpTrainingForwardOutputs(**self.window_inputs)
        out = MoeEpTrainingBackwardOutputs(**self.window_outputs, grad_activation=self.buffers["grad_activation"])
        self.run_base.zero_()
        for n in self.window_widths:
            # Include the final partial group only if it contains a live tile.
            threshold = n - self.group_tiles + 1 if self.bounded_groups else n
            torch.sub(self.live_tiles, self.run_base, out=self.remaining_tiles)
            torch.ge(self.remaining_tiles, threshold, out=self.more_tiles)
            with self.conditions.while_true(self.more_tiles) as update_condition:
                if self.bounded_groups:
                    self.routes.fn(
                        self.read_window.external,
                        self.tile_experts,
                        self.bounds,
                        record.route_slots,
                        self.window_experts,
                        n,
                        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                    )
                else:
                    self.read_window.copy(self.run_base, n, fields=1 if n in self.batches else None)
                    torch.add(self.window_ids[:n], self.run_base, out=self.window_index[:n])
                    torch.index_select(self.tile_experts, 0, self.window_index[:n], out=self.window_experts[:n])
                if n in self.batches:
                    self.batches[n].run(self, record, self.dgrad_divisors, banks, overlap, self.window_experts, n)
                else:
                    self._tile_run(record, out, self.dgrad_divisors, banks, overlap, n, self.window_experts)
                    self.write_window.copy(self.run_base, n)
                self.run_base.add_(n)
                torch.sub(self.live_tiles, self.run_base, out=self.remaining_tiles)
                torch.ge(self.remaining_tiles, threshold, out=self.more_tiles)
                update_condition()

    def public_symmetric_buffers(self):
        return {**self.buffers, "backward_input": self.exchange.dy, "dprob": self.exchange.dp}

    def validate_bundle(self, bundle, cls, *, retained=False):
        if not isinstance(bundle, cls):
            raise TypeError(f"expected {cls.__name__}")
        items = {f.name: getattr(bundle, f.name) for f in fields(cls)}
        if cls is MoeEpTrainingBackwardOutputs:
            if (bundle.wgrad_expert_offsets is None) != (bundle.wgrad_valid_route_counts is None):
                raise ValueError("supply both NVFP4 wgrad metadata buffers")
            if bundle.wgrad_expert_offsets is None:
                del items["wgrad_expert_offsets"], items["wgrad_valid_route_counts"]
        requirements = {name: self.requirements[name] for name in items}
        if retained and isinstance(bundle.route_slots, torch.Tensor):
            rows = bundle.route_slots.numel()
            if rows < TILE or rows > self.rows or rows % TILE:
                raise ValueError("retained NVFP4 record capacity must be a positive number of whole route tiles within the plan capacity")
            if rows < self.rows:
                for name in _ROUTE_ROW_FIELDS:
                    shape, stride, dtype, align = requirements[name]
                    requirements[name] = ((rows, *shape[1:]), stride, dtype, align)
        _validate_named_buffers(items, requirements, device=self.device)
        return items

    @torch.no_grad()
    def compact_record(self, record):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compact_training_record reads the live row count and allocates outside CUDA Graph capture")
        items = self.validate_bundle(record, MoeEpTrainingForwardOutputs)
        live = int(record.expert_offsets[-1].item())
        if not 0 <= live <= self.rows or live % TILE:
            raise ValueError("invalid padded expert end in NVFP4 forward record")
        # Keep a valid device address for kernel arguments on an empty expert
        # rank. The device bounds exclude this dummy tile.
        rows = max(TILE, live)
        return MoeEpTrainingForwardOutputs(**{name: (value[:rows] if name in _ROUTE_ROW_FIELDS else value).clone() for name, value in items.items()})

    def validate_inputs(self, value, ids, prob):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
            raise TypeError("NVFP4 training inputs must be BF16 tensors")
        return validate_training_input(
            self.resolved,
            "activation or grad_output",
            value,
            ids,
            prob,
            device=self.device,
            validate_expert_ids=self.resolved.public_config.validation_mode == "strict",
        )

    def check_weight(self, weights, backward):
        e, h, i = self.topology.experts_per_rank, self.h, self.i
        cls = MoeEpNvfp4BackwardWeights if backward else MoeEpNvfp4ForwardWeights
        if not isinstance(weights, cls):
            raise TypeError(f"expected {cls.__name__}")
        matrices = (
            ((weights.w1_transpose, (e, h, 2 * i)), (weights.w2_transpose, (e, i, h))) if backward else ((weights.fc1, (e, 2 * i, h)), (weights.fc2, (e, h, i)))
        )
        items = {}
        for idx, (matrix, shape) in enumerate(matrices):
            if matrix.format is not MoeFormat.NVFP4 or matrix.axis != 2 or matrix.shape != shape or matrix.device != self.device:
                raise ValueError(f"expected K-packed NVFP4 expert weights of shape {shape} on {self.device}")
            items[f"payload{idx}"], items[f"sf{idx}"] = matrix.data, matrix.scale
        if not backward:
            for name in _FORWARD_SCALES:
                t = getattr(weights, name)
                if t.device != self.device or t.dtype != torch.float32 or (t.numel() != 1 if name == "input_norm" else t.shape != (e,)):
                    raise ValueError(f"invalid FP32 checkpoint scale {name} for {e} experts")
        return items

    def gather(self, tensor, fill=0):
        padded = F.pad(tensor, (0, 0, 0, self.max_tokens - tensor.shape[0]), value=fill)
        if self.topology.ep_size == 1:
            return padded
        result = padded.new_empty((self.max_tokens * self.topology.ep_size, padded.shape[1]))
        dist.all_gather_into_tensor(result, padded, group=self.group)
        return result

    def tiles(self, ends):
        experts = torch.searchsorted(ends, self.tile_starts, right=True, out=self.tile_experts)
        experts.clamp_max_(self.topology.experts_per_rank - 1)
        for tile in range(self.rows // TILE):
            yield tile, slice(tile * TILE, (tile + 1) * TILE), experts[tile : tile + 1]

    def combine(self, routes, slots, dest, count, reduce_slots=True):
        global_tokens = self.max_tokens * self.topology.ep_size
        sentinel = global_tokens * self.k
        row = torch.arange(self.rows, device=slots.device)
        # Valid routes own distinct slots. Give each padding writer a private
        # index as well, so scatter never writes one address from multiple lanes.
        index = torch.where(slots >= 0, slots, sentinel + row)
        inverse = torch.full((sentinel + self.rows,), self.rows, dtype=torch.long, device=slots.device)
        inverse.scatter_(0, index, row)
        values = F.pad(routes, (0, 0, 0, 1)).index_select(0, inverse[:sentinel]).view(global_tokens, self.k, -1)
        if self.topology.ep_size != 1:
            dist.all_reduce(values, group=self.group)
        begin = self.topology.ep_rank * self.max_tokens
        local = values[begin : begin + count]
        if reduce_slots:
            result = local[:, 0].float()
            for slot in range(1, self.k):
                result = result + local[:, slot].float()
        else:
            result = local.squeeze(-1)
        dest[:count].copy_(result)
        return dest[:count]

    @torch.no_grad()
    def forward(self, x, ids, prob, weights, out):
        count = self.validate_inputs(x, ids, prob)
        writable = self.validate_bundle(out, MoeEpTrainingForwardOutputs)
        weight_tensors = self.check_weight(weights, False)
        validate_training_non_aliasing(dict(writable, x=x, ids=ids, prob=prob, **weight_tensors))
        validate_readonly_against_outputs({name: getattr(weights, name) for name in _FORWARD_SCALES}, writable)
        ids = self.gather(ids, self.resolved.public_config.model.num_experts)
        slots, counts, ends = route_table(ids, self.topology.ep_rank * self.topology.experts_per_rank, self.topology.experts_per_rank, self.rows)
        out.route_slots.copy_(slots)
        out.valid_route_counts.copy_(counts)
        out.expert_offsets.copy_(ends)
        out.activation_norms[:, 0].copy_(weights.input_norm.reshape(()))
        out.activation_norms[:, 1].copy_(weights.fc1_norm_const)
        if self.grouped_forward is not None:
            return self.grouped_forward.run(self.exchange, x, prob, weights, out)
        x, prob = self.gather(x), self.gather(prob).flatten()
        routes = x.new_empty((self.rows, self.h))
        fc1, fc2 = self.gemms[TILE, 2 * self.i, self.h], self.gemms[TILE, self.h, self.i]
        bank1, bank2 = fc1.bind_dgrad(weights.fc1), fc2.bind_dgrad(weights.fc2)
        for _, rows, e in self.tiles(ends):
            slot = slots[rows]
            ix = slot.clamp_min(0)
            xt = torch.where(slot[:, None] >= 0, x[ix // self.k], 0)
            xp = out.fc1_a[rows], out.fc1_sfa[rows]
            self.pack_x.run(xt, weights.input_norm, *xp)
            pre = fc1.dgrad(xp, bank1, e) * weights.fc31_alpha.index_select(0, e)
            out.fc1_preact[rows].copy_(pre)
            weighted = self.activation_fwd.run(pre, torch.where(slot >= 0, prob[ix], 0.0))
            ap = out.saved_a[rows], out.saved_sfa[rows]
            self.pack_a.run(weighted, weights.fc1_norm_const.index_select(0, e), *ap)
            routes[rows].copy_(fc2.dgrad(ap, bank2, e) * weights.fc2_alpha.index_select(0, e))
        return self.combine(routes, slots, out.output, count)

    @torch.no_grad()
    def backward(self, dy, ids, prob, weights, record, out, overlap, dw1, dw2):
        count = self.validate_inputs(dy, ids, prob)
        saved = self.validate_bundle(record, MoeEpTrainingForwardOutputs, retained=True)
        writable = self.validate_bundle(out, MoeEpTrainingBackwardOutputs)
        weight_tensors = self.check_weight(weights, True)
        divisor1 = positive_fp32(self.path.dgu_norm * weights.w1_norm, "FC1 dgrad divisor")
        divisor2 = positive_fp32(self.path.dy_norm * weights.w2_norm, "FC2 dgrad divisor")
        if (dw1 is None) != (dw2 is None):
            raise ValueError("supply both master gradient buffers")
        validate_training_non_aliasing(dict(saved, **writable, dy=dy, ids=ids, prob=prob, dw1=dw1, dw2=dw2, **weight_tensors))
        if dw1 is not None:
            self.check_gradients(dw1, dw2)
        if record.route_slots.numel() < self.rows:
            torch.le(record.expert_offsets[-1], record.route_slots.numel(), out=self.more_tiles)
            torch._assert_async(self.more_tiles, "NVFP4 expert offsets exceed retained record capacity")
        main = torch.cuda.current_stream()
        worker = self.weight_stream if overlap else main
        exchange = self.exchange
        capturing = torch.cuda.is_current_stream_capturing()
        bank2 = self.gemms[TILE, self.i, self.h].bind_dgrad(weights.w2_transpose, fixed_address=not capturing)
        bank1 = self.gemms[TILE, self.h, 2 * self.i].bind_dgrad(weights.w1_transpose, fixed_address=not capturing)
        banks = bank1, bank2
        exchange.begin(dy, prob, overlap)
        torch.searchsorted(record.expert_offsets, self.tile_starts, right=True, out=self.tile_experts)
        torch.div(record.expert_offsets[-1], TILE, rounding_mode="floor", out=self.live_tiles)
        worker.wait_stream(main)
        with torch.cuda.stream(worker):
            self.repack_x.run(
                record.fc1_a,
                record.fc1_sfa,
                out.fc1_wgrad_a,
                out.fc1_wgrad_sfa,
                record.activation_norms,
                self.tile_experts,
                self.norm[self.path.x_wgrad_norm],
                self.live_tiles,
            )
            self.repack_a.run(
                record.saved_a,
                record.saved_sfa,
                out.fc2_a,
                out.fc2_sfa,
                record.activation_norms,
                self.tile_experts,
                self.norm[self.path.a_wgrad_norm],
                self.live_tiles,
            )
            if dw1 is not None:
                self._bind_saved_wgrad(record.expert_offsets, out.fc1_wgrad_a, out.fc1_wgrad_sfa, out.fc2_a, out.fc2_sfa)
        self.read_window.bind(record)
        self.write_window.bind(out)
        self.dgrad_divisors[0].fill_(divisor1)
        self.dgrad_divisors[1].fill_(divisor2)
        if capturing:
            self._live_windows(banks, overlap)
        else:
            self.eager_graphs[overlap].replay()
        exchange.reader = exchange.writer = main
        if dw1 is not None:
            worker.wait_stream(main)
            with torch.cuda.stream(worker):
                self._grouped_wgrad(out.fc1_b, out.fc1_sfb, out.fc2_b, out.fc2_sfb, dw1, dw2)
                self.weight_done.record(worker)
                self.weight_join.record(worker)
        exchange.finish(out, count)
        grad_x, grad_p = out.grad_activation[:count], out.dprob[:count]
        self.dx_ready.record(main)
        if dw1 is not None:
            main.wait_event(self.weight_join)
        else:
            main.wait_stream(worker)
        # Packed operands and their segment boundaries must survive reuse of
        # the forward record, including when the captured routing changes.
        offsets, counts = out.wgrad_expert_offsets, out.wgrad_valid_route_counts
        if offsets is None:
            offsets, counts = record.expert_offsets.clone(), record.valid_route_counts.clone()
        else:
            offsets.copy_(record.expert_offsets)
            counts.copy_(record.valid_route_counts)
        operands = MoeEpTrainingWgradOperands(
            out.fc1_wgrad_a,
            out.fc1_wgrad_sfa,
            out.fc1_b,
            out.fc1_sfb,
            out.fc2_a,
            out.fc2_sfa,
            out.fc2_b,
            out.fc2_sfb,
            offsets,
            counts,
            self.encoding_norms,
        )
        return grad_x, grad_p, operands

    def _tile_run(self, record, out, divisors, banks, overlap, n, experts):
        main = torch.cuda.current_stream()
        exchange = self.exchange
        exchange.reader = exchange.get_stream if overlap else main
        exchange.writer = exchange.put_stream if overlap else main
        # All side-stream dependencies belong to this run (including when it
        # lives inside a conditional child graph).
        exchange.reader.wait_stream(main)
        exchange.writer.wait_stream(main)
        exchange.prefetch(0, record.route_slots[:TILE])
        pending_return = None
        for tile in range(n):
            rows = slice(tile * TILE, (tile + 1) * TILE)
            e = experts[tile : tile + 1]
            slot = record.route_slots[rows]
            yt, pt = exchange.take(tile)
            self.pack_token[self.h, yt.dtype, False].run(yt, self.norm[self.path.dy_norm], out.fc2_b[tile], out.fc2_sfb[tile])
            dy_pack = self.feature_scratch[self.h]
            self.pack_feature[self.h, yt.dtype].run(yt, self.norm[self.path.dy_norm], *dy_pack)
            dgrad2 = self.gemms[TILE, self.i, self.h]
            da = dgrad2.dgrad(dy_pack, banks[1], e, ready=exchange.mma_ready[tile])
            # Release the preceding tile alongside this tile's FC2 MMA. The MMA
            # start event also orders all writes to the preceding return slot.
            if pending_return is not None:
                exchange.send(*pending_return, ready=exchange.mma_ready[tile])
            if tile + 1 < n:
                exchange.prefetch(tile + 1, record.route_slots[rows.stop : rows.stop + TILE])
            da.div_(divisors[1])
            if tile >= 2:
                main.wait_event(exchange.sent[tile - 2])
            dest = slice(tile % 2 * TILE, (tile % 2 + 1) * TILE)
            dgu = self.activation_bwd.run(record.fc1_preact[rows], da, pt, self.dp[dest, 0])
            gu_pack = self.feature_scratch[2 * self.i]
            self.pack_feature[2 * self.i, dgu.dtype].run(dgu, self.norm[self.path.dgu_norm], *gu_pack)
            dgrad1 = self.gemms[TILE, self.h, 2 * self.i]
            block = dgrad1.dgrad(gu_pack, banks[0], e)
            torch.div(block, divisors[0], out=self.dx[dest])
            result = (tile, slot, self.dx[dest], self.dp[dest, 0], out.grad_activation)
            if overlap and tile + 1 < n:
                pending_return = result
            else:
                exchange.send(*result)
            self.pack_token[2 * self.i, dgu.dtype, False].run(dgu, self.norm[self.path.dgu_norm], out.fc1_b[tile], out.fc1_sfb[tile])
            exchange.events[1][tile].record(main)
        main.wait_stream(exchange.writer)
        # finish() drains the rank once in the parent graph, after the last run.
        exchange.writer = main
        exchange.reader = main

    @torch.no_grad()
    def wgrad(self, bundle, dw1, dw2):
        if not isinstance(bundle, MoeEpTrainingWgradOperands):
            raise TypeError("expected MoeEpTrainingWgradOperands")
        if bundle.encoding_norms != self.encoding_norms:
            raise ValueError("wgrad encoding norms must match the prepared NVFP4 configuration")
        tensors = {f.name: getattr(bundle, f.name) for f in fields(bundle) if f.name != "encoding_norms"}
        req = {name: self.requirements[{"fc1_a": "fc1_wgrad_a", "fc1_sfa": "fc1_wgrad_sfa"}.get(name, name)] for name in tensors}
        _validate_named_buffers(tensors, req, device=self.device)
        validate_training_non_aliasing(dict(tensors, dw1=dw1, dw2=dw2))
        self.check_gradients(dw1, dw2)
        torch.searchsorted(bundle.expert_offsets, self.tile_starts, right=True, out=self.tile_experts)
        self._bind_saved_wgrad(bundle.expert_offsets, bundle.fc1_a, bundle.fc1_sfa, bundle.fc2_a, bundle.fc2_sfa)
        self._grouped_wgrad(bundle.fc1_b, bundle.fc1_sfb, bundle.fc2_b, bundle.fc2_sfb, dw1, dw2)

    def _bind_saved_wgrad(self, ends, x, sx, a, sa):
        # X/A depend only on this microbatch's forward record. Prepare the
        # native K-grouped layout while the main stream produces dY and dGU.
        self.group_ends.copy_(ends)
        self.grouped[1].b.bind(a, sa, self.tile_experts, self.group_ends)
        self.grouped[0].b.bind(x, sx, self.tile_experts, self.group_ends)

    def _grouped_wgrad(self, gu, sgu, y, sy, dw1, dw2):
        self.grouped[1].run(dw2, y, sy, self.tile_experts, self.group_ends)
        self.grouped[0].run(dw1, gu, sgu, self.tile_experts, self.group_ends)

    def check_gradients(self, dw1, dw2):
        e = self.topology.experts_per_rank
        for weight, shape in ((dw1, (e, 2 * self.i, self.h)), (dw2, (e, self.h, self.i))):
            if weight.shape != shape or weight.dtype != torch.float32 or weight.device != self.device or not weight.is_contiguous():
                raise ValueError(f"expected contiguous FP32 weight gradient of shape {shape} on {self.device}")
            if weight.data_ptr() % 16:
                raise ValueError("NVFP4 weight gradient buffers must be 16-byte aligned")
