# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-capacity Blackwell NVFP4 training with deterministic 256-route K splits."""

from __future__ import annotations

from dataclasses import fields
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ._config import _required_padded_rows, MoeEpFc1WeightLayout
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
from ._validation import _validate_named_buffers, validate_training_input, validate_training_non_aliasing

TILE = 256


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
        self.rows = p = config.parallel.physical_recv_pool_rows or needed
        if p < needed or p % TILE:
            raise ValueError(f"NVFP4 physical_recv_pool_rows must cover {needed} rows in multiples of {TILE}")
        tiles = p // TILE
        u8, sf, f32 = torch.uint8, torch.float8_e4m3fn, torch.float32
        self.requirements = {}

        def spec(name, shape, dtype):
            stride = torch.empty(shape, device="meta").stride()
            self.requirements[name] = (shape, stride, dtype, 16)

        source_rows = t * self.topology.ep_size if self.path.fc1_ring_slots else p
        spec("fc1_a", (source_rows, h // 2), u8)
        spec("fc1_sfa", (source_rows, h // 16), sf)
        spec("saved_a", (p, i // 2), u8)
        spec("saved_sfa", (p, i // 16), sf)
        for name, dim in (("fc1_wgrad_a", h), ("fc1_b", 2 * i), ("fc2_a", i), ("fc2_b", h)):
            spec(name, (tiles, dim, TILE // 2), u8)
        for name, dim in (("fc1_wgrad_sfa", h), ("fc1_sfb", 2 * i), ("fc2_sfa", i), ("fc2_sfb", h)):
            spec(name, (tiles, dim, TILE // 16), sf)
        spec("activation_norms", (e, 2), f32)
        spec("expert_offsets", (e,), torch.int64)
        spec("valid_route_counts", (e,), torch.int64)
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
            shape: TileGemm(*shape, device, expert_weights=True, communication_sms=comm_sms)
            for shape in sorted({(TILE, 2 * i, h), (TILE, h, i), (TILE, i, h), (TILE, h, 2 * i)})
        }

        for n, k_dim in {(i, h), (h, 2 * i)}:
            self.gemms[TILE, n, k_dim].prepare_dgrad(e, n, k_dim, device)

        from ._nvfp4_wgrad import GroupedWgrad

        self.group_ends = torch.zeros(e, dtype=torch.int32, device=device)
        self.grouped = [GroupedWgrad(m, n, p, e, self.group_ends, device) for m, n in ((2 * i, h), (h, i))]
        from ._nvfp4_pack import PackTile, SwiGLUBackward, SwiGLUForward

        self.pack_x = PackTile(h, torch.bfloat16, False, False, forward=True)
        self.pack_a = PackTile(i, torch.float32, False, False, forward=True)
        self.activation_fwd = SwiGLUForward(i, self.path.gate_up_clamp, device)
        self.activation_bwd = SwiGLUBackward(i, self.path.gate_up_clamp, device)
        self.tile_starts = torch.arange(0, self.rows, TILE, device=device)
        self.tile_experts = torch.empty_like(self.tile_starts)
        self.tile_norms = torch.empty((1, 2), dtype=torch.float32, device=device)

        self.pack_feature = {key: PackTile(*key, False, False) for key in {(h, torch.bfloat16), (2 * i, torch.float32)}}
        self.pack_token = {
            key: PackTile(*key[:2], True, key[2])
            for key in {(h, torch.bfloat16, False), (2 * i, torch.float32, False), (h, torch.float32, True), (i, torch.float32, True)}
        }
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
        return_rows = 2 * TILE
        self.dx = torch.empty((return_rows, h), device=device, dtype=torch.float32)
        self.dp = torch.empty((return_rows, 1), device=device, dtype=torch.float32)

        # Process full 16-tile windows and the final live tail with device-side
        # loop bounds. Both the graph size and the window scratch are bounded.
        from ._nvfp4_schedule import RouteConditions, WindowBindings

        self.conditions = RouteConditions(device)
        self.window = min(16, tiles)
        self.window_ids = torch.arange(self.window, device=device)
        self.window_index = torch.empty_like(self.window_ids)
        self.window_experts = torch.empty_like(self.window_ids)
        self.live_tiles = torch.empty((), dtype=torch.int64, device=device)
        self.run_base = torch.empty_like(self.live_tiles)
        self.window_inputs = {}
        self.window_outputs = {}
        for name in ("route_slots", "fc1_preact", "fc1_a", "fc1_sfa", "saved_a", "saved_sfa"):
            shape, _, dtype, _ = self.requirements[name]
            self.window_inputs[name] = torch.empty((self.window * TILE, *shape[1:]), device=device, dtype=dtype)
        for name in ("fc1_wgrad_a", "fc1_wgrad_sfa", "fc1_b", "fc1_sfb", "fc2_a", "fc2_sfa", "fc2_b", "fc2_sfb"):
            shape, _, dtype, _ = self.requirements[name]
            self.window_outputs[name] = torch.empty((self.window, *shape[1:]), device=device, dtype=dtype)

        read_fields = dict(self.window_inputs)
        if self.path.fc1_ring_slots:
            for name in ("fc1_a", "fc1_sfa"):
                del read_fields[name]
        self.read_window = WindowBindings(read_fields, self.window, device, write=False)
        self.write_window = WindowBindings(self.window_outputs, self.window, device, write=True)
        self.record_norms = torch.empty((e, 2), dtype=torch.float32, device=device)
        self.dgrad_divisors = torch.ones(2, dtype=torch.float32, device=device)
        self.remaining_tiles = torch.empty_like(self.live_tiles)
        self.more_tiles = torch.empty((), dtype=torch.bool, device=device)
        # Eager uses the same device-side route count without a D2H read. The
        # two prepared graphs bind caller-owned records through a pointer table.
        self.fc1_ring = None
        if self.path.fc1_ring_slots:
            from ._nvfp4_ring import Fc1Ring

            self.fc1_ring = Fc1Ring(self, TILE, min(self.path.fc1_ring_slots, tiles), self.path.fc1_ring_overlap)
        eager_banks = tuple(self.gemms[TILE, n, kk].fallback_weights.view(torch.float4_e2m1fn_x2).permute(1, 2, 0) for n, kk in ((h, 2 * i), (i, h)))
        self.eager_graphs = {}
        for overlap in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._live_windows(eager_banks, overlap)
            self.eager_graphs[overlap] = graph

    def close(self):
        with torch.cuda.device(self.device):
            self.eager_graphs.clear()
            self.fc1_ring = None
            self.conditions.close()

    def _live_windows(self, banks, overlap):
        record = MoeEpTrainingForwardOutputs(**self.window_inputs, activation_norms=self.record_norms)
        out = MoeEpTrainingBackwardOutputs(**self.window_outputs, grad_activation=self.buffers["grad_activation"])
        self.run_base.zero_()
        widths = (self.window, 1) if self.window > 1 else (1,)
        for n in widths:
            torch.sub(self.live_tiles, self.run_base, out=self.remaining_tiles)
            torch.ge(self.remaining_tiles, n, out=self.more_tiles)
            with self.conditions.while_true(self.more_tiles) as update_condition:
                self.read_window.copy(self.run_base, n)
                if self.fc1_ring is not None:
                    for tile in range(n):
                        rows = slice(tile * TILE, (tile + 1) * TILE)
                        self.fc1_ring.source.load(record.route_slots[rows], record.fc1_a[rows], record.fc1_sfa[rows])
                torch.add(self.window_ids[:n], self.run_base, out=self.window_index[:n])
                torch.index_select(self.tile_experts, 0, self.window_index[:n], out=self.window_experts[:n])
                self._tile_run(record, out, self.dgrad_divisors, banks, overlap, n, self.window_experts)
                self.write_window.copy(self.run_base, n)
                self.run_base.add_(n)
                torch.sub(self.live_tiles, self.run_base, out=self.remaining_tiles)
                torch.ge(self.remaining_tiles, n, out=self.more_tiles)
                update_condition()

    def public_symmetric_buffers(self):
        return {**self.buffers, "backward_input": self.exchange.dy, "dprob": self.exchange.dp}

    def validate_bundle(self, bundle, cls):
        if not isinstance(bundle, cls):
            raise TypeError(f"expected {cls.__name__}")
        items = {f.name: getattr(bundle, f.name) for f in fields(cls)}
        _validate_named_buffers(items, {name: self.requirements[name] for name in items}, device=self.device)
        return items

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
        if backward:
            if not all(math.isfinite(s) and s > 0 for s in (weights.w1_norm, weights.w2_norm)):
                raise ValueError("backward weight norms must be finite and positive")
        else:
            for name in ("input_norm", "fc1_norm_const", "fc31_alpha", "fc2_alpha"):
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
        index = torch.where(slots >= 0, slots, sentinel)
        inverse = torch.full((sentinel + 1,), self.rows, dtype=torch.long, device=self.device)
        # Each valid slot has exactly one writer. All padding writers store the same sentinel row.
        inverse.scatter_(0, index, torch.where(slots >= 0, torch.arange(self.rows, device=self.device), self.rows))
        values = F.pad(routes, (0, 0, 0, 1)).index_select(0, inverse[:-1]).view(global_tokens, self.k, -1)
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
        ids = self.gather(ids, self.resolved.public_config.model.num_experts)
        x, prob = self.gather(x), self.gather(prob).flatten()
        slots, counts, ends = route_table(ids, self.topology.ep_rank * self.topology.experts_per_rank, self.topology.experts_per_rank, self.rows)
        out.route_slots.copy_(slots)
        out.valid_route_counts.copy_(counts)
        out.expert_offsets.copy_(ends)
        out.activation_norms[:, 0].copy_(weights.input_norm.reshape(()))
        out.activation_norms[:, 1].copy_(weights.fc1_norm_const)
        routes = x.new_empty((self.rows, self.h))
        if self.fc1_ring is not None:
            self.fc1_ring.run(self, x, prob, weights, out, routes)
            return self.combine(routes, slots, out.output, count)
        for _, rows, e in self.tiles(ends):
            slot = slots[rows]
            ix = slot.clamp_min(0)
            xt = torch.where(slot[:, None] >= 0, x[ix // self.k], 0)
            xp = out.fc1_a[rows], out.fc1_sfa[rows]
            self.pack_x.run(xt, weights.input_norm, *xp)
            fc1 = self.gemms[TILE, 2 * self.i, self.h]
            pre = fc1(xp, fc1.select_expert(weights.fc1, e)) * weights.fc31_alpha.index_select(0, e)
            out.fc1_preact[rows].copy_(pre)
            weighted = self.activation_fwd.run(pre, prob[ix])
            ap = out.saved_a[rows], out.saved_sfa[rows]
            self.pack_a.run(weighted, weights.fc1_norm_const.index_select(0, e), *ap)
            fc2 = self.gemms[TILE, self.h, self.i]
            routes[rows].copy_(fc2(ap, fc2.select_expert(weights.fc2, e)) * weights.fc2_alpha.index_select(0, e))
        return self.combine(routes, slots, out.output, count)

    @torch.no_grad()
    def backward(self, dy, ids, prob, weights, record, out, overlap, dw1, dw2):
        count = self.validate_inputs(dy, ids, prob)
        saved = self.validate_bundle(record, MoeEpTrainingForwardOutputs)
        writable = self.validate_bundle(out, MoeEpTrainingBackwardOutputs)
        weight_tensors = self.check_weight(weights, True)
        if (dw1 is None) != (dw2 is None):
            raise ValueError("supply both master gradient buffers")
        validate_training_non_aliasing(dict(saved, **writable, dy=dy, ids=ids, prob=prob, dw1=dw1, dw2=dw2, **weight_tensors))
        if dw1 is not None:
            self.check_gradients(dw1, dw2)
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
        self.read_window.bind(record)
        if self.fc1_ring is not None:
            self.fc1_ring.source.bind(record)
        self.write_window.bind(out)
        self.record_norms.copy_(record.activation_norms)
        self.dgrad_divisors[0].fill_(self.path.dgu_norm * weights.w1_norm)
        self.dgrad_divisors[1].fill_(self.path.dy_norm * weights.w2_norm)
        if capturing:
            self._live_windows(banks, overlap)
        else:
            self.eager_graphs[overlap].replay()
        exchange.reader = exchange.writer = main
        if dw1 is not None:
            worker.wait_stream(main)
            with torch.cuda.stream(worker):
                self._grouped_wgrad(
                    out.fc1_b, out.fc1_sfb, out.fc1_wgrad_a, out.fc1_wgrad_sfa, out.fc2_b, out.fc2_sfb, out.fc2_a, out.fc2_sfa, record.expert_offsets, dw1, dw2
                )
                self.weight_done.record(worker)
                self.weight_join.record(worker)
        exchange.finish(out, count)
        grad_x, grad_p = out.grad_activation[:count], out.dprob[:count]
        self.dx_ready.record(main)
        if dw1 is not None:
            main.wait_event(self.weight_join)
        operands = MoeEpTrainingWgradOperands(
            out.fc1_wgrad_a,
            out.fc1_wgrad_sfa,
            out.fc1_b,
            out.fc1_sfb,
            out.fc2_a,
            out.fc2_sfa,
            out.fc2_b,
            out.fc2_sfb,
            record.expert_offsets,
            record.valid_route_counts,
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
            torch.index_select(record.activation_norms, 0, e, out=self.tile_norms)
            norms = self.tile_norms[0]
            self.pack_token[self.i, torch.float32, True].run(
                record.saved_a[rows],
                self.norm[self.path.a_wgrad_norm],
                out.fc2_a[tile],
                out.fc2_sfa[tile],
                source_sf=record.saved_sfa[rows],
                source_norm=norms[1:2],
            )
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
            self.pack_token[self.h, torch.float32, True].run(
                record.fc1_a[rows],
                self.norm[self.path.x_wgrad_norm],
                out.fc1_wgrad_a[tile],
                out.fc1_wgrad_sfa[tile],
                source_sf=record.fc1_sfa[rows],
                source_norm=norms[:1],
            )
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
        self._grouped_wgrad(
            bundle.fc1_b,
            bundle.fc1_sfb,
            bundle.fc1_a,
            bundle.fc1_sfa,
            bundle.fc2_b,
            bundle.fc2_sfb,
            bundle.fc2_a,
            bundle.fc2_sfa,
            bundle.expert_offsets,
            dw1,
            dw2,
        )

    def _grouped_wgrad(self, gu, sgu, x, sx, y, sy, a, sa, ends, dw1, dw2):
        self.group_ends.copy_(ends)
        self.grouped[1].run(dw2, y, sy, a, sa, self.tile_experts, self.group_ends, self.path.dy_norm * self.path.a_wgrad_norm)
        self.grouped[0].run(dw1, gu, sgu, x, sx, self.tile_experts, self.group_ends, self.path.dgu_norm * self.path.x_wgrad_norm)

    def check_gradients(self, dw1, dw2):
        e = self.topology.experts_per_rank
        for weight, shape in ((dw1, (e, 2 * self.i, self.h)), (dw2, (e, self.h, self.i))):
            if weight.shape != shape or weight.dtype != torch.float32 or weight.device != self.device or not weight.is_contiguous():
                raise ValueError(f"expected contiguous FP32 weight gradient of shape {shape} on {self.device}")
