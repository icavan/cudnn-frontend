# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windowed FC1 dispatch for the 256-row NVFP4 training kernel."""

from types import SimpleNamespace

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch

from ._nvfp4_schedule import WindowBindings


class SourceRows:
    """Restore packed expert rows from the invocation's source-token record."""

    def __init__(self, rows, hidden, top_k, device):
        self.rows, self.hidden, self.top_k = rows, hidden, top_k
        self.pointers = torch.empty(2, dtype=torch.int64, device=device)

        def fake(dtype, *shape):
            return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=2)

        self.launch = cute.compile(
            self.gather,
            fake(cutlass.Int64, 2),
            fake(cutlass.Int64, rows),
            fake(cutlass.Int16, rows, hidden // 4),
            fake(cutlass.Int16, rows, hidden // 32),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    def bind(self, record):
        self.pointers[0].fill_(record.fc1_a.data_ptr())
        self.pointers[1].fill_(record.fc1_sfa.data_ptr())

    @cute.kernel
    def gather_rows(self, pointers, routes, payload, scales):
        row, segment, _ = cute.arch.block_idx()
        col = segment * 128 + cute.arch.thread_idx()[0]
        route = routes[row]
        if col < self.hidden // 4:
            packed = cutlass.Int16(0)
            if route >= 0:
                source = cute.make_ptr(cutlass.Int16, pointers[0], cute.AddressSpace.gmem)
                packed = cute.make_tensor(source + (route // self.top_k) * (self.hidden // 4), cute.make_layout(self.hidden // 4))[col]
            payload[row, col] = packed
        if col < self.hidden // 32:
            sf = cutlass.Int16(0)
            if route >= 0:
                source = cute.make_ptr(cutlass.Int16, pointers[1], cute.AddressSpace.gmem)
                sf = cute.make_tensor(source + (route // self.top_k) * (self.hidden // 32), cute.make_layout(self.hidden // 32))[col]
            scales[row, col] = sf

    @cute.jit
    def gather(self, pointers: cute.Tensor, routes: cute.Tensor, payload: cute.Tensor, scales: cute.Tensor, stream: cuda.CUstream):
        self.gather_rows(pointers, routes, payload, scales).launch(grid=(self.rows, cute.ceil_div(self.hidden // 4, 128), 1), block=(128, 1, 1), stream=stream)

    def load(self, routes, payload, scales):
        self.launch(self.pointers, routes, payload.view(torch.int16), scales.view(torch.int16), cuda.CUstream(torch.cuda.current_stream().cuda_stream))


class Fc1Ring:
    """Finite work packets with ordered admission and one ready/retired event pair per slot."""

    def __init__(self, state, tile, slots, overlap):
        dev = state.device
        self.tile, self.slots, self.overlap = tile, slots, overlap and slots > 1
        packet = self.packet = min(state.rows // tile, 2 * slots if self.overlap else slots)
        self.reader = torch.cuda.Stream(device=dev)
        self.ready = [torch.cuda.Event() for _ in range(slots)]
        self.retired = [torch.cuda.Event() for _ in range(slots)]
        self.source = SourceRows(tile, state.h, state.k, dev)
        self.payload = torch.empty((slots, tile, state.h // 2), dtype=torch.uint8, device=dev)
        self.sf = torch.empty((slots, tile, state.h // 16), dtype=torch.float8_e4m3fn, device=dev)
        self.routes = torch.empty((packet, tile), dtype=torch.int64, device=dev)
        self.prob = torch.empty(state.topology.ep_size * state.max_tokens * state.k, dtype=torch.float32, device=dev)
        self.alpha = torch.empty((3, state.topology.experts_per_rank), dtype=torch.float32, device=dev)
        self.read_routes = WindowBindings(dict(route_slots=self.routes), packet, dev, write=False)
        self.gemm1 = state.gemms[tile, 2 * state.i, state.h]
        self.gemm2 = state.gemms[tile, state.h, state.i]
        self.weights = WindowBindings(
            dict(w1=self.gemm1.weight_data, sf1=self.gemm1.weight_sf, w2=self.gemm2.weight_data, sf2=self.gemm2.weight_sf), 1, dev, write=False
        )
        self.results = {
            name: torch.empty((tile, width), device=dev, dtype=dtype)
            for name, width, dtype in (
                ("fc1_preact", 2 * state.i, torch.float32),
                ("saved_a", state.i // 2, torch.uint8),
                ("saved_sfa", state.i // 16, torch.float8_e4m3fn),
                ("route_output", state.h, torch.bfloat16),
            )
        }
        self.store = WindowBindings(self.results, 1, dev, write=True)
        self.indices = torch.arange(packet, device=dev)
        self.index = torch.empty_like(self.indices)
        self.experts = torch.empty_like(self.indices)
        self.scales = torch.empty((3, 1), dtype=torch.float32, device=dev)
        self.prob_index = torch.empty(tile, dtype=torch.int64, device=dev)
        self.tile_prob = torch.empty(tile, dtype=torch.float32, device=dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.windows(state)

    def windows(self, state):
        state.run_base.zero_()
        for n in dict.fromkeys((self.packet, self.slots, 1)):
            torch.sub(state.live_tiles, state.run_base, out=state.remaining_tiles)
            torch.ge(state.remaining_tiles, n, out=state.more_tiles)
            with state.conditions.while_true(state.more_tiles) as update:
                self.window(state, n)
                state.run_base.add_(n)
                torch.sub(state.live_tiles, state.run_base, out=state.remaining_tiles)
                torch.ge(state.remaining_tiles, n, out=state.more_tiles)
                update()

    def window(self, state, n):
        main = torch.cuda.current_stream()
        self.read_routes.copy(state.run_base, n)
        reader = self.reader if self.overlap and n > 1 else main
        if reader != main:
            reader.wait_stream(main)

        def dispatch(logical):
            slot = logical % self.slots
            with torch.cuda.stream(reader):
                if logical >= self.slots:
                    reader.wait_event(self.retired[slot])
                self.source.load(self.routes[logical], self.payload[slot], self.sf[slot])
                if reader != main:
                    self.ready[slot].record(reader)

        for logical in range(min(n, self.slots)):
            dispatch(logical)
        torch.add(self.indices[:n], state.run_base, out=self.index[:n])
        torch.index_select(state.tile_experts, 0, self.index[:n], out=self.experts[:n])
        for s in range(n):
            slot = s % self.slots
            expert = self.experts[s : s + 1]
            for i in range(3):
                torch.index_select(self.alpha[i], 0, expert, out=self.scales[i])
            self.weights.copy(expert, 1)
            if reader != main:
                main.wait_event(self.ready[slot])
            pre = self.gemm1((self.payload[slot], self.sf[slot]), (self.gemm1.weight_data[0], self.gemm1.weight_sf[0]))
            if s + self.slots < n:
                # The entire FC1 launch, including every output-feature tile,
                # precedes this event. FC2 and its logical record can outlive the slot.
                self.retired[slot].record(main)
                dispatch(s + self.slots)
            pre = torch.mul(pre, self.scales[0], out=self.results["fc1_preact"])
            torch.clamp_min(self.routes[s], 0, out=self.prob_index)
            torch.index_select(self.prob, 0, self.prob_index, out=self.tile_prob)
            weighted = state.activation_fwd.run(pre, self.tile_prob)
            ap = self.results["saved_a"], self.results["saved_sfa"]
            state.pack_a.run(weighted, self.scales[2], *ap)
            result = self.gemm2(ap, (self.gemm2.weight_data[0], self.gemm2.weight_sf[0]))
            result.mul_(self.scales[1])
            self.results["route_output"].copy_(result)
            self.store.copy(self.index[s : s + 1], 1)

    def run(self, state, x, prob, weights, record, routes):
        # The checkpoint input multiplier is local to the receiving plan.
        # Pack after all-gather so differing rank-local multipliers are preserved.
        for start in range(0, x.shape[0], self.tile):
            rows = slice(start, start + self.tile)
            src = x[rows]
            if src.shape[0] == self.tile:
                state.pack_x.run(src, weights.input_norm, record.fc1_a[rows], record.fc1_sfa[rows])
            else:
                padded = torch.nn.functional.pad(src, (0, 0, 0, self.tile - src.shape[0]))
                state.pack_x.run(padded, weights.input_norm, self.payload[0], self.sf[0])
                record.fc1_a[rows].copy_(self.payload[0, : src.shape[0]])
                record.fc1_sfa[rows].copy_(self.sf[0, : src.shape[0]])
        self.source.bind(record)
        self.read_routes.bind(record)
        self.store.bind(SimpleNamespace(**vars(record), route_output=routes))
        self.prob.copy_(prob)
        for dst, src in zip(self.alpha, (weights.fc31_alpha, weights.fc2_alpha, weights.fc1_norm_const)):
            dst.copy_(src)
        # WindowBindings moves aligned 64-bit words. An otherwise contiguous
        # checkpoint view may start at a byte offset inside its allocation.
        matrices = [
            a.clone(memory_format=torch.contiguous_format) if a.data_ptr() % 8 else a.contiguous()
            for w in (weights.fc1, weights.fc2)
            for a in (w.data, w.scale)
        ]
        self.weights.bind(SimpleNamespace(**dict(zip(self.weights.names, matrices))))
        torch.searchsorted(record.expert_offsets, state.tile_starts, right=True, out=state.tile_experts)
        torch.div(record.expert_offsets[-1], self.tile, rounding_mode="floor", out=state.live_tiles)
        if torch.cuda.is_current_stream_capturing():
            self.windows(state)
        else:
            self.graph.replay()
