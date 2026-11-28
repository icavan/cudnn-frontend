# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windowed expert dgrad. A 256-row segment belongs to exactly one expert."""

import torch
from cuda.bindings import driver as cuda

from ._nvfp4 import mma_operand
from ._nvfp4_pack import DualPack, SwiGLUBackward
from ._nvfp4_peer import PeerDgradStore, PeerKernels


class WindowDgrad:
    def __init__(self, state, tiles, n, k, input_type, data_name, scale_name, activation=None, peers=None, peer_store=None):
        rows = tiles * 256
        gemm = state.gemms[256, n, k]
        self.gemm = gemm
        self.direct = gemm.compile_dgrad(state.topology.experts_per_rank, n, k, tiles, live=state.bounded_groups, peer_store=peer_store)
        device = state.device
        if peer_store is None:
            self.out = torch.empty((1, rows, n), dtype=torch.float32, device=device).permute(1, 2, 0)
        else:
            # The GEMM needs the logical C layout. The custom store owns the
            # writes, so the local rank's FP32 route plane can back this view.
            self.out = state.exchange.dx[:rows].view(1, rows, n).permute(1, 2, 0)
        self.pack = DualPack(
            k, rows, input_type, device, state.write_window, data_name, scale_name, activation=activation, peers=peers, masked=state.bounded_groups
        )

    def run(self, x, norm, bank, experts, table, base, offset, mma_ready=None, dest=None):
        self.pack.run(x, norm, table, base, offset)
        return self.multiply(bank, experts, mma_ready, dest)

    def multiply(self, bank, experts, mma_ready=None, dest=None, packed=None, route=()):
        payload, sf = (self.pack.payload, self.pack.sf) if packed is None else packed
        out = self.out if dest is None else dest
        if mma_ready is not None:
            mma_ready.record(torch.cuda.current_stream())
        self.direct(
            mma_operand(payload),
            bank,
            sf,
            self.gemm.expert_sf,
            out,
            self.gemm.amax,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            experts,
            *route,
        )
        return out.squeeze(-1)


class BackwardWindow:
    def __init__(self, state, tiles):
        self.tiles = tiles
        h, i, rows, device = state.h, state.i, tiles * 256, state.device
        peers = (state.topology.ep_size, state.max_tokens, state.k) if state.bounded_groups or state.window >= 64 else None
        self.fc2 = WindowDgrad(state, tiles, i, h, torch.bfloat16, "fc2_b", "fc2_sfb", peers=peers)
        field = state.read_window.names.index("fc1_preact")
        activation = (state.path.gate_up_clamp, field, state.read_window.count) if i % 256 == 0 else None
        direct_return = state.direct_return
        peer_store = PeerDgradStore(state.topology.ep_size, state.max_tokens * state.k, h) if direct_return else None
        self.fc1 = WindowDgrad(state, tiles, h, 2 * i, torch.float32, "fc1_b", "fc1_sfb", activation=activation, peer_store=peer_store)
        self.activation = None
        if activation is None:
            self.activation = SwiGLUBackward(i, state.path.gate_up_clamp, device, rows, record_field=field, fields=state.read_window.count)
        self.packed = None
        if peers is None:
            self.dy = [torch.empty((rows, h), dtype=torch.bfloat16, device=device) for _ in range(2)]
        else:
            pack = self.fc2.pack
            self.packed = [(pack.payload, pack.sf), (torch.empty_like(pack.payload), torch.zeros_like(pack.sf))]
        self.p = [torch.empty(rows, dtype=torch.float32, device=device) for _ in range(2)]
        self.dp = [torch.empty_like(p) for p in self.p]
        k = PeerKernels(state.max_tokens, h, state.k)
        self.notify = None
        if direct_return:
            self.notify = state.exchange.compile_notification(k, rows)
            self.dx_peers = state.exchange.ptrs + k.dx
        else:
            self.get, self.put = state.exchange.compile_transfer(k, rows, scale_dx=True)
        self.received = [torch.cuda.Event() for _ in self.p]
        self.mma_ready = [torch.cuda.Event() for _ in self.p]
        self.ready = [torch.cuda.Event() for _ in self.p]
        self.sent = [torch.cuda.Event() for _ in self.p]
        self.dx = [self.fc1.out, self.fc1.out if direct_return else torch.empty_like(self.fc1.out)]

    def run(self, state, record, divisors, banks, overlap, experts, width):
        # Separate pairs of receive and return buffers allow both transfers
        # to run alongside dgrad. Join the writer before reusing a window.
        main = torch.cuda.current_stream()
        exchange = state.exchange
        reader = exchange.get_stream if overlap else main
        writer = exchange.put_stream if overlap else main
        reader.wait_stream(main)
        writer.wait_stream(main)
        t = self.tiles
        norm, path = state.norm, state.path
        rows = t * 256
        groups = width // t
        base = state.bounds if state.bounded_groups else state.run_base

        def receive(group):
            buffer = group % 2
            slots = record.route_slots[group * rows : (group + 1) * rows]
            if self.packed is None:
                self.get(exchange.ptrs, slots, self.dy[buffer], self.p[buffer], cuda.CUstream(reader.cuda_stream))
            else:
                self.fc2.pack.receive(
                    exchange.ptrs,
                    slots,
                    self.p[buffer],
                    self.packed[buffer],
                    norm[path.dy_norm],
                    state.write_window.external,
                    base,
                    group * t,
                    cuda.CUstream(reader.cuda_stream),
                )
            self.received[buffer].record(reader)

        def publish(group, dx):
            slots = record.route_slots[group * rows : (group + 1) * rows]
            stream = cuda.CUstream(writer.cuda_stream)
            if self.notify is None:
                self.put(exchange.ptrs, slots, dx, self.dp[group % 2], exchange.gen, divisors[:1], stream)
            else:
                self.notify(exchange.ptrs, slots, self.dp[group % 2], stream)
            self.sent[group % 2].record(writer)

        receive(0)
        for group in range(groups):
            buffer = group % 2
            offset = group * t
            e = experts[offset : offset + t]
            route = (base, offset) if state.bounded_groups else ()
            main.wait_event(self.received[buffer])
            if self.packed is None:
                da = self.fc2.run(self.dy[buffer], norm[path.dy_norm], banks[1], e, state.write_window.external, base, offset, self.mma_ready[buffer])
            else:
                da = self.fc2.multiply(banks[1], e, self.mma_ready[buffer], packed=self.packed[buffer], route=route)
            if group + 1 < groups:
                # FC2 begins after the previous activation consumed this input
                # buffer. The next transfer can reuse it during the current MMA.
                reader.wait_event(self.mma_ready[buffer])
                receive(group + 1)
            if group >= 2:
                # The previous owner of this dX/dp buffer may still be returning it.
                main.wait_event(self.sent[buffer])
            if self.activation is None:
                self.fc1.pack.run_swiglu(
                    state.read_window.external,
                    da,
                    self.p[buffer],
                    divisors[1:2],
                    norm[path.dgu_norm],
                    state.write_window.external,
                    base,
                    offset,
                    self.dp[buffer],
                )
                if self.notify is not None:
                    slots = record.route_slots[group * rows : (group + 1) * rows]
                    route = (*route, (self.dx_peers, slots, divisors[:1]))
                dx = self.fc1.multiply(banks[0], e, dest=self.dx[buffer], route=route)
            else:
                dgu = self.activation.run(state.read_window.external, da, self.p[buffer], self.dp[buffer], base.view(1), offset, divisors[1:2])
                dx = self.fc1.run(dgu, norm[path.dgu_norm], banks[0], e, state.write_window.external, base, offset, dest=self.dx[buffer])
            self.ready[buffer].record(main)
            writer.wait_event(self.ready[buffer])
            publish(group, dx)
            if group % 2 == 0:
                exchange.combine_ready(writer)
        main.wait_stream(writer)
