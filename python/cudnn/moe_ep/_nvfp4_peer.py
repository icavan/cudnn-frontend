# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-domain peer transport for 256-route dgrad tiles."""

import math

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch


class PeerKernels:
    def __init__(self, tokens, h, k):
        self.tokens, self.h, self.k = tokens, h, k
        self.chunk = 128
        self.chunks = (h + self.chunk - 1) // self.chunk
        # All offsets are in bytes relative to a peer's mapped allocation.
        self.prob = tokens * h * 2
        self.dx = self.prob + tokens * k * 4
        self.dp = self.dx + tokens * k * h * 4
        self.flags = self.dp + tokens * k * 4
        self.combined = self.flags + tokens * self.chunks * 4
        self.nbytes = self.combined + tokens * h * 2

    @cute.kernel
    def receive(self, base: cute.Tensor, slots: cute.Tensor, y: cute.Tensor, p: cute.Tensor):
        row, chunk, _ = cute.arch.block_idx()
        col = chunk * self.chunk + cute.arch.thread_idx()[0]
        slot = slots[row]
        if col < self.h:
            val = cutlass.BFloat16(0.0)
            if slot >= 0:
                token = slot // self.k
                ptr = cute.make_ptr(cutlass.BFloat16, base[token // self.tokens], cute.AddressSpace.gmem)
                val = cute.make_tensor(ptr, cute.make_layout(self.tokens * self.h))[(token % self.tokens) * self.h + col]
            y[row, col] = val
        if col == 0:
            prob = cutlass.Float32(0)
            if slot >= 0:
                ptr = cute.make_ptr(cutlass.Float32, base[slot // (self.tokens * self.k)] + self.prob, cute.AddressSpace.gmem)
                prob = cute.arch.load(ptr + slot % (self.tokens * self.k), cutlass.Float32)
            p[row] = prob

    @cute.kernel
    def publish(self, base: cute.Tensor, slots: cute.Tensor, dx: cute.Tensor, dp: cute.Tensor, generation: cute.Tensor):
        row, chunk, _ = cute.arch.block_idx()
        last = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(1), byte_alignment=4)
        lane = cute.arch.thread_idx()[0]
        col = chunk * self.chunk + lane
        slot = slots[row]
        if slot >= 0:
            peer = base[slot // (self.tokens * self.k)]
            local = slot % (self.tokens * self.k)
            ptr = cute.make_ptr(cutlass.Float32, peer + self.dx, cute.AddressSpace.gmem)
            if col < self.h:
                cute.arch.store(ptr + local * self.h + col, dx[row, col])
            if chunk == 0 and lane == 0:
                dprob = cute.make_ptr(cutlass.Float32, peer + self.dp, cute.AddressSpace.gmem)
                cute.arch.store(dprob + local, dp[row])
            # Each route has exactly one writer per chunk. The CTA barrier
            # orders all lanes' payload stores before the lane-0 system RMW.
            # Its acquire/release chain joins the previous K-1 writers; the
            # second CTA barrier propagates that acquire to the reading lanes.
            cute.arch.sync_threads()
            token = local // self.k
            if lane == 0:
                counts = cute.make_ptr(cutlass.Int32, peer + self.flags, cute.AddressSpace.gmem)
                prior = cute.arch.atomic_add(counts + token * self.chunks + chunk, cutlass.Int32(1), sem="acq_rel", scope="sys")
                last[0] = (prior == generation[0] * self.k + (self.k - 1)).to(cutlass.Int32)
            cute.arch.sync_threads()
            if last[0] != 0 and col < self.h:
                total = cutlass.Float32(0)
                for k in cutlass.range_constexpr(self.k):
                    total += cute.arch.load(ptr + (token * self.k + k) * self.h + col, cutlass.Float32)
                combined = cute.make_ptr(cutlass.Uint16, peer + self.combined, cute.AddressSpace.gmem)
                cute.arch.store(combined + token * self.h + col, total.to(cutlass.BFloat16).bitcast(cutlass.Uint16))

    @cute.jit
    def get(self, base: cute.Tensor, slots: cute.Tensor, y: cute.Tensor, p: cute.Tensor, stream: cuda.CUstream):
        self.receive(base, slots, y, p).launch(grid=(256, self.chunks, 1), block=(self.chunk, 1, 1), stream=stream)

    @cute.jit
    def put(self, base: cute.Tensor, slots: cute.Tensor, dx: cute.Tensor, dp: cute.Tensor, gen: cute.Tensor, stream: cuda.CUstream):
        self.publish(base, slots, dx, dp, gen).launch(grid=(256, self.chunks, 1), block=(self.chunk, 1, 1), stream=stream)


class TileExchange:
    def __init__(self, state):
        t, h, k = state.max_tokens, state.h, state.k
        kernel = PeerKernels(t, h, k)
        self.handle = None
        if state.topology.ep_size == 1:
            self.heap = torch.empty((kernel.nbytes,), dtype=torch.uint8, device=state.device)
            peers = [self.heap.data_ptr()]
        else:
            from torch.distributed import _symmetric_memory as symm

            self.heap = symm.empty(kernel.nbytes, dtype=torch.uint8, device=state.device)
            self.handle = symm.rendezvous(self.heap, state.group)
            peers = [self.handle.get_buffer(r, (kernel.nbytes,), torch.uint8).data_ptr() for r in range(state.topology.ep_size)]
        self.heap.zero_()

        def view(start, shape, dtype):
            size = math.prod(shape) * dtype.itemsize
            return self.heap[start : start + size].view(dtype).reshape(shape)

        self.dy = view(0, (t, h), torch.bfloat16)
        self.prob = view(kernel.prob, (t, k), torch.float32)
        self.dx = view(kernel.dx, (t * k, h), torch.float32)
        self.dp = view(kernel.dp, (t, k), torch.float32)
        self.flags = view(kernel.flags, (t * kernel.chunks,), torch.int32)
        self.combined = view(kernel.combined, (t, h), torch.bfloat16)
        self.top_k = k
        self.gen = torch.zeros(1, device=state.device, dtype=torch.int32)
        self.ptrs = torch.tensor(peers, device=state.device)
        self.inputs = [
            (torch.empty((256, h), dtype=torch.bfloat16, device=state.device), torch.empty(256, dtype=torch.float32, device=state.device)) for _ in range(2)
        ]
        self.mma_ready = [torch.cuda.Event() for _ in range(state.rows // 256)]
        self.get_stream = torch.cuda.Stream(device=state.device)
        self.put_stream = torch.cuda.Stream(device=state.device)
        self.events = [[torch.cuda.Event() for _ in range(state.rows // 256)] for _ in range(3)]
        self.sent = [torch.cuda.Event() for _ in range(state.rows // 256)]
        self.start = torch.cuda.Event()
        self.end = torch.cuda.Event()

        def fake(dtype, *shape):
            return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)

        stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        base = fake(cutlass.Int64, state.topology.ep_size)
        slots = fake(cutlass.Int64, 256)
        gen = fake(cutlass.Int32, 1)
        kw = {"options": "--enable-tvm-ffi"}
        self.get = cute.compile(kernel.get, base, slots, fake(cutlass.BFloat16, 256, h), fake(cutlass.Float32, 256), stream, **kw)
        self.put = cute.compile(kernel.put, base, slots, fake(cutlass.Float32, 256, h), fake(cutlass.Float32, 256), gen, stream, **kw)

    def begin(self, dy, prob, overlap):
        main = torch.cuda.current_stream()
        self.reader = self.get_stream if overlap else main
        self.writer = self.put_stream if overlap else main
        self.dy[: len(dy)].copy_(dy)
        self.prob[: len(prob)].copy_(prob)
        self.gen.add_(1)
        torch.mul(self.gen.expand_as(self.flags), self.top_k, out=self.flags)
        if self.handle is not None:
            self.handle.barrier()
        self.start.record(main)

    def prefetch(self, tile, slots):
        received, consumed, _ = self.events
        if tile > 0:
            self.reader.wait_event(self.mma_ready[tile - 1])
        if tile >= 2:
            self.reader.wait_event(consumed[tile - 2])
        self.get(self.ptrs, slots, *self.inputs[tile % 2], cuda.CUstream(self.reader.cuda_stream))
        received[tile].record(self.reader)

    def take(self, tile):
        torch.cuda.current_stream().wait_event(self.events[0][tile])
        return self.inputs[tile % 2]

    def send(self, tile, slots, dx, dp, out, *, ready=None):
        if ready is None:
            ready = self.events[2][tile]
            ready.record(torch.cuda.current_stream())
        self.writer.wait_event(ready)
        stream = cuda.CUstream(self.writer.cuda_stream)
        self.put(self.ptrs, slots, dx, dp, self.gen, stream)
        self.sent[tile].record(self.writer)

    def finish(self, out, count):
        main = torch.cuda.current_stream()
        self.end.record(self.writer)
        main.wait_event(self.end)
        if self.handle is not None:
            self.handle.barrier()
        out.grad_activation[:count].copy_(self.combined[:count])
        out.dprob[:count].copy_(self.dp[:count])
        # All ranks finish consuming this generation before the next reuse.
        if self.handle is not None:
            self.handle.barrier()
