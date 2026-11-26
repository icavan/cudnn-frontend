# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-domain peer transport for 256-route dgrad tiles."""

import math
from contextlib import ExitStack
from weakref import finalize

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch

from cudnn.frost.tile_dsl.pointwise import div_rn_f32
from ._nvfp4_schedule import _result


class PeerDgradStore:
    """Store a single-CTA MMA output subtile into FP32 source-order route slots."""

    def __init__(self, peer_count, slots_per_rank, hidden):
        self.peer_count = peer_count
        self.slots_per_rank = slots_per_rank
        self.hidden = hidden

    @cute.jit
    def store(self, shared, stage, thread, threads, tile, cta_shape, epi_tile, subtile, args):
        peers, slots, divisor = args
        m, n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
        # Accumulator subtiles advance in M before N. Each vector is wholly
        # inside a row because the prepared path uses full 128-column CTA tiles.
        rows_per_column = cta_shape[0] // m
        row_base = tile[0] * cta_shape[0] + subtile % rows_per_column * m
        col_base = tile[1] * cta_shape[1] + subtile // rows_per_column * n
        for part in cutlass.range_constexpr(m * n // (threads * 4)):
            element = (part * threads + thread) * 4
            row, col = element // n, element % n
            slot = slots[row_base + row]
            if slot >= 0:
                dst = cute.make_ptr(cutlass.Float32, peers[slot // self.slots_per_rank], cute.AddressSpace.gmem)
                offset = (slot % self.slots_per_rank) * self.hidden + col_base + col
                values = cute.make_rmem_tensor((4,), cutlass.Float32)
                for j in cutlass.range_constexpr(4):
                    values[j] = shared[row, col + j, stage]
                if divisor[0] != 1.0:
                    for j in cutlass.range_constexpr(4):
                        values[j] = div_rn_f32(values[j], divisor[0])
                cute.autovec_copy(values, cute.make_tensor((dst + offset).align(16), cute.make_layout(4)))


class PeerKernels:
    def __init__(self, tokens, h, k):
        self.tokens, self.h, self.k = tokens, h, k
        self.local_reduce = h >= 1024
        self.threads = 128
        self.chunk = 128
        if self.local_reduce:
            # Aim for two comparable transfers per row. Each iteration of the
            # vectorized drain covers 512 features; cap its unrolled work.
            self.chunk = min(4096, max(1024, ((h + 1023) // 1024) * 512))
        self.chunks = (h + self.chunk - 1) // self.chunk
        # All offsets are in bytes relative to a peer's mapped allocation.
        self.prob = tokens * h * 2
        self.dx = (self.prob + tokens * k * 4 + 15) // 16 * 16
        self.dp = self.dx + tokens * k * h * 4
        self.flags = self.dp + tokens * k * 4
        self.combined = (self.flags + tokens * self.chunks * 4 + 15) // 16 * 16
        self.nbytes = self.combined + tokens * h * 2

    @cute.kernel
    def receive(self, base: cute.Tensor, slots: cute.Tensor, y: cute.Tensor, p: cute.Tensor):
        row, chunk, _ = cute.arch.block_idx()
        lane = cute.arch.thread_idx()[0]
        slot = slots[row]
        if cutlass.const_expr(self.local_reduce):
            # A 64-bit transaction moves four contiguous BF16 activations.
            dst = cute.recast_ptr(y.iterator, dtype=cutlass.Int64)
            for part in cutlass.range_constexpr(self.chunk // (4 * self.threads)):
                word = chunk * (self.chunk // 4) + part * self.threads + lane
                if word < self.h // 4:
                    value = cutlass.Int64(0)
                    if slot >= 0:
                        token = slot // self.k
                        src = cute.make_ptr(cutlass.Int64, base[token // self.tokens], cute.AddressSpace.gmem)
                        value = cute.arch.load(src + (token % self.tokens) * (self.h // 4) + word, cutlass.Int64)
                    cute.arch.store(dst + row * (self.h // 4) + word, value)
        else:
            for part in cutlass.range_constexpr(self.chunk // self.threads):
                col = chunk * self.chunk + part * self.threads + lane
                if col < self.h:
                    val = cutlass.BFloat16(0.0)
                    if slot >= 0:
                        token = slot // self.k
                        ptr = cute.make_ptr(cutlass.BFloat16, base[token // self.tokens], cute.AddressSpace.gmem)
                        val = cute.make_tensor(ptr, cute.make_layout(self.tokens * self.h))[(token % self.tokens) * self.h + col]
                    y[row, col] = val
        if chunk == 0 and lane == 0:
            prob = cutlass.Float32(0)
            if slot >= 0:
                ptr = cute.make_ptr(cutlass.Float32, base[slot // (self.tokens * self.k)] + self.prob, cute.AddressSpace.gmem)
                prob = cute.arch.load(ptr + slot % (self.tokens * self.k), cutlass.Float32)
            p[row] = prob

    @cute.kernel
    def publish(self, base: cute.Tensor, slots: cute.Tensor, dx: cute.Tensor, dp: cute.Tensor, generation: cute.Tensor, divisor):
        row, chunk, _ = cute.arch.block_idx()
        last = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(1), byte_alignment=4)
        lane = cute.arch.thread_idx()[0]
        slot = slots[row]
        if slot >= 0:
            peer = base[slot // (self.tokens * self.k)]
            local = slot % (self.tokens * self.k)
            ptr = cute.make_ptr(cutlass.Float32, peer + self.dx, cute.AddressSpace.gmem)
            if cutlass.const_expr(self.local_reduce):
                src = cute.recast_ptr(dx.iterator, dtype=cutlass.Int64)
                dst = cute.make_ptr(cutlass.Int64, peer + self.dx, cute.AddressSpace.gmem)
                for part in cutlass.range_constexpr(self.chunk // (2 * self.threads)):
                    word = chunk * (self.chunk // 2) + part * self.threads + lane
                    if word < self.h // 2:
                        value = cute.arch.load(src + row * (self.h // 2) + word, cutlass.Int64)
                        if cutlass.const_expr(divisor is not None):
                            if divisor[0] != 1.0:
                                pair = cute.make_rmem_tensor((2,), cutlass.Float32)
                                cute.recast_tensor(pair, cutlass.Int64)[0] = value
                                for j in cutlass.range_constexpr(2):
                                    pair[j] = div_rn_f32(pair[j], divisor[0])
                                value = cute.recast_tensor(pair, cutlass.Int64)[0]
                        cute.arch.store(dst + local * (self.h // 2) + word, value)
            else:
                for part in cutlass.range_constexpr(self.chunk // self.threads):
                    col = chunk * self.chunk + part * self.threads + lane
                    if col < self.h:
                        value = dx[row, col]
                        if cutlass.const_expr(divisor is not None):
                            if divisor[0] != 1.0:
                                value = div_rn_f32(value, divisor[0])
                        cute.arch.store(ptr + local * self.h + col, value)
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
                if cutlass.const_expr(not self.local_reduce):
                    last[0] = (prior == generation[0] * self.k + (self.k - 1)).to(cutlass.Int32)
            if cutlass.const_expr(not self.local_reduce):
                cute.arch.sync_threads()
                if last[0] != 0:
                    for part in cutlass.range_constexpr(self.chunk // self.threads):
                        col = chunk * self.chunk + part * self.threads + lane
                        if col < self.h:
                            total = cutlass.Float32(0)
                            for k in cutlass.range_constexpr(self.k):
                                total += cute.arch.load(ptr + (token * self.k + k) * self.h + col, cutlass.Float32)
                            combined = cute.make_ptr(cutlass.Uint16, peer + self.combined, cute.AddressSpace.gmem)
                            cute.arch.store(combined + token * self.h + col, total.to(cutlass.BFloat16).bitcast(cutlass.Uint16))

    @cute.kernel
    def signal_return(self, base, slots, dp):
        row = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        chunk = cute.arch.block_idx()[1]
        slot = slots[row]
        if slot >= 0:
            peer = base[slot // (self.tokens * self.k)]
            local = slot % (self.tokens * self.k)
            if chunk == 0:
                dprob = cute.make_ptr(cutlass.Float32, peer + self.dp, cute.AddressSpace.gmem)
                cute.arch.store(dprob + local, dp[row])
            # The return stream waits for the entire FC1 grid. Its FP32 peer
            # stores happen before this cumulative system release.
            counts = cute.make_ptr(cutlass.Int32, peer + self.flags, cute.AddressSpace.gmem)
            cute.arch.atomic_add(counts + local // self.k * self.chunks + chunk, cutlass.Int32(1), sem="acq_rel", scope="sys")

    @cute.jit
    def notify(self, base: cute.Tensor, slots: cute.Tensor, dp: cute.Tensor, stream: cuda.CUstream):
        self.signal_return(base, slots, dp).launch(grid=(slots.shape[0] // 128, self.chunks, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def reduce_ready(self, counts, gen, done, dx, combined):
        tid = cute.arch.thread_idx()[0]
        entry = cute.arch.block_idx()[0]
        target = (gen[0] + 1) * self.k
        decision = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(1), byte_alignment=4)
        while entry < self.tokens * self.chunks:
            if tid == 0:
                ready = cutlass.Int32(0)
                if done[entry] == 0:
                    ready = (cute.arch.load(counts.iterator + entry, cutlass.Int32, sem="acquire", scope="sys") == target).to(cutlass.Int32)
                decision[0] = ready
                if ready:
                    done[entry] = 1
            # All four warps share one acquire decision for the whole chunk.
            cute.arch.sync_threads()
            if decision[0]:
                token = entry // self.chunks
                # Form long-window slot offsets in 64 bits before pointer addition.
                if cutlass.const_expr(self.tokens * self.k * self.h > 2**31):
                    token = token.to(cutlass.Int64)
                for part in cutlass.range_constexpr(self.chunk // (4 * self.threads)):
                    col = entry % self.chunks * self.chunk + (part * self.threads + tid) * 4
                    if col < self.h:
                        total = cute.make_rmem_tensor((4,), cutlass.Float32)
                        total.fill(0.0)
                        for slot in cutlass.range_constexpr(self.k):
                            ptr = dx.iterator + (token * self.k + slot) * self.h + col
                            vals = cute.make_rmem_tensor((4,), cutlass.Float32)
                            # Peer heap regions start on 16-byte boundaries, H
                            # is a multiple of 32, and col a multiple of four.
                            cute.autovec_copy(cute.make_tensor(ptr.align(16), cute.make_layout(4)), vals)
                            total.store(total.load() + vals.load())
                        result = cute.make_rmem_tensor((4,), cutlass.BFloat16)
                        result.store(total.load().to(cutlass.BFloat16))
                        ptr = combined.iterator + token * self.h + col
                        cute.autovec_copy(result, cute.make_tensor(ptr.align(8), cute.make_layout(4)))
            cute.arch.sync_threads()
            entry += cute.arch.grid_dim()[0]

    @cute.jit
    def drain(self, counts: cute.Tensor, gen: cute.Tensor, done: cute.Tensor, dx: cute.Tensor, combined: cute.Tensor, stream: cuda.CUstream):
        self.reduce_ready(counts, gen, done, dx, combined).launch(grid=(min(self.tokens * self.chunks, 1024), 1, 1), block=(128, 1, 1), stream=stream)

    @cute.jit
    def get(self, base: cute.Tensor, slots: cute.Tensor, y: cute.Tensor, p: cute.Tensor, stream: cuda.CUstream):
        self.receive(base, slots, y, p).launch(grid=(slots.shape[0], self.chunks, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def put(self, base: cute.Tensor, slots: cute.Tensor, dx: cute.Tensor, dp: cute.Tensor, gen: cute.Tensor, divisor, stream: cuda.CUstream):
        self.publish(base, slots, dx, dp, gen, divisor).launch(grid=(slots.shape[0], self.chunks, 1), block=(self.threads, 1, 1), stream=stream)


class TileExchange:
    @staticmethod
    def _destroy_stream(device, stream):
        with torch.cuda.device(device):
            _result(cuda.cuStreamDestroy(stream))

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
        # Pooled handles can alias the caller's capture stream after pool
        # wraparound. The child graph needs two distinct, owned handles.
        with torch.cuda.device(state.device), ExitStack() as cleanup:
            streams = []
            for _ in range(2):
                (raw,) = _result(cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING))
                cleanup.callback(TileExchange._destroy_stream, state.device, raw)
                streams.append(torch.cuda.ExternalStream(int(raw), device=state.device))
            self.get_stream, self.put_stream = streams
            self._release = finalize(self, cleanup.pop_all().close)
            self._release.atexit = False
        self.events = [[torch.cuda.Event() for _ in range(state.rows // 256)] for _ in range(3)]
        self.sent = [torch.cuda.Event() for _ in range(state.rows // 256)]
        self.start = torch.cuda.Event()
        self.end = torch.cuda.Event()

        self.drain = None
        if kernel.local_reduce:
            self.done = torch.empty_like(self.flags)

            def flat(dtype, length):
                return make_fake_compact_tensor(dtype, (length,), stride_order=(0,), assumed_align=4)

            self.drain = cute.compile(
                kernel.drain,
                flat(cutlass.Int32, self.flags.numel()),
                flat(cutlass.Int32, 1),
                flat(cutlass.Int32, self.flags.numel()),
                flat(cutlass.Float32, self.dx.numel()),
                flat(cutlass.BFloat16, self.combined.numel()),
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        self.get, self.put = self.compile_transfer(kernel, 256)

    def close(self):
        self._release()

    def compile_notification(self, kernel, rows):
        def vector(dtype, length):
            return make_fake_compact_tensor(dtype, (length,), stride_order=(0,), assumed_align=dtype.width // 8)

        return cute.compile(
            kernel.notify,
            vector(cutlass.Int64, self.ptrs.numel()),
            vector(cutlass.Int64, rows),
            vector(cutlass.Float32, rows),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    def compile_transfer(self, kernel, rows, *, scale_dx=False):
        def fake(dtype, *shape):
            return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)

        stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        base = fake(cutlass.Int64, self.ptrs.numel())
        slots = fake(cutlass.Int64, rows)
        gen = fake(cutlass.Int32, 1)
        return (
            cute.compile(kernel.get, base, slots, fake(cutlass.BFloat16, rows, kernel.h), fake(cutlass.Float32, rows), stream, options="--enable-tvm-ffi"),
            cute.compile(
                kernel.put,
                base,
                slots,
                fake(cutlass.Float32, rows, kernel.h),
                fake(cutlass.Float32, rows),
                gen,
                fake(cutlass.Float32, 1) if scale_dx else None,
                stream,
                options="--enable-tvm-ffi",
            ),
        )

    def begin(self, dy, prob, overlap):
        main = torch.cuda.current_stream()
        self.reader = self.get_stream if overlap else main
        self.writer = self.put_stream if overlap else main
        self.dy[: len(dy)].copy_(dy)
        self.prob[: len(prob)].copy_(prob)
        if self.drain is not None:
            self.done.zero_()
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
        self.put(self.ptrs, slots, dx, dp, self.gen, None, stream)
        self.sent[tile].record(self.writer)

    def combine_ready(self, stream):
        if self.drain is not None:
            self.drain(self.flags, self.gen, self.done, self.dx.view(-1), self.combined.view(-1), cuda.CUstream(stream.cuda_stream))

    def finish(self, out, count):
        main = torch.cuda.current_stream()
        self.end.record(self.writer)
        main.wait_event(self.end)
        if self.handle is not None:
            self.handle.barrier()
        self.combine_ready(main)
        out.grad_activation[:count].copy_(self.combined[:count])
        out.dprob[:count].copy_(self.dp[:count])
        # All ranks finish consuming this generation before the next reuse.
        if self.handle is not None:
            self.handle.barrier()
