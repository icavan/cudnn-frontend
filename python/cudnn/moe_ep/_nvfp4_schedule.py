# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-controlled route windows using explicit, non-pool capture streams."""

from contextlib import contextmanager
from ctypes import c_uint64
from weakref import finalize

import cutlass
from cutlass import cute
from cuda.bindings import driver as cuda
import torch


def _result(call):
    status, *values = call
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"NVFP4 route graph: {status.name}")
    return values


_SET_PREDICATE = b"""
.version 8.3
.target sm_80
.address_size 64
.extern .func cudaGraphSetConditional(.param .b64 h, .param .b32 v);
.visible .entry set_predicate(.param .u64 handle, .param .u64 pred) {
    .reg .b64 h, addr;
    .reg .b32 v;
    ld.param.u64 h, [handle];
    ld.param.u64 addr, [pred];
    ld.global.u8 v, [addr];
    { .param .b64 arg0; .param .b32 arg1;
      st.param.b64 [arg0], h; st.param.b32 [arg1], v;
      call.uni cudaGraphSetConditional, (arg0, arg1); }
    ret;
}
"""


class RouteConditions:
    def __init__(self, device):
        (module,) = _result(cuda.cuModuleLoadData(_SET_PREDICATE))
        try:
            (self.setter,) = _result(cuda.cuModuleGetFunction(module, b"set_predicate"))
            (self.context,) = _result(cuda.cuCtxGetCurrent())
            (self.raw_stream,) = _result(cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING))
        except Exception:
            _result(cuda.cuModuleUnload(module))
            raise
        self._release = finalize(self, self._destroy, device, self.raw_stream, module)
        self.stream = torch.cuda.ExternalStream(int(self.raw_stream), device=device)

    @staticmethod
    def _destroy(device, stream, module):
        with torch.cuda.device(device):
            try:
                _result(cuda.cuStreamDestroy(stream))
            finally:
                _result(cuda.cuModuleUnload(module))

    @contextmanager
    def while_true(self, predicate):
        parent = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _, _, graph, _, _, _ = _result(cuda.cuStreamGetCaptureInfo(parent))
        (handle,) = _result(cuda.cuGraphConditionalHandleCreate(graph, self.context, 0, 0))
        args = ((int(handle), predicate.data_ptr()), (c_uint64, c_uint64))

        def set_condition():
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _result(cuda.cuLaunchKernel(self.setter, 1, 1, 1, 1, 1, 1, 0, stream, args, 0))

        set_condition()
        _, _, _, deps, edges, count = _result(cuda.cuStreamGetCaptureInfo(parent))
        params = cuda.CUgraphNodeParams()
        params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.conditional.ctx = self.context
        params.conditional.handle = handle
        params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
        params.conditional.size = 1
        (node,) = _result(cuda.cuGraphAddNode(graph, deps, edges, count, params))
        _result(
            cuda.cuStreamUpdateCaptureDependencies(parent, [node], None, 1, cuda.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES)
        )
        _result(
            cuda.cuStreamBeginCaptureToGraph(
                self.raw_stream, params.conditional.phGraph_out[0], None, None, 0, cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            )
        )
        try:
            with torch.cuda.stream(self.stream):
                yield set_condition
        finally:
            _result(cuda.cuStreamEndCapture(self.raw_stream))

    def close(self):
        self._release()


class WindowBindings:
    """Rebind caller buffers at each invocation of an address-stable eager graph."""

    def __init__(self, tensors, window, device, *, write):
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

        self.names = tuple(tensors)
        self.write = write
        self.count = len(tensors)
        self.local = torch.tensor([t.data_ptr() for t in tensors.values()], dtype=torch.long, device=device)
        self.words = torch.tensor([t.nbytes // (8 * window) for t in tensors.values()], dtype=torch.long, device=device)
        self.external = torch.empty_like(self.local)
        table = make_fake_compact_tensor(cutlass.Int64, (self.count,), stride_order=(0,), assumed_align=8)
        scalar = make_fake_compact_tensor(cutlass.Int64, (1,), stride_order=(0,), assumed_align=8)
        self.fn = cute.compile(
            self.transfer, table, table, table, scalar, cutlass.Int32(1), make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi"
        )

    def bind(self, bundle):
        for i, name in enumerate(self.names):
            self.external[i].fill_(getattr(bundle, name).data_ptr())

    @cute.kernel
    def copy_window(self, external, local, words, base, n):
        cta, field, _ = cute.arch.block_idx()
        size = words[field]
        count = size * n
        big = cute.make_ptr(cutlass.Int64, external[field], cute.AddressSpace.gmem) + base[0] * size
        small = cute.make_ptr(cutlass.Int64, local[field], cute.AddressSpace.gmem)
        large_view = cute.make_tensor(big, cute.make_layout(count))
        small_view = cute.make_tensor(small, cute.make_layout(count))
        at = cutlass.Int64(cta) * 256 + cute.arch.thread_idx()[0]
        while at < count:
            if cutlass.const_expr(self.write):
                large_view[at] = small_view[at]
            else:
                small_view[at] = large_view[at]
            at += 256 * cute.arch.grid_dim()[0]

    @cute.jit
    def transfer(self, external: cute.Tensor, local: cute.Tensor, words: cute.Tensor, base: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.copy_window(external, local, words, base, n).launch(grid=(256, self.count, 1), block=(256, 1, 1), stream=stream)

    def copy(self, base, n):
        self.fn(self.external, self.local, self.words, base.view(1), n, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
