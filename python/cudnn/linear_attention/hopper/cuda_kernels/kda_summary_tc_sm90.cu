// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <mma.h>

// BT16 affine summaries. H and M have separate CTAs: M starts from I and
// never reads V/Q/dO/seed. State carry, residual and triangular solve are FP32;
// projection and rank-16 update use BF16 Tensor Core operands / FP32 accumulation.
// A tile whose log prefix leaves [-20,20] takes a direct FP32 recurrence.
// This bounds factorisation and avoids subtracting huge, rounded prefixes.
namespace wmma     = nvcuda::wmma;
using bf16         = __nv_bfloat16;
constexpr int Rows = 16, Threads = 128;
// Rotate successive rows across shared-memory banks. Every WMMA pitch retains
// the required 16-byte granularity, and every fragment base is 32-byte aligned.
constexpr int BfStride = 136, StateStride = 132, SmallStride = 24;

#define PORT(type, name) type *name, long long name##0, long long name##1, long long name##2, long long name##3
#define ARG(name) name, name##0, name##1, name##2, name##3
#define PORTS                                                                                                     \
    PORT(const bf16, q), PORT(const bf16, k), PORT(const bf16, v), PORT(const float, g), PORT(const float, beta), \
        PORT(const void, cu), PORT(const float, seed), PORT(float, state), PORT(float, transition), int heads,    \
        int cu64, float scale
#define ARGS \
    ARG(q), ARG(k), ARG(v), ARG(g), ARG(beta), ARG(cu), ARG(seed), ARG(state), ARG(transition), heads, cu64, scale

using AFragment = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>;
using BFragment = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major>;
using RFragment = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major>;
using CFragment = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float
warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset);
    return x;
}

__device__ __forceinline__ float
decay(float x, float log_g) {
    return fabsf(log_g) < 0.5f ? fmaf(x, expm1f(log_g), x) : x * expf(log_g);
}

// All 32 lanes participate. B is stored as [N,K] and read transposed.
template <bool Transposed = false, int OutputStride = SmallStride>
__device__ __forceinline__ void
project(const bf16 *a, const bf16 *b, float *c) {
    AFragment af;
    BFragment bf;
    CFragment cf;
    wmma::fill_fragment(cf, 0.0f);
#pragma unroll
    for (int d = 0; d < 128; d += 16) {
        wmma::load_matrix_sync(af, a + d, BfStride);
        wmma::load_matrix_sync(bf, b + d, BfStride);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c, cf, OutputStride, Transposed ? wmma::mem_col_major : wmma::mem_row_major);
}

// Compact global coefficients. Slot i for a sequence starts at
// floor(cu[i]/16)+i: ceil(length/16) tiles fit before the next base. Therefore
// floor(T/16)+N slots cover any valid THD distribution without reading lengths
// on the host. Empty/gap slots are never consumed.
template <bool Backward>
struct __align__(32) Tile {
    bf16 a[16][128], r[16][128];
    float l[16][16], decay[128], beta[16];
    int near[128], unsafe;
    __align__(32) bf16 qr[Backward ? 16 : 1][Backward ? 128 : 1];
    __align__(32) float qk[Backward ? 16 : 1][Backward ? 16 : 1];
};
static_assert(sizeof(Tile<false>) == 10400, "forward workspace ABI");
static_assert(sizeof(Tile<true>) == 15456, "reverse workspace ABI");

__device__ __forceinline__ long long
offset(const void *cu, long long stride, int cu64, int i) {
    return cu64 ? static_cast<const long long *>(cu)[i * stride] : static_cast<const int *>(cu)[i * stride];
}

#define PREP_PORTS PORTS, void *workspace, long long chunks, int sequences
#define PREP_ARGS ARGS, workspace, chunks, sequences

template <bool Backward>
__device__ void
prepare(PREP_PORTS) {
    const int tid = threadIdx.x, warp = tid / 32;
    const long long slot = blockIdx.x % chunks;
    const int head       = blockIdx.x / chunks;
    // Identical binary searches in all lanes keep every barrier CTA-uniform.
    int low = 0, high = sequences;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (offset(cu, cu3, cu64, mid) / 16 + mid <= slot)
            low = mid + 1;
        else
            high = mid;
    }
    const int seq = low - 1;
    if (seq < 0) return;
    const long long begin = offset(cu, cu3, cu64, seq), end = offset(cu, cu3, cu64, seq + 1);
    const long long pos = (slot - (begin / 16 + seq)) * 16;
    // Reject an unused slot before narrowing: capacity can exceed int32 even
    // when the live document is empty or much shorter than its allocation.
    if (pos >= end - begin) return;
    const int count     = int(min(16LL, end - begin - pos));
    Tile<Backward> *out = static_cast<Tile<Backward> *>(workspace) + (long long)head * chunks + slot;
    __shared__ __align__(32) bf16 a[16][BfStride], b[16][BfStride];
    __shared__ int unsafe;
    if (tid == 0) unsafe = 0;
    __syncthreads();
    float prefix[16], keys[16], sum = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        long long t = Backward ? end - 1 - pos - i : begin + pos + i;
        float gi    = i < count ? g[t * g1 + head * g2 + tid] : 0;
        prefix[i]   = Backward ? sum : sum + gi;
        sum += gi;
        keys[i] = i < count ? __bfloat162float(k[t * k1 + head * k2 + tid]) : 0;
        if (!isfinite(sum) || fabsf(sum) > 20.0f) atomicExch(&unsafe, 1);
    }
    __syncthreads();
    if (tid == 0) out->unsafe = unsafe;
    if (unsafe) return;
    float inverse[16], restore[16];
    const float span_decay = exp2f(sum * 1.4426950408889634f);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        float forward  = exp2f(prefix[i] * 1.4426950408889634f);
        inverse[i]     = 1.0f / forward;
        restore[i]     = span_decay * inverse[i];
        a[i][tid]      = __float2bfloat16_rn(keys[i] * forward);
        b[i][tid]      = __float2bfloat16_rn(keys[i] * inverse[i]);
        out->a[i][tid] = a[i][tid];
        out->r[i][tid] = __float2bfloat16_rn(keys[i] * restore[i]);
    }
    out->near[tid]  = fabsf(sum) < 0.5f;
    out->decay[tid] = out->near[tid] ? expm1f(sum) : expf(sum);
    if (tid < 16) {
        long long t    = Backward ? end - 1 - pos - tid : begin + pos + tid;
        out->beta[tid] = tid < count ? beta[t * beta2 + head * beta3] : 0;
    }
    __syncthreads();
    if (warp == 0) project<true, 16>(&a[0][0], &b[0][0], &out->l[0][0]);
    __syncthreads();
    if (Backward) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            long long t     = end - 1 - pos - i;
            float qi        = i < count ? scale * __bfloat162float(q[t * q1 + head * q2 + tid]) : 0;
            b[i][tid]       = __float2bfloat16_rn(qi * inverse[i]);
            out->qr[i][tid] = __float2bfloat16_rn(qi * restore[i]);
        }
        __syncthreads();
        if (warp == 0) project<true, 16>(&a[0][0], &b[0][0], &out->qk[0][0]);
    }
    // Same-stream kernel completion publishes coefficients before consumers.
}

template <bool Backward>
__device__ void
summary(PREP_PORTS) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int groups    = (transition ? 256 : 128) / Rows;
    const int first_row = blockIdx.x % groups * Rows;
    const int head = blockIdx.x / groups % heads, seq = blockIdx.x / (groups * heads);
    const bool is_transition = first_row >= 128;
    const int row_base       = first_row % 128;
    const long long begin =
        cu64 ? static_cast<const long long *>(cu)[seq * cu3] : static_cast<const int *>(cu)[seq * cu3];
    const long long end =
        cu64 ? static_cast<const long long *>(cu)[(seq + 1) * cu3] : static_cast<const int *>(cu)[(seq + 1) * cu3];

    __shared__ __align__(32) float s[Rows][StateStride], rhs[Rows][SmallStride], l[16][SmallStride],
        qk[16][SmallStride];
    __shared__ __align__(32) bf16 packed[Rows][BfStride], a[16][BfStride], b[16][BfStride], r[16][BfStride],
        u[Rows][SmallStride], value[Rows][SmallStride];
    __shared__ float total_g[128], betas[16];
    __shared__ bool near_total[128];

    for (int i = tid; i < Rows * 128; i += Threads) {
        int row = i / 128, d = i % 128;
        s[row][d] = is_transition ? float(row_base + row == d)
                                  : (seed ? seed[seq * seed0 + head * seed1 + (row_base + row) * seed2 + d] : 0);
    }
    __syncthreads();

    for (long long pos = 0; pos < end - begin; pos += 16) {
        const int count            = int(min(16LL, end - begin - pos));
        const long long slot       = begin / 16 + seq + pos / 16;
        const Tile<Backward> *tile = static_cast<const Tile<Backward> *>(workspace) + (long long)head * chunks + slot;
        if (tile->unsafe) {
            // Uniform CTA branch, no Tensor Core operation is outstanding.
            // Preserve FP32 per-token state operands for unsafe factorisation.
            for (int row = warp; row < Rows; row += Threads / 32) {
                float ss[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) ss[j] = s[row][lane + j * 32];
                for (int i = 0; i < count; ++i) {
                    long long t = Backward ? end - 1 - pos - i : begin + pos + i;
                    float val   = is_transition ? 0 : __bfloat162float(v[t * v1 + head * v2 + row_base + row]);
                    float kk[4], gg[4], dot = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int d = lane + j * 32;
                        kk[j] = __bfloat162float(k[t * k1 + head * k2 + d]);
                        gg[j] = g[t * g1 + head * g2 + d];
                        if (Backward) {
                            if (!is_transition)
                                ss[j] = fmaf(val, scale * __bfloat162float(q[t * q1 + head * q2 + d]), ss[j]);
                        } else
                            ss[j] = decay(ss[j], gg[j]);
                        dot = fmaf(ss[j], kk[j], dot);
                    }
                    dot          = warp_sum(dot);
                    float update = beta[t * beta2 + head * beta3] * (Backward ? -dot : val - dot);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        ss[j] = fmaf(update, kk[j], ss[j]);
                        if (Backward) ss[j] = decay(ss[j], gg[j]);
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) s[row][lane + j * 32] = ss[j];
            }
            __syncthreads();
            continue;
        }

        for (int i = tid; i < 16 * 128; i += Threads) {
            int token = i / 128, d = i % 128;
            a[token][d] = tile->a[token][d];
            r[token][d] = tile->r[token][d];
            if (Backward && !is_transition) b[token][d] = tile->qr[token][d];
        }
        for (int i = tid; i < 16 * 16; i += Threads) {
            l[i / 16][i % 16] = tile->l[i / 16][i % 16];
            if (Backward && !is_transition) qk[i / 16][i % 16] = tile->qk[i / 16][i % 16];
        }
        near_total[tid] = tile->near[tid];
        total_g[tid]    = tile->decay[tid];
        if (tid < 16) betas[tid] = tile->beta[tid];
        for (int i = tid; i < Rows * 128; i += Threads)
            packed[i / 128][i % 128] = __float2bfloat16_rn(s[i / 128][i % 128]);
        for (int i = tid; i < Rows * 16; i += Threads) {
            int row = i / 16, token = i % 16;
            long long t = Backward ? end - 1 - pos - token : begin + pos + token;
            value[row][token] =
                token < count && !is_transition ? v[t * v1 + head * v2 + row_base + row] : __float2bfloat16_rn(0);
        }
        __syncthreads();
        if (warp >= 1 && warp <= Rows / 16) project(&packed[(warp - 1) * 16][0], &a[0][0], &rhs[(warp - 1) * 16][0]);
        __syncthreads();

        // Two independent rows per warp; each half-warp solves in FP32.
        for (int row = tid / 16; row < Rows; row += Threads / 16) {
            const int token = lane % 16;
            float y         = token < count ? rhs[row][token] : 0;
            if (token < count) {
                if (Backward) {
                    if (!is_transition) {
                        for (int j = 0; j <= token; ++j) y = fmaf(__bfloat162float(value[row][j]), qk[j][token], y);
                    }
                } else
                    y = __bfloat162float(value[row][token]) - y;
                y *= betas[token];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                float solved = __shfl_sync(0xffffffff, y, j, 16);
                if (token > j && token < count) y = fmaf(-betas[token] * l[j][token], solved, y);
            }
            u[row][token] = __float2bfloat16_rn(Backward ? -y : y);
        }
        for (int i = tid; i < Rows * 128; i += Threads) {
            int row = i / 128, d = i % 128;
            s[row][d] = near_total[d] ? fmaf(s[row][d], total_g[d], s[row][d]) : s[row][d] * total_g[d];
        }
        __syncthreads();

        AFragment uf, vf;
        RFragment rf, qf;
        CFragment cf;
        constexpr int WarpsPerRow = (Threads / 32) / (Rows / 16);
        constexpr int ColsPerWarp = 128 / WarpsPerRow;
        const int row = warp / WarpsPerRow * 16, col_base = warp % WarpsPerRow * ColsPerWarp;
        wmma::load_matrix_sync(uf, &u[row][0], SmallStride);
        if (Backward && !is_transition) wmma::load_matrix_sync(vf, &value[row][0], SmallStride);
#pragma unroll
        for (int d = col_base; d < col_base + ColsPerWarp; d += 16) {
            wmma::load_matrix_sync(cf, &s[row][d], StateStride, wmma::mem_row_major);
            if (Backward && !is_transition) {
                wmma::load_matrix_sync(qf, &b[0][d], BfStride);
                wmma::mma_sync(cf, vf, qf, cf);
            }
            wmma::load_matrix_sync(rf, &r[0][d], BfStride);
            wmma::mma_sync(cf, uf, rf, cf);
            wmma::store_matrix_sync(&s[row][d], cf, StateStride, wmma::mem_row_major);
        }
        // Publish updated FP32 state and protect every SMEM operand before reuse.
        __syncthreads();
    }
    for (int i = tid; i < Rows * 128; i += Threads) {
        int row = row_base + i / 128, d = i % 128;
        if (is_transition)
            transition[seq * transition0 + head * transition1 + row * transition2 + d] = s[i / 128][d];
        else
            state[seq * state0 + head * state1 + row * state2 + d] = s[i / 128][d];
    }
}

extern "C" __global__ void
cudnn_kda_summary_tc_prepare_fwd(PREP_PORTS) {
    prepare<false>(PREP_ARGS);
}
extern "C" __global__ void
cudnn_kda_summary_tc_prepare_bwd(PREP_PORTS) {
    prepare<true>(PREP_ARGS);
}
extern "C" __global__ void
cudnn_kda_summary_tc_fwd(PREP_PORTS) {
    summary<false>(PREP_ARGS);
}
extern "C" __global__ void
cudnn_kda_summary_tc_bwd(PREP_PORTS) {
    summary<true>(PREP_ARGS);
}
