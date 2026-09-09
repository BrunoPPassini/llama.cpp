#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include "vecdotq.cuh"
#include "ggml-quants.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

// Device/stream-visible state word used by the persistent long-context
// prefill ring.  The upper bits are a monotonically increasing slot epoch and
// the low two bits encode ownership.  FREE is zero so a cudaMemset initializes
// every slot for epoch zero.  The copy stream performs FREE -> COPYING -> READY;
// the persistent MMA grid performs READY -> CONSUMING -> FREE.
enum ggml_cuda_ring_fsm_phase : uint32_t {
    GGML_CUDA_RING_FSM_FREE      = 0u,
    GGML_CUDA_RING_FSM_COPYING   = 1u,
    GGML_CUDA_RING_FSM_READY     = 2u,
    GGML_CUDA_RING_FSM_CONSUMING = 3u,
};

static constexpr __host__ __device__ uint32_t ggml_cuda_ring_fsm_word(
        const uint32_t epoch, const ggml_cuda_ring_fsm_phase phase) {
    return (epoch << 2) | uint32_t(phase);
}

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const char * __restrict__ K_cold,
        const char * __restrict__ V_cold,
        const int32_t cold_start,
        const int32_t nb12_cold,
        const int32_t nb22_cold,
        char * __restrict__ ring_vkq_state,
        float2 * __restrict__ ring_meta_state,
        const int32_t kv_start,
        const int32_t ne11_total,
        const bool ring_resume,
        const bool ring_intermediate);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};

// Coalesced GPU-side copy for sparse-VMM KV pages backed by host RAM.  CUDA's
// copy engine sees both aliases as CUDA virtual addresses and can select a slow
// device-to-device path for this case.  A wide grid of vector loads gives the
// GPU enough independent PCIe reads to hide host-memory latency.  This is a
// byte-for-byte transport only; no KV conversion or arithmetic is performed.
static __global__ void ggml_cuda_copy_host_vmm_2d_u4(
        char * __restrict__ dst,
        const char * __restrict__ src,
        const size_t plane_bytes,
        const size_t src_pitch,
        const size_t n_planes) {
    const size_t vecs_per_plane = plane_bytes / sizeof(uint4);
    const size_t n_vecs = vecs_per_plane * n_planes;

    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n_vecs; i += size_t(blockDim.x)*gridDim.x) {
        const size_t plane = i / vecs_per_plane;
        const size_t vec   = i - plane*vecs_per_plane;
        const uint4 value = *(const uint4 *)(src + plane*src_pitch + vec*sizeof(uint4));
        *(uint4 *)(dst + i*sizeof(uint4)) = value;
    }
}

static inline void ggml_cuda_copy_host_vmm_2d(
        char * dst,
        const char * src,
        const size_t plane_bytes,
        const size_t src_pitch,
        const size_t n_planes,
        cudaStream_t stream) {
    GGML_ASSERT(plane_bytes % sizeof(uint4) == 0);
    GGML_ASSERT((uintptr_t) dst % alignof(uint4) == 0);
    GGML_ASSERT((uintptr_t) src % alignof(uint4) == 0);
    GGML_ASSERT(src_pitch % alignof(uint4) == 0);

    const size_t n_vecs = plane_bytes / sizeof(uint4) * n_planes;
    int blocks = 1024;
    if (const char * value = getenv("LLAMA_KV_HOST_STAGE_COLD_KERNEL_BLOCKS")) {
        char * end = nullptr;
        const long parsed = strtol(value, &end, 10);
        if (end != value && parsed > 0) {
            blocks = int(std::min<long>(parsed, 4096));
        }
    }
    blocks = int(std::min<size_t>(blocks, (n_vecs + 255) / 256));
    ggml_cuda_copy_host_vmm_2d_u4<<<blocks, 256, 0, stream>>>(
            dst, src, plane_bytes, src_pitch, n_planes);
    CUDA_CHECK(cudaGetLastError());
}

// K and V use the same q4_0 -> f16 transform in the frozen Qwen profile.
// Launching them separately repeats launch/setup overhead for every KV tile.
// A single grid covers both tensors while preserving the exact per-element
// dequantization expression and destination layout used by convert.cu.
static __global__ void ggml_cuda_dequantize_kv_q4_0_f16_kernel(
        const void * __restrict__ K_q,
        const void * __restrict__ V_q,
        half * __restrict__ K_f16,
        half * __restrict__ V_f16,
        const int64_t K_ne00, const int64_t K_ne01, const int64_t K_ne02,
        const int64_t K_s01,  const int64_t K_s02,
        const int64_t V_ne00, const int64_t V_ne01, const int64_t V_ne02,
        const int64_t V_s01,  const int64_t V_s02) {
    const bool is_v = int64_t(blockIdx.z) >= K_ne02;
    const int64_t i02 = is_v ? int64_t(blockIdx.z) - K_ne02 : int64_t(blockIdx.z);
    const int64_t ne00 = is_v ? V_ne00 : K_ne00;
    const int64_t ne01 = is_v ? V_ne01 : K_ne01;
    const int64_t ne02 = is_v ? V_ne02 : K_ne02;
    if (i02 >= ne02) {
        return;
    }

    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);
    if (i00 >= ne00) {
        return;
    }

    const void * src = is_v ? V_q : K_q;
    half * dst = is_v ? V_f16 : K_f16;
    const int64_t s01 = is_v ? V_s01 : K_s01;
    const int64_t s02 = is_v ? V_s02 : K_s02;
    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        const int64_t ibx0 = i02*s02 + i01*s01;
        const int64_t ib = ibx0 + i00/QK4_0;
        const int64_t iqs = (i00 % QK4_0)/QR4_0;
        const int64_t iybs = i00 - i00 % QK4_0;

        float2 value;
        dequantize_q4_0(src, ib, iqs, value);

        const int64_t iy0 = (i02*ne01 + i01)*ne00 + iybs + iqs;
        dst[iy0] = ggml_cuda_cast<half>(value.x);
        dst[iy0 + QK4_0/2] = ggml_cuda_cast<half>(value.y);
    }
}

static inline void ggml_cuda_dequantize_kv_q4_0_f16(
        const void * K_q, const void * V_q,
        half * K_f16, half * V_f16,
        const int64_t K_ne00, const int64_t K_ne01, const int64_t K_ne02,
        const int64_t K_s01,  const int64_t K_s02,
        const int64_t V_ne00, const int64_t V_ne01, const int64_t V_ne02,
        const int64_t V_s01,  const int64_t V_s02,
        cudaStream_t stream) {
    const int64_t ne00_max = std::max(K_ne00, V_ne00);
    const int64_t ne01_max = std::max(K_ne01, V_ne01);
    // Qwen3.x uses a 128-wide K/V head. The generic 256-thread conversion
    // block therefore retires three quarters of its threads immediately: one
    // thread expands two q4 values, so only 64 lanes contain work. Keep a
    // whole-warp block, but size it to the actual head width. This changes only
    // launch geometry; every element uses the same dequantization expression.
    int dequant_threads = CUDA_DEQUANTIZE_BLOCK_SIZE;
    const char * narrow_block_env = getenv("LLAMA_KV_HOST_RING_MMA_Q4_NARROW_BLOCK");
    if (narrow_block_env != nullptr && strcmp(narrow_block_env, "0") != 0) {
        const int useful_threads = int((ne00_max + 1) / 2);
        dequant_threads = std::max(32, std::min(
            int(CUDA_DEQUANTIZE_BLOCK_SIZE), ((useful_threads + 31) / 32) * 32));
    }
    const dim3 blocks(
        (ne00_max + 2*dequant_threads - 1)/(2*dequant_threads),
        std::min<int64_t>(ne01_max, 65535),
        K_ne02 + V_ne02);
    ggml_cuda_dequantize_kv_q4_0_f16_kernel<<<blocks, dequant_threads, 0, stream>>>(
        K_q, V_q, K_f16, V_f16,
        K_ne00, K_ne01, K_ne02, K_s01, K_s02,
        V_ne00, V_ne01, V_ne02, V_s01, V_s02);
    CUDA_CHECK(cudaGetLastError());
}

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int        * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = ((ne30 - 1) / FATTN_KQ_STRIDE) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const int i0 = KV_max_sj + 2*tid;
            const int i1 = i0 + 1;
            const float x0 = i0 < ne30 ? __half2float(mask[j*s31 + i0]) : -INFINITY;
            const float x1 = i1 < ne30 ? __half2float(mask[j*s31 + i1]) : -INFINITY;
            all_inf = all_inf && int(isinf(x0)) && int(isinf(x1));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Combine independently-normalized stateful-ring partitions.  Unlike the
// regular parallel FA path, each plane below stores a normalized output plus
// its absolute online-softmax (max, rowsum).  Keeping planes contiguous lets
// every MMA block write with the original tensor strides; this final kernel is
// the only place that needs to understand the partitioned layout.
template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results_planar(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;
    const int row      = (sequence*ne01 + col)*ne02 + head;
    const int nrows    = ne01*ne02*gridDim.z;
    const int tid      = threadIdx.x;

    extern __shared__ float2 meta[];
    for (int part = tid; part < parallel_blocks; part += D) {
        meta[part] = VKQ_meta_ptr[part*nrows + row];
    }
    __syncthreads();

    float kqmax = meta[0].x;
    for (int part = 1; part < parallel_blocks; ++part) {
        kqmax = max(kqmax, meta[part].x);
    }

    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (int part = 0; part < parallel_blocks; ++part) {
        const float scale = expf(meta[part].x - kqmax);
        const float weight = scale * meta[part].y;
        numerator   += weight * VKQ_parts_ptr[(part*nrows + row)*D + tid];
        denominator += weight;
    }

    dst_ptr[row*D + tid] = numerator / denominator;
}

// Proof-of-concept exact cold-attention consumer for CUDA 13 HOST_NUMA VMM.
// The device-resident prefix is evaluated by the normal CUDA MMA kernel.  Its
// normalized output and (max, rowsum) are copied to the host, then the CPU
// continues the same online-softmax recurrence over the CPU-addressable cold
// pages.  No KV bytes cross PCIe during decode.  This is intentionally opt-in
// and decode-only; it establishes whether avoiding PCIe is worth the CPU work
// before committing to a graph-integrated persistent implementation.
template <int D>
static void ggml_cuda_cpu_cold_attention_q4_0(
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        ggml_tensor * KQV,
        const char * K_data,
        const char * V_data,
        const int64_t hot_prefix,
        const float scale,
        const float logit_softcap,
        const float2 * hot_meta_device,
        cudaStream_t stream) {
    const auto started = std::chrono::steady_clock::now();
    const int64_t q_rows = Q->ne[1];
    const int64_t q_heads = Q->ne[2];
    const int64_t kv_heads = K->ne[2];
    const int64_t gqa = q_heads / kv_heads;
    const int64_t cold_rows = K->ne[1] - hot_prefix;
    const int64_t output_rows = ggml_nrows(KQV);

    GGML_ASSERT(Q->ne[0] == D && K->ne[0] == D && V->ne[0] == D);
    GGML_ASSERT(Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1);
    GGML_ASSERT(K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(q_heads % kv_heads == 0);
    GGML_ASSERT(output_rows == q_rows*q_heads);

    std::vector<char> q_host(ggml_nbytes(Q));
    std::vector<float> output_host(ggml_nelements(KQV));
    std::vector<float2> meta_host(output_rows);

    CUDA_CHECK(cudaMemcpyAsync(
        q_host.data(), Q->data, q_host.size(), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        output_host.data(), KQV->data, output_host.size()*sizeof(float), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        meta_host.data(), hot_meta_device, meta_host.size()*sizeof(float2), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int requested_threads = int(std::thread::hardware_concurrency());
    if (const char * value = getenv("LLAMA_KV_CPU_COLD_THREADS")) {
        const int parsed = atoi(value);
        if (parsed > 0) {
            requested_threads = parsed;
        }
    }
    const int task_count = int(q_rows*q_heads);
    const int thread_count = std::max(1, std::min(task_count, requested_threads));
    std::atomic<int> next_task { 0 };

    auto worker = [&]() {
        std::array<float, D> q_scaled;
        std::array<float, D> key;
        std::array<float, D> value;
        std::array<float, D> numerator;

        for (;;) {
            const int task = next_task.fetch_add(1, std::memory_order_relaxed);
            if (task >= task_count) {
                return;
            }

            const int64_t q_row = task / q_heads;
            const int64_t q_head = task - q_row*q_heads;
            const int64_t kv_head = q_head / gqa;
            const float * q = reinterpret_cast<const float *>(
                q_host.data() + q_row*Q->nb[1] + q_head*Q->nb[2]);
            for (int d = 0; d < D; ++d) {
                // The MMA path materializes Q and quantized K/V as F16 before
                // multiplication.  Mirror that input precision on the CPU.
                q_scaled[d] = __half2float(__float2half_rn(q[d]*scale));
            }

            const int64_t out_row = q_row*q_heads + q_head;
            float * output = output_host.data() + out_row*D;
            float max_score = meta_host[out_row].x;
            float row_sum = meta_host[out_row].y;
            for (int d = 0; d < D; ++d) {
                numerator[d] = output[d]*row_sum;
            }

            for (int64_t token = hot_prefix; token < K->ne[1]; ++token) {
                const char * k_row = K_data + token*K->nb[1] + kv_head*K->nb[2];
                const char * v_row = V_data + token*V->nb[1] + kv_head*V->nb[2];
                dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(k_row), key.data(), D);
                dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(v_row), value.data(), D);

                float score = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float k = __half2float(__float2half_rn(key[d]));
                    score = fmaf(q_scaled[d], k, score);
                    value[d] = __half2float(__float2half_rn(value[d]));
                }
                if (logit_softcap != 0.0f) {
                    score = logit_softcap*tanhf(score);
                }

                const float candidate_max = score + FATTN_KQ_MAX_OFFSET;
                if (candidate_max > max_score) {
                    const float diff = max_score - candidate_max;
                    const float old_scale = diff >= SOFTMAX_FTZ_THRESHOLD ? expf(diff) : 0.0f;
                    row_sum *= old_scale;
                    for (int d = 0; d < D; ++d) {
                        numerator[d] *= old_scale;
                    }
                    max_score = candidate_max;
                }

                const float weight = expf(score - max_score);
                row_sum += weight;
                for (int d = 0; d < D; ++d) {
                    numerator[d] = fmaf(weight, value[d], numerator[d]);
                }
            }

            const float inv_sum = 1.0f / row_sum;
            for (int d = 0; d < D; ++d) {
                output[d] = numerator[d]*inv_sum;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(thread_count > 1 ? thread_count - 1 : 0);
    for (int i = 1; i < thread_count; ++i) {
        workers.emplace_back(worker);
    }
    worker();
    for (std::thread & thread : workers) {
        thread.join();
    }

    CUDA_CHECK(cudaMemcpyAsync(
        KQV->data, output_host.data(), output_host.size()*sizeof(float), cudaMemcpyDefault, stream));

    if (getenv("LLAMA_KV_CPU_COLD_TRACE")) {
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        fprintf(stderr,
            "KV_CPU_COLD_ATTN: q_rows=%lld q_heads=%lld kv_heads=%lld cold_rows=%lld threads=%d elapsed_ms=%.3f\n",
            (long long) q_rows, (long long) q_heads, (long long) kv_heads,
            (long long) cold_rows, thread_count, elapsed_ms);
        fflush(stderr);
    }
}

static inline bool ggml_cuda_fattn_inline_q4_d256(const ggml_tensor * dst, int cc, int ncols1, int ncols2) {
    const char * value = getenv("LLAMA_KV_HOST_RING_MMA_INLINE_Q4_D256");
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    return value != nullptr && strcmp(value, "0") != 0 &&
        cc == GGML_CUDA_CC_BLACKWELL && ncols1 == 1 && ncols2 == 8 &&
        Q->ne[1] <= 8 && Q->ne[2] == 24 && Q->ne[3] == 1 &&
        K->ne[0] == 256 && V->ne[0] == 256 && K->ne[2] == 4 && V->ne[2] == 4 &&
        K->ne[3] == 1 && V->ne[3] == 1 && K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE,
    const bool mma_ring_capable = false, const size_t mma_ring_state_bytes_per_thread = 0
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V);

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    ggml_cuda_pool_alloc<char>   K_host_stage(pool);
    ggml_cuda_pool_alloc<char>   V_host_stage(pool);
    ggml_cuda_pool_alloc<char>   K_cold_stage(pool);
    ggml_cuda_pool_alloc<char>   V_cold_stage(pool);
    ggml_cuda_pool_alloc<char>   K_ring_f16(pool);
    ggml_cuda_pool_alloc<char>   V_ring_f16(pool);
    ggml_cuda_pool_alloc<char>   ring_vkq_state(pool);
    ggml_cuda_pool_alloc<float2> ring_meta_state(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    const char * K_cold_data = nullptr;
    const char * V_cold_data = nullptr;
    int32_t cold_start = INT32_MAX;
    int32_t nb12_cold = 0;
    int32_t nb22_cold = 0;

    // A sparse CUDA VMM KV cache may have a device-backed hot prefix followed by
    // host-backed pages.  The quantized vector FA kernel otherwise reads those
    // host pages directly for every query head, which turns one sequential KV
    // scan into many small PCIe transactions.  For decode, optionally make one
    // bulk copy of the exact byte span consumed by this layer and run FA from the
    // device staging buffers.  No conversion or arithmetic is performed here;
    // the staged bytes are bit-identical to the cache.
    //
    // The tile/MMA paths already materialize K/V as F16 in device memory, so
    // staging them first would only add another copy.  Keep this optimization
    // restricted to the direct quantized vector path.
    const bool stage_host_kv = getenv("LLAMA_KV_HOST_STAGE_FA") != nullptr &&
            !need_f16_K && !need_f16_V &&
            K->ne[3] == 1 && V->ne[3] == 1 &&
            K->ne[1] > 0 && K->ne[2] > 0 &&
            V->ne[1] > 0 && V->ne[2] > 0;

    if (stage_host_kv) {
        const size_t K_row_bytes = ggml_row_size(K->type, K->ne[0]);
        const size_t V_row_bytes = ggml_row_size(V->type, V->ne[0]);
        const size_t K_span = (K->ne[1] - 1)*K->nb[1] + (K->ne[2] - 1)*K->nb[2] + K_row_bytes;
        const size_t V_span = (V->ne[1] - 1)*V->nb[1] + (V->ne[2] - 1)*V->nb[2] + V_row_bytes;

        K_host_stage.alloc(K_span);
        V_host_stage.alloc(V_span);

        CUDA_CHECK(cudaMemcpyAsync(K_host_stage.ptr, K_data, K_span, cudaMemcpyDefault, main_stream));
        CUDA_CHECK(cudaMemcpyAsync(V_host_stage.ptr, V_data, V_span, cudaMemcpyDefault, main_stream));

        K_data = K_host_stage.ptr;
        V_data = V_host_stage.ptr;
    }

    // Variant for a sparse CUDA VMM cache with a large device-backed prefix:
    // keep the hot prefix in place and bulk-copy only the host-backed tail into
    // compact device buffers.  cudaMemcpy2DAsync strips the hot prefix between
    // KV heads instead of copying the much larger strided address span.  The
    // vector FA kernel still performs one numerically identical full softmax;
    // it merely selects the compact tail pointer once a 128-token tile reaches
    // cold_start.
    const char * cold_stage_env = getenv("LLAMA_KV_HOST_STAGE_COLD_FA");
    const char * hot_prefix_env = getenv("LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS");
    const int64_t hot_prefix = hot_prefix_env ? strtoll(hot_prefix_env, nullptr, 10) : 0;
    const size_t K_row_bytes = ggml_row_size(K->type, K->ne[0]);
    const size_t V_row_bytes = ggml_row_size(V->type, V->ne[0]);
    const bool K_head_major = K->nb[1] == K_row_bytes;
    const bool V_head_major = V->nb[1] == V_row_bytes;
    const bool K_token_major = K->nb[2] == K_row_bytes && K->nb[1] == K_row_bytes*size_t(K->ne[2]);
    const bool V_token_major = V->nb[2] == V_row_bytes && V->nb[1] == V_row_bytes*size_t(V->ne[2]);
    const char * ring_env = getenv("LLAMA_KV_HOST_RING_FA");
    const char * ring_mma_env = getenv("LLAMA_KV_HOST_RING_MMA");
    const char * ring_mma_cold_only_env = getenv("LLAMA_KV_HOST_RING_MMA_COLD_ONLY");
    const char * ring_mma_compact_env = getenv("LLAMA_KV_HOST_RING_MMA_COMPACT_STATE");
    const bool ring_mma_cold_only = ring_mma_cold_only_env != nullptr &&
            strcmp(ring_mma_cold_only_env, "0") != 0;
    const bool ring_mma_has_cold_tail = hot_prefix > 0 && hot_prefix < K->ne[1] && hot_prefix < V->ne[1];
    const bool ring_mma_kv = mma_ring_capable && ring_mma_env != nullptr && strcmp(ring_mma_env, "0") != 0 &&
            (!ring_mma_cold_only || ring_mma_has_cold_tail) &&
            ggml_is_quantized(K->type) && ggml_is_quantized(V->type) &&
            K->ne[3] == 1 && V->ne[3] == 1 && K->ne[1] > 0 && V->ne[1] > 0 &&
            K->ne[1] == V->ne[1] && K_token_major && V_token_major;
    float cpu_cold_max_bias = 0.0f;
    memcpy(&cpu_cold_max_bias, (const float *) KQV->op_params + 1, sizeof(float));
    const char * cpu_cold_env = getenv("LLAMA_KV_CPU_COLD_ATTN");
    const bool cpu_cold_attention =
            cpu_cold_env != nullptr && strcmp(cpu_cold_env, "0") != 0 &&
            ring_mma_kv && ring_mma_has_cold_tail && Q->ne[1] <= 8 && Q->ne[3] == 1 &&
            Q->ne[0] == DV && K->ne[0] == DV && V->ne[0] == DV &&
            K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 &&
            !V_is_K_view && sinks == nullptr && cpu_cold_max_bias == 0.0f;
    const bool ring_mma_compact = ring_mma_kv &&
            ((ring_mma_compact_env != nullptr && strcmp(ring_mma_compact_env, "0") != 0) ||
             cpu_cold_attention);
    const bool ring_cold_kv = !stage_host_kv && ring_env != nullptr && strcmp(ring_env, "0") != 0 &&
            !need_f16_K && !need_f16_V && Q->ne[1] <= 8 &&
            K->ne[3] == 1 && V->ne[3] == 1 &&
            hot_prefix > 0 && hot_prefix < K->ne[1] && hot_prefix < V->ne[1] &&
            hot_prefix <= INT32_MAX && (hot_prefix % 128) == 0 &&
            ((K_head_major && V_head_major) || (K_token_major && V_token_major));
    const bool stage_cold_kv = !stage_host_kv && cold_stage_env != nullptr &&
            !ring_cold_kv &&
            !need_f16_K && !need_f16_V &&
            K->ne[3] == 1 && V->ne[3] == 1 &&
            hot_prefix > 0 && hot_prefix < K->ne[1] && hot_prefix < V->ne[1] &&
            hot_prefix <= INT32_MAX && (hot_prefix % 128) == 0 &&
            K->nb[1] == K_row_bytes && V->nb[1] == V_row_bytes;

    if (stage_cold_kv) {
        const size_t K_cold_rows = K->ne[1] - hot_prefix;
        const size_t V_cold_rows = V->ne[1] - hot_prefix;
        const size_t K_plane_bytes = K_cold_rows * K->nb[1];
        const size_t V_plane_bytes = V_cold_rows * V->nb[1];

        GGML_ASSERT(K_plane_bytes <= INT32_MAX);
        GGML_ASSERT(V_plane_bytes <= INT32_MAX);

        K_cold_stage.alloc(K_plane_bytes * K->ne[2]);
        V_cold_stage.alloc(V_plane_bytes * V->ne[2]);

        if (getenv("LLAMA_KV_HOST_STAGE_COLD_KERNEL") != nullptr) {
            ggml_cuda_copy_host_vmm_2d(
                K_cold_stage.ptr,
                K_data + hot_prefix*K->nb[1],
                K_plane_bytes, K->nb[2], K->ne[2], main_stream);
            ggml_cuda_copy_host_vmm_2d(
                V_cold_stage.ptr,
                V_data + hot_prefix*V->nb[1],
                V_plane_bytes, V->nb[2], V->ne[2], main_stream);
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                K_cold_stage.ptr, K_plane_bytes,
                K_data + hot_prefix*K->nb[1], K->nb[2],
                K_plane_bytes, K->ne[2], cudaMemcpyDefault, main_stream));
            CUDA_CHECK(cudaMemcpy2DAsync(
                V_cold_stage.ptr, V_plane_bytes,
                V_data + hot_prefix*V->nb[1], V->nb[2],
                V_plane_bytes, V->ne[2], cudaMemcpyDefault, main_stream));
        }

        K_cold_data = K_cold_stage.ptr;
        V_cold_data = V_cold_stage.ptr;
        cold_start = (int32_t) hot_prefix;
        nb12_cold = (int32_t) K_plane_bytes;
        nb22_cold = (int32_t) V_plane_bytes;
    }

    if (!ring_mma_kv && need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(f16_extra.K != 0);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (!ring_mma_kv && need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(f16_extra.V != 0);
            half * V_f16 = (half *) f16_extra.V;
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && (Q->ne[1] >= 1024 || Q->ne[3] > 1 || (ring_mma_kv && Q->ne[1] <= 8))) {
        const int64_t s31 = mask->nb[1] / sizeof(half);
        const int64_t s33 = mask->nb[3] / sizeof(half);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half *) mask->data, KV_max.ptr, int(K->ne[1]), s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    // The stateful host-ring path deliberately uses the regular partitioned
    // launch, even when the dispatcher would normally choose stream-K.  Its
    // persistent accumulator is indexed by a stable partition id across all
    // ring tiles; a stream-K grid changes that ownership between launches.
    if (ring_mma_kv) {
        // One stable CUDA block owns each output tile across every KV segment.
        // This is required for restoring the query-dependent online-softmax
        // accumulator without stream-K fixups changing block ownership.
        parallel_blocks = 1;
        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;
    } else if (stream_k && !ring_cold_kv) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if constexpr (ncols1 == 1) {
            const char * fixed_mtp_shape_env = getenv("LLAMA_MTP_MMA_FIXED_SHAPE");
            if (fixed_mtp_shape_env != nullptr && strcmp(fixed_mtp_shape_env, "0") != 0 &&
                    cc >= GGML_CUDA_CC_BLACKWELL && Q->ne[1] <= 5) {
                blocks_num.x = ntiles_dst;
            }
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    if (ring_mma_kv) {
        int64_t tile_rows = 8192;
        if (const char * value = getenv("LLAMA_KV_HOST_RING_MMA_TILE_TOKENS")) {
            char * end = nullptr;
            const long long parsed = strtoll(value, &end, 10);
            if (end != value && parsed >= nbatch_fa) {
                tile_rows = parsed;
            }
        }
        tile_rows = std::max<int64_t>(nbatch_fa, std::min<int64_t>(tile_rows, K->ne[1]));
        tile_rows = (tile_rows / nbatch_fa) * nbatch_fa;

        const size_t layer_cache_k_rows = K->nb[1] > 0
            ? ctx.ring_mma_layer_cache_k_capacity / K->nb[1] : 0;
        const size_t layer_cache_v_rows = V->nb[1] > 0
            ? ctx.ring_mma_layer_cache_v_capacity / V->nb[1] : 0;
        const size_t layer_cache_capacity_rows = std::min(layer_cache_k_rows, layer_cache_v_rows);
        const size_t layer_cache_required_rows = hot_prefix > 0 && K->ne[1] > hot_prefix
            ? size_t(K->ne[1] - hot_prefix) : 0;
        const bool layer_cache_active =
            Q->ne[1] > 8 &&
            ctx.ring_mma_layer_cache_k != nullptr &&
            ctx.ring_mma_layer_cache_v != nullptr &&
            layer_cache_required_rows > 0 &&
            layer_cache_required_rows <= layer_cache_capacity_rows &&
            K_token_major && V_token_major;

        int ring_partitions = 1;
        if (!ring_mma_compact && Q->ne[1] <= 8) {
            if (const char * value = getenv("LLAMA_KV_HOST_RING_MMA_PARTITIONS")) {
                char * end = nullptr;
                const long parsed = strtol(value, &end, 10);
                if (end != value && parsed > 1) {
                    ring_partitions = std::min<int>(16, parsed);
                }
            }
        }

        const char * pipeline_env = getenv("LLAMA_KV_HOST_RING_MMA_PIPELINE");
        const char * prefill_pipeline_env = getenv("LLAMA_KV_HOST_RING_MMA_PREFILL_PIPELINE");
        const char * prefill_fsm_env = getenv("LLAMA_KV_HOST_RING_MMA_PREFILL_FSM");
        const bool ring_prefill_fsm = Q->ne[1] > 8 && !layer_cache_active &&
            prefill_fsm_env != nullptr && strcmp(prefill_fsm_env, "0") != 0 &&
            K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 &&
            K->ne[0] == 128 && V->ne[0] == 128;
        const bool ring_prefill_pipeline = Q->ne[1] > 8 &&
            prefill_pipeline_env != nullptr && strcmp(prefill_pipeline_env, "0") != 0;
        const bool ring_pipeline = !ring_mma_compact && (Q->ne[1] <= 8 || ring_prefill_pipeline) &&
            pipeline_env != nullptr && strcmp(pipeline_env, "0") != 0;
        const char * triple_pipeline_env = getenv("LLAMA_KV_HOST_RING_MMA_TRIPLE_PIPELINE");
        // Prefill is compute-heavy enough for a two-slot producer/consumer
        // pipeline. Keep the third copy stream decode-only: it saves staging
        // memory at large Q while overlapping H2D+dequant with tensor-core FA.
        const bool ring_triple_pipeline = ring_pipeline && (Q->ne[1] <= 8 || ring_prefill_fsm) &&
            triple_pipeline_env != nullptr && strcmp(triple_pipeline_env, "0") != 0;
        const bool ring_fsm_active = ring_prefill_fsm && ring_triple_pipeline;
        const int ring_pipeline_slots = ring_triple_pipeline ? 3 : 2;
        const int ring_stage_slots = ring_pipeline ? ring_pipeline_slots : 1;

        const char * direct_hot_env = getenv("LLAMA_KV_HOST_RING_MMA_DIRECT_HOT");
        const bool direct_hot = direct_hot_env != nullptr && strcmp(direct_hot_env, "0") != 0;
        int64_t transfer_rows = tile_rows;
        if (const char * value = getenv("LLAMA_KV_HOST_RING_MMA_TRANSFER_TOKENS")) {
            char * end = nullptr;
            const long long parsed = strtoll(value, &end, 10);
            if (end != value && parsed >= tile_rows) {
                transfer_rows = std::min<int64_t>(parsed, K->ne[1]);
            }
        }
        transfer_rows = std::max<int64_t>(tile_rows, (transfer_rows / tile_rows) * tile_rows);
        const bool ring_bulk_transfer =
            Q->ne[1] <= 8 && ring_triple_pipeline && direct_hot &&
            transfer_rows > tile_rows && hot_prefix > 0 &&
            (hot_prefix % tile_rows) == 0 && K_token_major && V_token_major &&
            getenv("LLAMA_KV_HOST_RING_MMA_DIRECT_HOST_DEQUANT") == nullptr;

        cudaStream_t ring_stage_stream = main_stream;
        cudaStream_t ring_copy_stream = main_stream;
        if (ring_pipeline) {
            if (ctx.ring_mma_stage_stream == nullptr) {
                CUDA_CHECK(cudaStreamCreateWithFlags(
                    &ctx.ring_mma_stage_stream, cudaStreamNonBlocking));
            }
            if (ring_triple_pipeline && ctx.ring_mma_copy_stream == nullptr) {
                CUDA_CHECK(cudaStreamCreateWithFlags(
                    &ctx.ring_mma_copy_stream, cudaStreamNonBlocking));
            }
            if (ctx.ring_mma_fork_event == nullptr) {
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &ctx.ring_mma_fork_event, cudaEventDisableTiming));
            }
            for (int slot = 0; slot < ring_pipeline_slots; ++slot) {
                if (ring_triple_pipeline && ctx.ring_mma_copy_ready_events[slot] == nullptr) {
                    CUDA_CHECK(cudaEventCreateWithFlags(
                        &ctx.ring_mma_copy_ready_events[slot], cudaEventDisableTiming));
                }
                if (ctx.ring_mma_ready_events[slot] == nullptr) {
                    CUDA_CHECK(cudaEventCreateWithFlags(
                        &ctx.ring_mma_ready_events[slot], cudaEventDisableTiming));
                }
                if (ctx.ring_mma_consumed_events[slot] == nullptr) {
                    CUDA_CHECK(cudaEventCreateWithFlags(
                        &ctx.ring_mma_consumed_events[slot], cudaEventDisableTiming));
                }
            }
            if (ring_bulk_transfer) {
                for (int slot = 0; slot < ggml_backend_cuda_context::ring_mma_bulk_slots; ++slot) {
                    if (ctx.ring_mma_bulk_ready_events[slot] == nullptr) {
                        CUDA_CHECK(cudaEventCreateWithFlags(
                            &ctx.ring_mma_bulk_ready_events[slot], cudaEventDisableTiming));
                    }
                    if (ctx.ring_mma_bulk_consumed_events[slot] == nullptr) {
                        CUDA_CHECK(cudaEventCreateWithFlags(
                            &ctx.ring_mma_bulk_consumed_events[slot], cudaEventDisableTiming));
                    }
                }
            }
            ring_stage_stream = ctx.ring_mma_stage_stream;
            ring_copy_stream = ring_triple_pipeline ? ctx.ring_mma_copy_stream : ring_stage_stream;

            // Bridge all prior work on the main stream into the producer.
            // This also prevents a new graph execution from reusing staging
            // storage while the previous execution still consumes it.
            CUDA_CHECK(cudaEventRecord(ctx.ring_mma_fork_event, main_stream));
            CUDA_CHECK(cudaStreamWaitEvent(
                ring_stage_stream, ctx.ring_mma_fork_event, 0));
            if (ring_triple_pipeline) {
                CUDA_CHECK(cudaStreamWaitEvent(
                    ring_copy_stream, ctx.ring_mma_fork_event, 0));
            }
        }

        const int nthreads_ring = int(block_dim.x * block_dim.y);
        const size_t K_quant_bytes = size_t(tile_rows) * K->nb[1];
        const size_t V_quant_bytes = size_t(tile_rows) * V->nb[1];
        const size_t K_transfer_bytes = size_t(transfer_rows) * K->nb[1];
        const size_t V_transfer_bytes = size_t(transfer_rows) * V->nb[1];
        const size_t K_f16_bytes = size_t(tile_rows) * K->ne[0] * K->ne[2] * sizeof(half);
        const size_t V_f16_bytes = size_t(tile_rows) * V->ne[0] * V->ne[2] * sizeof(half);
        const char * inline_q4_env = getenv("LLAMA_KV_HOST_RING_MMA_INLINE_Q4");
        const bool inline_q4_d256 = !V_is_K_view && ggml_cuda_fattn_inline_q4_d256(dst, cc, ncols1, ncols2);
        const bool inline_q4_mma = inline_q4_d256 || ((ring_fsm_active ||
            (inline_q4_env != nullptr && strcmp(inline_q4_env, "0") != 0)) &&
            K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 &&
            K->ne[0] == 128 && V->ne[0] == 128 && K->ne[3] == 1 && V->ne[3] == 1);
        if (inline_q4_d256 && getenv("LLAMA_KV_INLINE_TRACE") != nullptr) {
            fprintf(stderr, "INLINE_Q4_D256: q=%lld kv=%lld slots=%d f16_bytes_removed=%zu\n",
                (long long) Q->ne[1], (long long) K->ne[1], ring_stage_slots,
                size_t(ring_stage_slots) * (K_f16_bytes + V_f16_bytes));
        }

        // Pool objects are stack/LIFO. Allocate in declaration order so their
        // destructors release in the exact reverse order.
        if (ring_partitions > 1) {
            dst_tmp.alloc(size_t(ring_partitions) * ggml_nelements(KQV));
            dst_tmp_meta.alloc(size_t(ring_partitions) * ggml_nrows(KQV));
        }

        K_cold_stage.alloc(ring_bulk_transfer
            ? size_t(ggml_backend_cuda_context::ring_mma_bulk_slots) * K_transfer_bytes
            : size_t(ring_stage_slots) * K_quant_bytes);
        V_cold_stage.alloc(ring_bulk_transfer
            ? size_t(ggml_backend_cuda_context::ring_mma_bulk_slots) * V_transfer_bytes
            : size_t(ring_stage_slots) * V_quant_bytes);
        if (!inline_q4_mma) {
            K_ring_f16.alloc(size_t(ring_stage_slots) * K_f16_bytes);
            V_ring_f16.alloc(size_t(ring_stage_slots) * V_f16_bytes);
        }

        if (ring_fsm_active) {
            // The persistent grid retains VKQ and online-softmax state in
            // registers. Global state is only the three slot words and three
            // block-completion counters used by the producer/consumer FSM.
            ring_vkq_state.alloc(size_t(ring_pipeline_slots) * sizeof(uint32_t));
            ring_meta_state.alloc(ring_pipeline_slots);
            CUDA_CHECK(cudaMemsetAsync(
                ring_vkq_state.ptr, 0,
                size_t(ring_pipeline_slots) * sizeof(uint32_t), main_stream));
            CUDA_CHECK(cudaMemsetAsync(
                ring_meta_state.ptr, 0,
                size_t(ring_pipeline_slots) * sizeof(float2), main_stream));
        } else if (ring_mma_compact) {
            // One (max, rowsum) pair per output row; KQV itself retains the
            // normalized accumulator between KV segments.
            ring_meta_state.alloc(ggml_nrows(KQV));
        } else {
            GGML_ASSERT(mma_ring_state_bytes_per_thread > 0);
            ring_vkq_state.alloc(size_t(ntiles_dst) * ring_partitions * nthreads_ring * mma_ring_state_bytes_per_thread);
            // NVIDIA MMA uses two softmax columns per thread. AMD currently
            // does not enter this experimental path, but two is a safe bound.
            ring_meta_state.alloc(size_t(ntiles_dst) * ring_partitions * nthreads_ring * 2);
        }

        if (layer_cache_active) {
            const uint64_t content_epoch = ggml_cuda_ring_mma_layer_cache_epoch(ctx.device);
            const bool same_layout =
                ctx.ring_mma_layer_cache_epoch      == content_epoch &&
                ctx.ring_mma_layer_cache_k_key      == K_data &&
                ctx.ring_mma_layer_cache_v_key      == V_data &&
                ctx.ring_mma_layer_cache_hot_prefix == size_t(hot_prefix) &&
                ctx.ring_mma_layer_cache_k_stride   == K->nb[1] &&
                ctx.ring_mma_layer_cache_v_stride   == V->nb[1];

            // A smaller required tail means the logical context was truncated
            // or rebuilt even if the sparse VMM addresses did not move.
            const bool content_shrank =
                layer_cache_required_rows < ctx.ring_mma_layer_cache_rows;

            if (!same_layout || content_shrank) {
                ctx.ring_mma_layer_cache_epoch      = content_epoch;
                ctx.ring_mma_layer_cache_k_key      = K_data;
                ctx.ring_mma_layer_cache_v_key      = V_data;
                ctx.ring_mma_layer_cache_hot_prefix = size_t(hot_prefix);
                ctx.ring_mma_layer_cache_k_stride   = K->nb[1];
                ctx.ring_mma_layer_cache_v_stride   = V->nb[1];
                ctx.ring_mma_layer_cache_rows       = 0;
            }

            if (ctx.ring_mma_layer_cache_rows < layer_cache_required_rows) {
                const size_t row0 = ctx.ring_mma_layer_cache_rows;
                const size_t rows = layer_cache_required_rows - row0;
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *) ctx.ring_mma_layer_cache_k + row0*K->nb[1],
                    K_data + (hot_prefix + row0)*K->nb[1],
                    rows*K->nb[1], cudaMemcpyDefault, main_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *) ctx.ring_mma_layer_cache_v + row0*V->nb[1],
                    V_data + (hot_prefix + row0)*V->nb[1],
                    rows*V->nb[1], cudaMemcpyDefault, main_stream));
                ctx.ring_mma_layer_cache_rows = layer_cache_required_rows;
            }
        }

        const dim3 ring_blocks_num(ntiles_dst, ring_partitions, 1);
        const ggml_cuda_kernel_launch_params ring_launch_params =
            ggml_cuda_kernel_launch_params(ring_blocks_num, block_dim, nbytes_shared, main_stream);

        const size_t K_ts = ggml_type_size(K->type);
        const size_t V_ts = ggml_type_size(V->type);
        to_fp16_nc_cuda_t K_to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
        to_fp16_nc_cuda_t V_to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);

        int64_t segment_start = 0;
        int64_t segment_index = 0;
        bool resume = false;
        bool ring_slot_used[ggml_backend_cuda_context::ring_mma_pipeline_slots_max] = { false, false, false };
        bool bulk_slot_used[ggml_backend_cuda_context::ring_mma_bulk_slots] = { false, false };
        const char * fused_q4_env = getenv("LLAMA_KV_HOST_RING_MMA_FUSED_Q4_DEQUANT");
        const bool fused_q4_dequant = fused_q4_env != nullptr && strcmp(fused_q4_env, "0") != 0 &&
            K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 &&
            K->ne[3] == 1 && V->ne[3] == 1;
        const char * direct_host_dequant_env = getenv("LLAMA_KV_HOST_RING_MMA_DIRECT_HOST_DEQUANT");
        const bool direct_host_dequant = !inline_q4_mma && fused_q4_dequant && direct_host_dequant_env != nullptr &&
            strcmp(direct_host_dequant_env, "0") != 0;
        const char * batch_copy_env = getenv("LLAMA_KV_HOST_RING_MMA_BATCH_COPY");
        const bool batch_copy = batch_copy_env != nullptr && strcmp(batch_copy_env, "0") != 0;

        if (ring_fsm_active) {
            GGML_ASSERT(ring_pipeline_slots == 3);
            GGML_ASSERT(inline_q4_mma);
            GGML_ASSERT(K_quant_bytes <= size_t(INT32_MAX));
            GGML_ASSERT(V_quant_bytes <= size_t(INT32_MAX));

            // Make initialization visible to both FSM branches. The main
            // stream owns the persistent consumer; the copy stream owns every
            // state transition through COPYING and READY.
            CUDA_CHECK(cudaEventRecord(ctx.ring_mma_ready_events[2], main_stream));
            CUDA_CHECK(cudaStreamWaitEvent(
                ring_copy_stream, ctx.ring_mma_ready_events[2], 0));

            uint32_t * const ring_fsm_state = (uint32_t *) ring_vkq_state.ptr;
            int64_t cold_segment = 0;
            for (int64_t start = hot_prefix; start < K->ne[1]; start += tile_rows, ++cold_segment) {
                const int32_t rows = int32_t(std::min<int64_t>(tile_rows, K->ne[1] - start));
                const int slot = int(cold_segment % ring_pipeline_slots);
                const uint32_t epoch = uint32_t(cold_segment / ring_pipeline_slots + 1);
                const CUdeviceptr state_address =
                    (CUdeviceptr) (ring_fsm_state + slot);

                // Exact-value waits make illegal transitions observable and
                // prevent DMA from overwriting a tile still used by any block.
                CU_CHECK(cuStreamWaitValue32(
                    (CUstream) ring_copy_stream, state_address,
                    ggml_cuda_ring_fsm_word(epoch - 1, GGML_CUDA_RING_FSM_FREE),
                    CU_STREAM_WAIT_VALUE_EQ));
                CU_CHECK(cuStreamWriteValue32(
                    (CUstream) ring_copy_stream, state_address,
                    ggml_cuda_ring_fsm_word(epoch, GGML_CUDA_RING_FSM_COPYING),
                    CU_STREAM_WRITE_VALUE_DEFAULT));

                char * const K_slot = K_cold_stage.ptr + size_t(slot) * K_quant_bytes;
                char * const V_slot = V_cold_stage.ptr + size_t(slot) * V_quant_bytes;
                CUDA_CHECK(cudaMemcpyAsync(
                    K_slot, K_data + start*K->nb[1],
                    size_t(rows)*K->nb[1], cudaMemcpyDefault, ring_copy_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    V_slot, V_data + start*V->nb[1],
                    size_t(rows)*V->nb[1], cudaMemcpyDefault, ring_copy_stream));

                CU_CHECK(cuStreamWriteValue32(
                    (CUstream) ring_copy_stream, state_address,
                    ggml_cuda_ring_fsm_word(epoch, GGML_CUDA_RING_FSM_READY),
                    CU_STREAM_WRITE_VALUE_DEFAULT));
            }
            CUDA_CHECK(cudaEventRecord(
                ctx.ring_mma_copy_ready_events[0], ring_copy_stream));

            const int32_t first_rows = int32_t(std::min<int64_t>(tile_rows, K->ne[1]));
            ggml_cuda_kernel_launch(fattn_kernel, ring_launch_params,
                (const char *) Q->data,
                K_data,
                V_data,
                mask ? ((const char *) mask->data) : nullptr,
                sinks ? ((const char *) sinks->data) : nullptr,
                KV_max.ptr,
                (float *) KQV->data,
                nullptr,
                scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K->ne[0], first_rows, K->ne[2], 1,
                int32_t(K->nb[1]), int32_t(K->nb[2]), int64_t(K->nb[3]),
                int32_t(V->nb[1]), int32_t(V->nb[2]), int64_t(V->nb[3]),
                mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
                K_cold_stage.ptr, V_cold_stage.ptr, int32_t(hot_prefix),
                -int32_t(K_quant_bytes), -int32_t(V_quant_bytes),
                ring_vkq_state.ptr, ring_meta_state.ptr,
                int32_t(tile_rows), int32_t(K->ne[1]), true, false);
            CUDA_CHECK(cudaGetLastError());

            // Join the producer branch back into the captured graph after the
            // consumer has retired. In steady state the kernel's FSM already
            // guarantees that all required copies completed.
            CUDA_CHECK(cudaStreamWaitEvent(
                main_stream, ctx.ring_mma_copy_ready_events[0], 0));
            return;
        }

        const int64_t ring_scan_end = cpu_cold_attention ? hot_prefix : K->ne[1];
        while (segment_start < ring_scan_end) {
            const int32_t segment_rows = int32_t(std::min<int64_t>(tile_rows, ring_scan_end - segment_start));
            const size_t K_segment_bytes = size_t(segment_rows) * K->nb[1];
            const size_t V_segment_bytes = size_t(segment_rows) * V->nb[1];

            const bool segment_is_hot = direct_hot && hot_prefix > 0 &&
                segment_start + segment_rows <= hot_prefix;
            const bool segment_is_cached = layer_cache_active && segment_start >= hot_prefix &&
                size_t(segment_start - hot_prefix + segment_rows) <= ctx.ring_mma_layer_cache_rows;
            const bool segment_uses_bulk = ring_bulk_transfer && !segment_is_hot &&
                !segment_is_cached && segment_start >= hot_prefix;
            int bulk_slot = -1;
            int64_t bulk_start = 0;
            int32_t bulk_rows = 0;
            int64_t bulk_row_offset = 0;
            bool bulk_last_subtile = false;
            const char * K_quant_tile = segment_is_cached
                ? (const char *) ctx.ring_mma_layer_cache_k + size_t(segment_start - hot_prefix)*K->nb[1]
                : K_data + segment_start*K->nb[1];
            const char * V_quant_tile = segment_is_cached
                ? (const char *) ctx.ring_mma_layer_cache_v + size_t(segment_start - hot_prefix)*V->nb[1]
                : V_data + segment_start*V->nb[1];
            const int ring_slot = ring_pipeline ? int(segment_index % ring_pipeline_slots) : 0;
            char * K_quant_stage_slot = K_cold_stage.ptr + size_t(ring_slot) * K_quant_bytes;
            char * V_quant_stage_slot = V_cold_stage.ptr + size_t(ring_slot) * V_quant_bytes;
            half * K_f16_slot = inline_q4_mma ? nullptr :
                (half *) (K_ring_f16.ptr + size_t(ring_slot) * K_f16_bytes);
            half * V_f16_slot = inline_q4_mma ? nullptr :
                (half *) (V_ring_f16.ptr + size_t(ring_slot) * V_f16_bytes);

            if (segment_uses_bulk) {
                const int64_t cold_row = segment_start - hot_prefix;
                const int64_t bulk_index = cold_row / transfer_rows;
                bulk_start = hot_prefix + bulk_index * transfer_rows;
                bulk_rows = int32_t(std::min<int64_t>(transfer_rows, ring_scan_end - bulk_start));
                bulk_row_offset = segment_start - bulk_start;
                bulk_slot = int(bulk_index % ggml_backend_cuda_context::ring_mma_bulk_slots);
                bulk_last_subtile = bulk_row_offset + segment_rows >= bulk_rows;

                char * const K_bulk_slot = K_cold_stage.ptr + size_t(bulk_slot) * K_transfer_bytes;
                char * const V_bulk_slot = V_cold_stage.ptr + size_t(bulk_slot) * V_transfer_bytes;
                if (bulk_row_offset == 0) {
                    if (bulk_slot_used[bulk_slot]) {
                        CUDA_CHECK(cudaStreamWaitEvent(
                            ring_copy_stream, ctx.ring_mma_bulk_consumed_events[bulk_slot], 0));
                    }
                    CUDA_CHECK(cudaMemcpyAsync(
                        K_bulk_slot, K_data + bulk_start*K->nb[1],
                        size_t(bulk_rows)*K->nb[1], cudaMemcpyDefault, ring_copy_stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        V_bulk_slot, V_data + bulk_start*V->nb[1],
                        size_t(bulk_rows)*V->nb[1], cudaMemcpyDefault, ring_copy_stream));
                    CUDA_CHECK(cudaEventRecord(
                        ctx.ring_mma_bulk_ready_events[bulk_slot], ring_copy_stream));
                    bulk_slot_used[bulk_slot] = true;
                }
                CUDA_CHECK(cudaStreamWaitEvent(
                    ring_stage_stream, ctx.ring_mma_bulk_ready_events[bulk_slot], 0));
                K_quant_tile = K_bulk_slot + size_t(bulk_row_offset)*K->nb[1];
                V_quant_tile = V_bulk_slot + size_t(bulk_row_offset)*V->nb[1];

                // Preserve the original F16 slot lifetime independently from
                // the larger q4 DMA slot lifetime.
                if (ring_slot_used[ring_slot]) {
                    CUDA_CHECK(cudaStreamWaitEvent(
                        ring_stage_stream, ctx.ring_mma_consumed_events[ring_slot], 0));
                }
            }

            if (ring_pipeline && ring_slot_used[ring_slot] && !ring_triple_pipeline) {
                // The producer may overwrite this slot only after the main
                // stream has finished the MMA kernel that consumed it.
                CUDA_CHECK(cudaStreamWaitEvent(
                    ring_stage_stream, ctx.ring_mma_consumed_events[ring_slot], 0));
            }

            // Token-major Qwen KV rows are contiguous across all heads. One
            // bulk transfer turns host-backed VMM pages into a compact device
            // tile before dequantization and tensor-core consumption.  Pages
            // wholly inside the device-backed prefix are already resident;
            // dequantize them in place instead of paying a redundant D2D copy.
            if (!segment_is_hot && !segment_is_cached && !segment_uses_bulk) {
                if (direct_host_dequant) {
                    // The quantized cold pages are host-backed CUDA VMM. A
                    // coalesced q4 kernel consumes every byte exactly once and
                    // writes the F16 staging tile, eliminating the intermediate
                    // q4 H2D copy and its second device-memory read.
                    if (ring_pipeline && ring_slot_used[ring_slot]) {
                        CUDA_CHECK(cudaStreamWaitEvent(
                            ring_stage_stream, ctx.ring_mma_consumed_events[ring_slot], 0));
                    }
                } else {
                cudaStream_t copy_stream = ring_triple_pipeline ? ring_copy_stream : ring_stage_stream;
                if (ring_triple_pipeline && ring_slot_used[ring_slot]) {
                    // Slot reuse is gated before the copy. The following
                    // copy-ready event therefore also makes the f16 staging
                    // region safe for the dequantizer to overwrite.
                    CUDA_CHECK(cudaStreamWaitEvent(
                        copy_stream, ctx.ring_mma_consumed_events[ring_slot], 0));
                }
#if CUDART_VERSION >= 13000
                if (batch_copy) {
                    void * destinations[2] = { K_quant_stage_slot, V_quant_stage_slot };
                    const void * sources[2] = { K_quant_tile, V_quant_tile };
                    const size_t sizes[2] = { K_segment_bytes, V_segment_bytes };
                    cudaMemcpyAttributes attributes = {};
                    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
                    attributes.flags = cudaMemcpyFlagPreferOverlapWithCompute;
                    size_t attributes_index = 0;
                    CUDA_CHECK(cudaMemcpyBatchAsync(
                        destinations, sources, sizes, 2,
                        &attributes, &attributes_index, 1, copy_stream));
                } else
#else
                GGML_UNUSED(batch_copy);
#endif
                {
                CUDA_CHECK(cudaMemcpyAsync(K_quant_stage_slot, K_quant_tile,
                                           K_segment_bytes, cudaMemcpyDefault, copy_stream));
                CUDA_CHECK(cudaMemcpyAsync(V_quant_stage_slot, V_quant_tile,
                                           V_segment_bytes, cudaMemcpyDefault, copy_stream));
                }
                K_quant_tile = K_quant_stage_slot;
                V_quant_tile = V_quant_stage_slot;
                if (ring_triple_pipeline) {
                    CUDA_CHECK(cudaEventRecord(
                        ctx.ring_mma_copy_ready_events[ring_slot], copy_stream));
                    CUDA_CHECK(cudaStreamWaitEvent(
                        ring_stage_stream, ctx.ring_mma_copy_ready_events[ring_slot], 0));
                }
                }
            } else if (ring_triple_pipeline && ring_slot_used[ring_slot]) {
                // Hot tiles bypass the copy stream, so the dequantizer itself
                // must wait for the previous consumer of this slot.
                CUDA_CHECK(cudaStreamWaitEvent(
                    ring_stage_stream, ctx.ring_mma_consumed_events[ring_slot], 0));
            }

            if (inline_q4_mma) {
                // q4 bytes are consumed directly by the MMA kernel. The stage
                // stream has only the H2D copy to finish for a cold segment.
            } else if (fused_q4_dequant) {
                ggml_cuda_dequantize_kv_q4_0_f16(
                    K_quant_tile, V_quant_tile, K_f16_slot, V_f16_slot,
                    K->ne[0], segment_rows, K->ne[2],
                    K->nb[1] / K_ts, K->nb[2] / K_ts,
                    V->ne[0], segment_rows, V->ne[2],
                    V->nb[1] / V_ts, V->nb[2] / V_ts,
                    ring_stage_stream);
            } else {
                K_to_fp16(K_quant_tile, K_f16_slot,
                          K->ne[0], segment_rows, K->ne[2], 1,
                          K->nb[1] / K_ts, K->nb[2] / K_ts, 0, ring_stage_stream);
                V_to_fp16(V_quant_tile, V_f16_slot,
                          V->ne[0], segment_rows, V->ne[2], 1,
                          V->nb[1] / V_ts, V->nb[2] / V_ts, 0, ring_stage_stream);
            }

            if (ring_pipeline) {
                CUDA_CHECK(cudaEventRecord(
                    ctx.ring_mma_ready_events[ring_slot], ring_stage_stream));
                CUDA_CHECK(cudaStreamWaitEvent(
                    main_stream, ctx.ring_mma_ready_events[ring_slot], 0));
            }

            const int32_t nb11_f16 = int32_t(K->ne[0] * sizeof(half));
            const int32_t nb12_f16 = int32_t(size_t(segment_rows) * nb11_f16);
            const int64_t nb13_f16 = int64_t(K->ne[2]) * nb12_f16;
            const int32_t nb21_f16 = int32_t(V->ne[0] * sizeof(half));
            const int32_t nb22_f16 = int32_t(size_t(segment_rows) * nb21_f16);
            const int64_t nb23_f16 = int64_t(V->ne[2]) * nb22_f16;
            const char * K_kernel_tile = inline_q4_mma ? K_quant_tile : (const char *) K_f16_slot;
            const char * V_kernel_tile = inline_q4_mma ? V_quant_tile : (const char *) V_f16_slot;
            const int32_t nb11_kernel = inline_q4_mma ? int32_t(K->nb[1]) : nb11_f16;
            const int32_t nb12_kernel = inline_q4_mma ? int32_t(K->nb[2]) : nb12_f16;
            const int64_t nb13_kernel = inline_q4_mma ? int64_t(K->nb[3]) : nb13_f16;
            const int32_t nb21_kernel = inline_q4_mma ? int32_t(V->nb[1]) : nb21_f16;
            const int32_t nb22_kernel = inline_q4_mma ? int32_t(V->nb[2]) : nb22_f16;
            const int64_t nb23_kernel = inline_q4_mma ? int64_t(V->nb[3]) : nb23_f16;
            const bool intermediate = segment_start + segment_rows < ring_scan_end;

            ggml_cuda_kernel_launch(fattn_kernel, ring_launch_params,
                (const char *) Q->data,
                K_kernel_tile,
                V_kernel_tile,
                mask ? ((const char *) mask->data) : nullptr,
                intermediate ? nullptr : (sinks ? ((const char *) sinks->data) : nullptr),
                KV_max.ptr,
                ring_partitions > 1 ? dst_tmp.ptr : (float *) KQV->data,
                ring_partitions > 1 ? dst_tmp_meta.ptr : nullptr,
                scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K->ne[0], segment_rows, K->ne[2], 1, nb11_kernel, nb12_kernel, nb13_kernel,
                nb21_kernel, nb22_kernel, nb23_kernel,
                mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
                nullptr, nullptr, INT32_MAX, 0, 0,
                ring_mma_compact ? nullptr : ring_vkq_state.ptr, ring_meta_state.ptr,
                int32_t(segment_start), int32_t(K->ne[1]), resume, intermediate);
            CUDA_CHECK(cudaGetLastError());

            if (ring_pipeline) {
                CUDA_CHECK(cudaEventRecord(
                    ctx.ring_mma_consumed_events[ring_slot], main_stream));
                ring_slot_used[ring_slot] = true;
            }
            if (segment_uses_bulk && bulk_last_subtile) {
                CUDA_CHECK(cudaEventRecord(
                    ctx.ring_mma_bulk_consumed_events[bulk_slot], main_stream));
            }

            resume = true;
            segment_start += segment_rows;
            segment_index += 1;
        }

        if (cpu_cold_attention) {
            ggml_cuda_cpu_cold_attention_q4_0<DV>(
                Q, K, V, KQV, K_data, V_data, hot_prefix,
                scale, logit_softcap, ring_meta_state.ptr, main_stream);
            return;
        }

        if (ring_partitions > 1) {
            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
            const size_t nbytes_shared_combine = size_t(ring_partitions) * sizeof(float2);
            const ggml_cuda_kernel_launch_params combine_launch_params =
                ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
            ggml_cuda_kernel_launch(flash_attn_combine_results_planar<DV>, combine_launch_params,
                dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, ring_partitions);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    if (ring_cold_kv) {
        int64_t tile_rows = 8192;
        if (const char * value = getenv("LLAMA_KV_HOST_RING_TILE_TOKENS")) {
            char * end = nullptr;
            const long long parsed = strtoll(value, &end, 10);
            if (end != value && parsed >= 128) {
                tile_rows = parsed;
            }
        }
        tile_rows = std::max<int64_t>(128, std::min<int64_t>(tile_rows, K->ne[1] - hot_prefix));
        tile_rows = (tile_rows / 128) * 128;

        const int nthreads_ring = int(block_dim.x * block_dim.y);
        const int nthreads_V_ring = std::min(DV/4, 32);
        const int vkq_regs_per_thread = (DV/2) / nthreads_V_ring;
        const size_t nrows_kqv = ggml_nrows(KQV);

        const size_t K_stage_plane = size_t(tile_rows) * K->nb[1];
        const size_t V_stage_plane = size_t(tile_rows) * V->nb[1];
        const size_t K_stage_bytes = K_token_major ? K_stage_plane : K_stage_plane * K->ne[2];
        const size_t V_stage_bytes = V_token_major ? V_stage_plane : V_stage_plane * V->ne[2];

        if (parallel_blocks == 1) {
            dst_tmp.alloc(ggml_nelements(KQV));
            dst_tmp_meta.alloc(ggml_nrows(KQV));
        }

        // ggml_cuda_pool is stack/LIFO: allocate in declaration order so the
        // pool_alloc destructors release in the exact reverse order.
        K_cold_stage.alloc(K_stage_bytes);
        V_cold_stage.alloc(V_stage_bytes);
        ring_vkq_state.alloc(nrows_kqv * parallel_blocks * nthreads_ring * vkq_regs_per_thread * sizeof(float2));
        ring_meta_state.alloc(nrows_kqv * parallel_blocks * nthreads_ring);

        const dim3 ring_blocks_num(ntiles_x, parallel_blocks, ntiles_z_gqa*K->ne[2]*Q->ne[3]);
        const ggml_cuda_kernel_launch_params ring_launch_params = ggml_cuda_kernel_launch_params(ring_blocks_num, block_dim, nbytes_shared, main_stream);
        const float * ring_dst = dst_tmp.ptr;

        auto launch_ring_segment = [&](const char * K_segment, const char * V_segment,
                                       const int32_t segment_start, const int32_t segment_rows,
                                       const int32_t K_head_stride, const int32_t V_head_stride,
                                       const bool resume, const bool intermediate) {
            ggml_cuda_kernel_launch(fattn_kernel, ring_launch_params,
                (const char *) Q->data,
                K_segment,
                V_segment,
                mask ? ((const char *) mask->data) : nullptr,
                intermediate ? nullptr : (sinks ? ((const char *) sinks->data) : nullptr),
                nullptr,
                (float *) ring_dst, dst_tmp_meta.ptr,
                scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
                K->ne[0], segment_rows, K->ne[2], K->ne[3], K->nb[1], K_head_stride, K->nb[3],
                V->nb[1], V_head_stride, V->nb[3],
                mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
                nullptr, nullptr, INT32_MAX, 0, 0,
                ring_vkq_state.ptr, ring_meta_state.ptr,
                segment_start, int32_t(K->ne[1]), resume, intermediate);
            CUDA_CHECK(cudaGetLastError());
        };

        launch_ring_segment(K_data, V_data, 0, int32_t(hot_prefix), int32_t(K->nb[2]), int32_t(V->nb[2]), false, true);

        int64_t segment_start = hot_prefix;
        while (segment_start < K->ne[1]) {
            const int32_t segment_rows = int32_t(std::min<int64_t>(tile_rows, K->ne[1] - segment_start));
            const size_t K_plane_bytes = size_t(segment_rows) * K->nb[1];
            const size_t V_plane_bytes = size_t(segment_rows) * V->nb[1];

            if (K_token_major) {
                CUDA_CHECK(cudaMemcpyAsync(
                    K_cold_stage.ptr, K_data + segment_start*K->nb[1],
                    K_plane_bytes, cudaMemcpyDefault, main_stream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    K_cold_stage.ptr, K_plane_bytes,
                    K_data + segment_start*K->nb[1], K->nb[2],
                    K_plane_bytes, K->ne[2], cudaMemcpyDefault, main_stream));
            }
            if (V_token_major) {
                CUDA_CHECK(cudaMemcpyAsync(
                    V_cold_stage.ptr, V_data + segment_start*V->nb[1],
                    V_plane_bytes, cudaMemcpyDefault, main_stream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    V_cold_stage.ptr, V_plane_bytes,
                    V_data + segment_start*V->nb[1], V->nb[2],
                    V_plane_bytes, V->ne[2], cudaMemcpyDefault, main_stream));
            }

            const bool intermediate = segment_start + segment_rows < K->ne[1];
            launch_ring_segment(
                K_cold_stage.ptr, V_cold_stage.ptr,
                int32_t(segment_start), segment_rows,
                K_token_major ? int32_t(K->nb[2]) : int32_t(K_plane_bytes),
                V_token_major ? int32_t(V->nb[2]) : int32_t(V_plane_bytes),
                true, intermediate);
            segment_start += segment_rows;
        }

        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);
        const ggml_cuda_kernel_launch_params combine_launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, combine_launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
        ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        K_cold_data, V_cold_data, cold_start, nb12_cold, nb22_cold,
        nullptr, nullptr, 0, int32_t(K->ne[1]), false, false
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
