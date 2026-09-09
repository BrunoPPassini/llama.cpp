#pragma once

#include "common.cuh"
#include "mma.cuh"

// XOR swizzle for K/V shared-memory tiles. It removes row padding and avoids
// bank conflicts when the half2 stride is a multiple of 32 (Turing+).
namespace ggml_cuda_fattn_smem_swizzle {

// The original PR enabled swizzling from the shared-memory stride alone.  A
// follow-up from the same author made this a per-kernel configuration: some
// fallback/MHA shapes can happen to have a bank-aligned internal tile without
// using an ldmatrix layout that is compatible with the XOR map.  Keep the
// optimization only on the shapes explicitly enabled by that follow-up.
static __host__ __device__ constexpr bool shape_supported(const int dkq, const int dv) {
    return (dkq ==  64 && dv ==  64) ||
           (dkq == 128 && dv == 128) ||
           (dkq == 192 && dv == 128) ||
           (dkq == 256 && dv == 256) ||
           (dkq == 320 && dv == 256) ||
           (dkq == 512 && dv == 512) ||
           (dkq == 576 && dv == 512);
}

static __host__ __device__ constexpr bool bank_aligned(const int nbatch_2) {
    return nbatch_2 >= 32 && nbatch_2 % 32 == 0;
}

static __device__ constexpr bool enabled(const int nbatch_2) {
#if defined(TURING_MMA_AVAILABLE)
    return bank_aligned(nbatch_2);
#else
    GGML_UNUSED(nbatch_2);
    return false;
#endif
}

static __host__ bool enabled(const int nbatch_2, const int cc) {
#ifdef GGML_USE_HIP
    GGML_UNUSED(nbatch_2);
    GGML_UNUSED(cc);
    return false;
#else
    return turing_mma_available(cc) && bank_aligned(nbatch_2);
#endif
}

static __device__ constexpr int tile_stride(const int nbatch_2) {
    return enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}

static __host__ int tile_stride(const int nbatch_2, const int cc) {
    return enabled(nbatch_2, cc) ? nbatch_2 : nbatch_2 + 4;
}

template<int stride_h2>
static __device__ __forceinline__ int bytes_rc(const int row, const int col_h2) {
    static_assert(bank_aligned(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    return ((row * stride_h2 + col_h2) * (int) sizeof(half2)) ^ ((row & 7) << 4);
}

static __device__ __forceinline__ void ldmatrix_x4(int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ldmatrix_x4_trans(int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif
}

template<int stride_h2>
static __device__ __forceinline__ const half2 * lane_addr(
        const half2 * tile_base, const int base_row, const int base_col_h2, const int I, const int J) {
    static_assert(bank_aligned(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    const int lane_row = threadIdx.x % I;
    const int lane_col = (threadIdx.x / I) * (J / 2);
    uint32_t byte_off = (uint32_t) ((base_row + lane_row)*stride_h2 + base_col_h2 + lane_col) * (uint32_t) sizeof(half2);
    byte_off ^= (uint32_t) (((base_row + lane_row) & 7) << 4);
    return (const half2 *) ((const char *) tile_base + byte_off);
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix(
        TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, ggml_cuda_mma::tile<16, 8, half2>>,
            "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix(TileT & t, const half2 * tile_base, const int off_h2) {
    if constexpr (swz) {
        load_ldmatrix<stride_h2, swz>(t, tile_base, off_h2 / stride_h2, off_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix(t, tile_base + off_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix_trans(
        TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, ggml_cuda_mma::tile<16, 8, half2>>,
            "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4_trans((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix_trans(TileT & t, const half2 * tile_base, const int off_h2) {
    if constexpr (swz) {
        load_ldmatrix_trans<stride_h2, swz>(t, tile_base, off_h2 / stride_h2, off_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + off_h2, stride_h2);
    }
}

} // namespace ggml_cuda_fattn_smem_swizzle
