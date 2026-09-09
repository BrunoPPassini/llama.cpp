#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "fattn-swizzle.cuh"

using namespace ggml_cuda_mma;

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 4,  64,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 4,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  32,  96,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    // TODO tune specifically for Volta
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  64,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  64,  96,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 128, 2,  32, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_cdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 1,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 256, 4,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 256, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 256, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 256, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 256, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 512, 1,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 512, 1,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  64, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  64, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

// Blackwell can profitably distribute Qwen3.8's MTP verification shape over
// four warps. Keep this override out of the inherited Ampere table so older
// architectures retain their established launch geometry.
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_blackwell(
        const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 8, 128, 2, 64, 128, 128, 128, 2, true);
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (blackwell_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_blackwell(DKQ, DV, ncols);
    }
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    if (amd_mfma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
    }
    if (amd_wmma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_blackwell(DKQ, DV, ncols);
#elif defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(AMD_MFMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    return 1; // AMD has a single column per thread.
#else
    return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static __host__ int get_cols_per_warp(const int cc) {
    if (turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc)) {
        return 16;
    } else {
        // Volta
        return 32;
    }
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc) : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2) {
#ifdef CP_ASYNC_AVAILABLE
    return ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2) : 0;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV, const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity is 16 bytes.
    constexpr int h2_per_chunk = 16/sizeof(half2);
    const int chunks_per_row = D2 / h2_per_chunk;
    if constexpr (use_cp_async) {
        static_assert(warp_size == 32, "bad warp_size");
        static_assert(!oob_check, "OOB check not compatible with cp_async");
        constexpr int preload = 64;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = warp_size >> n;
            const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    if constexpr (swz) {
                        const int smem_offs_b = ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, k*h2_per_chunk);
                        cp_async_cg_16<preload>(tile_KV_32 + smem_offs_b, KV + i*stride_KV + k*h2_per_chunk);
                    } else {
                        cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i*stride_KV + k*h2_per_chunk);
                    }
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        const half2 zero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        auto load = [&] __device__ (const int n) {
            const int stride_k = 32 >> n;
            const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    if constexpr (swz) {
                        ggml_cuda_memcpy_1<16>((char *) tile_KV + ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, k*h2_per_chunk),
                            !oob_check || i < i_sup ? KV + i*stride_KV + k*h2_per_chunk : zero);
                    } else {
                        ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4,
                            !oob_check || i < i_sup ? KV + i*stride_KV + k*h2_per_chunk : zero);
                    }
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    }
}

// Load q4_0 K/V directly into the F16 shared-memory tile consumed by MMA.
// This removes the global F16 staging write/read from the ring path while
// preserving the exact q4_0 -> F16 value presented to the tensor cores.
// `stride_KV` is measured in bytes for q4_0 and in half2 elements for F16.
template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_q4_0_load_tile(
        const char * const __restrict__ KV,
        half2 * const __restrict__ tile_KV,
        const int D2,
        const int stride_KV,
        const int i_sup,
        const int64_t row_start,
        const int h2_start) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads = nwarps * warp_size;
    const int tid = int(threadIdx.y) * warp_size + int(threadIdx.x);

    for (int linear = tid; linear < nbatch_fa * D2; linear += nthreads) {
        const int row = linear / D2;
        const int h2  = linear - row * D2;

        if constexpr (oob_check) {
            if (row >= i_sup) {
                if constexpr (swz) {
                    *reinterpret_cast<half2 *>((char *) tile_KV + ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(row, h2)) = make_half2(0.0f, 0.0f);
                } else {
                    tile_KV[row * stride_tile + h2] = make_half2(0.0f, 0.0f);
                }
                continue;
            }
        }

        const int elem0 = 2 * (h2_start + h2);
        const int elem1 = elem0 + 1;
        const block_q4_0 * const blocks = reinterpret_cast<const block_q4_0 *>(
            KV + (row_start + row) * int64_t(stride_KV));
        const int ib = elem0 / QK4_0;
        const int within0 = elem0 % QK4_0;
        const int within1 = elem1 % QK4_0;
        const float d = blocks[ib].d;
        const int packed0 = blocks[ib].qs[within0 % (QK4_0 / 2)];
        const int packed1 = blocks[ib].qs[within1 % (QK4_0 / 2)];
        const int q0 = within0 < QK4_0 / 2 ? packed0 & 0x0f : packed0 >> 4;
        const int q1 = within1 < QK4_0 / 2 ? packed1 & 0x0f : packed1 >> 4;
        const half v0 = ggml_cuda_cast<half>((q0 - 8.0f) * d);
        const half v1 = ggml_cuda_cast<half>((q1 - 8.0f) * d);
        if constexpr (swz) {
            *reinterpret_cast<half2 *>((char *) tile_KV + ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(row, h2)) = make_half2(v0, v1);
        } else {
            tile_KV[row * stride_tile + h2] = make_half2(v0, v1);
        }
    }
}

template<bool KV_q4_0, int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_load_tile(
        const half2 * const __restrict__ KV,
        half2 * const __restrict__ tile_KV,
        const int D2,
        const int stride_KV,
        const int i_sup,
        const int64_t row_start,
        const int h2_start,
        const bool kv_q4_active) {
    if constexpr (KV_q4_0) {
        if (kv_q4_active) {
            GGML_UNUSED(use_cp_async);
            flash_attn_ext_q4_0_load_tile<stride_tile, swz, nwarps, nbatch_fa, oob_check>(
                reinterpret_cast<const char *>(KV), tile_KV, D2, stride_KV,
                i_sup, row_start, h2_start);
        } else {
            flash_attn_ext_f16_load_tile<stride_tile, swz, nwarps, nbatch_fa, use_cp_async, oob_check>(
                KV + row_start * int64_t(stride_KV) + h2_start,
                tile_KV, D2, stride_KV, i_sup);
        }
    } else {
        GGML_UNUSED(kv_q4_active);
        flash_attn_ext_f16_load_tile<stride_tile, swz, nwarps, nbatch_fa, use_cp_async, oob_check>(
            KV + row_start * int64_t(stride_KV) + h2_start,
            tile_KV, D2, stride_KV, i_sup);
    }
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int i_sup, const int j0, const uint3 ne01) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*warp_size && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + int64_t(j_vram)*stride_mask + i);
        }
    } else if constexpr (oob_check) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + i] : half(0.0f);
            }
        }
    } else if constexpr (nbatch_fa < 2*warp_size) {
        constexpr int cols_per_warp = 2*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (warp_size/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*warp_size) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + i);
            }
        }
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
         bool use_logit_softcap, bool V_is_K_view, bool KV_q4_0, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup,
        const bool kv_q4_active) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages  (DKQ, DV, ncols1, ncols2);

    constexpr int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2);
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2);
    constexpr bool swz_shape = ggml_cuda_fattn_smem_swizzle::shape_supported(DKQ, DV);
    constexpr bool swz_K = swz_shape && ggml_cuda_fattn_smem_swizzle::enabled(nbatch_K2);
    constexpr bool swz_V = V_is_K_view ? swz_K : ggml_cuda_fattn_smem_swizzle::enabled(nbatch_V2);

    const int k_VKQ_0 = kb0 * nbatch_fa;
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    if constexpr (nstages > 1) {
        static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        flash_attn_ext_load_tile<KV_q4_0, stride_tile_V, swz_V, nwarps, nbatch_fa, use_cp_async, oob_check>
            (V_h2, tile_V, nbatch_V2, stride_V, k_VKQ_sup, k_VKQ_0, 0, kv_q4_active);
    } else {
        constexpr bool use_cp_async = nstages == 1;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                (mask_h + k_VKQ_0, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
        }
    }

    // For MLA K and V have the same data.
    // Therefore, iterate over K in reverse and later re-use the data if possible.
#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;

        if constexpr (nstages <= 1) {
            const int k0_diff = k0_stop - k0_start;
            constexpr bool use_cp_async = nstages == 1;
            flash_attn_ext_load_tile<KV_q4_0, stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                (K_h2, tile_K, k0_diff, stride_K, k_VKQ_sup, k_VKQ_0, k0_start, kv_q4_active);
            if (use_cp_async) {
                cp_async_wait_all();
            }
            __syncthreads();
        }

        // Calculate tile of KQ:
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_K, swz_K>(
                        K_A, tile_K, i_KQ_0, k_KQ_0 - k0_start);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        } else {
            constexpr int stride_tile_Q = DKQ/2 + 4;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                    T_A_KQ K_A;
                    ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_K, swz_K>(
                        K_A, tile_K, i_KQ_0, k_KQ_0 - k0_start);

                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        }

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

                // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
                // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory layout.
#ifdef RDNA3
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_j(l);
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
#else
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
#endif // RDNA3
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#elif defined(AMD_MFMA_AVAILABLE)
            // MFMA: 4 threads per Q column (threadIdx.x % 16 == col, spaced by 16).
            constexpr int offset_first = 32;
            constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#else // Volta
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    if constexpr (nstages > 1) {
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (ncols2 > 1 || mask_h) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (mask_h + k_VKQ_0 + nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
            }
            flash_attn_ext_load_tile<KV_q4_0, stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                (K_h2, tile_K, nbatch_K2, stride_K, k_VKQ_sup, k_VKQ_0 + nbatch_fa, 0, kv_q4_active);
        }
    }


    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;

        if constexpr (nstages <= 1) {
            const int i0_diff = i0_stop - i0_start;
            if (!V_is_K_view || i0_stop > 2*nbatch_K2) {
                constexpr bool use_cp_async = nstages == 1;
                flash_attn_ext_load_tile<KV_q4_0, stride_tile_V, swz_V, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (V_h2, tile_V, i0_diff/2, stride_V, k_VKQ_sup, k_VKQ_0, i0_start/2, kv_q4_active);
                if (use_cp_async) {
                    cp_async_wait_all();
                }
                __syncthreads();
            }
        }
        const half2 * tile_V_i = !V_is_K_view || i0_stop > 2*nbatch_K2 ? tile_V : tile_V + i0_start/2;

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                ggml_cuda_fattn_smem_swizzle::load_ldmatrix_trans<stride_tile_V, swz_V>(
                    A, tile_V, (int) (tile_V_i - tile_V) + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
#else
                    // swap A and B for CUDA.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_V, swz_V>(
                    A, tile_V, (int) (tile_V_i - tile_V) + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

#if defined(TURING_MMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int DV> struct mma_tile_sizes<DV, 8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#else
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED>; // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#else // Volta
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

static __device__ __forceinline__ void ggml_cuda_ring_fsm_wait_ready(
        uint32_t * const state, const uint32_t expected_epoch) {
    const uint32_t epoch_mask = ~uint32_t(3);
    const uint32_t expected = expected_epoch << 2;
    for (;;) {
        const uint32_t observed = atomicAdd(state, 0u);
        const uint32_t phase = observed & 3u;
        if ((observed & epoch_mask) == expected &&
                (phase == GGML_CUDA_RING_FSM_READY || phase == GGML_CUDA_RING_FSM_CONSUMING)) {
            if (phase == GGML_CUDA_RING_FSM_READY) {
                atomicCAS(state, observed,
                    ggml_cuda_ring_fsm_word(expected_epoch, GGML_CUDA_RING_FSM_CONSUMING));
            }
            return;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
}

static __device__ __forceinline__ void ggml_cuda_ring_fsm_release(
        uint32_t * const state, uint32_t * const done,
        const uint32_t epoch, const uint32_t block_count) {
    const uint32_t prior = atomicAdd(done, 1u);
    if (prior + 1u == block_count) {
        atomicExch(done, 0u);
        __threadfence_system();
        atomicExch(state, ggml_cuda_ring_fsm_word(epoch, GGML_CUDA_RING_FSM_FREE));
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool KV_q4_0, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const bool kv_q4_active,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop,
        char * const __restrict__ ring_vkq_state,
        float2 * const __restrict__ ring_meta_state,
        const bool ring_resume,
        const bool ring_intermediate,
        const char * const __restrict__ ring_fsm_K_stage,
        const char * const __restrict__ ring_fsm_V_stage,
        uint32_t * const __restrict__ ring_fsm_state,
        uint32_t * const __restrict__ ring_fsm_done,
        const int ring_fsm_hot_prefix,
        const int ring_fsm_tile_rows,
        const int ring_fsm_total_rows,
        const int ring_fsm_slots,
        const int ring_fsm_K_slot_bytes,
        const int ring_fsm_V_slot_bytes,
        const bool ring_fsm_active) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<DV, ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<DV, ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<DV, ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<DV, ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<DV, ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2);
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2);
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;
    constexpr bool swz_shape = ggml_cuda_fattn_smem_swizzle::shape_supported(DKQ, DV);
    constexpr bool swz_K = swz_shape && ggml_cuda_fattn_smem_swizzle::enabled(nbatch_K2);
    constexpr bool swz_V = V_is_K_view ? swz_K : ggml_cuda_fattn_smem_swizzle::enabled(nbatch_V2);

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);

    T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    T_C_VKQ VKQ_C[DV % 32 != 0       ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // A ring launch assigns one stable CUDA block to each output tile. Preserve
    // the raw tensor-core fragment plus the per-thread online-softmax state so
    // the next fixed-size KV segment resumes exactly where this one stopped.
    // This state depends on Q, but not on total context length.
    const int ring_tid = int(threadIdx.y) * warp_size + int(threadIdx.x);
    const size_t ring_block = size_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t ring_thread = ring_block * (nwarps * warp_size) + ring_tid;
    static_assert(sizeof(VKQ_C) <= size_t(DV) * sizeof(float2), "MMA ring state bound is too small");
    static_assert(sizeof(VKQ_C) % sizeof(uint32_t) == 0, "MMA fragment must be word aligned");

    // A null fragment pointer with a live meta pointer selects compact state:
    // keep one normalized output plus (max, rowsum) per logical output row.
    // The regular ring path retains the raw MMA fragments per CUDA thread.
    const bool ring_compact = ring_vkq_state == nullptr && ring_meta_state != nullptr;

    if (ring_resume && !ring_compact) {
        const uint32_t * src = (const uint32_t *) (ring_vkq_state + ring_thread * sizeof(VKQ_C));
        uint32_t * dst = (uint32_t *) VKQ_C;
#pragma unroll
        for (int i = 0; i < int(sizeof(VKQ_C) / sizeof(uint32_t)); ++i) {
            dst[i] = src[i];
        }
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float2 meta = ring_meta_state[ring_thread * 2 + col];
            KQ_max[col] = meta.x;
            KQ_rowsum[col] = meta.y;
        }
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    // In the persistent mode one CUDA block owns the same output tile for the
    // complete KV scan. Q, the online softmax scalars and the VKQ fragment stay
    // resident while the block advances through the state-machine slots. This
    // removes the old per-segment fragment writeback/reload and Q reload.
    const int ring_fsm_segments = ring_fsm_active ?
        (ring_fsm_total_rows + ring_fsm_tile_rows - 1) / ring_fsm_tile_rows : 1;

    for (int ring_fsm_segment = 0; ring_fsm_segment < ring_fsm_segments; ++ring_fsm_segment) {
        const int segment_start = ring_fsm_active ? ring_fsm_segment * ring_fsm_tile_rows : 0;
        const int segment_rows = ring_fsm_active ?
            min(ring_fsm_tile_rows, ring_fsm_total_rows - segment_start) : ne11;
        const bool segment_hot = !ring_fsm_active ||
            segment_start + segment_rows <= ring_fsm_hot_prefix;

        const half2 * K_iter = K_h2;
        const half2 * V_iter = V_h2;
        const half * mask_iter = mask_h;
        int ring_fsm_slot = -1;
        uint32_t ring_fsm_epoch = 0;

        if (ring_fsm_active) {
            if (segment_hot) {
                K_iter = (const half2 *) ((const char *) K_h2 + int64_t(segment_start) * stride_K);
                V_iter = V_is_K_view ? K_iter :
                    (const half2 *) ((const char *) V_h2 + int64_t(segment_start) * stride_V);
            } else {
                const int cold_segment = (segment_start - ring_fsm_hot_prefix) / ring_fsm_tile_rows;
                ring_fsm_slot = cold_segment % ring_fsm_slots;
                ring_fsm_epoch = uint32_t(cold_segment / ring_fsm_slots + 1);
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    ggml_cuda_ring_fsm_wait_ready(ring_fsm_state + ring_fsm_slot, ring_fsm_epoch);
                }
                __syncthreads();
                K_iter = (const half2 *) (ring_fsm_K_stage + int64_t(ring_fsm_slot) * ring_fsm_K_slot_bytes);
                V_iter = V_is_K_view ? K_iter :
                    (const half2 *) (ring_fsm_V_stage + int64_t(ring_fsm_slot) * ring_fsm_V_slot_bytes);
            }
            if (mask_h) {
                mask_iter = mask_h + segment_start;
            }
        }

        int kb0 = ring_fsm_active ? 0 : kb0_start;
        const int kb0_end = ring_fsm_active ?
            (segment_rows + nbatch_fa - 1) / nbatch_fa : kb0_stop;

        // Some canonical P8 partitions are temporarily empty near a segment
        // boundary.  Their accumulator may still contain all preceding ring
        // segments, so skip the KV iteration but continue through the normal
        // fragment materialization below.  The old unconditional last
        // iteration read beyond the staged tail for such partitions.
        if (kb0 < kb0_end) {
        // Preload mask and K data for first iteration when using cp_async with multiple stages:
        if constexpr (nstages > 1) {
            static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
            constexpr bool use_cp_async = true;
            constexpr bool oob_check    = false;
            constexpr int  k_VKQ_sup    = nbatch_fa;
            if (ncols2 > 1 || mask_iter) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (mask_iter + kb0*nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
            }
            flash_attn_ext_load_tile<KV_q4_0, stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                (K_iter, tile_K, nbatch_K2, stride_K, k_VKQ_sup, int64_t(kb0)*nbatch_fa, 0, kv_q4_active);
        }

        // kb0 is always smaller than kb0_end so the last iter is unconditional.
        if constexpr (ncols2 == 1) {
            constexpr bool oob_check = true;
            for (; kb0 < kb0_end-1; ++kb0) {
                constexpr bool last_iter = false;
                constexpr int  k_VKQ_sup = nbatch_fa;
                flash_attn_ext_f16_iter
                    <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup, last_iter, oob_check,
                     T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                    (Q_f2, K_iter, V_iter, mask_iter, dstk, dstk_fixup, scale, slope, logit_softcap,
                     ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                     KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, kv_q4_active);
            }
            constexpr bool last_iter = true;
            const int k_VKQ_sup = segment_rows - kb0*nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup, last_iter, oob_check,
                  T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_iter, V_iter, mask_iter, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, kv_q4_active);
        } else {
            constexpr bool oob_check = false;
            for (; kb0 < kb0_end-1; ++kb0) {
                constexpr bool last_iter = false;
                constexpr int  k_VKQ_sup = nbatch_fa;
                flash_attn_ext_f16_iter
                    <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup, last_iter, oob_check,
                     T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                    (Q_f2, K_iter, V_iter, mask_iter, dstk, dstk_fixup, scale, slope, logit_softcap,
                     ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                     KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, kv_q4_active);
            }
            constexpr bool last_iter = true;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_iter, V_iter, mask_iter, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, kv_q4_active);
        }
        }

        // Multi-stage loads may still reference shared memory. Every block must
        // retire the tile before the last block releases its slot to DMA.
        if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
            __syncthreads();
        }
        if (ring_fsm_active && !segment_hot) {
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                ggml_cuda_ring_fsm_release(
                    ring_fsm_state + ring_fsm_slot,
                    ring_fsm_done + ring_fsm_slot,
                    ring_fsm_epoch,
                    uint32_t(gridDim.x) * uint32_t(gridDim.y) * uint32_t(gridDim.z));
            }
            __syncthreads();
        }
    }

    if (ring_intermediate && !ring_compact) {
        const uint32_t * src = (const uint32_t *) VKQ_C;
        uint32_t * dst = (uint32_t *) (ring_vkq_state + ring_thread * sizeof(VKQ_C));
#pragma unroll
        for (int i = 0; i < int(sizeof(VKQ_C) / sizeof(uint32_t)); ++i) {
            dst[i] = src[i];
        }
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            ring_meta_state[ring_thread * 2 + col] = make_float2(KQ_max[col], KQ_rowsum[col]);
        }
        return;
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = (threadIdx.y/np)*cols_per_warp + (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col));
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1) {
        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        float KQ_cmn;
        float KQ_cms[nmeta];
        float KQ_crs;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;

        if (threadIdx.y % np == 0) {
            // Combine the meta data for parallel warps via shared memory.
            float2 meta[nmeta];
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
            }

            KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
                }
            }

#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
            }

            KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_crs += KQ_cms[imeta]*meta[imeta].y;
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
                }
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // Write back combined meta data:
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                    // Combined KQ max scale + rowsum.
                    meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
                    // Compact cross-segment state also needs the absolute max.
                    float * meta_values = (float *) (meta_ptr + imeta * warp_size * tile_stride/2);
                    meta_values[2] = KQ_cmn;
                }
            }

            // Combined KQ max + rowsum.
            static_assert(cols_per_warp <= warp_size);
            if (needs_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>, "bad VKQ type");
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
            if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
                if constexpr (T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR) {
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                        for (int l = 0; l < T_C_VKQ::ne; ++l) {
                            const int j = j0 + T_C_VKQ::get_i(l);
                            const int k = k1 + T_C_VKQ::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                        }
                    }
                } else {
                    static_assert(T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR_SCRAMBLED, "bad T_C_VKQ data layout");
                    using T_C_VKQ_us = tile<T_C_VKQ::I, T_C_VKQ::J, half2, DATA_LAYOUT_I_MAJOR>; // us == unscrambled
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
                        const T_C_VKQ_us VKQ_C_us = unscramble(VKQ_C[(k00 + k1)/T_C_VKQ::J]);
#pragma unroll
                        for (int l = 0; l < T_C_VKQ_us::ne; ++l) {
                            const int j = j0 + T_C_VKQ_us::get_i(l);
                            const int k = k1 + T_C_VKQ_us::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C_us.x[l];
                        }
                    }
                }
            } else {
                static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
                half * tile_Q_h = (half *) tile_Q;
#pragma unroll
                for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J/2) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = 2*k1 + T_C_VKQ::get_j(l);

                        tile_Q_h[j*(2*tile_stride) + k] = VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const int row = (jt*ncols1 + j_dst)*ne02 + c_dst;
                            const float current_max = np == 1 ? meta_j[0] : meta_j[2];
                            const float current_sum = meta_j[1];
                            if (ring_compact && ring_resume) {
                                const float2 prior_meta = ring_meta_state[row];
                                const float combined_max = fmaxf(prior_meta.x, current_max);
                                const float prior_scale = expf(prior_meta.x - combined_max);
                                const float current_scale = expf(current_max - combined_max);
                                const float combined_sum = prior_scale*prior_meta.y + current_scale*current_sum;
                                const float2 prior_value = dstk[row*(DV/2) + k00 + k];
                                dstk_val.x = (prior_scale*prior_meta.y*prior_value.x + current_scale*dstk_val.x) / combined_sum;
                                dstk_val.y = (prior_scale*prior_meta.y*prior_value.y + current_scale*dstk_val.y) / combined_sum;
                            } else {
                                dstk_val.x /= current_sum;
                                dstk_val.y /= current_sum;
                            }
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }

    const bool ring_partition_final = ring_vkq_state != nullptr && !ring_intermediate && dstk_fixup != nullptr;
    if (ring_compact || ring_partition_final) {
        // Every output element has consumed the previous metadata before it is
        // replaced. One thread commits the small semantic state for this tile.
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int jc_dst = 0; jc_dst < ncols; ++jc_dst) {
                const int j_dst = jc_dst / ncols2;
                const int c_dst = jc_dst % ncols2;
                if ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) ||
                    (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio)) {
                    continue;
                }
                const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;
                const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
                const int row = (jt*ncols1 + j_dst)*ne02 + c_dst;
                const float current_max = np == 1 ? meta_j[0] : meta_j[2];
                float2 combined = make_float2(current_max, meta_j[1]);
                if (ring_compact && ring_resume) {
                    const float2 prior = ring_meta_state[row];
                    combined.x = fmaxf(prior.x, combined.x);
                    combined.y = expf(prior.x - combined.x)*prior.y + expf(current_max - combined.x)*meta_j[1];
                }
                if (ring_compact) {
                    ring_meta_state[row] = combined;
                } else {
                    dstk_fixup[row] = combined;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop, ring_vkq_state, ring_meta_state, ring_resume, ring_intermediate,
        ring_fsm_K_stage, ring_fsm_V_stage, ring_fsm_state, ring_fsm_done,
        ring_fsm_hot_prefix, ring_fsm_tile_rows, ring_fsm_total_rows, ring_fsm_slots,
        ring_fsm_K_slot_bytes, ring_fsm_V_slot_bytes, ring_fsm_active);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool KV_q4_0 = false>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
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
        const char * K_cold_ptr,
        const char * V_cold_ptr,
        const int32_t cold_start,
        const int32_t nb12_cold,
        const int32_t nb22_cold,
        char * ring_vkq_state_ptr,
        float2 * ring_meta_state_ptr,
        const int32_t kv_start,
        const int32_t ne11_total,
        const bool ring_resume,
        const bool ring_intermediate) {
    GGML_UNUSED(ne11_total);
    ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
        NO_DEVICE_CODE;
        return;
    }
    if (DKQ == 192 && ncols2 != 8 && ncols2 != 16) {
        NO_DEVICE_CODE;
        return;
    }
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 128) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_MFMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / warp_size;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const bool ring_fsm_active = KV_q4_0 && K_cold_ptr != nullptr && V_cold_ptr != nullptr &&
        ring_vkq_state_ptr != nullptr && ring_meta_state_ptr != nullptr &&
        nb12_cold < 0 && nb22_cold < 0 && kv_start > 0 && ne11_total > 0;
    const bool kv_q4_active = KV_q4_0 &&
        (ring_fsm_active || ring_vkq_state_ptr != nullptr || ring_meta_state_ptr != nullptr);
    const int stride_K    = kv_q4_active ? nb11 : nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = V_is_K_view ? stride_K : (kv_q4_active ? nb21 : nb21 / sizeof(half2));

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    // Stateful MMA ring with a fixed number of persistent KV partitions.  The
    // x dimension continues to own one logical output tile, while y selects a
    // stable disjoint interval inside every staged KV tile.  Each partition
    // resumes its own raw tensor-core/online-softmax state across launches.
    // The host combines the normalized planes once after the last segment.
    if (!ring_fsm_active && ring_vkq_state_ptr != nullptr && gridDim.y > 1) {
        const int64_t ntiles = int64_t(iter_j) * iter_z_gqa * ne12 * ne03;
        const int64_t tile = int64_t(blockIdx.x) * ntiles / gridDim.x;
        const int sequence = tile / (iter_j*iter_z_gqa*ne12);
        const int tile_rem0 = tile - int64_t(sequence)*iter_j*iter_z_gqa*ne12;
        const int z_KV = tile_rem0 / (iter_j*iter_z_gqa);
        const int tile_rem1 = tile_rem0 - int64_t(z_KV)*iter_j*iter_z_gqa;
        const int zt_gqa = tile_rem1 / iter_j;
        const int jt = tile_rem1 - zt_gqa*iter_j;
        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2;

        // Partition the causally-visible prefix for this query, not the common
        // padded K allocation of the whole verification batch.  KV_max is
        // derived from the actual mask, so it remains valid with checkpoint
        // padding and gives logical token t identical P8 ownership in BASE(Q=1)
        // and MTP(Q>1).  Fall back to the shape formula for unmasked callers.
        const int query_index = jt*ncols1;
        const int visible_total = KV_max != nullptr
            ? KV_max[sequence*iter_j + jt]
            : max(0, ne11_total - int(ne01.z) + query_index + 1);
        const int visible_segment_rows = max(0, min(ne11, visible_total - kv_start));
        const int iter_k_visible = (visible_segment_rows + (nbatch_fa - 1)) / nbatch_fa;
        const int kb0_start = int64_t(iter_k_visible) * blockIdx.y / gridDim.y;
        const int kb0_stop  = int64_t(iter_k_visible) * (blockIdx.y + 1) / gridDim.y;

        const float2 * Q_f2 = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2 = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half2  * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33)) + kv_start;
        const float * sinks_f = sinks && blockIdx.y == 0 ? (const float *) sinks + zt_Q : nullptr;

        const size_t nrows = size_t(ne01.z) * ne02 * ne03;
        const size_t row_base = size_t(sequence)*ne01.z*ne02 + zt_Q;
        float2 * dstk = ((float2 *) dst) +
            (size_t(blockIdx.y)*nrows + row_base) * (DV/2);
        float2 * dst_meta_part = dst_meta ? dst_meta + size_t(blockIdx.y)*nrows + row_base : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;
        constexpr bool needs_fixup = false;
        constexpr bool is_fixup = false;
        flash_attn_ext_f16_process_tile
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup>
            (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta_part,
             scale, slope, logit_softcap, ne01, ne02, gqa_ratio, ne11,
             stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, kv_q4_active,
             jt, zt_gqa, kb0_start, kb0_stop,
             ring_vkq_state_ptr, ring_meta_state_ptr, ring_resume, ring_intermediate,
             nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, false);
        return;
    }

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33)) + (ring_fsm_active ? 0 : kv_start);
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);
        char * ring_accum_tile = ring_fsm_active ? nullptr : ring_vkq_state_ptr;
        float2 * ring_meta_tile = ring_fsm_active ? nullptr : ring_meta_state_ptr;
        if (!ring_fsm_active && ring_vkq_state_ptr == nullptr && ring_meta_tile != nullptr) {
            ring_meta_tile += sequence*ne01.z*ne02 + zt_Q;
        }

        const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
        const char * ring_fsm_K_head = ring_fsm_active ? K_cold_ptr + nb12*z_KV : nullptr;
        const char * ring_fsm_V_head = ring_fsm_active ? V_cold_ptr + nb22*z_KV : nullptr;
        uint32_t * ring_fsm_state = ring_fsm_active ? (uint32_t *) ring_vkq_state_ptr : nullptr;
        uint32_t * ring_fsm_done  = ring_fsm_active ? (uint32_t *) ring_meta_state_ptr : nullptr;
        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, kv_q4_active, jt, zt_gqa, kb0_start, kb0_stop,
                 ring_accum_tile, ring_meta_tile, ring_resume, ring_intermediate,
                 ring_fsm_K_head, ring_fsm_V_head, ring_fsm_state, ring_fsm_done,
                 cold_start, kv_start, ne11_total, 3, -nb12_cold, -nb22_cold, ring_fsm_active);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, kv_q4_active, jt, zt_gqa, kb0_start, kb0_stop,
                 ring_accum_tile, ring_meta_tile, ring_resume, ring_intermediate,
                 ring_fsm_K_head, ring_fsm_V_head, ring_fsm_state, ring_fsm_done,
                 cold_start, kv_start, ne11_total, 3, -nb12_cold, -nb22_cold, ring_fsm_active);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33)) + (ring_fsm_active ? 0 : kv_start);
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);
    char * ring_accum_tile = ring_fsm_active ? nullptr : ring_vkq_state_ptr;
    float2 * ring_meta_tile = ring_fsm_active ? nullptr : ring_meta_state_ptr;
    if (!ring_fsm_active && ring_vkq_state_ptr == nullptr && ring_meta_tile != nullptr) {
        ring_meta_tile += sequence*ne01.z*ne02 + zt_Q;
    }

    const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const char * ring_fsm_K_head = ring_fsm_active ? K_cold_ptr + nb12*z_KV : nullptr;
    const char * ring_fsm_V_head = ring_fsm_active ? V_cold_ptr + nb22*z_KV : nullptr;
    uint32_t * ring_fsm_state = ring_fsm_active ? (uint32_t *) ring_vkq_state_ptr : nullptr;
    uint32_t * ring_fsm_done  = ring_fsm_active ? (uint32_t *) ring_meta_state_ptr : nullptr;
    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, KV_q4_0, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, kv_q4_active, jt, zt_gqa, kb0_start, kb0_stop,
         ring_accum_tile, ring_meta_tile, ring_resume, ring_intermediate,
         ring_fsm_K_head, ring_fsm_V_head, ring_fsm_state, ring_fsm_done,
         cold_start, kv_start, ne11_total, 3, -nb12_cold, -nb22_cold, ring_fsm_active);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
        K_cold_ptr, V_cold_ptr, cold_start, nb12_cold, nb22_cold,
        ring_vkq_state_ptr, ring_meta_state_ptr, kv_start, ne11_total, ring_resume, ring_intermediate);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

    const bool swz_shape = ggml_cuda_fattn_smem_swizzle::shape_supported(DKQ, DV);
    const int stride_tile_K = swz_shape ? ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc) : nbatch_K2 + 4;
    const int stride_tile_V = V_is_K_view ? stride_tile_K :
        (swz_shape ? ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc) : nbatch_V2 + 4);
    const size_t nbytes_shared_KV_1stage = nbatch_fa * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa * (stride_tile_K + stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    const char * inline_q4_env = getenv("LLAMA_KV_HOST_RING_MMA_INLINE_Q4");
    const bool inline_q4_mma = !V_is_K_view && (ggml_cuda_fattn_inline_q4_d256(dst, cc, ncols1, ncols2) ||
        (DKQ == 128 && DV == 128 && inline_q4_env != nullptr && strcmp(inline_q4_env, "0") != 0 &&
        dst->src[1]->type == GGML_TYPE_Q4_0 && dst->src[2]->type == GGML_TYPE_Q4_0));
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        if constexpr (!V_is_K_view && ((DKQ == 128 && DV == 128) ||
                (DKQ == 256 && DV == 256 && ncols1 == 1 && ncols2 == 8))) {
            fattn_kernel = inline_q4_mma ?
                flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, true> :
                flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false>;
        } else {
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false>;
        }

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES][2] = {};
        if (!shared_memory_limit_raised[id][inline_q4_mma]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id][inline_q4_mma] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        if constexpr (!V_is_K_view && ((DKQ == 128 && DV == 128) ||
                (DKQ == 256 && DV == 256 && ncols1 == 1 && ncols2 == 8))) {
            fattn_kernel = inline_q4_mma ?
                flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, true> :
                flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false>;
        } else {
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false>;
        }

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES][2] = {};
        if (!shared_memory_limit_raised[id][inline_q4_mma]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id][inline_q4_mma] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    using ring_T_C_VKQ = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;
    constexpr size_t ring_state_count_i = DV / ring_T_C_VKQ::I;
    constexpr size_t ring_state_count_j = DV / (2 * ring_T_C_VKQ::J);
    constexpr size_t ring_state_bytes_per_thread =
        (ring_state_count_i > ring_state_count_j ? ring_state_count_i : ring_state_count_j) * sizeof(ring_T_C_VKQ);

    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, warp_size_host, true, ring_state_bytes_per_thread);
}


#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 32,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  1,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  8);

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);

// Mistral Small 4 (DKQ=320, DV=256), GQA=32-only build:
extern DECL_FATTN_MMA_F16_CASE(320, 256,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(320, 256,  2, 32);

// For GLM 4.7 Flash
extern DECL_FATTN_MMA_F16_CASE(576, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  2, 32);
