#include "common.cuh"
#include "ggml-cuda-gdn-transaction.h"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

bool ggml_cuda_gdn_replay(const ggml_cuda_gdn_replay_args * args, int32_t count);
bool ggml_cuda_gdn_replay_on_stream(
        const ggml_cuda_gdn_replay_args * args, int32_t count, cudaStream_t stream);
bool ggml_cuda_gdn_log_prepare_on_stream(
        const ggml_cuda_gdn_log_prepare_args * args, int32_t count,
        int32_t n_keep, int32_t n_slots, cudaStream_t stream);
