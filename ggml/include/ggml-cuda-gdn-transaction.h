#pragma once

#include "ggml-backend.h"

#include <cstdint>

// Private CUDA backend ABI for exact recurrent-state transaction replay.
struct ggml_cuda_gdn_replay_args {
    const float * base;
    float *       dst;
    const float * log;
    int64_t       state_size;
    int64_t       state_dim;
    int64_t       head_count;
    int32_t       n_keep;
};

struct ggml_cuda_gdn_log_prepare_args {
    const float * src;
    float *       dst;
    int64_t       log_stride;
    int64_t       head_count;
};

using ggml_cuda_gdn_replay_t = bool (*)(const ggml_cuda_gdn_replay_args * args, int32_t count);

// Enqueue replay on stream 0 of the supplied CUDA backend. The next graph on
// the same backend is ordered after the replay without a host synchronization.
using ggml_cuda_gdn_replay_async_t = bool (*)(
        const ggml_cuda_gdn_replay_args * args, int32_t count, ggml_backend_t backend);

using ggml_cuda_gdn_log_prepare_async_t = bool (*)(
        const ggml_cuda_gdn_log_prepare_args * args, int32_t count,
        int32_t n_keep, int32_t n_slots, ggml_backend_t backend);
