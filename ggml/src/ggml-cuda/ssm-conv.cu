#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// Exact decode specialization for a four-token Qwen MTP3 target step.  The
// generic graph first concatenates [c0,c1,c2] with [q0,q1,q2,q3], copies four
// overlapping three-wide rollback windows, and then runs a width-4 SSM
// convolution.  Keep the same multiply/add order while avoiding all
// intermediate traffic.
static __global__ void ssm_conv_qwen_mtp3_f32(
        const char * conv_state, const char * qkv_new, const char * weights,
        char * output,
        float * snapshot0, float * snapshot1, float * snapshot2, float * snapshot3,
        int64_t channels,
        int64_t conv_nb0, int64_t conv_nb1,
        int64_t qkv_nb0, int64_t qkv_nb1,
        int64_t weight_nb1, int64_t output_nb1) {
    ggml_cuda_pdl_lc();
    const int64_t channel = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }

    ggml_cuda_pdl_sync();
    const char * c_row = conv_state + channel * conv_nb1;
    const float c0 = *(const float *) (c_row + 0 * conv_nb0);
    const float c1 = *(const float *) (c_row + 1 * conv_nb0);
    const float c2 = *(const float *) (c_row + 2 * conv_nb0);

    const float q0 = *(const float *) (qkv_new + 0 * qkv_nb0 + channel * qkv_nb1);
    const float q1 = *(const float *) (qkv_new + 1 * qkv_nb0 + channel * qkv_nb1);
    const float q2 = *(const float *) (qkv_new + 2 * qkv_nb0 + channel * qkv_nb1);
    const float q3 = *(const float *) (qkv_new + 3 * qkv_nb0 + channel * qkv_nb1);

    const char * w_row = weights + channel * weight_nb1;
    const float w0 = *(const float *) (w_row + 0 * sizeof(float));
    const float w1 = *(const float *) (w_row + 1 * sizeof(float));
    const float w2 = *(const float *) (w_row + 2 * sizeof(float));
    const float w3 = *(const float *) (w_row + 3 * sizeof(float));

    float sum0 = 0.0f;
    sum0 += c0 * w0; sum0 += c1 * w1; sum0 += c2 * w2; sum0 += q0 * w3;
    float sum1 = 0.0f;
    sum1 += c1 * w0; sum1 += c2 * w1; sum1 += q0 * w2; sum1 += q1 * w3;
    float sum2 = 0.0f;
    sum2 += c2 * w0; sum2 += q0 * w1; sum2 += q1 * w2; sum2 += q2 * w3;
    float sum3 = 0.0f;
    sum3 += q0 * w0; sum3 += q1 * w1; sum3 += q2 * w2; sum3 += q3 * w3;

    *(float *) (output + 0 * output_nb1 + channel * sizeof(float)) = ggml_cuda_op_silu_single(sum0);
    *(float *) (output + 1 * output_nb1 + channel * sizeof(float)) = ggml_cuda_op_silu_single(sum1);
    *(float *) (output + 2 * output_nb1 + channel * sizeof(float)) = ggml_cuda_op_silu_single(sum2);
    *(float *) (output + 3 * output_nb1 + channel * sizeof(float)) = ggml_cuda_op_silu_single(sum3);

    const int64_t out = channel * 3;
    snapshot0[out + 0] = c1; snapshot0[out + 1] = c2; snapshot0[out + 2] = q0;
    snapshot1[out + 0] = c2; snapshot1[out + 1] = q0; snapshot1[out + 2] = q1;
    snapshot2[out + 0] = q0; snapshot2[out + 1] = q1; snapshot2[out + 2] = q2;
    snapshot3[out + 0] = q1; snapshot3[out + 1] = q2; snapshot3[out + 2] = q3;
}

template <size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_qwen_mtp3_snapshots_f32(
        const float * src0_ptr, const float * src1_ptr,
        float * dst_ptr,
        float * snapshot0, float * snapshot1, float * snapshot2, float * snapshot3,
        int src0_nb1, int src0_nb2, int src1_nb1,
        int dst_nb0, int dst_nb1, int dst_nb2) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;

    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float * y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);
    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };
    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; ++j) {
        w[j] = w_block[tid * stride_w + j];
        x[j] = x_block[tid * stride_x + j];
    }

    const int64_t channel = bidy * split_d_inner + tid;
    float * snapshots[4] = { snapshot0, snapshot1, snapshot2, snapshot3 };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float sumf = 0.0f;
        if (i > 0) {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }
#pragma unroll
        for (size_t j = 0; j < d_conv; ++j) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        y_block[i * stride_y + tid] = ggml_cuda_op_silu_single(sumf);

        float * snapshot = snapshots[i] + channel * 3;
        snapshot[0] = x[(i + 1) % d_conv];
        snapshot[1] = x[(i + 2) % d_conv];
        snapshot[2] = x[(i + 3) % d_conv];
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

void ggml_cuda_op_ssm_conv_qwen_mtp3(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * ssm_conv,
        ggml_tensor * silu_dst,
        const ggml_tensor * conv_state,
        const ggml_tensor * qkv_new,
        ggml_tensor * const snapshots[4]) {
    const ggml_tensor * weights = ssm_conv->src[1];
    const int64_t channels = ssm_conv->ne[0];

    GGML_ASSERT(ssm_conv->src[0]->op == GGML_OP_CONCAT);
    GGML_ASSERT(ssm_conv->src[0]->ne[0] == 7 && ssm_conv->src[0]->ne[1] == channels);
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && weights->ne[0] == 4 && weights->ne[1] == channels);
    GGML_ASSERT(conv_state->type == GGML_TYPE_F32 && conv_state->ne[0] == 3 && conv_state->ne[1] == channels);
    GGML_ASSERT(qkv_new->type == GGML_TYPE_F32 && qkv_new->ne[0] == 4 && qkv_new->ne[1] == channels);
    GGML_ASSERT(silu_dst->type == GGML_TYPE_F32 && silu_dst->ne[0] == channels && silu_dst->ne[1] == 4);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(snapshots[i]->type == GGML_TYPE_F32 && ggml_is_contiguous(snapshots[i]));
        GGML_ASSERT(ggml_nelements(snapshots[i]) == 3 * channels);
    }

    constexpr int threads = 128;
    const int64_t blocks = (channels + threads - 1) / threads;
    GGML_ASSERT(blocks <= INT_MAX);
    const ggml_cuda_kernel_launch_params launch_params((dim3) blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(ssm_conv_qwen_mtp3_f32, launch_params,
        (const char *) conv_state->data, (const char *) qkv_new->data, (const char *) weights->data,
        (char *) silu_dst->data,
        (float *) snapshots[0]->data, (float *) snapshots[1]->data,
        (float *) snapshots[2]->data, (float *) snapshots[3]->data,
        channels,
        conv_state->nb[0], conv_state->nb[1],
        qkv_new->nb[0], qkv_new->nb[1],
        weights->nb[1], silu_dst->nb[1]);
}

void ggml_cuda_op_ssm_conv_qwen_mtp3_snapshots(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * ssm_conv,
        ggml_tensor * silu_dst,
        ggml_tensor * const snapshots[4]) {
    const ggml_tensor * src0    = ssm_conv->src[0];
    const ggml_tensor * weights = ssm_conv->src[1];
    const int64_t channels = ssm_conv->ne[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src0->op == GGML_OP_CONCAT);
    GGML_ASSERT(src0->ne[0] == 7 && src0->ne[1] == channels && src0->ne[2] == 1);
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && weights->ne[0] == 4 && weights->ne[1] == channels);
    GGML_ASSERT(silu_dst->type == GGML_TYPE_F32 && silu_dst->ne[0] == channels && silu_dst->ne[1] == 4);
    GGML_ASSERT(channels % 128 == 0);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(snapshots[i]->type == GGML_TYPE_F32 && ggml_is_contiguous(snapshots[i]));
        GGML_ASSERT(ggml_nelements(snapshots[i]) == 3 * channels);
    }

    constexpr int threads = 128;
    const dim3 blocks(1, (unsigned int) (channels / threads), 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(ssm_conv_qwen_mtp3_snapshots_f32<threads, 4>, launch_params,
        (const float *) src0->data, (const float *) weights->data, (float *) silu_dst->data,
        (float *) snapshots[0]->data, (float *) snapshots[1]->data,
        (float *) snapshots[2]->data, (float *) snapshots[3]->data,
        src0->nb[1], src0->nb[2], weights->nb[1],
        silu_dst->nb[0], silu_dst->nb[1], silu_dst->nb[2]);
}
