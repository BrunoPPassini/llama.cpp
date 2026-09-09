#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// Dense Qwen MTP3 decode fast path.  It consumes the two CONCAT inputs
// directly, writes the four rollback snapshots, and produces SiLU(SSM_CONV)
// without materializing the seven-wide convolution input.
void ggml_cuda_op_ssm_conv_qwen_mtp3(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * ssm_conv,
        ggml_tensor * silu_dst,
        const ggml_tensor * conv_state,
        const ggml_tensor * qkv_new,
        ggml_tensor * const snapshots[4]);

// Conservative variant: retain the materialized CONCAT and the original SSM
// arithmetic, but scatter the four rollback windows from the SSM registers.
void ggml_cuda_op_ssm_conv_qwen_mtp3_snapshots(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * ssm_conv,
        ggml_tensor * silu_dst,
        ggml_tensor * const snapshots[4]);
