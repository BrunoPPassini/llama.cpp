#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// Copy the four Qwen MTP3 convolution rollback windows with one launch.  Each
// source is a strided [3, channels] F32 view and each destination is a
// contiguous F32 cache view.  The graph matcher owns all structural checks.
void ggml_cuda_cpy_qwen_conv_snapshots(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * const src[4],
              ggml_tensor * const dst[4]);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
