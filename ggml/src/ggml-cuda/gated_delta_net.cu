#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstring>

__device__ __forceinline__ float gdn_state_update(float decay, float state, float key, float delta) {
    return decay * state + key * delta;
}

__device__ __forceinline__ float gdn_state_replay_update(float decay, float state, float key, float delta) {
    return __fmaf_rn(decay, state, __fmul_rn(key, delta));
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                      float *       txn_log,
                                      int64_t       txn_token_stride,
                                      const float * prior_log,
                                      float *       committed_state,
                                      int           prior_slots,
                                      int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // Each warp owns eight columns. q/k/g/beta are identical for all eight, so this
    // removes most of their load/instruction overhead while preserving the
    // exact per-column arithmetic order.
    constexpr int columns_per_warp = 8;
    const int      lane     = threadIdx.x;
    const int      col_base = (blockIdx.z * blockDim.y + threadIdx.y) * columns_per_warp;

    // Preserve the original modulo path when Q/K are already expanded.  When
    // the graph supplies compact grouped Q/K, map each contiguous value-head
    // group to its source Q/K head without materializing GGML_OP_REPEAT.
    const uint32_t iq1 = neqk1_magic.z == H
        ? fastmodulo(h_idx, neqk1_magic)
        : h_idx / (H / neqk1_magic.z);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    if (committed_state != nullptr) {
        committed_state += state_out_offset;
    }
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[columns_per_warp][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < columns_per_warp; ++c) {
        const int col = col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[c][r] = curr_state[col * S_v + i];
        }
    }

    if constexpr (!KDA) {
        // Apply the previously accepted speculative transaction while this
        // layer's recurrent state is already resident in registers.  The
        // arithmetic is deliberately identical to the exact eager replay.
        // prior_slots is graph-stable; unused slots are prepared as identity
        // updates (decay=1, key=delta=0).
        if (prior_log != nullptr) {
            for (int t = 0; t < prior_slots; ++t) {
                const float * token_log = prior_log + (int64_t) t * txn_token_stride;
                const float decay = token_log[h_idx];
                float key_reg[rows_per_lane];
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int row = r * warp_size + lane;
                    key_reg[r] = token_log[H + h_idx * S_v + row];
                }
#pragma unroll
                for (int c = 0; c < columns_per_warp; ++c) {
                    const int col = col_base + c;
                    const float delta = token_log[H + H * S_v + h_idx * S_v + col];
#pragma unroll
                    for (int r = 0; r < rows_per_lane; ++r) {
                        s_shard[c][r] = gdn_state_replay_update(decay, s_shard[c][r], key_reg[r], delta);
                    }
                }
            }

            // Persist exactly the accepted prefix.  Subsequent speculative
            // tokens continue from the same register image but are not stored
            // until their own transaction is accepted.
#pragma unroll
            for (int c = 0; c < columns_per_warp; ++c) {
                const int col = col_base + c;
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int row = r * warp_size + lane;
                    committed_state[col * S_v + row] = s_shard[c][r];
                }
            }
        }
    }

#pragma unroll 2
    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float g_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
            if constexpr (KDA) {
                // g is shared by every output column owned by this warp.  The
                // previous four-column kernel evaluated expf(g[i]) twice per
                // column (eight identical evaluations).  Keep the exact same
                // expf result in a register and reuse it across all columns.
                g_reg[r] = expf(g_t[i]);
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);
            float kv_shard[columns_per_warp] = {};
#pragma unroll
            for (int c = 0; c < columns_per_warp; ++c) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard[c] += s_shard[c][r] * k_reg[r];
                }
            }

            float kv_col[columns_per_warp];
#pragma unroll
            for (int p = 0; p < columns_per_warp / 2; ++p) {
                const float2 pair = warp_reduce_sum<warp_size>(make_float2(kv_shard[2*p], kv_shard[2*p + 1]));
                kv_col[2*p]     = pair.x;
                kv_col[2*p + 1] = pair.y;
            }
            float delta_col[columns_per_warp];
            float attn_partial[columns_per_warp] = {};

            if (txn_log != nullptr && blockIdx.z == 0 && threadIdx.y == 0) {
                float * token_log = txn_log + t * txn_token_stride;
                if (lane == 0) {
                    token_log[h_idx] = g_val;
                }
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int i = r * warp_size + lane;
                    token_log[H + h_idx * S_v + i] = k_reg[r];
                }
            }

#pragma unroll
            for (int c = 0; c < columns_per_warp; ++c) {
                const int col = col_base + c;
                delta_col[c] = (v_t[col] - g_val * kv_col[c]) * beta_val;
                if (txn_log != nullptr && lane == 0) {
                    float * token_log = txn_log + t * txn_token_stride;
                    token_log[H + H * S_v + h_idx * S_v + col] = delta_col[c];
                }
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = gdn_state_update(g_val, s_shard[c][r], k_reg[r], delta_col[c]);
                    attn_partial[c] += s_shard[c][r] * q_reg[r];
                }
            }

            float attn_col[columns_per_warp];
#pragma unroll
            for (int p = 0; p < columns_per_warp / 2; ++p) {
                const float2 pair = warp_reduce_sum<warp_size>(make_float2(attn_partial[2*p], attn_partial[2*p + 1]));
                attn_col[2*p]     = pair.x;
                attn_col[2*p + 1] = pair.y;
            }
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < columns_per_warp; ++c) {
                    attn_data[col_base + c] = attn_col[c] * scale;
                }
            }
        } else {
            // KDA has a distinct decay per state row, but adjacent output
            // columns still share the same q/k/g inputs.  Reduce two columns
            // with float2 shuffles: each component follows exactly the same
            // add order as the scalar reduction, while halving shuffle issue
            // count and exposing independent FMAs to the scheduler.
#pragma unroll
            for (int p = 0; p < columns_per_warp / 2; ++p) {
                constexpr int pair_width = 2;
                const int c0 = pair_width * p;
                const int c1 = c0 + 1;
                const int col0 = col_base + c0;
                const int col1 = col_base + c1;
                float kv0 = 0.0f;
                float kv1 = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    // Keep the original multiply association.  Hoisting g*k
                    // increased register pressure and was slower on SM120.
                    kv0 += g_reg[r] * s_shard[c0][r] * k_reg[r];
                    kv1 += g_reg[r] * s_shard[c1][r] * k_reg[r];
                }
                const float2 kv = warp_reduce_sum<warp_size>(make_float2(kv0, kv1));
                const float delta0 = (v_t[col0] - kv.x) * beta_val;
                const float delta1 = (v_t[col1] - kv.y) * beta_val;
                float attn0 = 0.0f;
                float attn1 = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c0][r] = g_reg[r] * s_shard[c0][r] + k_reg[r] * delta0;
                    s_shard[c1][r] = g_reg[r] * s_shard[c1][r] + k_reg[r] * delta1;
                    attn0 += s_shard[c0][r] * q_reg[r];
                    attn1 += s_shard[c1][r] * q_reg[r];
                }
                const float2 attn = warp_reduce_sum<warp_size>(make_float2(attn0, attn1));
                if (lane == 0) {
                    attn_data[col0] = attn.x * scale;
                    attn_data[col1] = attn.y * scale;
                }
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
    #pragma unroll
                for (int c = 0; c < columns_per_warp; ++c) {
                    const int col = col_base + c;
                    float * curr_state_out = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state_out[col * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }

    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < columns_per_warp; ++c) {
            const int col = col_base + c;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                state[col * S_v + i] = s_shard[c][r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
         float scale, int64_t state_slot_stride, float * txn_log, int64_t txn_token_stride,
        const float * prior_log, float * committed_state, int prior_slots,
        int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    constexpr int columns_per_warp = 8;
    const int num_warps = std::min(4, (int) S_v / columns_per_warp);
    dim3      grid_dims(H, n_seqs, (S_v + num_warps * columns_per_warp - 1) /
                                      (num_warps * columns_per_warp));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride,
                txn_log, txn_token_stride, prior_log, committed_state, prior_slots, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride,
                txn_log, txn_token_stride, prior_log, committed_state, prior_slots, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride,
                txn_log, txn_token_stride, prior_log, committed_state, prior_slots, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride,
                txn_log, txn_token_stride, prior_log, committed_state, prior_slots, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];
    ggml_tensor * src_log   = dst->src[6];
    ggml_tensor * src_prior = dst->src[7];
    ggml_tensor * src_commit = dst->src[8];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;
    GGML_ASSERT(H % neqk1 == 0);

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;
    float *       log_d = src_log != nullptr ? (float *) src_log->data : nullptr;
    const float * prior_d = src_prior != nullptr ? (const float *) src_prior->data : nullptr;
    float *       commit_d = src_commit != nullptr ? (float *) src_commit->data : nullptr;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(src_log == nullptr || (src_log->type == GGML_TYPE_F32 && ggml_is_contiguous(src_log)));
    GGML_ASSERT(src_prior == nullptr || (src_prior->type == GGML_TYPE_F32 && ggml_is_contiguous(src_prior)));
    GGML_ASSERT(src_commit == nullptr || (src_commit->type == GGML_TYPE_F32 && ggml_is_contiguous(src_commit)));
    GGML_ASSERT((src_prior == nullptr) == (src_commit == nullptr));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const int prior_slots = ggml_get_op_params_i32(dst, 1);
    const bool keep_rs = K > 1;
    const int64_t txn_token_stride = H * (1 + 2 * S_v);

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride,
                log_d, txn_token_stride, prior_d, commit_d, prior_slots, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride,
                log_d, txn_token_stride, prior_d, commit_d, prior_slots, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride,
                log_d, txn_token_stride, prior_d, commit_d, prior_slots, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride,
                log_d, txn_token_stride, prior_d, commit_d, prior_slots, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

template <int S_v>
__device__ __forceinline__ void gdn_replay_one(
        const ggml_cuda_gdn_replay_args & a, int h_idx, int lane, int col) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ?
        ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int rows_per_lane = S_v / warp_size;
    const int64_t state_off = (int64_t) h_idx * S_v * S_v + col * S_v;
    const int64_t token_stride = a.head_count * (1 + 2 * S_v);
    float shard[rows_per_lane];

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        shard[r] = a.base[state_off + r * warp_size + lane];
    }

    // Keep the exact per-element token order used by the original replay
    // kernel.  Layers are independent, so only their launch packaging changes.
    for (int t = 0; t < a.n_keep; ++t) {
        const float * token_log = a.log + (int64_t) t * token_stride;
        const float decay = token_log[h_idx];
        const float delta = token_log[a.head_count + a.head_count * S_v + h_idx * S_v + col];
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * warp_size + lane;
            const float key = token_log[a.head_count + h_idx * S_v + row];
            shard[r] = gdn_state_replay_update(decay, shard[r], key, delta);
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        a.dst[state_off + r * warp_size + lane] = shard[r];
    }
}

template <int S_v>
__global__ void gdn_replay_cuda(
        const float * base, float * dst, const float * log, int64_t H, int n_keep) {
    const int h_idx = blockIdx.x;
    const int lane  = threadIdx.x;
    const int col   = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= S_v) {
        return;
    }

    const ggml_cuda_gdn_replay_args a = {
        base,
        dst,
        log,
        H * S_v * S_v,
        S_v,
        H,
        n_keep,
    };
    gdn_replay_one<S_v>(a, h_idx, lane, col);
}

// Qwen3.8 has 48 independent recurrent layers.  Passing their small
// descriptors as a kernel-parameter pack turns 48 tiny launches into one while
// preserving the arithmetic order within every state element.
static constexpr int GDN_REPLAY_LAYERS_PER_LAUNCH = 48;

struct gdn_replay_layer_pack {
    int32_t count;
    ggml_cuda_gdn_replay_args layers[GDN_REPLAY_LAYERS_PER_LAUNCH];
};
static_assert(sizeof(gdn_replay_layer_pack) <= 4096,
        "GDN replay kernel arguments must fit the CUDA 4 KiB parameter limit");

template <int S_v>
__global__ void gdn_replay_layers_cuda(gdn_replay_layer_pack pack) {
    const int layer_idx = blockIdx.z;
    if (layer_idx >= pack.count) {
        return;
    }

    const ggml_cuda_gdn_replay_args & a = pack.layers[layer_idx];
    const int h_idx = blockIdx.x;
    const int lane  = threadIdx.x;
    const int col   = blockIdx.y * blockDim.y + threadIdx.y;
    if (h_idx >= a.head_count || col >= S_v) {
        return;
    }

    gdn_replay_one<S_v>(a, h_idx, lane, col);
}

template <int S_v>
static void gdn_replay_launch_group(
        const ggml_cuda_gdn_replay_args * args, int32_t count, int warp_size, int num_warps,
        cudaStream_t stream) {
    gdn_replay_layer_pack pack = {};
    int64_t max_heads = 0;

    auto flush = [&]() {
        if (pack.count == 0) {
            return;
        }
        const dim3 grid(
            (uint32_t) max_heads,
            (uint32_t) ((S_v + num_warps - 1) / num_warps),
            (uint32_t) pack.count);
        const dim3 block(
            (uint32_t) (warp_size <= S_v ? warp_size : S_v),
            (uint32_t) num_warps,
            1);
        const ggml_cuda_kernel_launch_params launch_params(grid, block, 0, stream);
        ggml_cuda_kernel_launch(gdn_replay_layers_cuda<S_v>, launch_params, pack);
        pack = {};
        max_heads = 0;
    };

    for (int32_t i = 0; i < count; ++i) {
        const auto & a = args[i];
        if (a.state_dim != S_v) {
            continue;
        }
        if (pack.count == GDN_REPLAY_LAYERS_PER_LAUNCH) {
            flush();
        }
        pack.layers[pack.count++] = a;
        max_heads = std::max<int64_t>(max_heads, a.head_count);
    }
    flush();
}

static bool ggml_cuda_gdn_replay_impl(
        const ggml_cuda_gdn_replay_args * args, int32_t count, cudaStream_t stream, bool synchronize) {
    if (args == nullptr || count <= 0 || args[0].base == nullptr || args[0].dst == nullptr || args[0].log == nullptr) {
        return false;
    }

    cudaPointerAttributes attrs = {};
    if (cudaPointerGetAttributes(&attrs, args[0].dst) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const int old_device = ggml_cuda_get_device();
    ggml_cuda_set_device(attrs.device);

    // Validate the complete transaction before launching anything.  This
    // prevents a malformed later descriptor from leaving an earlier layer
    // committed only halfway through the operation.
    for (int32_t i = 0; i < count; ++i) {
        const auto & a = args[i];
        if (a.base == nullptr || a.dst == nullptr || a.log == nullptr || a.n_keep <= 0 || a.head_count <= 0 ||
            a.state_size != a.head_count * a.state_dim * a.state_dim ||
            (a.state_dim != 16 && a.state_dim != 32 && a.state_dim != 64 && a.state_dim != 128)) {
            ggml_cuda_set_device(old_device);
            return false;
        }
    }

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;

    const char * fused_env = getenv("LLAMA_GDN_REPLAY_FUSED");
    const bool fused = fused_env == nullptr || strcmp(fused_env, "0") != 0;
    if (fused) {
        gdn_replay_launch_group<16> (args, count, warp_size, num_warps, stream);
        gdn_replay_launch_group<32> (args, count, warp_size, num_warps, stream);
        gdn_replay_launch_group<64> (args, count, warp_size, num_warps, stream);
        gdn_replay_launch_group<128>(args, count, warp_size, num_warps, stream);
    } else {
        // Reference path retained for profiler and bit-parity comparisons.
        for (int32_t i = 0; i < count; ++i) {
            const auto & a = args[i];
            dim3 grid(a.head_count, (a.state_dim + num_warps - 1) / num_warps, 1);
            dim3 block(warp_size <= a.state_dim ? warp_size : a.state_dim, num_warps, 1);
            const ggml_cuda_kernel_launch_params launch_params(grid, block, 0, stream);
            switch (a.state_dim) {
                case 16:  ggml_cuda_kernel_launch(gdn_replay_cuda<16>,  launch_params, a.base, a.dst, a.log, a.head_count, a.n_keep); break;
                case 32:  ggml_cuda_kernel_launch(gdn_replay_cuda<32>,  launch_params, a.base, a.dst, a.log, a.head_count, a.n_keep); break;
                case 64:  ggml_cuda_kernel_launch(gdn_replay_cuda<64>,  launch_params, a.base, a.dst, a.log, a.head_count, a.n_keep); break;
                case 128: ggml_cuda_kernel_launch(gdn_replay_cuda<128>, launch_params, a.base, a.dst, a.log, a.head_count, a.n_keep); break;
            }
        }
    }

    if (synchronize) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ggml_cuda_set_device(old_device);
    return true;
}

bool ggml_cuda_gdn_replay(const ggml_cuda_gdn_replay_args * args, int32_t count) {
    return ggml_cuda_gdn_replay_impl(args, count, nullptr, true);
}

bool ggml_cuda_gdn_replay_on_stream(
        const ggml_cuda_gdn_replay_args * args, int32_t count, cudaStream_t stream) {
    return ggml_cuda_gdn_replay_impl(args, count, stream, false);
}

static constexpr int GDN_LOG_PREPARE_LAYERS_PER_LAUNCH = 48;

struct gdn_log_prepare_layer_pack {
    int32_t count;
    ggml_cuda_gdn_log_prepare_args layers[GDN_LOG_PREPARE_LAYERS_PER_LAUNCH];
};
static_assert(sizeof(gdn_log_prepare_layer_pack) <= 4096,
        "GDN log prepare kernel arguments must fit the CUDA 4 KiB parameter limit");

__global__ void gdn_log_prepare_cuda(
        gdn_log_prepare_layer_pack pack, int32_t n_keep, int32_t n_slots) {
    const int layer = blockIdx.z;
    if (layer >= pack.count || blockIdx.y >= (uint32_t) n_slots) {
        return;
    }
    const auto & a = pack.layers[layer];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.log_stride) {
        return;
    }

    float value;
    if ((int32_t) blockIdx.y < n_keep) {
        value = a.src[(int64_t) blockIdx.y * a.log_stride + i];
    } else {
        // Identity replay slot: state <- 1*state + 0*0.
        value = i < a.head_count ? 1.0f : 0.0f;
    }
    a.dst[(int64_t) blockIdx.y * a.log_stride + i] = value;
}

bool ggml_cuda_gdn_log_prepare_on_stream(
        const ggml_cuda_gdn_log_prepare_args * args, int32_t count,
        int32_t n_keep, int32_t n_slots, cudaStream_t stream) {
    if (args == nullptr || count <= 0 || count > GDN_LOG_PREPARE_LAYERS_PER_LAUNCH ||
            n_keep <= 0 || n_slots < n_keep) {
        return false;
    }

    cudaPointerAttributes attrs = {};
    if (cudaPointerGetAttributes(&attrs, args[0].dst) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const int old_device = ggml_cuda_get_device();
    ggml_cuda_set_device(attrs.device);
    int64_t max_stride = 0;
    gdn_log_prepare_layer_pack pack = {};
    pack.count = count;
    for (int32_t i = 0; i < count; ++i) {
        const auto & a = args[i];
        if (a.src == nullptr || a.dst == nullptr || a.log_stride <= 0 ||
                a.head_count <= 0 || a.head_count > a.log_stride) {
            ggml_cuda_set_device(old_device);
            return false;
        }
        pack.layers[i] = a;
        max_stride = std::max(max_stride, a.log_stride);
    }

    constexpr int threads = 256;
    const dim3 grid(
        (uint32_t) ((max_stride + threads - 1) / threads),
        (uint32_t) n_slots,
        (uint32_t) count);
    const ggml_cuda_kernel_launch_params launch_params(grid, threads, 0, stream);
    ggml_cuda_kernel_launch(gdn_log_prepare_cuda, launch_params, pack, n_keep, n_slots);
    ggml_cuda_set_device(old_device);
    return true;
}
