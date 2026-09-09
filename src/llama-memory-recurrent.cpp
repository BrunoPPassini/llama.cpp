#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "ggml-cuda-gdn-transaction.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);
    const char * txn_env = std::getenv("LLAMA_RS_TRANSACTION_LOG");
    use_gdn_txn = txn_env != nullptr && strcmp(txn_env, "0") != 0 &&
        model.arch == LLM_ARCH_QWEN35 && offload && type_s == GGML_TYPE_F32 &&
        mem_size == 1 && n_seq_max == 1 && n_rs_seq > 0;
    const char * phase_arena_env = std::getenv("LLAMA_RS_PHASE_ARENA");
    use_phase_arena = use_gdn_txn && phase_arena_env != nullptr && strcmp(phase_arena_env, "0") != 0;
    const char * lazy_replay_env = std::getenv("LLAMA_RS_LAZY_REPLAY");
    use_lazy_replay = use_phase_arena && lazy_replay_env != nullptr && strcmp(lazy_replay_env, "0") != 0;
    if (use_lazy_replay) {
        txn_lazy_slots = n_rs_seq + 1;
        if (const char * slots_env = std::getenv("LLAMA_RS_LAZY_REPLAY_SLOTS")) {
            char * end = nullptr;
            const unsigned long requested = std::strtoul(slots_env, &end, 10);
            if (end == slots_env || *end != '\0' || requested == 0 || requested > n_rs_seq + 1) {
                throw std::runtime_error("invalid LLAMA_RS_LAZY_REPLAY_SLOTS");
            }
            txn_lazy_slots = (uint32_t) requested;
        }
    }
    const char * verify_env = std::getenv("LLAMA_RS_TRANSACTION_VERIFY_LAYER");
    if (use_gdn_txn && verify_env != nullptr) {
        txn_verify_layer = std::atoi(verify_env);
        if (txn_verify_layer < 0 || txn_verify_layer >= n_layer) {
            throw std::runtime_error("invalid LLAMA_RS_TRANSACTION_VERIFY_LAYER");
        }
    }
    txn_active.assign(n_seq_max, 0);
    txn_last_tokens.assign(n_seq_max, 0);
    txn_pending.assign(n_seq_max, 0);
    txn_lazy_pending.assign(n_seq_max, 0);
    split_s_snapshots = !use_gdn_txn && getenv("LLAMA_RS_SPLIT_F16") != nullptr && n_rs_seq > 0 && type_s == GGML_TYPE_F32;

    cells.clear();
    cells.resize(mem_size);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(7u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    s_snap_l.resize(n_layer);
    txn_l.resize(n_layer);
    txn_prior_l.resize(n_layer);
    txn_verify_l.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload && getenv("LLAMA_RS_HOST_GPU") != nullptr) {
            auto * dev = model.dev_layer(i);
            if (auto * host_buft = ggml_backend_dev_host_buffer_type(dev)) {
                buft = host_buft;
                dev_name = ggml_backend_buft_name(host_buft);
            }
        } else if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows_r = mem_size * (1 + n_rs_seq);
        const uint32_t n_rows_s = mem_size * (use_gdn_txn ? (use_phase_arena ? 1 : 2) : (1 + n_rs_seq));
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows_r);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(),
                split_s_snapshots ? mem_size : n_rows_s);
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;

        if (split_s_snapshots) {
            ggml_tensor * s_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                    hparams.n_embd_s(), mem_size * n_rs_seq);
            ggml_format_name(s_snap, "cache_s_snap_l%d", i);
            s_snap_l[i] = s_snap;
        }

        if (use_gdn_txn) {
            const uint32_t state_dim  = hparams.ssm_d_state;
            const uint32_t head_count = hparams.ssm_d_inner / state_dim;
            const uint32_t log_stride = head_count * (1 + 2 * state_dim);
            ggml_tensor * txn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, log_stride, n_rs_seq + 1);
            ggml_format_name(txn, "cache_gdn_txn_l%d", i);
            txn_l[i] = txn;

            if (use_lazy_replay) {
                ggml_tensor * prior = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, log_stride, txn_lazy_slots);
                ggml_format_name(prior, "cache_gdn_txn_prior_l%d", i);
                txn_prior_l[i] = prior;
            }

            if (i == txn_verify_layer) {
                ggml_tensor * verify = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.n_embd_s(), n_rs_seq + 1);
                ggml_format_name(verify, "cache_gdn_txn_verify_l%d", i);
                txn_verify_l[i] = verify;
            }
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), split_s_snapshots ? "+F16 snapshots" : "",
                (float)memory_size_s / (1024.0f * 1024.0f));
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
    }

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
    std::fill(txn_active.begin(), txn_active.end(), 0);
    std::fill(txn_last_tokens.begin(), txn_last_tokens.end(), 0);
    std::fill(txn_pending.begin(), txn_pending.end(), 0);
    std::fill(txn_lazy_pending.begin(), txn_lazy_pending.end(), 0);
    txn_batch = false;
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (p0 >= p1) {
        return true;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();

    // A recurrent state is an aggregate of the complete token history.  It
    // cannot be edited by deleting or moving a historical interval.  The only
    // exact partial operation supported here is a bounded rollback of the
    // sequence tail using the per-token transaction log.
    if (seq_id >= (llama_seq_id) size) {
        return false;
    }

    if (seq_id < 0) {
        // A negative sequence id means all sequences.  Only a complete clear
        // can be represented exactly for recurrent state.
        if (!rm_all) {
            return false;
        }
    } else if (!rm_all) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id < 0) {
            return true;
        }

        auto & cell = cells[tail_id];
        if (!cell.has_seq_id(seq_id)) {
            return false;
        }

        // The requested range lies strictly after the current tail.
        if (p0 > cell.pos) {
            return true;
        }

        // Deleting anything before the tail while retaining the tail would
        // require replaying every surviving token after the edit.
        if (p1 <= cell.pos) {
            return false;
        }

        // The range includes the tail.  Roll it back only when every removed
        // token is represented in the transaction log.
        if (p0 > 0) {
            const llama_pos rollback = cell.pos - (p0 - 1);
            if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                if (use_gdn_txn && !txn_rollback(seq_id, (uint32_t) rollback)) {
                    return false;
                }
                set_rs_idx(seq_id, (uint32_t) rollback);
                cell.pos = p0 - 1;
                return true;
            }
            return false;
        }

        // [0, p1) covers the complete known sequence.  Treat it as a full
        // per-sequence clear even when p1 is finite.
    }

    if (rm_all || (seq_id >= 0 && p0 == 0)) {
        if (seq_id >= 0) {
            set_rs_idx(seq_id, 0);
            if ((size_t) seq_id < txn_active.size()) {
                txn_active[seq_id] = 0;
                txn_last_tokens[seq_id] = 0;
                txn_pending[seq_id] = 0;
                txn_lazy_pending[seq_id] = 0;
            }
        } else {
            std::fill(rs_idx.begin(), rs_idx.end(), 0);
            std::fill(txn_active.begin(), txn_active.end(), 0);
            std::fill(txn_last_tokens.begin(), txn_last_tokens.end(), 0);
            std::fill(txn_pending.begin(), txn_pending.end(), 0);
            std::fill(txn_lazy_pending.begin(), txn_lazy_pending.end(), 0);
        }

        // A complete removal must reset the recurrent state on-device in
        // BOTH transaction and non-transaction mode.  In transaction mode this
        // clears both S planes, the R state and the compact log.  In
        // non-transaction mode the S state is single-buffered and updated
        // in-place, so it must be explicitly zeroed here or stale
        // pre-compaction state leaks into the post-compaction context (the
        // model keeps "remembering" the old conversation through the
        // linear-attention state and produces incoherent output after
        // compaction).  Backend clear maps to a device memset and does not
        // transfer the cache through host memory.
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                cells[i].tail = -1;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    if (seq_id >= 0 && (uint32_t) seq_id < size) {
        cells[seq_id].tail = -1;
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0 || (size_t) seq_id >= rs_idx.size()) {
        return;
    }
    rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx;
}

void llama_memory_recurrent::set_recurrent_transaction(bool enabled) {
    if (!txn_phase_arena_enabled()) {
        return;
    }
    if (enabled) {
        for (uint8_t pending : txn_pending) {
            GGML_ASSERT(!pending);
        }
    }
    txn_batch = enabled;
}

int32_t llama_memory_recurrent::txn_s_copy(int i, llama_seq_id seq) const {
    const uint32_t cell_idx = i + head;
    const int32_t src0 = cells[cell_idx].src0;
    if (!use_gdn_txn) {
        return src0;
    }
    if (use_phase_arena) {
        return src0;
    }
    GGML_ASSERT(seq >= 0 && (size_t) seq < txn_active.size());
    return (int32_t) (txn_active[seq] * size) + src0;
}

int32_t llama_memory_recurrent::txn_s_work(int i, llama_seq_id seq) const {
    const uint32_t cell_idx = i + head;
    const int32_t src0 = cells[cell_idx].src0;
    if (!use_gdn_txn) {
        return src0;
    }
    if (use_phase_arena) {
        return src0;
    }
    GGML_ASSERT(seq >= 0 && (size_t) seq < txn_active.size());
    return (int32_t) ((txn_active[seq] ^ 1u) * size) + src0;
}

ggml_tensor * llama_memory_recurrent::get_txn_l(int32_t il) const {
    return txn_l[il];
}

ggml_tensor * llama_memory_recurrent::get_txn_prior_l(int32_t il) const {
    return txn_prior_l[il];
}

ggml_tensor * llama_memory_recurrent::get_txn_verify_l(int32_t il) const {
    return txn_verify_l[il];
}

bool llama_memory_recurrent::txn_rollback(llama_seq_id seq_id, uint32_t rollback) {
    GGML_ASSERT(use_gdn_txn && seq_id >= 0 && (size_t) seq_id < txn_active.size());
    const uint32_t n_tokens = txn_last_tokens[seq_id];
    if (use_phase_arena) {
        if (!txn_pending[seq_id]) {
            return true;
        }
        if (rollback > n_tokens) {
            LLAMA_LOG_ERROR("%s: rollback %u exceeds last transaction size %u\n", __func__, rollback, n_tokens);
            return false;
        }
        return recurrent_transaction_accept(seq_id, n_tokens - rollback, nullptr);
    }
    if (rollback > n_tokens) {
        LLAMA_LOG_ERROR("%s: rollback %u exceeds last transaction size %u\n", __func__, rollback, n_tokens);
        return false;
    }

    const uint32_t n_keep = n_tokens - rollback;
    if (n_keep == 0) {
        txn_active[seq_id] ^= 1u;
        txn_last_tokens[seq_id] = 0;
        return true;
    }

    const int64_t state_dim  = hparams.ssm_d_state;
    const int64_t head_count = hparams.ssm_d_inner / state_dim;
    const int64_t state_size = hparams.n_embd_s();
    std::vector<ggml_cuda_gdn_replay_args> args;
    args.reserve(s_l.size());

    ggml_backend_reg_t reg = nullptr;
    for (size_t il = 0; il < s_l.size(); ++il) {
        if (s_l[il] == nullptr || txn_l[il] == nullptr) {
            continue;
        }
        if (reg == nullptr) {
            const auto buft = ggml_backend_buffer_get_type(s_l[il]->buffer);
            const auto dev  = ggml_backend_buft_get_device(buft);
            reg = ggml_backend_dev_backend_reg(dev);
        }
        float * states = (float *) s_l[il]->data;
        const uint32_t active = txn_active[seq_id];
        args.push_back({
            states + active * state_size,
            states + (active ^ 1u) * state_size,
            (const float *) txn_l[il]->data,
            state_size,
            state_dim,
            head_count,
            (int32_t) n_keep,
        });
    }

    auto replay = reg != nullptr ? (ggml_cuda_gdn_replay_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_cuda_gdn_replay") : nullptr;
    if (replay == nullptr || !replay(args.data(), (int32_t) args.size())) {
        LLAMA_LOG_ERROR("%s: CUDA GDN replay is unavailable or failed\n", __func__);
        return false;
    }

    if (txn_verify_layer >= 0 && txn_verify_l[txn_verify_layer] != nullptr) {
        const size_t bytes = (size_t) state_size * sizeof(float);
        std::vector<uint32_t> expected(state_size);
        std::vector<uint32_t> actual(state_size);
        ggml_backend_tensor_get(txn_verify_l[txn_verify_layer], expected.data(), (size_t) rollback * bytes, bytes);
        ggml_backend_tensor_get(s_l[txn_verify_layer], actual.data(), (size_t) (txn_active[seq_id] ^ 1u) * bytes, bytes);
        size_t first = expected.size();
        size_t different = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (expected[i] != actual[i]) {
                if (first == expected.size()) {
                    first = i;
                }
                ++different;
            }
        }
        if (different == 0) {
            LLAMA_LOG_INFO("%s: verify layer=%d rollback=%u: BIT-EXACT (%zu floats)\n",
                    __func__, txn_verify_layer, rollback, expected.size());
        } else {
            LLAMA_LOG_ERROR("%s: verify layer=%d rollback=%u: MISMATCH %zu/%zu first=%zu expected=0x%08x actual=0x%08x\n",
                    __func__, txn_verify_layer, rollback, different, expected.size(), first,
                    expected[first], actual[first]);
        }
    }

    txn_last_tokens[seq_id] = n_keep;
    return true;
}

bool llama_memory_recurrent::recurrent_transaction_accept(
        llama_seq_id seq_id, uint32_t n_keep, ggml_backend_t backend) {
    if (!txn_phase_arena_enabled()) {
        return true;
    }
    if (seq_id < 0 || (size_t) seq_id >= txn_pending.size()) {
        return false;
    }
    if (!txn_pending[seq_id]) {
        return true;
    }

    const uint32_t n_tokens = txn_last_tokens[seq_id];
    if (n_keep > n_tokens) {
        LLAMA_LOG_ERROR("%s: keep %u exceeds transaction size %u\n", __func__, n_keep, n_tokens);
        return false;
    }

    if (n_keep > 0 && use_lazy_replay && backend != nullptr && n_keep <= txn_lazy_slots) {
        LLAMA_LOG_WARN("LAZY_ACCEPT seq=%d keep=%u verified=%u\n", (int) seq_id, n_keep, n_tokens);
        const int64_t state_dim  = hparams.ssm_d_state;
        const int64_t head_count = hparams.ssm_d_inner / state_dim;
        const int64_t log_stride = head_count * (1 + 2 * state_dim);
        std::vector<ggml_cuda_gdn_log_prepare_args> args;
        args.reserve(s_l.size());
        ggml_backend_reg_t reg = nullptr;

        for (size_t il = 0; il < txn_l.size(); ++il) {
            if (txn_l[il] == nullptr || txn_prior_l[il] == nullptr) {
                continue;
            }
            if (reg == nullptr) {
                const auto buft = ggml_backend_buffer_get_type(txn_l[il]->buffer);
                const auto dev  = ggml_backend_buft_get_device(buft);
                reg = ggml_backend_dev_backend_reg(dev);
            }
            args.push_back({
                (const float *) txn_l[il]->data,
                (float *) txn_prior_l[il]->data,
                log_stride,
                head_count,
            });
        }

        auto prepare_async = reg != nullptr ? (ggml_cuda_gdn_log_prepare_async_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_cuda_gdn_log_prepare_async") : nullptr;
        if (prepare_async != nullptr && prepare_async(
                args.data(), (int32_t) args.size(), (int32_t) n_keep,
                (int32_t) txn_lazy_slots, backend)) {
            // Diagnostic switch: prove whether the compact-log hand-off and
            // the following graph are ordered on the same CUDA stream.  The
            // optimized path leaves this disabled and will use an explicit
            // stream dependency if the synchronized control restores parity.
            const char * sync_env = std::getenv("LLAMA_RS_LAZY_PREPARE_SYNC");
            if (sync_env != nullptr && strcmp(sync_env, "0") != 0) {
                ggml_backend_synchronize(backend);
            }
            txn_lazy_pending[seq_id] = (uint8_t) n_keep;
            txn_pending[seq_id] = 0;
            txn_last_tokens[seq_id] = 0;
            return true;
        }
        LLAMA_LOG_WARN("%s: lazy GDN replay unavailable; using eager replay\n", __func__);
    }

    if (n_keep > 0) {
        const int64_t state_dim  = hparams.ssm_d_state;
        const int64_t head_count = hparams.ssm_d_inner / state_dim;
        const int64_t state_size = hparams.n_embd_s();
        std::vector<ggml_cuda_gdn_replay_args> args;
        args.reserve(s_l.size());
        ggml_backend_reg_t reg = nullptr;

        for (size_t il = 0; il < s_l.size(); ++il) {
            if (s_l[il] == nullptr || txn_l[il] == nullptr) {
                continue;
            }
            if (reg == nullptr) {
                const auto buft = ggml_backend_buffer_get_type(s_l[il]->buffer);
                const auto dev  = ggml_backend_buft_get_device(buft);
                reg = ggml_backend_dev_backend_reg(dev);
            }
            args.push_back({
                (const float *) s_l[il]->data,
                (float *) s_l[il]->data,
                (const float *) txn_l[il]->data,
                state_size,
                state_dim,
                head_count,
                (int32_t) n_keep,
            });
        }

        auto replay_async = reg != nullptr ? (ggml_cuda_gdn_replay_async_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_cuda_gdn_replay_async") : nullptr;
        if (backend != nullptr && replay_async != nullptr &&
                replay_async(args.data(), (int32_t) args.size(), backend)) {
            txn_pending[seq_id] = 0;
            txn_last_tokens[seq_id] = 0;
            return true;
        }

        // Compatibility path for backends without ordered async replay.
        if (backend != nullptr) {
            ggml_backend_synchronize(backend);
        }
        auto replay = reg != nullptr ? (ggml_cuda_gdn_replay_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_cuda_gdn_replay") : nullptr;
        if (replay == nullptr || !replay(args.data(), (int32_t) args.size())) {
            LLAMA_LOG_ERROR("%s: CUDA GDN replay is unavailable or failed\n", __func__);
            return false;
        }
    }

    txn_pending[seq_id] = 0;
    txn_last_tokens[seq_id] = 0;
    txn_lazy_pending[seq_id] = 0;
    return true;
}

void llama_memory_recurrent::txn_commit(const llama_ubatch & ubatch) {
    if (!use_gdn_txn || ubatch.n_seqs != 1 || ubatch.n_tokens == 0) {
        return;
    }
    const llama_seq_id seq_id = ubatch.seq_id[0][0];
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < txn_active.size());
    if (use_phase_arena) {
        if (txn_batch) {
            GGML_ASSERT(!txn_pending[seq_id]);
            txn_pending[seq_id] = 1;
            txn_last_tokens[seq_id] = ubatch.n_seq_tokens;
        }
        return;
    }
    txn_active[seq_id] ^= 1u;
    txn_last_tokens[seq_id] = ubatch.n_seq_tokens;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    // Moving position metadata is trivial, but a generic context shift first
    // removes a historical interval.  The aggregate recurrent state cannot be
    // transformed to represent that edited history without restoring a
    // checkpoint and replaying the surviving suffix.
    return false;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }
    for (const auto & s : s_snap_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> r_ranges; // logical source row ranges
    std::vector<std::pair<uint32_t, uint32_t>> s_ranges;
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t physical_id = cell.src >= 0 ? cell.src : (int32_t) i;
            const uint32_t r_cell_id = rs_idx_cur * size + physical_id;
            if (r_ranges.empty() || r_ranges.back().second != r_cell_id) {
                r_ranges.emplace_back(r_cell_id, r_cell_id + 1);
            } else {
                r_ranges.back().second++;
            }

            uint32_t s_cell_id = r_cell_id;
            if (use_gdn_txn) {
                const llama_seq_id state_seq = seq_id != -1 ? seq_id : *cell.seq_id.begin();
                GGML_ASSERT(state_seq >= 0 && (size_t) state_seq < txn_active.size());
                // txn_active identifies the plane containing the committed
                // recurrent state.  The opposite plane is only scratch for
                // the next transaction and must never be checkpointed.
                s_cell_id = txn_active[state_seq] * size + physical_id;
            }
            if (s_ranges.empty() || s_ranges.back().second != s_cell_id) {
                s_ranges.emplace_back(s_cell_id, s_cell_id + 1);
            } else {
                s_ranges.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : r_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : s_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, r_ranges, s_ranges);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    res = res && state_read_meta(io, cell_count, seq_id);

    try {
        res = res && state_read_data(io, cell_count);
    } catch (...) {
        res = false;
    }

    if (!res) {
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        if (seq_id == -1) {
            std::fill(rs_idx.begin(), rs_idx.end(), 0);
        } else {
            set_rs_idx(seq_id, 0);
        }
    }
    if (use_gdn_txn) {
        if (seq_id == -1) {
            // state_read_data restores S contiguously into plane 0.
            std::fill(txn_active.begin(), txn_active.end(), 0);
            std::fill(txn_last_tokens.begin(), txn_last_tokens.end(), 0);
        } else if ((size_t) seq_id < txn_active.size()) {
            txn_active[seq_id] = 0;
            txn_last_tokens[seq_id] = 0;
        }
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(
        llama_io_write_i & io,
        const std::vector<std::pair<uint32_t, uint32_t>> & r_ranges,
        const std::vector<std::pair<uint32_t, uint32_t>> & s_ranges) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : r_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane.
            for (const auto & range : s_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : s_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) :
    status(LLAMA_MEMORY_STATUS_SUCCESS),
    mem(mem),
    ubatches(std::move(ubatches)),
    views(this->ubatches.size()) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    mem->find_slot(ubatches[i_next]);
    mem->txn_commit(ubatches[i_next]);

    // Snapshot only the graph descriptor created by find_slot().  In normal
    // ubatch-major execution the next microbatch overwrites these fields only
    // after the current graph is finished.  Layer-major execution revisits an
    // older microbatch after that overwrite, so retaining the descriptor is
    // required for the exact same recurrent-state rows to be selected.
    auto & view = views[i_next];
    view.n_rs = mem->n;
    view.head = mem->head;
    view.rs_z = mem->rs_z;
    view.s_copy.resize(view.n_rs);
    view.txn_copy.resize(view.n_rs);
    view.txn_work.resize(view.n_rs);

    const auto & ubatch = ubatches[i_next];
    if (mem->txn_lazy_replay_enabled() && ubatch.n_seqs == 1 && ubatch.n_tokens > 0) {
        const llama_seq_id seq_id = ubatch.seq_id[0][0];
        if (seq_id >= 0 && (size_t) seq_id < mem->txn_lazy_pending.size()) {
            view.txn_lazy_replay = mem->txn_lazy_pending[seq_id];
            if (view.txn_lazy_replay > 0) {
                LLAMA_LOG_WARN("LAZY_APPLY seq=%d count=%u tokens=%u txn=%d\n",
                        (int) seq_id, view.txn_lazy_replay, ubatch.n_tokens, (int) mem->txn_batch);
            }
            mem->txn_lazy_pending[seq_id] = 0;
        }
    }

    for (uint32_t i = 0; i < view.n_rs; ++i) {
        const uint32_t cell_idx = i + view.head;
        const int32_t src0 = mem->cells[cell_idx].src0;

        uint32_t rollback_idx = 0;
        if (mem->n_rs_seq != 0 && !mem->cells[cell_idx].seq_id.empty()) {
            const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
            if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
                rollback_idx = mem->rs_idx[seq];
                mem->rs_idx[seq] = 0;
            }
        }
        view.s_copy[i] = (int32_t) (rollback_idx * mem->size) + src0;

        const uint32_t seq_index = std::min<uint32_t>(i, ubatch.n_seqs - 1);
        const llama_seq_id seq_id = ubatch.seq_id[seq_index * ubatch.n_seq_tokens][0];
        view.txn_copy[i] = mem->txn_s_copy(i, seq_id);
        view.txn_work[i] = mem->txn_s_work(i, seq_id);
    }
    view.valid = true;

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

bool llama_memory_recurrent_context::select_ubatch(size_t index) {
    if (index >= ubatches.size()) {
        return false;
    }

    i_next = index;
    return true;
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    if (is_full) {
        return mem->size;
    }
    if (!views.empty() && views[i_next].valid) {
        return views[i_next].n_rs;
    }
    return mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    if (is_full) {
        return 0;
    }
    if (!views.empty() && views[i_next].valid) {
        return views[i_next].head;
    }
    return mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    if (is_full) {
        return 0;
    }
    if (!views.empty() && views[i_next].valid) {
        return views[i_next].rs_z;
    }
    return mem->rs_z;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_snap_l(int32_t il) const {
    return mem->s_snap_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_txn_l(int32_t il) const {
    return mem->get_txn_l(il);
}

ggml_tensor * llama_memory_recurrent_context::get_txn_prior_l(int32_t il) const {
    return mem->get_txn_prior_l(il);
}

ggml_tensor * llama_memory_recurrent_context::get_txn_verify_l(int32_t il) const {
    return mem->get_txn_verify_l(il);
}

bool llama_memory_recurrent_context::has_split_s_snapshots() const {
    return mem->split_s_snapshots;
}

bool llama_memory_recurrent_context::txn_enabled() const {
    return mem->txn_enabled();
}

bool llama_memory_recurrent_context::txn_phase_arena_enabled() const {
    return mem->txn_phase_arena_enabled();
}

bool llama_memory_recurrent_context::txn_lazy_replay_enabled() const {
    return mem->txn_lazy_replay_enabled();
}

uint32_t llama_memory_recurrent_context::txn_lazy_replay_pending() const {
    if (is_full) {
        return false;
    }
    if (!views.empty() && views[i_next].valid) {
        return views[i_next].txn_lazy_replay;
    }
    return false;
}

uint32_t llama_memory_recurrent_context::txn_lazy_replay_slots() const {
    return mem->txn_lazy_slots;
}

bool llama_memory_recurrent_context::txn_batch_enabled() const {
    return !is_full && mem->txn_batch_enabled();
}

int32_t llama_memory_recurrent_context::txn_s_copy(int i, llama_seq_id seq_id) const {
    if (!views.empty() && views[i_next].valid) {
        GGML_ASSERT(i >= 0 && (size_t) i < views[i_next].txn_copy.size());
        return views[i_next].txn_copy[i];
    }
    return mem->txn_s_copy(i, seq_id);
}

int32_t llama_memory_recurrent_context::txn_s_work(int i, llama_seq_id seq_id) const {
    if (!views.empty() && views[i_next].valid) {
        GGML_ASSERT(i >= 0 && (size_t) i < views[i_next].txn_work.size());
        return views[i_next].txn_work[i];
    }
    return mem->txn_s_work(i, seq_id);
}

uint32_t llama_memory_recurrent_context::get_n_rs_seq() const {
    return mem->n_rs_seq;
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    if (!views.empty() && views[i_next].valid) {
        GGML_ASSERT(i >= 0 && (size_t) i < views[i_next].s_copy.size());
        return views[i_next].s_copy[i];
    }

    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}
