#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <map>
#include <cinttypes>
#include <random>
#include <sstream>
#include <type_traits>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    virtual bool needs_prompt_history() const { return true; }

    virtual void reset(llama_seq_id /*seq_id*/) {}

    virtual const std::vector<std::vector<llama_token_data>> * draft_probs(
            llama_seq_id /*seq_id*/) const { return nullptr; }

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // skipping the embedding batches leaves a hole in the draft's cache and
        // the next injection fails to initialize.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather this chunk's target features, interleaved by extract layer
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                // fuse extracted features through DFlash encoder
                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                // inject the DFlash decoder K/V cache at the tokens' target positions
                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

                for (int32_t i = 0; i < n_chunk; ++i) {
                    batch_inject.pos[i]       = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.n_past;

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    bool rejection_sampling = false;
    float proposal_temp = 1.0f;
    bool rank_proposal = false;
    uint32_t rank_seed = 424242;
    bool position_temp_proposal = false;
    float position_temps[3] = { 1.0f, 1.0f, 1.0f };
    std::vector<std::mt19937> rank_rng;
    std::vector<std::vector<std::vector<llama_token_data>>> sampled_draft_probs;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    int32_t defer_prefill_tail = 0;
    std::vector<bool> deferred_prefill_ready;
    std::vector<std::vector<llama_token>> deferred_tokens;
    std::vector<std::vector<llama_pos>> deferred_positions;
    std::vector<std::vector<float>> deferred_h;
    std::vector<float> last_second_confidence;
    std::vector<float> last_third_confidence;
    std::vector<float> last_fourth_confidence;
    uint64_t third_conf_attempts[10] = {};
    uint64_t third_conf_accepts[10] = {};
    uint64_t third_conf_reports = 0;
    uint64_t fourth_conf_attempts[10] = {};
    uint64_t fourth_conf_accepts[10] = {};
    uint64_t fifth_conf_attempts[10] = {};
    uint64_t fifth_conf_accepts[10] = {};

    bool needs_prompt_history() const override { return false; }

    // Online MTP-depth controller. The target still verifies every returned
    // token, so changing the number of proposals cannot bypass target logits.
    // Reward is measured over the entire draft+verify cycle, not inferred from
    // proposal confidence alone: (one target token + accepted drafts) / time.
    bool     adaptive_depth_enabled        = false;
    int32_t  adaptive_depth_min            = 1;
    int32_t  adaptive_depth_max            = 1;
    uint64_t adaptive_warmup_per_depth     = 4;
    uint64_t adaptive_probe_interval       = 32;
    double   adaptive_ema_alpha            = 0.20;
    double   adaptive_switch_hysteresis    = 0.02;
    uint64_t adaptive_decisions            = 0;
    double   adaptive_tps_ema[5]           = {};
    double   adaptive_accept_ema[5]        = {};
    uint64_t adaptive_samples[5]           = {};
    uint64_t adaptive_pos_attempts[5]      = {};
    uint64_t adaptive_pos_accepts[5]       = {};
    double   adaptive_pos_ema[5]           = {};
    uint64_t adaptive_pos_samples[5]       = {};
    double   adaptive_pos3_enter           = 0.30;
    double   adaptive_pos3_exit            = 0.40;
    std::vector<int32_t> adaptive_depth;
    std::vector<int32_t> adaptive_last_depth;
    std::vector<int32_t> adaptive_last_proposed;
    std::vector<int64_t> adaptive_cycle_start_us;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        // Preserve the frozen production MTP batching contract. The later
        // prefill-specific capacity changed proposal behavior and throughput.
        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        rejection_sampling = [] {
            const char * value = std::getenv("LLAMA_SPEC_REJECTION_SAMPLING");
            return value != nullptr && std::strcmp(value, "0") != 0;
        }();
        if (const char * value = std::getenv("LLAMA_MTP_PROPOSAL_TEMP")) {
            proposal_temp = std::max(0.01f, std::strtof(value, nullptr));
        }
        if (const char * value = std::getenv("LLAMA_MTP_RANK_PROPOSAL")) {
            rank_proposal = std::strcmp(value, "0") != 0;
        }
        if (const char * value = std::getenv("LLAMA_MTP_RANK_SEED")) {
            rank_seed = (uint32_t) std::strtoul(value, nullptr, 10);
        }
        if (const char * value = std::getenv("LLAMA_MTP_POSITION_TEMPS")) {
            const char * cursor = value;
            bool valid = true;
            for (size_t pos = 0; pos < 3; ++pos) {
                char * end = nullptr;
                position_temps[pos] = std::strtof(cursor, &end);
                if (end == cursor || !(position_temps[pos] > 0.0f)) {
                    valid = false;
                    break;
                }
                cursor = *end == ',' ? end + 1 : end;
            }
            position_temp_proposal = valid;
        }

        adaptive_depth_enabled = [] {
            const char * value = std::getenv("LLAMA_MTP_ADAPTIVE_DEPTH");
            return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
        }();
        if (adaptive_depth_enabled) {
            const auto env_i32 = [](const char * name, int32_t fallback) {
                const char * value = std::getenv(name);
                if (value == nullptr) {
                    return fallback;
                }
                char * end = nullptr;
                const long parsed = std::strtol(value, &end, 10);
                return end != value && *end == '\0' ? (int32_t) parsed : fallback;
            };
            const auto env_f64 = [](const char * name, double fallback) {
                const char * value = std::getenv(name);
                if (value == nullptr) {
                    return fallback;
                }
                char * end = nullptr;
                const double parsed = std::strtod(value, &end);
                return end != value && *end == '\0' ? parsed : fallback;
            };

            adaptive_depth_min = std::max(1, env_i32("LLAMA_MTP_ADAPTIVE_MIN", 1));
            adaptive_depth_max = std::min({ 4, this->params.n_max,
                    std::max(adaptive_depth_min, env_i32("LLAMA_MTP_ADAPTIVE_MAX", this->params.n_max)) });
            adaptive_depth_min = std::min(adaptive_depth_min, adaptive_depth_max);
            adaptive_warmup_per_depth = (uint64_t) std::max(1,
                    env_i32("LLAMA_MTP_ADAPTIVE_WARMUP", (int32_t) adaptive_warmup_per_depth));
            adaptive_probe_interval = (uint64_t) std::max(8,
                    env_i32("LLAMA_MTP_ADAPTIVE_PROBE_INTERVAL", (int32_t) adaptive_probe_interval));
            adaptive_ema_alpha = std::max(0.01, std::min(1.0,
                    env_f64("LLAMA_MTP_ADAPTIVE_EMA", adaptive_ema_alpha)));
            adaptive_switch_hysteresis = std::max(0.0, std::min(0.25,
                    env_f64("LLAMA_MTP_ADAPTIVE_HYSTERESIS", adaptive_switch_hysteresis)));
            adaptive_pos3_enter = std::max(0.0, std::min(1.0,
                    env_f64("LLAMA_MTP_ADAPTIVE_POS3_ENTER", adaptive_pos3_enter)));
            adaptive_pos3_exit = std::max(adaptive_pos3_enter, std::min(1.0,
                    env_f64("LLAMA_MTP_ADAPTIVE_POS3_EXIT", adaptive_pos3_exit)));

            SPC_WRN("adaptive MTP depth enabled: min=%d max=%d warmup=%llu probe=%llu ema=%.3f hysteresis=%.3f\n",
                    adaptive_depth_min, adaptive_depth_max,
                    (unsigned long long) adaptive_warmup_per_depth,
                    (unsigned long long) adaptive_probe_interval,
                    adaptive_ema_alpha, adaptive_switch_hysteresis);
        }

        smpls.resize(n_seq);
        const uint32_t proposal_seed = rank_seed ^ 0x9e3779b9u;
        for (uint32_t seq = 0; seq < n_seq; ++seq) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.temp     = proposal_temp;
            sparams.seed     = proposal_seed + seq;
            sparams.samplers = rejection_sampling
                ? std::vector<common_sampler_type> {
                        COMMON_SAMPLER_TYPE_TEMPERATURE,
                        COMMON_SAMPLER_TYPE_TOP_K }
                : std::vector<common_sampler_type> { COMMON_SAMPLER_TYPE_TOP_K };
            smpls[seq].reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }
        sampled_draft_probs.resize(n_seq);
        rank_rng.reserve(n_seq);
        for (uint32_t seq = 0; seq < n_seq; ++seq) {
            rank_rng.emplace_back(proposal_seed + seq);
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        if (const char * value = std::getenv("LLAMA_MTP_DEFER_PREFILL_TAIL")) {
            char * end = nullptr;
            const long requested = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && requested > 0) {
                defer_prefill_tail = (int32_t) requested;
            }
        }
        deferred_prefill_ready.assign(n_seq, defer_prefill_tail == 0);
        deferred_tokens.resize(n_seq);
        deferred_positions.resize(n_seq);
        deferred_h.resize(n_seq);
        last_second_confidence.assign(n_seq, -1.0f);
        last_third_confidence.assign(n_seq, -1.0f);
        last_fourth_confidence.assign(n_seq, -1.0f);
        if (adaptive_depth_enabled) {
            const int32_t adaptive_start = std::min(3, adaptive_depth_max);
            adaptive_depth.assign(n_seq, adaptive_start);
            adaptive_last_depth.assign(n_seq, adaptive_start);
            adaptive_last_proposed.assign(n_seq, 0);
            adaptive_cycle_start_us.assign(n_seq, 0);
        }
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    static constexpr uint32_t state_magic   = 0x3150544dU; // "MTP1" in little endian
    static constexpr uint32_t state_version = 1;

    template<typename T>
    static void state_append(std::vector<uint8_t> & data, const T & value) {
        static_assert(std::is_trivially_copyable<T>::value, "state value must be trivially copyable");
        const size_t old_size = data.size();
        data.resize(old_size + sizeof(T));
        std::memcpy(data.data() + old_size, &value, sizeof(T));
    }

    template<typename T>
    static bool state_read(const std::vector<uint8_t> & data, size_t & offset, T & value) {
        static_assert(std::is_trivially_copyable<T>::value, "state value must be trivially copyable");
        if (offset > data.size() || data.size() - offset < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, data.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }

    template<typename T>
    static void state_append_vector(std::vector<uint8_t> & data, const std::vector<T> & values) {
        static_assert(std::is_trivially_copyable<T>::value, "state vector must be trivially copyable");
        const uint64_t count = values.size();
        state_append(data, count);
        if (!values.empty()) {
            const size_t bytes = values.size() * sizeof(T);
            const size_t old_size = data.size();
            data.resize(old_size + bytes);
            std::memcpy(data.data() + old_size, values.data(), bytes);
        }
    }

    template<typename T>
    static bool state_read_vector(
            const std::vector<uint8_t> & data, size_t & offset, std::vector<T> & values,
            uint64_t max_count) {
        static_assert(std::is_trivially_copyable<T>::value, "state vector must be trivially copyable");
        uint64_t count = 0;
        if (!state_read(data, offset, count) || count > max_count ||
                count > (data.size() - offset) / sizeof(T)) {
            return false;
        }
        values.resize((size_t) count);
        const size_t bytes = values.size() * sizeof(T);
        if (bytes > 0) {
            std::memcpy(values.data(), data.data() + offset, bytes);
            offset += bytes;
        }
        return true;
    }

    void reset(llama_seq_id seq_id) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        if (chain_heads) {
            chain_h[seq_id].clear();
        }

        i_last[seq_id]      = -1;
        i_batch_beg[seq_id] = -1;
        i_batch_end[seq_id] = -1;

        deferred_prefill_ready[seq_id] = defer_prefill_tail == 0;
        deferred_tokens[seq_id].clear();
        deferred_positions[seq_id].clear();
        deferred_h[seq_id].clear();

        sampled_draft_probs[seq_id].clear();
        last_second_confidence[seq_id] = -1.0f;
        last_third_confidence[seq_id]  = -1.0f;
        last_fourth_confidence[seq_id] = -1.0f;

        common_sampler_reset(smpls[seq_id].get());
        if (backend_chains[seq_id]) {
            llama_sampler_reset(backend_chains[seq_id]);
        }
        rank_rng[seq_id].seed((rank_seed ^ 0x9e3779b9u) + (uint32_t) seq_id);

        if (adaptive_depth_enabled) {
            const int32_t adaptive_start = std::min(3, adaptive_depth_max);
            adaptive_depth[seq_id]          = adaptive_start;
            adaptive_last_depth[seq_id]     = adaptive_start;
            adaptive_last_proposed[seq_id]  = 0;
            adaptive_cycle_start_us[seq_id] = 0;

            // The production profile is single-sequence.  Reset the shared
            // controller too, so a queued request starts from the same state
            // as the identical request in a fresh process.  Do not disturb a
            // genuinely concurrent multi-sequence controller.
            if (n_seq == 1) {
                adaptive_decisions = 0;
                std::fill(std::begin(adaptive_tps_ema),       std::end(adaptive_tps_ema),       0.0);
                std::fill(std::begin(adaptive_accept_ema),    std::end(adaptive_accept_ema),    0.0);
                std::fill(std::begin(adaptive_samples),       std::end(adaptive_samples),       0);
                std::fill(std::begin(adaptive_pos_attempts),  std::end(adaptive_pos_attempts),  0);
                std::fill(std::begin(adaptive_pos_accepts),   std::end(adaptive_pos_accepts),   0);
                std::fill(std::begin(adaptive_pos_ema),       std::end(adaptive_pos_ema),       0.0);
                std::fill(std::begin(adaptive_pos_samples),   std::end(adaptive_pos_samples),   0);
            }
        }
    }

    bool adaptive_is_enabled() const {
        const char * value = std::getenv("LLAMA_MTP_ADAPTIVE_DEPTH");
        return adaptive_depth_enabled && value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    int32_t adaptive_select_next(int32_t current) {
        if (adaptive_depth_max < 3) {
            return adaptive_depth_max;
        }
        if (adaptive_pos_samples[3] < adaptive_warmup_per_depth) {
            return 3;
        }

        ++adaptive_decisions;
        if (current < 3) {
            if (adaptive_pos_ema[3] >= adaptive_pos3_exit ||
                    (adaptive_decisions % adaptive_probe_interval) == 0) {
                return 3;
            }
            return std::max(2, adaptive_depth_min);
        }

        return adaptive_pos_ema[3] < adaptive_pos3_enter
                ? std::max(2, adaptive_depth_min)
                : 3;
    }

    void adaptive_observe(llama_seq_id seq_id, uint16_t n_accepted) {
        if (!adaptive_is_enabled() || adaptive_last_proposed[seq_id] <= 0 ||
                adaptive_cycle_start_us[seq_id] <= 0) {
            return;
        }

        const int32_t depth = adaptive_last_depth[seq_id];
        const int32_t proposed = adaptive_last_proposed[seq_id];
        const int64_t elapsed_us = std::max<int64_t>(1,
                ggml_time_us() - adaptive_cycle_start_us[seq_id]);
        const int32_t accepted = std::min<int32_t>(n_accepted, proposed);
        const double cycle_tps = 1e6 * (1.0 + accepted) / elapsed_us;
        const double accept_ratio = (double) accepted / proposed;

        const double alpha = adaptive_samples[depth] == 0 ? 1.0 : adaptive_ema_alpha;
        adaptive_tps_ema[depth] = alpha * cycle_tps + (1.0 - alpha) * adaptive_tps_ema[depth];
        adaptive_accept_ema[depth] = alpha * accept_ratio +
                (1.0 - alpha) * adaptive_accept_ema[depth];
        adaptive_samples[depth]++;
        for (int32_t pos = 1; pos <= proposed && pos <= 4; ++pos) {
            adaptive_pos_attempts[pos]++;
            const double hit = accepted >= pos ? 1.0 : 0.0;
            if (hit > 0.0) {
                adaptive_pos_accepts[pos]++;
            }
            const double pos_alpha = adaptive_pos_samples[pos] == 0 ? 1.0 : adaptive_ema_alpha;
            adaptive_pos_ema[pos] = pos_alpha * hit + (1.0 - pos_alpha) * adaptive_pos_ema[pos];
            adaptive_pos_samples[pos]++;
        }

        adaptive_depth[seq_id] = adaptive_select_next(depth);
        adaptive_last_proposed[seq_id] = 0;
        adaptive_cycle_start_us[seq_id] = 0;

        uint64_t total_samples = 0;
        for (int32_t d = adaptive_depth_min; d <= adaptive_depth_max; ++d) {
            total_samples += adaptive_samples[d];
        }
        if ((total_samples % 32) == 0) {
            std::fprintf(stderr, "adaptive MTP: samples=%llu depth=%d->%d cycle=%.2f tok/s accepted=%d/%d "
                    "arms=[1:%.2f,2:%.2f,3:%.2f,4:%.2f] pos_ema=[1:%.3f,2:%.3f,3:%.3f,4:%.3f]\n",
                    (unsigned long long) total_samples, depth, adaptive_depth[seq_id], cycle_tps,
                    accepted, proposed,
                    adaptive_tps_ema[1], adaptive_tps_ema[2], adaptive_tps_ema[3], adaptive_tps_ema[4],
                    adaptive_pos_ema[1], adaptive_pos_ema[2], adaptive_pos_ema[3], adaptive_pos_ema[4]);
            std::fflush(stderr);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        if (rejection_sampling) {
            common_sampler_reset(smpls[seq_id].get());
            if (backend_chains[seq_id]) {
                llama_sampler_reset(backend_chains[seq_id]);
            }
            sampled_draft_probs[seq_id].clear();
            rank_rng[seq_id].seed((rank_seed ^ 0x9e3779b9u) + (uint32_t) seq_id);
        }

        auto * ctx_dft = this->params.ctx_dft;

        if (defer_prefill_tail > 0 && !deferred_prefill_ready[seq_id]) {
            common_batch_clear(batch);
            const auto & tokens = deferred_tokens[seq_id];
            const auto & positions = deferred_positions[seq_id];
            const auto & hidden = deferred_h[seq_id];
            GGML_ASSERT(tokens.size() == positions.size());
            GGML_ASSERT(hidden.size() == tokens.size() * (size_t) n_embd);

            for (size_t i = 0; i < tokens.size(); ++i) {
                common_batch_add(batch, tokens[i], positions[i], { seq_id }, 0);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                        hidden.data() + i * (size_t) n_embd,
                        (size_t) n_embd * sizeof(float));
            }

            auto * mem_dft = llama_get_memory(ctx_dft);
            llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }
                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("deferred MTP prefill head=%d failed rc=%d (tokens=%d)\n",
                            head, (int) rc, batch.n_tokens);
                    ok = false;
                    break;
                }
            }
            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0);
            }
            if (!ok) {
                return;
            }

            deferred_prefill_ready[seq_id] = true;
            deferred_tokens[seq_id].clear();
            deferred_positions[seq_id].clear();
            deferred_h[seq_id].clear();
        }
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // Position zero is an unambiguous sequence boundary.  Reset the
        // cross-batch hidden carry before pairing the first token; otherwise a
        // queued request can consume the final hidden row of the previous
        // request when the slot is reused.
        std::vector<bool> reset_done(n_seq, false);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq &&
                    batch_in.pos[k] == 0 && !reset_done[seq_id]) {
                reset(seq_id);
                reset_done[seq_id] = true;
            }
        }

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        auto * mem_dft = llama_get_memory(ctx_dft);
        bool defer_batch = false;
        if (defer_prefill_tail > 0) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                if (batch_in.pos[i_batch_beg[seq_id]] == 0) {
                    deferred_prefill_ready[seq_id] = false;
                    deferred_tokens[seq_id].clear();
                    deferred_positions[seq_id].clear();
                    deferred_h[seq_id].clear();
                    llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
                }
                defer_batch = defer_batch || !deferred_prefill_ready[seq_id];
            }
        }

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (defer_batch && !is_mem_shared) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0 || deferred_prefill_ready[seq_id]) {
                    continue;
                }
                const int32_t beg = i_batch_beg[seq_id];
                const int32_t end = i_batch_end[seq_id];
                auto & tokens = deferred_tokens[seq_id];
                auto & positions = deferred_positions[seq_id];
                auto & hidden = deferred_h[seq_id];

                // Only the final tail can survive until begin().  Copying every
                // hidden row and then erasing the prefix made deferred prefill
                // substantially more expensive than the MTP work it removed.
                // Select the useful rows before the host copy instead.  When a
                // complete tail is present in this target microbatch it fully
                // replaces the previous one; a shorter final microbatch is
                // appended and trimmed so the rolling tail remains contiguous.
                const int32_t start = std::max(beg, end - defer_prefill_tail + 1);
                if (end - beg + 1 >= defer_prefill_tail) {
                    tokens.clear();
                    positions.clear();
                    hidden.clear();
                }
                for (int32_t k = start; k <= end; ++k) {
                    if (batch_in.seq_id[k][0] != seq_id) {
                        continue;
                    }
                    const float * h_prev = k > beg
                            ? llama_get_embeddings_nextn_ith(ctx_tgt, k - 1)
                            : pending_h[seq_id].data();
                    tokens.push_back(batch_in.token[k]);
                    positions.push_back(batch_in.pos[k]);
                    hidden.insert(hidden.end(), h_prev, h_prev + n_embd);
                }
                if (tokens.size() > (size_t) defer_prefill_tail) {
                    const size_t drop = tokens.size() - (size_t) defer_prefill_tail;
                    tokens.erase(tokens.begin(), tokens.begin() + drop);
                    positions.erase(positions.begin(), positions.begin() + drop);
                    hidden.erase(hidden.begin(), hidden.begin() + drop * (size_t) n_embd);
                }
            }
        } else if (!is_mem_shared) {
            common_batch_clear(batch);

            int32_t prefill_tail = 0;
            if (const char * value = std::getenv("LLAMA_MTP_PREFILL_TAIL")) {
                char * end = nullptr;
                const long requested = std::strtol(value, &end, 10);
                if (end != value && *end == '\0' && requested > 0) {
                    prefill_tail = (int32_t) requested;
                }
            }

            // Pair each selected token x_p with h_(p-1).  During a large
            // prompt prefill it is sufficient to retain a small tail from each
            // target microbatch: h already carries the full target context,
            // while the MTP attention cache only supplies local draft history.
            // Small verification batches remain exact.
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                const int32_t beg = i_batch_beg[seq_id];
                const int32_t end = i_batch_end[seq_id];
                const int32_t start = prefill_tail > 0 ? std::max(beg, end - prefill_tail + 1) : beg;
                for (int32_t k = start; k <= end; ++k) {
                    if (batch_in.seq_id[k][0] != seq_id) {
                        continue;
                    }
                    common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { seq_id }, 0);
                    const float * h_prev = k > beg
                            ? llama_get_embeddings_nextn_ith(ctx_tgt, k - 1)
                            : pending_h[seq_id].data();
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_prev, row_bytes);
                }
            }

            // The native MTP helper only needs a rolling attention history.  If
            // its independently clamped context is smaller than the target
            // prompt, evict the oldest draft KV before decoding this chunk.
            // Positions stay absolute, so RoPE and the target hidden-state
            // alignment are unchanged; only stale MTP attention history is
            // discarded.  Leave a small reserve for speculative tokens.
            const int32_t n_ctx_dft = (int32_t) llama_n_ctx(ctx_dft);
            const int32_t reserve_dft = std::max(8, this->params.n_max + 2);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                const int32_t beg = i_batch_beg[seq_id];
                const int32_t end = i_batch_end[seq_id];
                const int32_t start = prefill_tail > 0 ? std::max(beg, end - prefill_tail + 1) : beg;
                const int32_t n_rows = end - start + 1;
                const int32_t max_prior = std::max(0, n_ctx_dft - n_rows - reserve_dft);
                const llama_pos pos_first = batch_in.pos[start];
                const llama_pos cutoff = pos_first - max_prior;
                if (cutoff > 0 && !llama_memory_seq_rm(mem_dft, seq_id, -1, cutoff)) {
                    SPC_ERR("failed to roll MTP memory for seq=%d before pos=%d (cutoff=%d)\n",
                            (int) seq_id, (int) pos_first, (int) cutoff);
                    return false;
                }
            }

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // Prompt prefill can contain thousands of hidden rows. Once the
            // MTP context has consumed them, only the final row is carried into
            // the next cycle. Verification batches are bounded by n_max + 1
            // and still retain every row for accept().
            const char * compact_prefill_h_env = std::getenv("LLAMA_MTP_COMPACT_PREFILL_VERIFY_H");
            const bool compact_prefill_h = compact_prefill_h_env != nullptr &&
                    std::strcmp(compact_prefill_h_env, "0") != 0;
            if (compact_prefill_h && n_rows > params.n_max + 1) {
                verify_h_rows[seq_id] = 1;
                verify_h[seq_id].resize((size_t) n_embd);
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_end[seq_id]);
                std::memcpy(verify_h[seq_id].data(), h, row_bytes);
                std::memcpy(pending_h[seq_id].data(), h, row_bytes);
            } else {
                verify_h_rows[seq_id] = n_rows;
                verify_h[seq_id].resize((size_t) n_rows * n_embd);

                for (int32_t i = 0; i < n_rows; ++i) {
                    const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                    std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
                }

                std::memcpy(pending_h[seq_id].data(),
                        verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            if (adaptive_is_enabled()) {
                adaptive_last_depth[seq_id] = adaptive_depth[seq_id];
                adaptive_last_proposed[seq_id] = 0;
                adaptive_cycle_start_us[seq_id] = ggml_time_us();
            }
            if (rejection_sampling) {
                sampled_draft_probs[seq_id].clear();
            } else {
                common_sampler_reset(smpls[seq_id].get());
            }
            last_second_confidence[seq_id] = -1.0f;
            last_third_confidence[seq_id] = -1.0f;
            last_fourth_confidence[seq_id] = -1.0f;

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                std::vector<llama_token_data> rank_dist;
                llama_token id;
                if (rejection_sampling && (rank_proposal || position_temp_proposal)) {
                    // Calibrated target-mass priors for MTP ranks 1..10. The
                    // ranking still comes from the current MTP logits; only q's
                    // calibration differs per recursively drafted position.
                    static constexpr double rank_weights[3][10] = {
                        { 0.880, 0.055, 0.042, 0.016, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001 },
                        { 0.769, 0.072, 0.023, 0.060, 0.025, 0.0102, 0.0102, 0.0102, 0.0102, 0.0102 },
                        { 0.876, 0.059, 0.0035, 0.027, 0.024, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021 },
                    };
                    const size_t pos = std::min<size_t>(2, result.size());
                    rank_dist.assign(cur_p->data, cur_p->data + cur_p->size);
                    std::vector<double> active_weights(rank_dist.size(), 0.0);
                    double sum = 0.0;
                    if (position_temp_proposal) {
                        double max_logit = -INFINITY;
                        for (const auto & item : rank_dist) {
                            max_logit = std::max(max_logit,
                                    (double) item.logit * proposal_temp / position_temps[pos]);
                        }
                        for (size_t rank = 0; rank < rank_dist.size(); ++rank) {
                            const double w = std::exp(
                                    (double) rank_dist[rank].logit * proposal_temp / position_temps[pos] - max_logit);
                            active_weights[rank] = w;
                            sum += w;
                        }
                    } else {
                        for (size_t rank = 0; rank < rank_dist.size(); ++rank) {
                            const double w = rank < 10 ? rank_weights[pos][rank] : 0.0;
                            active_weights[rank] = w;
                            sum += w;
                        }
                    }
                    GGML_ASSERT(sum > 0.0);
                    for (size_t rank = 0; rank < rank_dist.size(); ++rank) {
                        rank_dist[rank].p = (float) (active_weights[rank] / sum);
                    }
                    std::discrete_distribution<size_t> choose(active_weights.begin(), active_weights.end());
                    id = rank_dist[choose(rank_rng[seq_id])].id;
                } else {
                    id = rejection_sampling
                        ? cur_p->data[cur_p->selected].id
                        : cur_p->data[0].id;
                }

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                if (rejection_sampling) {
                    sampled_draft_probs[seq_id].push_back((rank_proposal || position_temp_proposal)
                        ? std::move(rank_dist)
                        : std::vector<llama_token_data>(cur_p->data, cur_p->data + cur_p->size));
                }

                result.push_back(id);

                if (result.size() == 2) {
                    last_second_confidence[seq_id] = cur_p->data[0].p;
                }
                if (result.size() == 3) {
                    last_third_confidence[seq_id] = cur_p->data[0].p;
                }
                if (result.size() == 4) {
                    last_fourth_confidence[seq_id] = cur_p->data[0].p;
                }

                // Qwen's third MTP head is often the least reliable.  Unlike
                // --spec-draft-p-min, this gate never drops the first or
                // second proposal: it only avoids launching the third head
                // when the second-head confidence is too low.  The target
                // still verifies every returned token, so this is purely a
                // scheduling optimization.
                static const float mtp_third_trigger_p_min = [] {
                    const char * value = getenv("LLAMA_MTP_THIRD_TRIGGER_P_MIN");
                    return value != nullptr ? strtof(value, nullptr) : 0.0f;
                }();
                if (result.size() == 2 && mtp_third_trigger_p_min > 0.0f &&
                    cur_p->data[0].p < mtp_third_trigger_p_min) {
                    if (getenv("LLAMA_MTP_PAD_SKIPPED_DRAFT") != nullptr) {
                        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_dft));
                        const llama_token sentinel = llama_vocab_bos(vocab);
                        if (sentinel >= 0) {
                            while (result.size() < (size_t) params.n_max) {
                                result.push_back(sentinel);
                            }
                        }
                    }
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                // The recurrent fourth proposal is substantially less useful
                // than the first three. Skip only that final decode when the
                // third proposal's own confidence is low; target verification
                // remains unchanged, so this is a scheduling optimization.
                static const float mtp_fourth_trigger_p_min = [] {
                    const char * value = getenv("LLAMA_MTP_FOURTH_TRIGGER_P_MIN");
                    return value != nullptr ? strtof(value, nullptr) : 0.0f;
                }();
                if (result.size() == 3 && mtp_fourth_trigger_p_min > 0.0f &&
                    cur_p->data[0].p < mtp_fourth_trigger_p_min) {
                    if (getenv("LLAMA_MTP_PAD_SKIPPED_DRAFT") != nullptr) {
                        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_dft));
                        const llama_token sentinel = llama_vocab_bos(vocab);
                        if (sentinel >= 0) {
                            // Keep the verifier batch shape identical to MTP4.
                            // The target validates this special-token proposal;
                            // it is never emitted unless the target itself picks it.
                            while (result.size() < (size_t) params.n_max) {
                                result.push_back(sentinel);
                            }
                        }
                    }
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                static const float mtp_fifth_trigger_p_min = [] {
                    const char * value = getenv("LLAMA_MTP_FIFTH_TRIGGER_P_MIN");
                    return value != nullptr ? strtof(value, nullptr) : 0.0f;
                }();
                if (result.size() == 4 && mtp_fifth_trigger_p_min > 0.0f &&
                    cur_p->data[0].p < mtp_fifth_trigger_p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                int32_t draft_limit = params.n_max;
                if (adaptive_is_enabled()) {
                    draft_limit = adaptive_depth[seq_id];
                    if (dp.n_max > 0) {
                        draft_limit = std::min(draft_limit, dp.n_max);
                    }
                }
                if (draft_limit <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
            if (adaptive_is_enabled()) {
                adaptive_last_proposed[seq_id] = (int32_t) dp.result->size();
                const char * fixed_verify = std::getenv("LLAMA_MTP_ADAPTIVE_FIXED_VERIFY");
                if (fixed_verify != nullptr && fixed_verify[0] != '\0' &&
                        std::strcmp(fixed_verify, "0") != 0 && !dp.result->empty()) {
                    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_dft));
                    const llama_token sentinel = llama_vocab_bos(vocab);
                    if (sentinel >= 0) {
                        while (dp.result->size() < (size_t) params.n_max) {
                            dp.result->push_back(sentinel);
                        }
                    }
                }
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);

        if (!is_other && adaptive_is_enabled()) {
            adaptive_observe(seq_id, n_accepted);
        }

        if (!is_other && getenv("LLAMA_SPEC_ACCEPT_STATS") != nullptr &&
                last_second_confidence[seq_id] >= 0.0f) {
            const int bin = std::max(0, std::min(9,
                    (int) (last_second_confidence[seq_id] * 10.0f)));
            third_conf_attempts[bin]++;
            if (n_accepted >= 3) {
                third_conf_accepts[bin]++;
            }
            if (last_third_confidence[seq_id] >= 0.0f) {
                const int fourth_bin = std::max(0, std::min(9,
                        (int) (last_third_confidence[seq_id] * 10.0f)));
                fourth_conf_attempts[fourth_bin]++;
                if (n_accepted >= 4) {
                    fourth_conf_accepts[fourth_bin]++;
                }
            }
            if (last_fourth_confidence[seq_id] >= 0.0f) {
                const int fifth_bin = std::max(0, std::min(9,
                        (int) (last_fourth_confidence[seq_id] * 10.0f)));
                fifth_conf_attempts[fifth_bin]++;
                if (n_accepted >= 5) {
                    fifth_conf_accepts[fifth_bin]++;
                }
            }
            ++third_conf_reports;
            if ((third_conf_reports % 32) == 0) {
                fprintf(stderr, "MTP_THIRD_CONF_STATS: calls=%llu bins=",
                        (unsigned long long) third_conf_reports);
                for (int i = 0; i < 10; ++i) {
                    fprintf(stderr, "%s%d:%llu/%llu", i == 0 ? "" : ",", i,
                            (unsigned long long) third_conf_accepts[i],
                            (unsigned long long) third_conf_attempts[i]);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "MTP_FOURTH_CONF_STATS: calls=%llu bins=",
                        (unsigned long long) third_conf_reports);
                for (int i = 0; i < 10; ++i) {
                    fprintf(stderr, "%s%d:%llu/%llu", i == 0 ? "" : ",", i,
                            (unsigned long long) fourth_conf_accepts[i],
                            (unsigned long long) fourth_conf_attempts[i]);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "MTP_FIFTH_CONF_STATS: calls=%llu bins=",
                        (unsigned long long) third_conf_reports);
                for (int i = 0; i < 10; ++i) {
                    fprintf(stderr, "%s%d:%llu/%llu", i == 0 ? "" : ",", i,
                            (unsigned long long) fifth_conf_accepts[i],
                            (unsigned long long) fifth_conf_attempts[i]);
                }
                fprintf(stderr, "\n");
                fflush(stderr);
            }
        }
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }

        data.clear();
        state_append(data, state_magic);
        state_append(data, state_version);
        state_append(data, n_embd);

        uint32_t flags = 0;
        flags |= rejection_sampling      ? 1u << 0 : 0;
        flags |= adaptive_depth_enabled  ? 1u << 1 : 0;
        flags |= chain_heads             ? 1u << 2 : 0;
        flags |= is_mem_shared           ? 1u << 3 : 0;
        state_append(data, flags);

        state_append_vector(data, pending_h[seq_id]);

        const uint32_t deferred_ready = deferred_prefill_ready[seq_id] ? 1u : 0u;
        state_append(data, deferred_ready);
        state_append_vector(data, deferred_tokens[seq_id]);
        state_append_vector(data, deferred_positions[seq_id]);
        state_append_vector(data, deferred_h[seq_id]);

        const int32_t depth = adaptive_depth_enabled ? adaptive_depth[seq_id] : -1;
        const int32_t last_depth = adaptive_depth_enabled ? adaptive_last_depth[seq_id] : -1;
        const int32_t last_proposed = adaptive_depth_enabled ? adaptive_last_proposed[seq_id] : 0;
        state_append(data, depth);
        state_append(data, last_depth);
        state_append(data, last_proposed);
        state_append(data, adaptive_decisions);
        for (int i = 0; i < 5; ++i) state_append(data, adaptive_tps_ema[i]);
        for (int i = 0; i < 5; ++i) state_append(data, adaptive_accept_ema[i]);
        for (int i = 0; i < 5; ++i) state_append(data, adaptive_samples[i]);
        for (int i = 0; i < 5; ++i) state_append(data, adaptive_pos_attempts[i]);
        for (int i = 0; i < 5; ++i) state_append(data, adaptive_pos_accepts[i]);
        for (int i = 0; i < 5; ++i) state_append(data, adaptive_pos_ema[i]);
        for (int i = 0; i < 5; ++i) state_append(data, adaptive_pos_samples[i]);

        std::ostringstream rng_stream;
        rng_stream << rank_rng[seq_id];
        const std::string rng_text = rng_stream.str();
        const std::vector<uint8_t> rng_bytes(rng_text.begin(), rng_text.end());
        state_append_vector(data, rng_bytes);

        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // Empty state is the explicit reset representation used after a full
        // memory clear or when a cached prefix cannot be restored exactly.
        reset(seq_id);
        if (data.empty()) {
            return;
        }

        size_t offset = 0;
        uint32_t magic = 0;
        uint32_t version = 0;
        int32_t stored_n_embd = 0;
        uint32_t flags = 0;

        std::vector<float> stored_pending_h;
        uint32_t deferred_ready = 0;
        std::vector<llama_token> stored_deferred_tokens;
        std::vector<llama_pos> stored_deferred_positions;
        std::vector<float> stored_deferred_h;
        int32_t stored_depth = -1;
        int32_t stored_last_depth = -1;
        int32_t stored_last_proposed = 0;
        uint64_t stored_decisions = 0;
        double stored_tps_ema[5] = {};
        double stored_accept_ema[5] = {};
        uint64_t stored_samples[5] = {};
        uint64_t stored_pos_attempts[5] = {};
        uint64_t stored_pos_accepts[5] = {};
        double stored_pos_ema[5] = {};
        uint64_t stored_pos_samples[5] = {};
        std::vector<uint8_t> rng_bytes;

        uint32_t expected_flags = 0;
        expected_flags |= rejection_sampling     ? 1u << 0 : 0;
        expected_flags |= adaptive_depth_enabled ? 1u << 1 : 0;
        expected_flags |= chain_heads            ? 1u << 2 : 0;
        expected_flags |= is_mem_shared          ? 1u << 3 : 0;

        bool ok =
            state_read(data, offset, magic) &&
            state_read(data, offset, version) &&
            state_read(data, offset, stored_n_embd) &&
            state_read(data, offset, flags) &&
            magic == state_magic && version == state_version && stored_n_embd == n_embd &&
            flags == expected_flags &&
            state_read_vector(data, offset, stored_pending_h, (uint64_t) n_embd) &&
            stored_pending_h.size() == (size_t) n_embd &&
            state_read(data, offset, deferred_ready) &&
            state_read_vector(data, offset, stored_deferred_tokens, 1u << 20) &&
            state_read_vector(data, offset, stored_deferred_positions, 1u << 20);

        const uint64_t max_hidden = std::min<uint64_t>(
                (uint64_t) 1u << 31,
                (uint64_t) std::max<size_t>(1, stored_deferred_tokens.size()) * (uint64_t) n_embd);
        ok = ok && state_read_vector(data, offset, stored_deferred_h, max_hidden) &&
            stored_deferred_tokens.size() == stored_deferred_positions.size() &&
            stored_deferred_h.size() == stored_deferred_tokens.size() * (size_t) n_embd &&
            state_read(data, offset, stored_depth) &&
            state_read(data, offset, stored_last_depth) &&
            state_read(data, offset, stored_last_proposed) &&
            state_read(data, offset, stored_decisions);

        for (int i = 0; i < 5 && ok; ++i) ok = state_read(data, offset, stored_tps_ema[i]);
        for (int i = 0; i < 5 && ok; ++i) ok = state_read(data, offset, stored_accept_ema[i]);
        for (int i = 0; i < 5 && ok; ++i) ok = state_read(data, offset, stored_samples[i]);
        for (int i = 0; i < 5 && ok; ++i) ok = state_read(data, offset, stored_pos_attempts[i]);
        for (int i = 0; i < 5 && ok; ++i) ok = state_read(data, offset, stored_pos_accepts[i]);
        for (int i = 0; i < 5 && ok; ++i) ok = state_read(data, offset, stored_pos_ema[i]);
        for (int i = 0; i < 5 && ok; ++i) ok = state_read(data, offset, stored_pos_samples[i]);
        ok = ok && state_read_vector(data, offset, rng_bytes, 1u << 20) && offset == data.size();

        if (!ok) {
            SPC_WRN("invalid MTP checkpoint state for seq_id=%d; using reset state\n", (int) seq_id);
            reset(seq_id);
            return;
        }

        pending_h[seq_id] = std::move(stored_pending_h);
        deferred_prefill_ready[seq_id] = deferred_ready != 0;
        deferred_tokens[seq_id] = std::move(stored_deferred_tokens);
        deferred_positions[seq_id] = std::move(stored_deferred_positions);
        deferred_h[seq_id] = std::move(stored_deferred_h);

        if (adaptive_depth_enabled && stored_depth >= adaptive_depth_min && stored_depth <= adaptive_depth_max &&
                stored_last_depth >= adaptive_depth_min && stored_last_depth <= adaptive_depth_max) {
            adaptive_depth[seq_id] = stored_depth;
            adaptive_last_depth[seq_id] = stored_last_depth;
            adaptive_last_proposed[seq_id] = std::max(0, stored_last_proposed);
            adaptive_cycle_start_us[seq_id] = 0;
            if (n_seq == 1) {
                adaptive_decisions = stored_decisions;
                std::copy(std::begin(stored_tps_ema),      std::end(stored_tps_ema),      std::begin(adaptive_tps_ema));
                std::copy(std::begin(stored_accept_ema),   std::end(stored_accept_ema),   std::begin(adaptive_accept_ema));
                std::copy(std::begin(stored_samples),      std::end(stored_samples),      std::begin(adaptive_samples));
                std::copy(std::begin(stored_pos_attempts), std::end(stored_pos_attempts), std::begin(adaptive_pos_attempts));
                std::copy(std::begin(stored_pos_accepts),  std::end(stored_pos_accepts),  std::begin(adaptive_pos_accepts));
                std::copy(std::begin(stored_pos_ema),      std::end(stored_pos_ema),      std::begin(adaptive_pos_ema));
                std::copy(std::begin(stored_pos_samples),  std::end(stored_pos_samples),  std::begin(adaptive_pos_samples));
            }
        }

        if (!rng_bytes.empty()) {
            const std::string rng_text(rng_bytes.begin(), rng_bytes.end());
            std::istringstream rng_stream(rng_text);
            std::mt19937 restored;
            if (rng_stream >> restored) {
                rank_rng[seq_id] = restored;
            }
        }
    }

    const std::vector<std::vector<llama_token_data>> * draft_probs(
            llama_seq_id seq_id) const override {
        return rejection_sampling ? &sampled_draft_probs.at(seq_id) : nullptr;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    if (has_draft) {
        result.devices               = params_spec.devices;
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;

        // The target may expose a long operational context while the one-layer
        // MTP helper only needs a shorter rolling history for useful drafts.
        // Keeping both caches at the target size wastes scarce device memory
        // and prevents a larger target prefill micro-batch.  Make the draft
        // capacity independently clampable; positions remain absolute and the
        // normal memory implementation is responsible for recycling cells.
        if (const char * value = std::getenv("LLAMA_MTP_CTX_SIZE")) {
            char * end = nullptr;
            const long requested = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && requested > 0) {
                const uint32_t old_ctx = cparams.n_ctx;
                cparams.n_ctx = std::min<uint32_t>(old_ctx, (uint32_t) requested);
                LOG_INF("%s: MTP-only context clamp: %u -> %u\n",
                        __func__, old_ctx, cparams.n_ctx);
            } else {
                LOG_WRN("%s: ignoring invalid LLAMA_MTP_CTX_SIZE='%s'\n", __func__, value);
            }
        }

        // The MTP context normally inherits the target micro-batch size even
        // though drafting only decodes a handful of tokens at a time.  On
        // memory-constrained GPUs that reserves a large, mostly idle compute
        // buffer and prevents the target context from using a larger ubatch
        // for prefill.  Allow an explicit MTP-only clamp; n_batch is kept
        // unchanged because common_speculative_impl_draft_mtp allocates its
        // reusable input batch from it.
        if (const char * value = std::getenv("LLAMA_MTP_UBATCH")) {
            char * end = nullptr;
            const long requested = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && requested > 0) {
                const uint32_t old_ubatch = cparams.n_ubatch;
                cparams.n_ubatch = std::min<uint32_t>(old_ubatch, (uint32_t) requested);
                LOG_INF("%s: MTP-only ubatch clamp: %u -> %u\n",
                        __func__, old_ubatch, cparams.n_ubatch);
            } else {
                LOG_WRN("%s: ignoring invalid LLAMA_MTP_UBATCH='%s'\n", __func__, value);
            }
        }
    }

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

bool common_speculative_needs_prompt_history(const common_speculative * spec) {
    GGML_ASSERT(spec);
    for (const auto & impl : spec->impls) {
        if (impl->needs_prompt_history()) {
            return true;
        }
    }
    return false;
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }

    if (getenv("LLAMA_SPEC_ACCEPT_STATS") != nullptr) {
        static size_t report_counter = 0;
        ++report_counter;
        if ((report_counter % 32) == 0) {
            for (const auto & impl_stats : spec->impls) {
                fprintf(stderr,
                        "SPEC_ACCEPT_STATS: report=%zu impl=%s calls=%zu gen_drafts=%zu gen_tokens=%zu acc_drafts=%zu acc_tokens=%zu per_pos=",
                        report_counter,
                        common_speculative_type_to_str(impl_stats->type).c_str(),
                        impl_stats->n_call_accept,
                        impl_stats->n_gen_drafts,
                        impl_stats->n_gen_tokens,
                        impl_stats->n_acc_drafts,
                        impl_stats->n_acc_tokens);
                for (size_t pos = 0; pos < impl_stats->n_acc_tokens_per_pos.size(); ++pos) {
                    fprintf(stderr, "%s%zu", pos == 0 ? "" : ",",
                            impl_stats->n_acc_tokens_per_pos[pos]);
                }
                fprintf(stderr, "\n");
            }
            fflush(stderr);
        }
    }
}

const std::vector<std::vector<llama_token_data>> * common_speculative_get_draft_probs(
        const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->impl_last.size()) {
        return nullptr;
    }
    const common_speculative_impl * impl = spec->impl_last[seq_id];
    return impl ? impl->draft_probs(seq_id) : nullptr;
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_reset(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->reset(seq_id);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
