#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    int32_t moe_slot_count;
    int32_t moe_slot_moves;
    int32_t moe_slot_window;
    int32_t moe_slot_stability_window;
    float   moe_slot_stability_threshold;
    int32_t moe_slot_protect_recent;
    int32_t moe_slot_bootstrap;
    int32_t moe_slot_prefill;
    int32_t moe_slot_log;
    const struct ggml_tensor * const * moe_slot_expert_to_slot = nullptr; // [n_layer] each I32 [n_expert], local to layer slot view
    const struct ggml_tensor * const * moe_slot_up_exps        = nullptr; // [n_layer] per-layer slot-bank view
    const struct ggml_tensor * const * moe_slot_gate_exps      = nullptr; // [n_layer] per-layer slot-bank view
    const struct ggml_tensor * const * moe_slot_gate_up_exps   = nullptr; // [n_layer] per-layer slot-bank view
    const struct ggml_tensor * const * moe_slot_down_exps      = nullptr; // [n_layer] per-layer slot-bank view

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
