/*
* llamac - A pure C runtime for LLaMA models, optimized for edge devices.
 *
 * This file is part of the llamac project: https://github.com/llamac-lab/llamac
 *
 * This is a derivative work based on llama.cpp by Georgi Gerganov (MIT License).
 * Significant changes have been made to modularize, refactor, and port the code to C.
 *
 * Original project: https://github.com/ggerganov/llama.cpp
 *
 * Copyright (c) 2024–2025 Ervin Bosenbacher and contributors.
 * Licensed under the MIT License. See LICENSE file in the repository root.
 */


#ifndef LLAMAC_SAMPLER_H
#define LLAMAC_SAMPLER_H

#ifdef __cplusplus
extern "C" {
#endif

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        // accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
    //        llama_sampler_accept(smpl, id);
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //

    typedef void * llamac_sampler_context_t;

    // user code can implement the interface below in order to create custom llamac_sampler
    // _i -> interface
    struct llamac_sampler_i {
        const char *           (*name)  (const struct llamac_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llamac_sampler * smpl, llamac_token token);              // can be NULL
        void                   (*apply) (      struct llamac_sampler * smpl, llamac_token_data_array * cur_p); // required
        void                   (*reset) (      struct llamac_sampler * smpl);                                 // can be NULL
        struct llamac_sampler* (*clone) (const struct llamac_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llamac_sampler * smpl);                                 // can be NULL if ctx is NULL

        // todo: the below todo from the original codebase
        // todo: API for internal libllama usage for appending the sampling to an existing ggml_cgraph
        //void (*apply_llcml) (struct llamac_sampler * smpl, ...);
    };

    struct llamac_sampler {
        const struct llamac_sampler_i * iface;
        llamac_sampler_context_t        ctx;
    };

// llamac_sampler_free
// llamac_sampler_chain_init
// llamac_sampler_chain_add
// llamac_sampler_init_min_p
// llamac_sampler_init_temp
// llamac_sampler_init_top_p
// llamac_sampler_init_dist
//
// llamac_model_load
//

#ifdef __cplusplus
}
#endif


#endif //LLAMAC_SAMPLER_H
