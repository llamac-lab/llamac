/*
* GGML - A pure C runtime for LLaMA models, optimized for edge devices.
 *
 * This file is part of the GGML project: https://github.com/GGML-lab/GGML
 *
 * This is a derivative work based on llama.cpp by Georgi Gerganov (MIT License).
 * Significant changes have been made to modularize, refactor, and port the code to C.
 *
 * Original project: https://github.com/ggerganov/llama.cpp
 *
 * Copyright (c) 2024–2025 Ervin Bosenbacher and contributors.
 * Licensed under the MIT License. See LICENSE file in the repository root.
 */

#ifndef LLCML_H
#define LLCML_H


#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include "../llmclm/include/matmul.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// --------------------- structs and enums
#define llamac_LLM_API

#define MAX_MESSAGES 64
#define MAX_MESSAGE_LEN 1024
#define MAX_FORMATTED_LEN (2048 * 4)

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_MESSAGE_LEN 1024
#define MAX_MESSAGES    64

    // --- Opaque forward declarations
    typedef struct llama_model llama_model;
    typedef struct llama_context llama_context;
    typedef struct llama_sampler llama_sampler;
    typedef struct llama_vocab llama_vocab;

    // --- Chat message structure
    typedef struct {
        const char *role;
        char content[MAX_MESSAGE_LEN];
    } llamac_chat_message;

    typedef struct {
        llamac_chat_message messages[MAX_MESSAGES];
        int count;
        int prev_len;
    } llamac_chat_state;

    // --- Runtime struct
    typedef struct llamac_runtime {
        llama_model     *model;
        llama_context   *ctx;
        llama_sampler   *sampler;
        int             max_tokens;

        float           temperature;
        float           top_p;
        float           min_p;

        llamac_chat_state history;
    } llamac_runtime;


    // -----------------------------------------------
    // --- API functions
    // -----------------------------------------------
    int llamac_runtime_init(void);

    int llamac_model_load(
        const char      *model_path,
        int             n_gpu_layers,
        int             n_context,
        int             n_batch,
        float           min_p,
        float           temperature,
        llamac_runtime    *runtime);

    int llamac_one_shot(
        const char      *prompt,
        const char      *role,
        llamac_runtime    *rt,
        char            *out_buf,
        size_t          out_len,
        int             *token_count);

    int llamac_history_shot (
        const char      *prompt,
        const char      *role,
        llamac_runtime    *rt,
        char            *out_buf,
        size_t          out_len,
        int             *token_count);

    int llamac_chat(llamac_runtime *rt, const char *role);
    void llamac_free(llamac_runtime *runtime);

    void llamac_kv_cache_clear(llama_context *ctx);
    void llamac_sampler_rebuild(llamac_runtime *rt);
    void llamac_reset_history(llamac_runtime *rt);

    int llamac_chat_history(llamac_runtime *rt, const char *role);
    int llamac_runtime_init_with_log_level(int min_level);

#ifdef __cplusplus
}
#endif

#endif //LLCML_H
