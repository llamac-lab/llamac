//
// Created by ervin on 7/13/25.
//

#pragma once

#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <pthread.h>
#include <time.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../include/llamac.h"
#include "../include/llamac_log.h"

typedef struct {
    const char *role;
    char        content[LLAMAC_MAX_MESSAGE_LEN];
} llamac_chat_message;

typedef struct {
    llamac_chat_message messages[LLAMAC_MAX_MESSAGES];
    int count;
    int prev_len;
} llamac_chat_state;

struct llamac_runtime {
    llama_model   *model;
    llama_context *ctx;
    llama_sampler *sampler;
    int            max_tokens;
    float          temperature, top_p, min_p;
    llamac_chat_state history;
};

void llamac_logging_init(void);