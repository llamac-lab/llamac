//
// Created by ervin on 7/13/25.
//
#include "llamac_internal.h"
#include <unistd.h>
#include <threads.h>
#include <string.h>
static once_flag g_init_once = ONCE_FLAG_INIT;

/* one-time global init: back-ends + logging */
static void runtime_global_init(void)
{
    setenv("GGML_CUDA_FORCE_MMQ",    "1", 1);
    setenv("GGML_CUDA_FORCE_CUBLAS", "1", 1);
    ggml_backend_load_all();         /* GGML device discovery */
    llamac_logging_init();           /* log proxy */
}

// constructor
llamac_runtime *llamac_runtime_create(void)
{
    call_once(&g_init_once, runtime_global_init);

    llamac_runtime *rt = calloc(1, sizeof *rt);
    if (!rt) {
        fprintf(stderr, "[llamac] out of memory creating runtime\n");
        return NULL;
    }

    // some defaults / maybe set them to zero or drive them from config/globals?
    rt->max_tokens  = LLAMAC_MAX_TOKENS;
    rt->temperature = 0.80f;
    rt->top_p       = 0.95f;
    rt->min_p       = 0.05f;

    rt->history.count    = 0;
    rt->history.prev_len = 0;

    return rt;
}

static void free_chat_history(llamac_runtime *rt)
{
    for (int i = 0; i < rt->history.count; ++i) {
        free((void *)rt->history.messages[i].role);
        rt->history.messages[i].role = NULL;
    }
    rt->history.count    = 0;
    rt->history.prev_len = 0;
}

// destructor
void llamac_runtime_destroy(llamac_runtime *rt)
{
    if (!rt)
        return;

    free_chat_history(rt);

    if (rt->sampler)    llama_sampler_free(rt->sampler);
    if (rt->ctx)        llama_free(rt->ctx);
    if (rt->model)      llama_model_free(rt->model);

    free(rt);
}

// void llamac_free(llamac_runtime *rt) {
//     if (!rt) return;
//
//     // Free chat history roles
//     for (int i = 0; i < rt->history.count; ++i) {
//         if (rt->history.messages[i].role != NULL) {
//             free((void *)rt->history.messages[i].role);
//             rt->history.messages[i].role = NULL;
//         }
//     }
//
//     rt->history.count = 0;
//     rt->history.prev_len = 0;
//
//     // Free model-related components
//     if (rt->sampler) llama_sampler_free(rt->sampler);
//     if (rt->ctx) llama_free(rt->ctx);
//     if (rt->model) llama_model_free(rt->model);
// }
