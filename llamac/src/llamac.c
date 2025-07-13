//
// Created by ervin on 7/5/25.
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <llama.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <dirent.h>


#include "llamac.h"


// --- the callback used by the chat loop
typedef int (*generate_fn_t)(
    const llama_vocab *vocab,
    llama_sampler *smpl,
    const char *prompt,
    void *ctx,
    char *output,
    size_t output_size,
    int *token_count
);



#define LLAMAC_DEBUG
// ------------ logging
struct log_config {
    int level;
};

static void llamac_log_proxy(enum ggml_log_level level, const char * text, void * user_data) {
    struct log_config *cfg = (struct log_config *) user_data;
    if (level >= cfg->level) {
        fprintf(stderr, "%s", text);
    }
}

int llamac_runtime_init_with_log_level(int min_level) {
    static struct log_config cfg;
    cfg.level = min_level;

    llama_log_set(llamac_log_proxy, &cfg);
    return 0;
}
// ------------ logging

const int MAX_RESPONSE_SIZE = 2048 * 8;
#define LLAMAC_MAX_TOKENS 512

int count_threads() {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/task", getpid());
    DIR *dir = opendir(path);
    int count = 0;
    if (dir) {
        while (readdir(dir)) count++;
        closedir(dir);
    }
    return count - 2; // skip "." and ".."
}

int generate_story(
    const llama_vocab *vocab,
    llama_sampler *smpl,
    const char *prompt,
    void *ctx,
    char *output,
    size_t output_size,
    int *token_count) {

    // run inference
    llama_context *local_ctx = (llama_context *) ctx;

    //strcpy(output, "");
    //*token_count = 42;

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

    // tokenize the prompt
    const int32_t prompt_len = (int32_t) strlen(prompt);

    const int n_prompt_tokens = -llama_tokenize(
        vocab,
        prompt,
        prompt_len,
        NULL,
        0,
        is_first,
        true);

    if (n_prompt_tokens <= 0) {
        fprintf(stderr, "[error] failed to count tokens\n");
        return 1;
    }

    // Allocate token buffer
    llama_token *prompt_tokens = malloc(n_prompt_tokens * sizeof(llama_token));
    if (!prompt_tokens) {
        fprintf(stderr, "[error] out of memory\n");
        return 1;
    }

    const int rc = llama_tokenize(
        vocab,
        prompt,
        prompt_len,
        prompt_tokens,
        n_prompt_tokens,
        is_first,
        true);

    if (rc < 0) {
        fprintf(stderr, "[error] failed to tokenize\n");
        free(prompt_tokens);
        return 1;
    }
    //for (int i = 0; i < n_prompt_tokens; i++) {
    //    printf("[debug]: %d\n", *(prompt_tokens + i));
    //}
    llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt_tokens);
    llama_token new_token_id = 0;


    char *out_ptr = output;         // where to write next
    size_t remaining = output_size; // max

    while (true) {
        if (*token_count >= LLAMAC_MAX_TOKENS) {
            printf("[llamac] hit max token limit (%d)\n", *token_count);
            break;
        }

        const uint32_t n_ctx = llama_n_ctx(ctx);

        const llama_pos n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            printf("\033[0m\n");
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }
        if (llama_decode(ctx, batch)) {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        *token_count += 1;

        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
           break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        const int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf) - 1, 0, true);
        if (n < 0) {
            GGML_ABORT("failed to convert token to piece\n");
        }
        buf[n] = '\0';
        //printf("%s", buf);
        fflush(stdout);

        // Accumulate to output buffer
        if ((size_t)n < remaining) {
            memcpy(out_ptr, buf, n);
            out_ptr += n;
            remaining -= n;
            *out_ptr = '\0';
        } else {
            fprintf(stderr, "[warn] output buffer full — result truncated\n");
            break;
        }

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    free(prompt_tokens);
    return 0; // success
}

void llamac_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    // Adjust threshold here
    if (level >= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text); // Only log warnings and errors
    }
}

// -----------
int llamac_runtime_init() {
    setenv("GGML_CUDA_FORCE_MMQ", "1", 1);
    setenv("GGML_CUDA_FORCE_CUBLAS", "1", 1);
    ggml_backend_load_all();

    llama_log_set(llamac_log_callback, NULL);
    //llamac_runtime_init_with_log_level(GGML_LOG_LEVEL_DEBUG);

    return 0;
}

void llamac_sampler_rebuild(llamac_runtime *rt) {
    if (rt->sampler) {
        llama_sampler_free(rt->sampler);
    }

    rt->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(rt->sampler, llama_sampler_init_min_p(rt->min_p, 1));
    llama_sampler_chain_add(rt->sampler, llama_sampler_init_temp(rt->temperature));
    llama_sampler_chain_add(rt->sampler, llama_sampler_init_top_p(rt->top_p, 1));
    llama_sampler_chain_add(rt->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
#ifdef llamac_DEBUG
    printf("[llamac] Sampler rebuilt: min_p = %.2f, top_p = %.2f, temp = %.2f\n",
        rt->min_p, rt->top_p, rt->temperature);
#endif
}

// public interface
int llamac_model_load(
    const char      *model_path,
    int             n_gpu_layers,
    int             n_context,
    int             n_batch,
    float           min_p,
    float           temperature,
    llamac_runtime    *rt) {

    // initialize the model
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    rt->model = llama_model_load_from_file(model_path, model_params);
    if (!rt->model) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }
    // initialize the context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_context;
    ctx_params.n_batch = n_batch;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = (int32_t) sysconf(_SC_NPROCESSORS_ONLN);; // safe unless one is using millions of threads

    rt->ctx = llama_init_from_model(rt->model, ctx_params);
    if (!rt->ctx) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // Init sampler
    llamac_sampler_rebuild(rt);

#ifdef llamac_DEBUG
    printf("[llamac] Model loaded: %s\n", model_path);
    printf("[llamac] Threads = %d | Context = %d | Batch = %d | GPU layers = %d\n",
        ctx_params.n_threads, ctx_params.n_ctx, ctx_params.n_batch, n_gpu_layers);
  //  printf("[llamac] Sampler: min_p = %.2f, temp = %.2f, top_p = %.2f\n",
    //    rt->min_p, rt->temperature, rt->top_p);
#endif

    return 0;
}

int llamac_history_shot (
    const char      *prompt,
    const char      *role,
    llamac_runtime    *rt,
    char            *out_buf,
    size_t          out_len,
    int             *token_count) {

    // 1. check the inputs, no funny business here
    if (!prompt || !role || !rt || !out_buf || out_len == 0) {
        fprintf(stderr, "[llamac] Invalid parameters to llamac_history_shot\n");
        return 1;
    }

    // get the history
    llamac_chat_state *chat = &rt->history;
    // Enforce room for user + assistant messages
    // bounds check, do we have enough space in history?
    if (chat->count + 2 >= MAX_MESSAGES) {
        fprintf(stderr, "[llamac] Reached max message history\n");
        return 1;
    }
    // reset token count
    *token_count = 0;
    // reset output buffer
    memset(out_buf, 0, out_len);
    // clear the cache
    llamac_kv_cache_clear(rt->ctx);



    // --- Add user message to chat ---
    chat->messages[chat->count].role = strdup(role);
    strncpy(chat->messages[chat->count].content, prompt, MAX_MESSAGE_LEN - 1);
    chat->messages[chat->count].content[MAX_MESSAGE_LEN - 1] = '\0';
    chat->count++;

    // --- Prepare llama message format ---
    llama_chat_message llama_msgs[MAX_MESSAGES];
    for (int i = 0; i < chat->count; ++i) {
        llama_msgs[i].role    = chat->messages[i].role;
        llama_msgs[i].content = chat->messages[i].content;
    }

    // Format the full chat
    const char *tmpl = llama_model_chat_template(rt->model, NULL);
    char formatted[MAX_FORMATTED_LEN];

#ifdef LLAMC_DEBUG
    // apply the template
    printf("[llamac-debug] Chat state:\n");
    printf("  message count: %d\n", chat->count);
    for (int i = 0; i < chat->count; i++) {
        if (!chat->messages[i].role) {
            printf("  message[%d] role is NULL!\n", i);
        } else {
            printf("  message[%d] role = %s\n", i, chat->messages[i].role);
        }

        printf("  message[%d] content = %.*s\n", i, 60, chat->messages[i].content); // truncate
    }
    fflush(stdout);
#endif

     const int new_len = llama_chat_apply_template(
         tmpl,
         llama_msgs,
         chat->count,
         true,
         formatted,
         sizeof(formatted)
     );
     if (new_len < 0) {
         fprintf(stderr, "[llamac] Chat template formatting failed\n");
         return 1;
     }

    const char *formatted_prompt = formatted;
    chat->prev_len = new_len;

#ifdef LLAMC_DEBUG
    printf("[debug] ---- Prompt Context ----\n%s\n", formatted);
    printf("[llamac] role=%s | prompt_len=%d\n", role, new_len);
    printf("[debug] ---- Prompt Context ----\n%s\n", formatted_prompt);
#endif

    // get the vocab
    const llama_vocab * vocab = llama_model_get_vocab(rt->model);
    // get the generate pointer
    generate_fn_t generate = &generate_story;

    // some stats
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    printf("\033[33m");
    const int result = generate(
        vocab,
        rt->sampler,
        formatted_prompt,
        rt->ctx,
        out_buf,
        out_len,
        token_count
    );
    printf("\n\033[0m");
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (result == 0) {
        // Add assistant response to chat history
        chat->messages[chat->count].role = "assistant";
        strncpy(chat->messages[chat->count].content, out_buf, MAX_MESSAGE_LEN - 1);
        chat->messages[chat->count].content[MAX_MESSAGE_LEN - 1] = '\0';
        chat->count++;

        // Metrics
        double duration = (double)(end.tv_sec - start.tv_sec) +
                          (double)(end.tv_nsec - start.tv_nsec) / 1e9;
        double tps = *token_count / duration;

        printf("Tokens: %d\n", *token_count);
        printf("Time: %.2f seconds\n", duration);
        printf("TPS: %.2f tokens/sec\n", tps);
    }

    return result;
}

int llamac_one_shot(
    const char      *prompt,
    const char      *role,
    llamac_runtime    *rt,
    char            *out_buf,
    size_t          out_len,
    int             *token_count) {


    if (out_len == 0 || out_buf == NULL) {
        fprintf(stderr, "[llamac] Invalid output buffer\n");
        return 1;
    }
    // -----------------------
    // get the vocab
    const llama_vocab * vocab = llama_model_get_vocab(rt->model);

    // vars we need, no malloc
    llamac_chat_state chat = {0}; // <-- critical, we have only one here

    char formatted[MAX_FORMATTED_LEN];
    llama_chat_message messages[1];
    messages[0].role = role;
    messages[0].content = prompt;
    *token_count = 0;
    memset(out_buf, 0, out_len);
    llamac_kv_cache_clear(rt->ctx);
    // get the generate pointer
    generate_fn_t generate = &generate_story;

    // -----------------------

    // get the chat template
    const char *tmpl = llama_model_chat_template(rt->model, /* name */ NULL);
    // apply the template
    const int new_len = llama_chat_apply_template(
            tmpl,
            messages,
            1,
            true,
            formatted,
            sizeof(formatted)
        );
    if (new_len < 0) {
        fprintf(stderr, "template apply failed\n");
        return 1;
    }

    const char *formatted_prompt = formatted + chat.prev_len;
    int prompt_len = new_len - chat.prev_len;
    chat.prev_len = new_len;
#ifdef llamac_DEBUG
    printf("[llamac] role=%s | prompt_len=%d\n", role, prompt_len);
#endif
    // some stats
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    printf("\033[33m");
    const int result = generate(
        vocab,
        rt->sampler,
        formatted_prompt,
        rt->ctx,
        out_buf,
        out_len,
        token_count);

    printf("\n\033[0m");
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (result == 0) {
        double duration =
            (double) (end.tv_sec - start.tv_sec) +
            (double) (end.tv_nsec - start.tv_nsec) / 1e9;

        const double tps = *token_count / duration;
//#ifdef llamac_DEBUG
        printf("\n\033[0m");
        printf("Tokens: %d\n", *token_count);
        printf("Time: %.2f seconds\n", duration);
        printf("TPS: %.2f tokens/sec\n", tps);
//#endif
    }

    return 0;
}

int llamac_chat_history(llamac_runtime *rt, const char *role) {
    char input[1024];
    char output[8192];
    int token_count = 0;

    while (1) {
        printf("\033[32m> \033[0m");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;

        if (strlen(input) == 0 || strcmp(input, ".quit") == 0) break;

        if (llamac_history_shot(
            input,
            role,
            rt,
            output,
            sizeof(output),
            &token_count) == 0) {
#ifdef llamac_DEBUG
            printf("\033[33m%s\033[0m\n", output);
            printf("[tokens: %d]\n", token_count);
#endif
        } else {
            fprintf(stderr, "[error] inference failed\n");
        }
    }

    return 0;
}

int llamac_chat(llamac_runtime *rt, const char *role) {
    char input[1024];
    char output[8192];
    int token_count = 0;

    while (1) {
        printf("\033[32m> \033[0m");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;

        if (strlen(input) == 0 || strcmp(input, ".quit") == 0) break;

        if (llamac_one_shot(
            input,
            role,
            rt,
            output,
            sizeof(output),
            &token_count) == 0) {
#ifdef llamac_DEBUG
            printf("\033[33m%s\033[0m\n", output);
            printf("[tokens: %d]\n", token_count);
#endif
        } else {
            fprintf(stderr, "[error] inference failed\n");
        }
    }

    return 0;
}

void llamac_free(llamac_runtime *rt) {
    if (!rt) return;

    // Free chat history roles
    for (int i = 0; i < rt->history.count; ++i) {
        if (rt->history.messages[i].role != NULL) {
            free((void *)rt->history.messages[i].role);
            rt->history.messages[i].role = NULL;
        }
    }

    rt->history.count = 0;
    rt->history.prev_len = 0;

    // Free model-related components
    if (rt->sampler) llama_sampler_free(rt->sampler);
    if (rt->ctx) llama_free(rt->ctx);
    if (rt->model) llama_model_free(rt->model);
}

void llamac_kv_cache_clear(struct llama_context * ctx) {
    GGML_ASSERT(ctx != NULL);
    llama_memory_clear(llama_get_memory(ctx), true);
}

void llamac_chat_reset(llamac_runtime *rt) {
    if (!rt) return;

    for (int i = 0; i < rt->history.count; ++i) {
        if (rt->history.messages[i].role != NULL) {
            free((void *)rt->history.messages[i].role);
            rt->history.messages[i].role = NULL;
        }
    }

    rt->history.count = 0;
    rt->history.prev_len = 0;
}