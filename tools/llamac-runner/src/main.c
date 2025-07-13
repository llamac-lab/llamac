#include <stdio.h>
#include "ggml.h"
#include "llamac.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llamac.h"

/* example call

./llamac_runner \
  --model  ~/tinyllama-1.1b-chat-v1.0.Q4_0.gguf  \
  --ctx 1024 \
  --tokens 512 \
  --temp 0.7 \
  --top_p 0.9 \
  --prompt "Describe a future where AI governs a floating city."

*/

// -- kiss
int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "Write a short story about a lonely dragon who finds a lost city.";

    int ctx_size = 2048;
    int num_tokens = 2048;
    float temperature = 0.05f;
    float top_p = 0.8f;

    // --- args processing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) num_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc) top_p = atof(argv[++i]);
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s --model <path> [--ctx N] [--tokens N] [--temp F] [--top_p F] [--prompt STR]\n", argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Missing required --model argument.\n");
        return 1;
    }

    // and lets inference
    //llamac_runtime runtime = {0};
    llamac_runtime *rt = llamac_runtime_create();
    if (!rt) return 1;

    //llamac_runtime_init();

    if (llamac_model_load(model_path, 32, ctx_size, num_tokens, temperature, top_p, rt) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    printf("\n[one-shot prompt test]\n");
    char buffer[8192];
    int token_count = 0;

    if (llamac_history_shot(prompt, "user", rt, buffer, sizeof(buffer), &token_count) == 0) {
        printf("[output] %s\n", buffer);
    }

    printf("\n[entering chat mode]\n");
    llamac_chat_history(rt, "user");

    llamac_runtime_destroy(rt);
    return 0;
}
