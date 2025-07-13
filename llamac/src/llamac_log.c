//
// Created by ervin on 7/13/25.
//

#include <stdio.h>
#include "llamac_internal.h"
#include "llamac_log.h"


static struct {
    int min_level;
} log_cfg = { .min_level = 2 };          /* default = warn */


// struct log_config {
//     int level;
// };
//
// static void llamac_log_proxy(enum ggml_log_level level, const char * text, void * user_data) {
//     struct log_config *cfg = (struct log_config *) user_data;
//     if (level >= cfg->level) {
//         fprintf(stderr, "%s", text);
//     }
// }


static void llamac_log_proxy(enum ggml_log_level lvl,
                             const char *msg,
                             void *user) {
    (void) user;
    if ((int)lvl >= log_cfg.min_level) {
        fputs(msg, stderr);
    }
}

void llamac_log_callback(enum ggml_log_level level, const char * text, void * user_data);


/* Public façade (exported if you installed llamac_log.h) */
void llamac_set_log_level(int min_level) {
    log_cfg.min_level = min_level;
}

/* Called once from llamac_runtime_init() */
void llamac_logging_init(void) {
    llama_log_set(llamac_log_proxy, NULL);
}

// int llamac_runtime_init_with_log_level(int min_level) {
//     static struct log_config cfg;
//     cfg.level = min_level;
//
//     llama_log_set(llamac_log_proxy, &cfg);
//     return 0;
// }


// static void llamac_log_proxy(enum ggml_log_level lvl,
//                              const char *msg,
//                              void *user) {
//     (void) user;
//     if ((int)lvl >= log_cfg.min_level) {
//         fputs(msg, stderr);
//     }
// }

/* Public façade (exported if you installed llamac_log.h) */
// void llamac_set_log_level(int min_level) {
//     log_cfg.min_level = min_level;
// }

/* Called once from llamac_runtime_init() */
// void llamac_logging_init(void) {
//     llama_log_set(llamac_log_proxy, NULL);
// }





// void llamac_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
//     // Adjust threshold here
//     if (level >= GGML_LOG_LEVEL_WARN) {
//         fprintf(stderr, "%s", text); // Only log warnings and errors
//     }
// }
//
// void llamac_logging_init(void) {
//     llama_log_set(llamac_log_proxy, NULL);
// }