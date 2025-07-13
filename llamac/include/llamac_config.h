//
// Created by ervin on 7/13/25.
//

// #ifndef LLAMAC_CONFIG_H
// #define LLAMAC_CONFIG_H
#pragma once

/* tunables (change-once, propagate everywhere) */
#define LLAMAC_MAX_MESSAGES      64
#define LLAMAC_MAX_MESSAGE_LEN   1024
#define LLAMAC_MAX_RESPONSE      (2048 * 8)
#define LLAMAC_MAX_TOKENS        512

/* ANSI helpers */
#define LLAMAC_CLR(grp)      "\033[" grp "m"
#define CLR_RESET            LLAMAC_CLR("0")
#define CLR_USER             LLAMAC_CLR("32")
#define CLR_ASSISTANT        LLAMAC_CLR("33")

// #endif //LLAMAC_CONFIG_H
