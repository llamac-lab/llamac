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
    int lib();


#ifdef __cplusplus
}
#endif

#endif //LLCML_H
