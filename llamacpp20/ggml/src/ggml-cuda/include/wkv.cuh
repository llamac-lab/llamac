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
#include "common.cuh"
#include "../../../include/ggml.h"


#ifndef WKV_CUH
#define WKV_CUH


#ifdef __cplusplus
extern "C" {
#endif

#include "common.cuh"

#define CUDA_WKV_BLOCK_SIZE 64

    void ggml_cuda_op_rwkv_wkv6(struct ggml_backend_cuda_context *ctx, struct ggml_tensor *dst);
    void ggml_cuda_op_rwkv_wkv7(struct ggml_backend_cuda_context *ctx, struct ggml_tensor *dst);

#ifdef __cplusplus
}
#endif


#endif // WKV_CUH
