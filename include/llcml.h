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

/*
 * llcml - llamac - machine learning library. pure C implementation targeting various hardware backends
 *
 * rules, zero copy and zero move as much as possible. efficient memory utilization as much as possible,
 * think embedded hardware
 *
 * some areas covered (critical for LLMs especially)
 * - linear regression
 * - support vector machines
 * - neural networks
 * - tensor library
 *
 * Tensor example:
 * define the function: f(x) = a*x^2 + b
 *
 * {
 *     struct ggml_init_params params = {
 *        .mem_size   = 16*1024*1024,
 *        .mem_buffer = NULL,
 *     };
 *
 *     // memory allocation happens here
 *     struct ggml_context * ctx = ggml_init(params);
 *
 *     struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
 *
 *     ggml_set_param(ctx, x); // x is an input variable
 *
 *     struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
 *     struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
 *     struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
 *     struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
 *
 *     ...
 *     }
 */

#ifndef LLCML_H
#define LLCML_H

#include <stdbool.h>

#ifdef  __cplusplus
extern "C" {
#endif



 // --------------- TENSOR LIB
 enum llmc_tensor_flag {
   LLCML_TENSOR_FLAG_INPUT   =  1, // ...is an input for the compute graph
   LLMCL_TENSOR_FLAG_OUTPUT  =  2, // ...is an output for the compute graph
   GGML_TENSOR_FLAG_PARAM    =  4, // ...contains trainable parameters
   GGML_TENSOR_FLAG_LOSS     =  8, // ...defines loss for numerical optimization (multiple loss tensors add up)
};

 struct llcml_init_params {
   // memory pool
   size_t mem_size;   // bytes
   void * mem_buffer; // if NULL, memory will be allocated internally
   bool   no_alloc;   // don't allocate memory for the tensor data
 };

#define LLCML_ROPE_TYPE_NEOX   2
#define LLCML_ROPE_TYPE_MROPE  8
#define LLCML_ROPE_TYPE_VISION 24

#ifdef  __cplusplus
}
#endif


#endif //LLCML_H
