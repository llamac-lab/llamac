//
// Created by ervin on 7/11/25.
//

#ifndef LLAMAC_H
#define LLAMAC_H

#include "llcml.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

    // --------------------- structs and enums


    struct llamac_vocab;
    struct llamac_model;
    struct llamac_context;
    struct llamac_sampler;

    typedef struct llamac_memory_i * llama_memory_t;

    struct llamac_kv_cache; // DEPRECATED (use llama_memory instead)

    typedef int32_t llamac_pos;
    typedef int32_t llamac_token;
    typedef int32_t llamac_seq_id;

    enum llamac_vocab_type {
        LLAMAC_VOCAB_TYPE_NONE = 0, // For models without vocab
        LLAMAC_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMAC_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMAC_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
        LLAMAC_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
        LLAMAC_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
    };

    // pre-tokenization types
    enum llamac_vocab_pre_type {
        LLAMAC_VOCAB_PRE_TYPE_DEFAULT        = 0,
        LLAMAC_VOCAB_PRE_TYPE_LLAMA3         = 1,
        LLAMAC_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
        LLAMAC_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
        LLAMAC_VOCAB_PRE_TYPE_FALCON         = 4,
        LLAMAC_VOCAB_PRE_TYPE_MPT            = 5,
        LLAMAC_VOCAB_PRE_TYPE_STARCODER      = 6,
        LLAMAC_VOCAB_PRE_TYPE_GPT2           = 7,
        LLAMAC_VOCAB_PRE_TYPE_REFACT         = 8,
        LLAMAC_VOCAB_PRE_TYPE_COMMAND_R      = 9,
        LLAMAC_VOCAB_PRE_TYPE_STABLELM2      = 10,
        LLAMAC_VOCAB_PRE_TYPE_QWEN2          = 11,
        LLAMAC_VOCAB_PRE_TYPE_OLMO           = 12,
        LLAMAC_VOCAB_PRE_TYPE_DBRX           = 13,
        LLAMAC_VOCAB_PRE_TYPE_SMAUG          = 14,
        LLAMAC_VOCAB_PRE_TYPE_PORO           = 15,
        LLAMAC_VOCAB_PRE_TYPE_CHATGLM3       = 16,
        LLAMAC_VOCAB_PRE_TYPE_CHATGLM4       = 17,
        LLAMAC_VOCAB_PRE_TYPE_VIKING         = 18,
        LLAMAC_VOCAB_PRE_TYPE_JAIS           = 19,
        LLAMAC_VOCAB_PRE_TYPE_TEKKEN         = 20,
        LLAMAC_VOCAB_PRE_TYPE_SMOLLM         = 21,
        LLAMAC_VOCAB_PRE_TYPE_CODESHELL      = 22,
        LLAMAC_VOCAB_PRE_TYPE_BLOOM          = 23,
        LLAMAC_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24,
        LLAMAC_VOCAB_PRE_TYPE_EXAONE         = 25,
        LLAMAC_VOCAB_PRE_TYPE_CHAMELEON      = 26,
        LLAMAC_VOCAB_PRE_TYPE_MINERVA        = 27,
        LLAMAC_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28,
        LLAMAC_VOCAB_PRE_TYPE_GPT4O          = 29,
        LLAMAC_VOCAB_PRE_TYPE_SUPERBPE       = 30,
        LLAMAC_VOCAB_PRE_TYPE_TRILLION       = 31,
        LLAMAC_VOCAB_PRE_TYPE_BAILINGMOE     = 32,
        LLAMAC_VOCAB_PRE_TYPE_LLAMA4         = 33,
        LLAMAC_VOCAB_PRE_TYPE_PIXTRAL        = 34,
        LLAMAC_VOCAB_PRE_TYPE_SEED_CODER     = 35,
    };

    enum llama_rope_type {
        LLAMAC_ROPE_TYPE_NONE   = -1,
        LLAMAC_ROPE_TYPE_NORM   = 0,
        LLAMAC_ROPE_TYPE_NEOX   = LLCML_ROPE_TYPE_NEOX,
        LLAMAC_ROPE_TYPE_MROPE  = LLCML_ROPE_TYPE_MROPE,
        LLAMAC_ROPE_TYPE_VISION = LLCML_ROPE_TYPE_VISION,
    };

    enum llamac_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLAMAC_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMAC_TOKEN_TYPE_NORMAL       = 1,
        LLAMAC_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMAC_TOKEN_TYPE_CONTROL      = 3,
        LLAMAC_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMAC_TOKEN_TYPE_UNUSED       = 5,
        LLAMAC_TOKEN_TYPE_BYTE         = 6,
    };

    enum llamac_token_attr {
        LLAMAC_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMAC_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMAC_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMAC_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMAC_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLAMAC_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMAC_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMAC_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMAC_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMAC_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMAC_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llamac_ftype {
        LLAMAC_FTYPE_ALL_F32              = 0,
        LLAMAC_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLAMAC_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMAC_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMAC_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMAC_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LLAMAC_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMAC_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMAC_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LLAMAC_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LLAMAC_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors

        LLAMAC_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum llamac_rope_scaling_type {
        LLAMAC_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMAC_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMAC_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMAC_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMAC_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMAC_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMAC_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llamac_pooling_type {
        LLAMAC_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMAC_POOLING_TYPE_NONE = 0,
        LLAMAC_POOLING_TYPE_MEAN = 1,
        LLAMAC_POOLING_TYPE_CLS  = 2,
        LLAMACPOOLING_TYPE_LAST = 3,
        LLAMAC_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum llamac_attention_type {
        LLAMAC_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMAC_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMAC_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llamac_split_mode {
        LLAMAC_SPLIT_MODE_NONE  = 0, // single GPU
        LLAMAC_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
        LLAMAC_SPLIT_MODE_ROW   = 2, // split layers and KV across GPUs, use tensor parallelism if supported
    };

    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llamac_token_data {
     //   llamac_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llamac_token_data;

    typedef struct llamac_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
      //  llamac_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;
    } llamac_token_data_array;

    typedef bool (*llamac_progress_callback)(float progress, void * user_data);

    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llamac_batch {
        int32_t n_tokens;

     //   llama_token  *  token;
        float        *  embd;
      //  llama_pos    *  pos;
        int32_t      *  n_seq_id;
      //  llama_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } llamac_batch;

    enum llamac_model_kv_override_type {
        LLAMAC_KV_OVERRIDE_TYPE_INT,
        LLAMAC_KV_OVERRIDE_TYPE_FLOAT,
        LLAMAC_KV_OVERRIDE_TYPE_BOOL,
        LLAMAC_KV_OVERRIDE_TYPE_STR,
    };

    struct llamac_model_kv_override {
        enum llamac_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llamac_model_tensor_buft_override {
        const char * pattern;
        //ggml_backend_buffer_type_t buft;
    };



    // --------------------- interface

#ifdef __cplusplus
}
#endif

#endif //LLAMAC_H
