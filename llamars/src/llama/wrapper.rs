#![allow(dead_code)]

use crate::types::{LlamaBatch, LlamaContextParams, LlamaModelParams};
use std::os::raw::{c_char, c_int, c_void};

pub type LlamaMemory = *mut c_void;
pub type LlamaSeqId = c_int; // usually 0
pub type LlamaPos = c_int;

#[link(name = "llama")]
unsafe extern "C" {
    pub(crate) fn llama_model_default_params() -> LlamaModelParams;
    pub(crate) fn llama_context_default_params() -> LlamaContextParams;
    pub(crate) fn llama_model_load_from_file(
        path: *const c_char,
        params: LlamaModelParams,
    ) -> *mut c_void;
    pub(crate) fn llama_init_from_model(
        model: *mut c_void,
        params: LlamaContextParams,
    ) -> *mut c_void;
    pub(crate) fn llama_model_free(model: *mut c_void);
    pub(crate) fn llama_free(ctx: *mut c_void);
    pub(crate) fn llama_sampler_chain_init(params: *mut c_void) -> *mut c_void;
    pub(crate) fn llama_sampler_free(sampler: *mut c_void);
    pub(crate) fn llama_tokenize(
        vocab: *const c_void,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut c_int,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;
    pub(crate) fn llama_decode(ctx: *mut c_void, batch: LlamaBatch) -> c_int;
    pub(crate) fn llama_sampler_sample(sampler: *mut c_void, ctx: *mut c_void, idx: c_int)
    -> c_int;
    pub(crate) fn llama_token_to_piece(
        vocab: *const c_void,
        token: c_int,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;

    pub fn llama_vocab_is_eog(vocab: *const c_void, token: c_int) -> bool;

    //pub (crate) fn llama_n_ctx(ctx: *mut c_void) -> c_int;
    pub(crate) fn ggml_backend_load_all();

    pub(crate) fn llama_backend_init();

    pub(crate) fn ggml_backend_load(path: *const c_char);

    pub fn llama_model_get_vocab(model: *const c_void) -> *const c_void;

    // todo:
    // #[repr(C)]
    // pub struct LlamaVocab;
    // extern "C" {
    //     pub fn llama_model_get_vocab(model: *const c_void) -> *const LlamaVocab;
    // }

    pub fn llama_batch_get_one(token: *const c_int, n_token: c_int) -> LlamaBatch;

    // sampling
    pub fn llama_sampler_chain_default_params() -> *mut c_void;

    pub fn llama_sampler_chain_add(chain: *mut c_void, sampler: *mut c_void);

    pub fn llama_sampler_init_min_p(p: f32, k: i32) -> *mut c_void;
    pub fn llama_sampler_init_temp(temp: f32) -> *mut c_void;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut c_void;

    pub fn llama_vocab_type(vocab: *const c_void) -> c_int;

    pub fn llama_get_memory(ctx: *const c_void) -> LlamaMemory;

    pub fn llama_memory_seq_pos_max(mem: LlamaMemory, seq_id: LlamaSeqId) -> LlamaPos;

    pub fn llama_n_ctx(ctx: *const c_void) -> u32;
    pub fn llama_batch_init(n_tokens: c_int, embd: c_int, n_seq_max: c_int) -> LlamaBatch;

    pub fn llama_model_chat_template(model: *mut c_void, name: *const c_char) -> *const c_char;

    pub fn llama_chat_apply_template(
        tmpl: *const c_char,
        messages: *const LlamaChatMessage,
        n_messages: usize,
        add_bos: bool,
        buf: *mut c_char,
        buf_size: usize,
    ) -> c_int;
}

//LLAMA_API llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id);
//LLAMA_API llama_memory_t   llama_get_memory  (const struct llama_context * ctx);

// typedef int32_t llama_seq_id;
// typedef struct llama_memory_i * llama_memory_t;

#[repr(C)]
pub struct LlamaChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}
