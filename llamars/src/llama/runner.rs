#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(dead_code)]

// todo: needs some love later
// todo: factor out chat_message dependency which is not directly llamacpp stuff

use crate::chat_message;
use crate::types::LlamaContextParams;
use crate::types::{ChatMessage, LlamaOptions, LlamaOptionsBuilder, LlamaRunner};
use crate::types::{LlamaModelParams, LlamaVocabType};
use crate::wrapper::LlamaChatMessage;
use std::collections::VecDeque;
use std::ffi::CString;
use std::io;
use std::io::Write;
use std::mem::{align_of, size_of};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::time::Instant;

use crate::wrapper::{
    ggml_backend_load_all, llama_backend_init, llama_batch_get_one, llama_batch_init,
    llama_chat_apply_template, llama_context_default_params, llama_decode, llama_free,
    llama_get_memory, llama_init_from_model, llama_memory_seq_pos_max, llama_model_chat_template,
    llama_model_default_params, llama_model_free, llama_model_get_vocab,
    llama_model_load_from_file, llama_n_ctx, llama_sampler_chain_add,
    llama_sampler_chain_default_params, llama_sampler_chain_init, llama_sampler_free,
    llama_sampler_init_dist, llama_sampler_init_min_p, llama_sampler_init_temp,
    llama_sampler_sample, llama_token_to_piece, llama_tokenize, llama_vocab_is_eog,
    llama_vocab_type,
};

const LLAMA_DEFAULT_SEED: u32 = 0xFFFFFFFF;

unsafe impl Send for LlamaRunner {}
unsafe impl Sync for LlamaRunner {}

impl LlamaRunner {
    /// Create a new LlamaRunner with the given options
    pub fn new(options: LlamaOptions) -> Result<Self, Box<dyn std::error::Error>> {
        if options.model_path.is_empty() {
            return Err("Model path is required".into());
        }

        unsafe {
            ggml_backend_load_all();

            let params = LlamaParamsRaw::from_options(&options);

            let model_path_c = CString::new(options.model_path.clone())
                .map_err(|_| "Model path contains null byte(s)")?;
            println!(
                "[DEBUG] CString model path ptr: {:?}",
                model_path_c.as_ptr()
            );
            println!(
                "[DEBUG] CString model path bytes: {:?}",
                model_path_c.as_bytes_with_nul()
            );
            println!(
                "sizeof::<LlamaModelParams>() = {}",
                size_of::<LlamaModelParams>()
            );
            println!(
                "align_of::<LlamaModelParams>() = {}",
                align_of::<LlamaModelParams>()
            );

            // Initialize model parameters
            let mut model_params = llama_model_default_params();
            println!("[DEBUG] model_params.devices: {:?}", model_params.devices);
            println!(
                "[DEBUG] model_params.tensor_split: {:?}",
                model_params.tensor_split
            );
            // fix the crash: make sure .devices is null
            model_params.devices = std::ptr::null_mut();
            // optional: set other fields
            model_params.n_gpu_layers = options.n_gpu_layers.unwrap_or(0);

            if let Some(ngl) = options.n_gpu_layers {
                model_params.n_gpu_layers = ngl;
            }

            let model = llama_model_load_from_file(model_path_c.as_ptr(), params.model_params);
            if model.is_null() {
                return Err("Failed to load model".into());
            }

            // Load model
            // let model_path_c = CString::new(options.model_path.clone())
            //     .map_err(|_| "Invalid model path (contains null byte)")?;
            //

            // let model = unsafe {
            //     llama_model_load_from_file(model_path_c.as_ptr(), model_params)
            // };
            // if model.is_null() {
            //     return Err(format!("Failed to load model: {}", options.model_path).into());
            // }

            // Initialize context parameters
            let mut ctx_params = llama_context_default_params();
            if let Some(ctx_size) = options.context_size {
                ctx_params.n_ctx = ctx_size as u32;
                ctx_params.n_batch = ctx_size as u32;
            }
            if let Some(threads) = options.n_threads {
                ctx_params.n_threads = threads;
                ctx_params.n_threads_batch = threads;
            }

            //let context = llama_init_from_model(model, params.context_params);
            let context = llama_init_from_model(model, ctx_params);
            if context.is_null() {
                llama_model_free(model);
                return Err("Failed to initialize context".into());
            }

            // Create sampler
            // 2. Create sampler chain
            let sampler_params = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(sampler_params);
            if sampler.is_null() {
                llama_free(context);
                llama_model_free(model);
                return Err("Failed to create sampler".into());
            }

            // 3. Add samplers to the chain

            let min_p = llama_sampler_init_min_p(0.05, 1);
            let temp = llama_sampler_init_temp(0.8);
            let dist = llama_sampler_init_dist(LLAMA_DEFAULT_SEED);

            llama_sampler_chain_add(sampler, min_p);
            llama_sampler_chain_add(sampler, temp);
            llama_sampler_chain_add(sampler, dist);

            Ok(Self {
                model,
                context,
                sampler,
                options,
                messages: VecDeque::new(),
                _message_strings: Vec::new(),
            })
        }
    }

    // /// Add a message to the conversation history
    // pub fn add_message(&mut self, message: ChatMessage) {
    //     self.messages.push_back(message);
    // }

    /// Clear the conversation history
    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self._message_strings.clear();
    }

    /// Tokenize text into tokens
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        let text_c = CString::new(text)?;
        let text_len = text.len() as c_int;

        unsafe {
            // First call to get the required number of tokens
            let mut tokens = vec![0i32; text_len as usize + 256]; // Generous buffer
            let n_tokens = llama_tokenize(
                self.model,
                text_c.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                tokens.len() as c_int,
                true, // add_special
                true, // parse_special
            );

            if n_tokens < 0 {
                // Need more space
                tokens.resize((-n_tokens) as usize, 0);
                let n_tokens = llama_tokenize(
                    self.model,
                    text_c.as_ptr(),
                    text_len,
                    tokens.as_mut_ptr(),
                    tokens.len() as c_int,
                    true,
                    true,
                );
                if n_tokens < 0 {
                    return Err("Tokenization failed".into());
                }
                tokens.truncate(n_tokens as usize);
            } else {
                tokens.truncate(n_tokens as usize);
            }

            Ok(tokens)
        }
    }

    /// Convert a token back to text
    pub fn token_to_text(&self, token: i32) -> Result<String, Box<dyn std::error::Error>> {
        unsafe {
            let mut buffer = [0u8; 256];
            let n = llama_token_to_piece(
                self.model,
                token,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as c_int,
                0,
                true,
            );

            if n < 0 {
                return Err("Failed to convert token to text".into());
            }

            let text = std::str::from_utf8(&buffer[..n as usize])?;

            Ok(text.to_string())
        }
    }

    /// Start an interactive chat session
    pub fn chat_interactive(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        use std::io::{self, Write};

        println!("Starting interactive chat. Type '/bye' to exit.");

        loop {
            print!("> ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input == "/bye" {
                break;
            }

            if input.is_empty() {
                continue;
            }

            // Add user message
            self.add_message(ChatMessage::user(input));

            // Generate response
            match self.generate(input) {
                Ok(response) => {
                    println!("Assistant: {}", response);
                    self.add_message(ChatMessage::assistant(response));
                }
                Err(e) => {
                    eprintln!("Error generating response: {}", e);
                }
            }
        }

        Ok(String::new())
    }

    /// Get current context size
    pub fn context_size(&self) -> u32 {
        unsafe { llama_n_ctx(self.context) }
    }

    /// Get the conversation history
    // pub fn messages(&self) -> &VecDeque<ChatMessage> {
    //     &self.messages
    // }

    /// Get the model options
    pub fn options(&self) -> &LlamaOptions {
        &self.options
    }
}

/// Convenience functions for common use cases
impl LlamaRunner {
    pub fn ggml_backend_load_all() {
        unsafe { ggml_backend_load_all() }
    }

    pub fn llama_backend_init() {
        unsafe { llama_backend_init() }
    }
    /// Quick setup for a basic chat model
    pub fn quick_chat(model_path: impl Into<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let options = LlamaOptionsBuilder::new(model_path)
            .context_size(2048)
            .n_threads(4)
            .temperature(0.7)
            .build();

        Self::new(options)
    }

    /// Setup for high-performance inference
    pub fn performance_setup(
        model_path: impl Into<String>,
        gpu_layers: i32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let options = LlamaOptionsBuilder::new(model_path)
            .context_size(4096)
            .n_gpu_layers(gpu_layers)
            .n_threads(8)
            .temperature(0.7)
            .build();

        Self::new(options)
    }
}

fn nyra_load_all_backends() {
    unsafe { ggml_backend_load_all() };
}

impl LlamaRunner {
    // -----------------------------

    pub fn interactive_chat(options: LlamaOptions) -> Result<String, Box<dyn std::error::Error>> {
        // 0
        let llama_params = LlamaParamsRaw::from_options(&options);

        println!("interactive_chat");
        let ngl: i32 = 99;
        let n_ctx: u32 = 2048;

        // 1.
        unsafe { ggml_backend_load_all() };

        // 2.
        let mut model_params: LlamaModelParams = unsafe { llama_model_default_params() };
        model_params.n_gpu_layers = ngl;
        //println!("[debug] model_params:\n{}", model_params);
        println!("[debug] model_params (pretty):\n{}", model_params);

        // 3.
        let model_path_c = CString::new(options.model_path.clone())
            .map_err(|_| "Model path contains null byte(s)")?;

        let model =
            unsafe { llama_model_load_from_file(model_path_c.as_ptr(), llama_params.model_params) };
        if model.is_null() {
            return Err("Failed to load model".into());
        }
        println!("[debug] model (pretty):\n{:p}", model);

        // 4.
        let vocab = unsafe { llama_model_get_vocab(model) };
        if vocab.is_null() {
            return Err("vocab is null".into());
        }
        println!("[debug] Vocab pointer: (pretty):\n{:p}", vocab);

        // 5.
        let mut ctx_params: LlamaContextParams = unsafe { llama_context_default_params() };
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx;
        println!("[debug] context default params: (pretty):\n{:}", ctx_params);

        // 6.
        let ctx = unsafe { llama_init_from_model(model, ctx_params) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            return Err("error: failed to create the llama_contex".into());
        }
        println!("[debug] init model ctx pointer: (pretty):\n{:p}", ctx);

        // 7.

        let smpl = unsafe { llama_sampler_chain_init(llama_sampler_chain_default_params()) };
        if smpl.is_null() {
            unsafe {
                llama_free(ctx);
                llama_model_free(model);
            }
            return Err("Failed to create sampler".into());
        }

        unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05, 1)) };
        unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8)) };
        unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED)) };

        println!("[debug] sampler: (pretty):\n{:p}", smpl);

        // ----------- generate start
        let _generate = |prompt: &CString| -> Result<(String, usize), Box<dyn std::error::Error>> {
            let mut token_count = 0;
            // ... build up the response
            println!("[debug] Prompt: {:?}", prompt.as_bytes_with_nul());
            let prompt_ptr = prompt.as_ptr();
            let prompt_len = prompt.as_bytes().len() as c_int;
            println!("[debug] prompt_ptr: {:?}", prompt_ptr);
            println!("[debug] prompt_len: {:?}", prompt_len);

            let memory = unsafe { llama_get_memory(ctx) };
            let pos = unsafe { llama_memory_seq_pos_max(memory, 0) }; // default seq_id = 0

            let is_first = pos == -1;
            // 3. Preflight token count
            let n_prompt_tokens = unsafe {
                -llama_tokenize(
                    vocab,
                    prompt_ptr,
                    prompt_len,
                    std::ptr::null_mut(),
                    0,
                    is_first,
                    true, // parse_special = true for chat mode
                )
            };
            println!("[debug] n_prompt_tokens = {n_prompt_tokens}");
            if n_prompt_tokens <= 0 {
                return Err(format!(
                    "[error] tokenizer preflight failed (count = {n_prompt_tokens})"
                )
                .into());
            }

            // 4. get the tokens
            let mut prompt_tokens: Vec<c_int> = vec![0; n_prompt_tokens as usize];

            let result = unsafe {
                llama_tokenize(
                    vocab,
                    prompt_ptr,
                    prompt_len,
                    prompt_tokens.as_mut_ptr(),
                    prompt_tokens.len() as c_int,
                    is_first,
                    true,
                )
            };
            println!("[debug] prompt_tokens: {:?}", prompt_tokens);

            // 5. prefill/full decode
            let mut batch = unsafe {
                llama_batch_get_one(prompt_tokens.as_mut_ptr(), prompt_tokens.len() as c_int)
            };
            println!("[debug] batch: {}", batch);

            let mut response = String::new();
            let mut token_id: c_int = 0;

            loop {
                // 1. Check if there's room left in the context
                let n_ctx = unsafe { llama_n_ctx(ctx) };
                let mem = unsafe { llama_get_memory(ctx) };
                let n_ctx_used = unsafe { llama_memory_seq_pos_max(mem, 0) }; // seq_id = 0

                if (n_ctx_used + batch.n_tokens) > n_ctx as i32 {
                    println!();
                    eprintln!("context size exceeded");
                    std::process::exit(1);
                }

                // 2. decode
                let decode_result = unsafe { llama_decode(ctx, batch) };
                if decode_result != 0 {
                    panic!("failed to decode");
                }
                //println!("[debug] decode result: {}", decode_result);

                // 3. sample
                let new_token_id = unsafe { llama_sampler_sample(smpl, ctx, -1) };
                token_count += 1;

                //println!("[debug] new_token_id: {}", new_token_id);

                // 4. is it an end of generation?
                if unsafe { llama_vocab_is_eog(vocab, new_token_id) } {
                    break;
                }
                // failsafe
                //if token_count >= n_ctx { break; }

                // 5. convert the token to a string, print it and add it to the response
                let mut buf = vec![0u8; 512];
                let n = unsafe {
                    llama_token_to_piece(
                        vocab,
                        new_token_id,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len() as c_int,
                        0,
                        true,
                    )
                };
                if n > 0 {
                    let s = String::from_utf8_lossy(&buf[..n as usize]);
                    print!("{}", s);
                    response.push_str(&s);
                }

                // --- memory debugging
                //   let mem = unsafe { llama_get_memory(ctx) };
                //  let n_ctx = unsafe { llama_n_ctx(ctx) };
                //  let n_ctx_used = unsafe { llama_memory_seq_pos_max(mem, 0) };

                // println!(
                // "[debug] context usage: {} / {}, batch tokens: {}",
                // n_ctx_used, n_ctx, batch.n_tokens
                // );
                // --- memory debugging

                batch = unsafe { llama_batch_get_one(&new_token_id as *const c_int, 1) };
            }

            //Ok(String::new())
            Ok((response, token_count))
        };
        // ----------- generate end

        // 1. init
        let mut messages: Vec<LlamaChatMessage> = Vec::new();
        let mut formatted = unsafe { vec![0u8; llama_n_ctx(ctx) as usize] };
        let mut prev_len = 0;

        // 2. user loop
        loop {
            //3. user input
            print!("\x1b[32m> \x1b[0m");
            io::stdout().flush().ok();
            let mut user = String::new();
            io::stdin().read_line(&mut user)?;

            if user.trim().is_empty() {
                break;
            }

            // 4. apply chat template
            let tmpl_ptr = unsafe { llama_model_chat_template(model, ptr::null()) };
            let role_c = CString::new("user")?;
            let content_c = CString::new(user.trim())?;
            messages.push(LlamaChatMessage {
                role: role_c.as_ptr(),
                content: content_c.as_ptr(),
            });
            let new_len = unsafe {
                llama_chat_apply_template(
                    tmpl_ptr,
                    messages.as_ptr(),
                    messages.len(),
                    true,
                    formatted.as_mut_ptr() as *mut c_char,
                    formatted.len(),
                )
            };
            if new_len < 0 {
                panic!("[error] chat template failed");
            }
            let slice = &formatted[prev_len..new_len as usize];
            let prompt_cstr = CString::new(slice)?;
            let prompt_ptr = prompt_cstr.as_ptr();
            println!("[debug] {:?}", prompt_cstr);

            // 5. generate
            // ---------------------- GENERATE
            let start = Instant::now();
            println!("\x1b[33m");
            let response = _generate(&prompt_cstr)?; // call your generate logic
            println!("\n\x1b[0m");
            // ----------------------- GENERATE
            let duration = start.elapsed();

            // telemetry approx
            let seconds = duration.as_secs_f64();
            let (response, token_count) = _generate(&prompt_cstr)?;
            // let token_count = response.split_whitespace().count();
            // let estimated_tokens = (response.split_whitespace().count() as f64 * 1.3) as usize;
            // let elapsed_secs = duration.as_secs_f64();
            let tps = token_count as f64 / duration.as_secs_f64();
            println!(
                "[perf] Generated {} tokens in {:.2} seconds ({:.2} tokens/sec)",
                token_count, seconds, tps
            );

            // 6. process
            println!("[debug] {:?}", response);
            let role = CString::new("assistant")?;
            let content = CString::new(response.clone())?;

            let role_ptr = role.as_ptr();
            let content_ptr = content.as_ptr();

            messages.push(LlamaChatMessage {
                role: role_ptr,
                content: content_ptr,
            });

            let len = unsafe {
                llama_chat_apply_template(
                    tmpl_ptr,
                    messages.as_ptr(),
                    messages.len(),
                    false.into(),
                    std::ptr::null_mut(),
                    0,
                )
            };
            //Ok(String::new())
        }

        // todo: mockup response replacement
        let response = String::new();
        Ok(response)
    }

    // -----------------------------

    pub fn one_shot(model_path: &str, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let options = LlamaOptionsBuilder::new(model_path)
            .context_size(2048)
            .n_gpu_layers(32)
            .temperature(0.7)
            .n_threads(6)
            .build();

        let mut runner = Self::new(options)?;
        Ok(runner.generate(prompt)?)
    }
}

impl LlamaRunner {
    pub fn generate(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        use std::ffi::CString;
        use std::io::Write;

        let vocab = unsafe { llama_model_get_vocab(self.model) };
        if vocab.is_null() {
            return Err("vocab is null".into());
        }
        println!("[DEBUG] Vocab pointer: {vocab:?}");
        assert!(!vocab.is_null(), "🚨 llama_model_get_vocab returned NULL");

        let vocab_type_raw = unsafe { llama_vocab_type(vocab) };
        let vocab_type = LlamaVocabType::from_i32(vocab_type_raw);
        println!("[DEBUG] vocab type: {:?}", vocab_type);

        // -------------- GENERATE
        let generate_internal = |prompt: &str| -> Result<String, Box<dyn std::error::Error>> {
            // Step 1: determine token count
            let prompt_c = CString::new(prompt)?;
            println!("[DEBUG] Prompt: {:?}", prompt_c.as_bytes_with_nul());

            let prompt_ptr = prompt_c.as_ptr();
            let prompt_len = prompt_c.as_bytes().len() as c_int;

            let memory = unsafe { llama_get_memory(self.context) };
            let pos = unsafe { llama_memory_seq_pos_max(memory, 0) }; // default seq_id = 0

            let is_first = pos == -1;
            // 3. Preflight token count
            let n_prompt_tokens = unsafe {
                -llama_tokenize(
                    vocab,
                    prompt_ptr,
                    prompt_len,
                    std::ptr::null_mut(),
                    0,
                    is_first,
                    true, // parse_special = true for chat mode
                )
            };
            println!("[DEBUG] n_prompt_tokens = {n_prompt_tokens}");
            if n_prompt_tokens <= 0 {
                return Err(
                    format!("Tokenizer preflight failed (count = {n_prompt_tokens})").into(),
                );
            }

            // Step 2: tokenize
            let mut tokens: Vec<c_int> = vec![0; n_prompt_tokens as usize];

            let result = unsafe {
                llama_tokenize(
                    vocab,
                    prompt_c.as_ptr(),
                    prompt_c.as_bytes().len() as c_int,
                    tokens.as_mut_ptr(),
                    tokens.len() as c_int,
                    is_first,
                    true,
                )
            };

            if result < 0 {
                return Err("Tokenization failed (step 2)".into());
            }
            //
            // Step 3: prepare batch and decode prompt
            // let mut batch = unsafe { llama_batch_init(tokens.len() as c_int, 0, 1) };
            //
            // for i in 0..tokens.len() {
            //     unsafe {
            //         *batch.token.add(i) = tokens[i];
            //         *batch.pos.add(i) = i as c_int;
            //         *batch.n_seq_id.add(i) = 1;
            //         *batch.seq_id.add(i) = &mut 0;  // seq_id = 0
            //         *batch.logits.add(i) = if i == tokens.len() - 1 { 1 } else { 0 };
            //     }
            // }
            let mut batch =
                unsafe { llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as c_int) };

            let mut response = String::new();
            let mut token_id: c_int = 0;

            // Step 4: Autoregressive loop
            loop {
                // 1. Check if there's room left in the context
                let n_ctx = unsafe { llama_n_ctx(self.context) };
                let mem = unsafe { llama_get_memory(self.context) };
                let n_ctx_used = unsafe { llama_memory_seq_pos_max(mem, 0) }; // seq_id = 0

                if (n_ctx_used + batch.n_tokens) > n_ctx as i32 {
                    println!();
                    eprintln!("context size exceeded");
                    std::process::exit(1);
                }

                println!("[DEBUG] batch.n_tokens = {}", batch.n_tokens);
                println!("[DEBUG] token = {:?}", batch.token);
                println!("[DEBUG] logits ptr = {:?}", batch.logits);
                println!("[DEBUG] embd ptr = {:?}", batch.embd);
                println!("[DEBUG] pos ptr = {:?}", batch.pos);
                let vocab = unsafe { llama_model_get_vocab(self.model) };
                let vocab_type = unsafe { llama_vocab_type(vocab) };
                println!("[DEBUG] vocab_type = {}", vocab_type);
                // 2. Run model forward pass
                let decode_result = unsafe { llama_decode(self.context, batch) };
                if decode_result != 0 {
                    panic!("failed to decode");
                }
                break;
            }
            Ok(String::new())
        };

        let mut messages: Vec<LlamaChatMessage> = Vec::new();
        let mut formatted = unsafe { vec![0u8; llama_n_ctx(self.context) as usize] };
        let mut prev_len = 0;

        loop {
            print!("\x1b[32m> \x1b[0m");
            io::stdout().flush().ok();
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if input.trim().is_empty() {
                break;
            }

            let tmpl_ptr = unsafe { llama_model_chat_template(self.model, ptr::null()) };
            let role_c = CString::new("user")?;
            let content_c = CString::new(input.trim())?;
            messages.push(LlamaChatMessage {
                role: role_c.as_ptr(),
                content: content_c.as_ptr(),
            });
            let new_len = unsafe {
                llama_chat_apply_template(
                    tmpl_ptr,
                    messages.as_ptr(),
                    messages.len(),
                    true,
                    formatted.as_mut_ptr() as *mut c_char,
                    formatted.len(),
                )
            };
            if new_len < 0 {
                panic!("chat template failed");
            }
            let prompt =
                String::from_utf8_lossy(&formatted[prev_len..new_len as usize]).to_string();
            println!("{:?}", prompt);
            println!("\x1b[33m");
            let response = generate_internal(&prompt); // call your generate logic
            println!("\n\x1b[0m");
            break;
        }
        let response = String::new();
        Ok(response)
    }
}

impl LlamaRunner {
    pub fn add_message(&mut self, msg: ChatMessage) {
        self._message_strings
            .push(CString::new(msg.content.clone()).unwrap());
        self.messages.push_back(msg);
    }

    pub fn messages(&self) -> &VecDeque<ChatMessage> {
        &self.messages
    }
}

////////////////////////// FFI FIX FOR NOW
pub struct LlamaParamsRaw {
    pub model_params: LlamaModelParams,
    pub context_params: LlamaContextParams,
}

impl LlamaParamsRaw {
    pub fn from_options(options: &LlamaOptions) -> Self {
        unsafe {
            let mut model_params = llama_model_default_params();
            let mut context_params = llama_context_default_params();

            // Prevent SEGFAULTs by nulling unsafe fields if unused
            //     model_params.devices = ptr::null();
            // leave tensor_split as-is unless you know what you’re doing

            // Now override safe fields
            if let Some(n) = options.n_gpu_layers {
                model_params.n_gpu_layers = n;
            }

            if let Some(n) = options.context_size {
                context_params.n_batch = n as u32;
                context_params.n_ctx = n as u32;
            }

            if let Some(n) = options.n_threads {
                context_params.n_threads = n;
                context_params.n_threads_batch = n;
            }

            if let Some(_) = options.temperature {
                // not needed here unless passed to sampler
            }

            Self {
                model_params,
                context_params,
            }
        }
    }
}

impl Drop for LlamaRunner {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.context);
            llama_model_free(self.model);
            llama_sampler_free(self.sampler);
        }
    }
}
