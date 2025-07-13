#[allow(non_camel_case_types)]
use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int, c_void};
//use std::ptr;
use anyhow::{Error, anyhow};
use std::collections::VecDeque;
use std::fmt::{self, Display, Formatter};

// FFI bindings to llama.cpp C functions
#[repr(C)]
pub struct LlamaModel(*mut c_void);

#[repr(C)]
pub struct LlamaContext(*mut c_void);

#[repr(C)]
pub struct LlamaSampler(*mut c_void);

pub type LlamaProgressCallback =
    Option<extern "C" fn(progress: f32, user_data: *mut c_void) -> bool>;

#[repr(C)]
#[derive(Debug)]
pub struct LlamaModelParams {
    pub devices: *mut c_void,                 // ggml_backend_dev_t*
    pub tensor_buft_overrides: *const c_void, // llama_model_tensor_buft_override*

    pub n_gpu_layers: c_int,
    pub split_mode: c_int,
    pub main_gpu: c_int,

    pub tensor_split: *const c_float,

    pub progress_callback: LlamaProgressCallback,
    pub progress_callback_user_data: *mut c_void,

    pub kv_overrides: *const c_void, // llama_model_kv_override*

    // bools packed at the end
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,

    // Add explicit padding to match 8-byte alignment
    pub _padding: [u8; 4],
}
impl Display for LlamaModelParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "LlamaModelParams {{")?;
        writeln!(f, "  devices: {:p}", self.devices)?;
        writeln!(
            f,
            "  tensor_buft_overrides: {:p}",
            self.tensor_buft_overrides
        )?;
        writeln!(f, "  n_gpu_layers: {}", self.n_gpu_layers)?;
        writeln!(f, "  split_mode: {}", self.split_mode)?;
        writeln!(f, "  main_gpu: {}", self.main_gpu)?;
        writeln!(f, "  tensor_split: {:p}", self.tensor_split)?;
        writeln!(
            f,
            "  progress_callback: {}",
            if self.progress_callback.is_some() {
                "Some(...)"
            } else {
                "None"
            }
        )?;
        writeln!(
            f,
            "  progress_callback_user_data: {:p}",
            self.progress_callback_user_data
        )?;
        writeln!(f, "  kv_overrides: {:p}", self.kv_overrides)?;
        writeln!(f, "  vocab_only: {}", self.vocab_only)?;
        writeln!(f, "  use_mmap: {}", self.use_mmap)?;
        writeln!(f, "  use_mlock: {}", self.use_mlock)?;
        writeln!(f, "  check_tensors: {}", self.check_tensors)?;
        writeln!(f, "  _padding: {:?}", self._padding)?;
        write!(f, "}}")
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    pub rope_scaling_type: LlamaRopeScalingType,
    pub pooling_type: LlamaPoolingType,
    pub llama_attention_type: LlamaAttentionType,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,

    pub type_k: GgmlType,
    pub type_v: GgmlType,

    pub abort_callback: *mut c_void, // ggml_abort_callback (function pointer)
    pub abort_callback_data: *mut c_void,

    pub embeddings: bool,
    pub offload_kqv: bool,
    pub flash_attn: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,

    pub seed: u32, // can go here, as it’s after the booleans
}
impl Display for LlamaContextParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "LlamaContextParams {{")?;
        writeln!(f, "  n_ctx: {}", self.n_ctx)?;
        writeln!(f, "  n_batch: {}", self.n_batch)?;
        writeln!(f, "  n_ubatch: {}", self.n_ubatch)?;
        writeln!(f, "  n_seq_max: {}", self.n_seq_max)?;
        writeln!(f, "  n_threads: {}", self.n_threads)?;
        writeln!(f, "  n_threads_batch: {}", self.n_threads_batch)?;
        writeln!(f, "  rope_scaling_type: {}", self.rope_scaling_type)?;
        writeln!(f, "  pooling_type: {}", self.pooling_type)?;
        writeln!(f, "  llama_attention_type: {}", self.llama_attention_type)?;
        writeln!(f, "  rope_freq_base: {}", self.rope_freq_base)?;
        writeln!(f, "  rope_freq_scale: {}", self.rope_freq_scale)?;
        writeln!(f, "  yarn_ext_factor: {}", self.yarn_ext_factor)?;
        writeln!(f, "  yarn_attn_factor: {}", self.yarn_attn_factor)?;
        writeln!(f, "  yarn_beta_fast: {}", self.yarn_beta_fast)?;
        writeln!(f, "  yarn_beta_slow: {}", self.yarn_beta_slow)?;
        writeln!(f, "  yarn_orig_ctx: {}", self.yarn_orig_ctx)?;
        writeln!(f, "  defrag_thold: {}", self.defrag_thold)?;
        writeln!(f, "  cb_eval: {:p}", self.cb_eval)?;
        writeln!(f, "  cb_eval_user_data: {:p}", self.cb_eval_user_data)?;
        writeln!(f, "  type_k: {}", self.type_k)?;
        writeln!(f, "  type_v: {}", self.type_v)?;
        writeln!(f, "  abort_callback: {:p}", self.abort_callback)?;
        writeln!(f, "  abort_callback_data: {:p}", self.abort_callback_data)?;
        writeln!(f, "  embeddings: {}", self.embeddings)?;
        writeln!(f, "  offload_kqv: {}", self.offload_kqv)?;
        writeln!(f, "  flash_attn: {}", self.flash_attn)?;
        writeln!(f, "  no_perf: {}", self.no_perf)?;
        writeln!(f, "  op_offload: {}", self.op_offload)?;
        writeln!(f, "  swa_full: {}", self.swa_full)?;
        writeln!(f, "  seed: {}", self.seed)?;
        write!(f, "}}")
    }
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LlamaRopeScalingType {
    Unspecified = -1,
    None = 0,
    Linear = 1,
    Yarn = 2,
    LongRope = 3,
    //MaxValue = 3
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LlamaAttentionType {
    Unspecified = -1,
    Causal = 0,
    NonCausal = 1,
}

// ---
impl TryFrom<c_int> for LlamaRopeScalingType {
    type Error = anyhow::Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::Unspecified),
            0 => Ok(Self::None),
            1 => Ok(Self::Linear),
            2 => Ok(Self::Yarn),
            3 => Ok(Self::LongRope),
            _ => Err(anyhow::anyhow!("Invalid rope scaling type: {}", value)),
        }
    }
}
impl std::fmt::Display for LlamaRopeScalingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---
impl TryFrom<c_int> for LlamaPoolingType {
    type Error = Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::Unspecified),
            0 => Ok(Self::None),
            1 => Ok(Self::Mean),
            2 => Ok(Self::Cls),
            3 => Ok(Self::Last),
            4 => Ok(Self::Rank),
            _ => Err(anyhow!("Invalid pooling type: {}", value)),
        }
    }
}

impl fmt::Display for LlamaPoolingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Unspecified => "Unspecified",
            Self::None => "None",
            Self::Mean => "Mean",
            Self::Cls => "Cls",
            Self::Last => "Last",
            Self::Rank => "Rank",
        };
        write!(f, "{}", s)
    }
}

// ---
impl TryFrom<c_int> for LlamaAttentionType {
    type Error = Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::Unspecified),
            0 => Ok(Self::Causal),
            1 => Ok(Self::NonCausal),
            _ => Err(anyhow!("Invalid attention type: {}", value)),
        }
    }
}

impl fmt::Display for LlamaAttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Unspecified => "Unspecified",
            Self::Causal => "Causal",
            Self::NonCausal => "NonCausal",
        };
        write!(f, "{}", s)
    }
}

// ----------------
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // removed
    // Q4_3 = 5, // removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // Q4_0_4_4 = 31, // removed
    // Q4_0_4_8 = 32,
    // Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    // IQ4_NL_4_4 = 36,
    // IQ4_NL_4_8 = 37,
    // IQ4_NL_8_8 = 38,
    Count = 39,
}
impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::Count => "COUNT",
        };
        write!(f, "{}", name)
    }
}
impl TryFrom<c_int> for GgmlType {
    type Error = Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            39 => Ok(Self::Count),
            _ => Err(anyhow!("Invalid GGML type ID: {}", value)),
        }
    }
}

#[repr(C)]
pub struct LlamaChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaBatch {
    pub n_tokens: c_int,
    pub token: *mut c_int,
    pub embd: *mut c_float,
    pub pos: *mut c_int,
    pub n_seq_id: *mut c_int,
    pub seq_id: *mut *mut c_int,
    pub logits: *mut i8,
}

impl Display for LlamaBatch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "LlamaBatch {{")?;
        writeln!(f, "  n_tokens: {}", self.n_tokens)?;

        // Display token IDs
        if !self.token.is_null() {
            let tokens = unsafe { std::slice::from_raw_parts(self.token, self.n_tokens as usize) };
            writeln!(f, "  token: {:?}", tokens)?;
        } else {
            writeln!(f, "  token: null")?;
        }

        // Display positions
        if !self.pos.is_null() {
            let pos = unsafe { std::slice::from_raw_parts(self.pos, self.n_tokens as usize) };
            writeln!(f, "  pos: {:?}", pos)?;
        } else {
            writeln!(f, "  pos: null")?;
        }

        // Display logits pointer only
        writeln!(f, "  logits: {:p}", self.logits)?;

        // Optional: Show sequence info
        if !self.n_seq_id.is_null() && !self.seq_id.is_null() {
            let n_seq =
                unsafe { std::slice::from_raw_parts(self.n_seq_id, self.n_tokens as usize) };
            writeln!(f, "  n_seq_id: {:?}", n_seq)?;

            writeln!(f, "  seq_id:")?;
            for i in 0..(self.n_tokens as usize) {
                let count = n_seq[i];
                let ptr = unsafe { *self.seq_id.add(i) };
                if !ptr.is_null() {
                    let ids = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
                    writeln!(f, "    [{}]: {:?}", i, ids)?;
                } else {
                    writeln!(f, "    [{}]: null", i)?;
                }
            }
        } else {
            writeln!(f, "  n_seq_id or seq_id is null")?;
        }

        // Optional: Show embeddings if populated
        if !self.embd.is_null() {
            writeln!(f, "  embd: {:p}", self.embd)?;
        } else {
            writeln!(f, "  embd: null")?;
        }

        writeln!(f, "}}")
    }
}

/// Configuration options for the LLaMA model
#[derive(Debug, Clone)]
pub struct LlamaOptions {
    pub model_path: String,
    pub chat_template_file: Option<String>,
    pub context_size: Option<i32>,
    pub n_gpu_layers: Option<i32>,
    pub n_threads: Option<i32>,
    pub temperature: Option<f32>,
    pub use_jinja: bool,
    pub verbose: bool,
}

impl Default for LlamaOptions {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            chat_template_file: None,
            context_size: None,
            n_gpu_layers: None,
            n_threads: None,
            temperature: None,
            use_jinja: false,
            verbose: false,
        }
    }
}

/// Represents a chat message with role and content
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Main wrapper around llama.cpp functionality
pub struct LlamaRunner {
    pub model: *mut c_void,
    pub context: *mut c_void,
    pub sampler: *mut c_void,
    pub options: LlamaOptions,
    pub messages: VecDeque<ChatMessage>,
    // Keep CStrings alive for the duration of the struct
    pub _message_strings: Vec<CString>,
}

/// Builder pattern for LlamaOptions
pub struct LlamaOptionsBuilder {
    pub(crate) options: LlamaOptions,
}

#[repr(i32)]
#[derive(Debug)]
pub enum LlamaVocabType {
    Spm = 0,
    Bpe = 1,
    Wpm = 2,
    Ugm = 3,
    Rwkv = 4,
    None = 99,
}

impl LlamaVocabType {
    pub fn from_i32(value: i32) -> Option<Self> {
        use LlamaVocabType::*;
        Some(match value {
            0 => Spm,
            1 => Bpe,
            2 => Wpm,
            3 => Ugm,
            4 => Rwkv,
            99 => None,
            _ => return Some(None),
        })
    }
}
