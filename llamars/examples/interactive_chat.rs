use env::args;
use llamars::types::{LlamaOptionsBuilder, LlamaRunner};
use std::env;

fn main() -> anyhow::Result<()> {
    let _model_path = "~/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

    println!("----- interactive chat interface -----");
    let mut args = args();

    // Skip the first arg (the binary name)
    let _bin_name = args.next();

    let model_path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("Usage: interactive_chat <model_path>");
            std::process::exit(1);
        }
    };

    let options = LlamaOptionsBuilder::new(model_path)
        .context_size(2048)
        .n_gpu_layers(32)
        .temperature(0.7)
        .n_threads(6)
        .build();
    let _response: String = LlamaRunner::interactive_chat(options)
        .map_err(|e| anyhow::anyhow!("chat failed: {}", e))?;

    Ok(())
}
