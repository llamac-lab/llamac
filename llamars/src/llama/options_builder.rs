use crate::types::{LlamaOptions, LlamaOptionsBuilder};

impl LlamaOptionsBuilder {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            options: LlamaOptions {
                model_path: model_path.into(),
                ..Default::default()
            },
        }
    }

    pub fn chat_template_file(mut self, path: impl Into<String>) -> Self {
        self.options.chat_template_file = Some(path.into());
        self
    }

    pub fn context_size(mut self, size: i32) -> Self {
        self.options.context_size = Some(size);
        self
    }

    pub fn n_gpu_layers(mut self, layers: i32) -> Self {
        self.options.n_gpu_layers = Some(layers);
        self
    }

    pub fn n_threads(mut self, threads: i32) -> Self {
        self.options.n_threads = Some(threads);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.options.temperature = Some(temp);
        self
    }

    pub fn use_jinja(mut self, use_jinja: bool) -> Self {
        self.options.use_jinja = use_jinja;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.options.verbose = verbose;
        self
    }

    pub fn build(self) -> LlamaOptions {
        self.options
    }
}
