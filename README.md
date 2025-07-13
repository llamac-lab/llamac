# llamac

[![Build (CUDA - no GPU)](https://github.com/llamac-lab/llamac/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/llamac-lab/llamac/actions/workflows/cuda-build.yml)

**llamac-lab** is a pure C runtime for LLaMA-based models, built for tiny devices, embedded environments, and maximum portability.

Think [`llama.cpp`](https://github.com/ggerganov/llama.cpp), but:
- Flattened into C
- Optimized for small memory
- Easy to integrate with any stack (Rust, Python, etc.)
- Born for the edge

note: work in progress, not a functioning anything yet :)

working inferencing (currently using llamacpp as a backend)
<img width="1793" height="387" alt="image" src="https://github.com/user-attachments/assets/f88d8620-19f1-490a-861f-599fca071e1f" />
