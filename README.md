# llamac

[![Build (CUDA - no GPU)](https://github.com/llamac-lab/llamac/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/llamac-lab/llamac/actions/workflows/cuda-build.yml)

**llamac-lab** is a lightweight, portable C runtime for LLaMA-based models - designed (being designed, have the goal) for tiny devices, embedded systems, and people who like things fast, flat, and minimal.

 Think [`llama.cpp`](https://github.com/ggerganov/llama.cpp), but:

 * Flattened into C (no C++)
 * Slimmed down for constrained environments
 * Easy to embed into Rust, Python, etc.
 * Born for edge and embedded AI work


**Work in progress**: not production-ready, not stable — but *it runs*.


## Current Components

### `llamac-runner` (C)

> Basic inference runner for LLaMA models
> Tested with [tinyllama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main), working on direct GGUF integration.

### `gguf-reader` (C)

> Utility to parse and print GGUF metadata
> Currently supports GGUF v3 files. Helpful for tooling/debugging and figuring out what's inside that 4GB `gguf`.


## `llamars` (Rust)

Experimental Rust wrapper for the llama.cpp C API. **Early days**, but:

* Can load models, tokenize, and run inference
* Comes with `llamars-runner` for interactive terminal chat
* Tested with CUDA builds
* Known issue: greedy sampling sometimes collapses into... garlic quesadilla recursion 🍽️


## Example Output

> Inference running via `llamars-runner`

<img width="1793" height="387" alt="image" src="https://github.com/user-attachments/assets/f88d8620-19f1-490a-861f-599fca071e1f" />


## Status

This repo is:

* Highly Experimental
* Educational
* Actively developed (on weekends or whenever feel that way, coffee permitting)

Want to play with LLaMA on embedded Linux, micro-servers, or inside your Rust app? This project might help.


## License

MIT - use at your own risk, submit PRs, and feel free to break it better.

