#!/bin/bash

set -e

mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DDBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FORCE_CUBLAS=ON \
      -DLLAMA_STANDALONE=ON \
      -DGGML_THREADS=ON   ..
make -j$(nproc)



