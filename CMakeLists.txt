cmake_minimum_required(VERSION 3.10)
project(c_cuda_matmul C CXX CUDA)

set(CMAKE_C_STANDARD 11)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86 89)

add_executable(c_cuda_matmul
    src/main.c
    src/cuda-kernel/matmul.cu
)

target_include_directories(c_cuda_matmul PRIVATE include)

