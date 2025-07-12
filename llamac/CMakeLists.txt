cmake_minimum_required(VERSION 3.18)      # 3.18+ has first-class CUDA support
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(c_cuda_matmul LANGUAGES C CUDA)   # <- CUDA lives here again

add_library(llamac STATIC
        src/llamac.c
        llmclm/src/matmul.cu
)


target_include_directories(llamac PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include   # for llamac.h
)

target_include_directories(llamac PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/llmclm/include
)