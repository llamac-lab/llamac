cmake_minimum_required(VERSION 3.18)
project(llcml LANGUAGES C CUDA)

add_library(llcml STATIC
        cuda-kernel/src/matmul.cu
        cuda-kernel/include/mean.cuh
        cuda-kernel/include/wkv.cuh
        cuda-kernel/include/common.cuh
        cuda-kernel/src/wkv.cu
        cuda-kernel/include/vecdotq.cuh
        cuda-kernel/include/upscale.cuh
        cuda-kernel/include/unary.cuh
        cuda-kernel/include/tsembed.cuh
        cuda-kernel/include/sumrows.cuh
)

target_include_directories(llcml PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda-kernel/include
)