cmake_minimum_required(VERSION 3.18)
project(ggml LANGUAGES C CUDA)

file(GLOB_RECURSE GGML_SOURCES
        ggml-cuda/src/*.cu
)

add_library(ggml STATIC ${GGML_SOURCES})

target_include_directories(llamacpp20 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/ggml-cuda/include
)