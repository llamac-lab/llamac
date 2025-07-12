cmake_minimum_required(VERSION 3.18)      # 3.18+ has first-class CUDA support
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(llamac_runner LANGUAGES C CUDA)   # <- CUDA lives here again

add_executable(llamac_runner
        src/main.c
)

target_include_directories(llamac_runner PRIVATE
        ${CMAKE_SOURCE_DIR}/llamac/include
        ${CMAKE_SOURCE_DIR}/llamac/llmclm/include
        ${CMAKE_SOURCE_DIR}/llamacpp20/ggml/include
)

target_link_libraries(llamac_runner PRIVATE llamac)

