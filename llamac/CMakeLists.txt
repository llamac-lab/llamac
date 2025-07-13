cmake_minimum_required(VERSION 3.18)      # 3.18+ has first-class CUDA support
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(llamac LANGUAGES C CUDA)   # <- CUDA lives here again


find_package(Threads REQUIRED)

option(GGML_CUDA             "Enable CUDA backend in ggml" ON)
option(GGML_CUDA_FORCE_CUBLAS "Force cuBLAS kernels"     OFF)

# -----------
set(LLAMAC_SRC  src/llamac.c
        src/generate_stub.c)
set(LLAMAC_PRIVATE_LIBS Threads::Threads )

if (GGML_CUDA)
    enable_language(CUDA)
    list(APPEND LLAMAC_SRC llmclm/src/matmul.cu)
    list(APPEND LLAMAC_PRIVATE_LIBS ggml-cuda ggml-base)
    set(CUDA_SEP_COMP ON)
else()
    list(APPEND LLAMAC_SRC src/generate_stub.c)
    list(APPEND LLAMAC_PRIVATE_LIBS ggml-cpu  ggml-base)
    set(CUDA_SEP_COMP OFF)
endif()

# -----------
add_library(llamac STATIC ${LLAMAC_SRC})

target_include_directories(llamac
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/llmclm/include)

set_target_properties(llamac PROPERTIES
        CUDA_SEPARABLE_COMPILATION ${CUDA_SEP_COMP})

# -------------
target_link_libraries(llamac
        PUBLIC  llama
        PRIVATE ${LLAMAC_PRIVATE_LIBS})

add_library(llamac::llamac ALIAS llamac)