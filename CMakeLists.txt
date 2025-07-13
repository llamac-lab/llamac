cmake_minimum_required(VERSION 3.18)      # 3.18+ has first-class CUDA support
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(c_cuda_matmul LANGUAGES C CUDA)   # <- CUDA lives here again

set(CMAKE_C_STANDARD 11)

set(CUDA_MIN_VER 12.4)                                    # whatever you need
find_package(CUDAToolkit ${CUDA_MIN_VER} REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS CUDA_MIN_VER)
    message(FATAL_ERROR "Need CUDA ≥ ${CUDA_MIN_VER}")
endif()

# GPU architectures
set(DEFAULT_ARCHS "89")
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${DEFAULT_ARCHS}
            CACHE STRING "GPU arch list" FORCE)       # :contentReference[oaicite:0]{index=0}
endif()

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/llamacpp/include
        ${CMAKE_SOURCE_DIR}/llamacpp/ggml/include
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Forward llama.cpp config to subdir
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TOOLS ON CACHE BOOL "" FORCE)
#set(GGML_CUDA ON CACHE BOOL "" FORCE)
#set(GGML_CUDA_FORCE_CUBLAS ON CACHE BOOL "" FORCE)
option(GGML_CUDA "Enable CUDA backend in ggml" ON)      # default ON for dev boxes
option(GGML_CUDA_FORCE_CUBLAS "Force cuBLAS kernels" OFF)

set(LLAMA_STANDALONE ON CACHE BOOL "" FORCE)
set(GGML_THREADS ON CACHE BOOL "" FORCE)


add_subdirectory(llamacpp)
## core/tensor etc
add_subdirectory(llamac)
## tools/examples etc
add_subdirectory(tools/llamac-runner)
add_subdirectory(tools/guuf-reader)

