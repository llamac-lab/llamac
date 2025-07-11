project(llcml LANGUAGES C CUDA)

add_library(llcml STATIC
        src/cuda-kernel/matmul.cu
)

target_include_directories(llcml PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)