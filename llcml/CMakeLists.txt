cmake_minimum_required(VERSION 3.18)
project(llcml LANGUAGES C CUDA)

file(GLOB_RECURSE LLCML_SOURCES
        llcml-cuda/src/*.cu
)

add_library(llcml STATIC ${LLCML_SOURCES})

target_include_directories(llcml PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/llcml-cuda/include
)