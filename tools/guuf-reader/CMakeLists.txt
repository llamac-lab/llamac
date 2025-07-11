cmake_minimum_required(VERSION 3.31)
project(gguf_reader C)

set(CMAKE_C_STANDARD 23)

add_executable(gguf_reader
        src/main.c
        src/gguf_reader.c
)

target_include_directories(gguf_reader PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)


#add_executable(gguf_reader
#        src/main.c
#        src/gguf_reader.c
#)
#
#target_include_directories(gguf_reader PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/include
#        ${CMAKE_CURRENT_SOURCE_DIR}/src
#)