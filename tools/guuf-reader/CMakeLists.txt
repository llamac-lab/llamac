cmake_minimum_required(VERSION 3.18)
project(gguf_reader C)

set(CMAKE_C_STANDARD 23)

#add_executable(gguf_reader
#        src/main.c
#        src/gguf_reader.c
#        ../../llamac/src/llamac.c
#)
#

add_executable(gguf_reader
        src/main.c
        src/gguf_reader.c
)


target_include_directories(gguf_reader PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/llcml/include
        ${CMAKE_SOURCE_DIR}/llcml/llcml-cuda/include
)
#
#target_link_libraries(gguf_reader PRIVATE llamac)
#

# tools/gguf-reader/CMakeLists.txt

target_link_libraries(gguf_reader PRIVATE

        llamac
        $<LINK_ONLY:Threads::Threads>
        $<LINK_ONLY:${CMAKE_DL_LIBS}>
#        $<LINK_ONLY:rt>
)