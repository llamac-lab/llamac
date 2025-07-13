# llamac/CMakeLists.txt
# Option 1: no sub-project declaration at all (simpler)
# Option 2: project(llamac LANGUAGES C CUDA)

find_package(Threads REQUIRED)          # must come before use

add_library(llamac STATIC
        src/llamac.c
        llmclm/src/matmul.cu
)

target_include_directories(llamac
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/llmclm/include
)

set_target_properties(llamac PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(llamac
        PUBLIC  llama
        PRIVATE Threads::Threads ${CMAKE_DL_LIBS}
)

add_library(llamac::llamac ALIAS llamac)
