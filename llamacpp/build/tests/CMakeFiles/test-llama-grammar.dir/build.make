# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test-llama-grammar.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-llama-grammar.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-llama-grammar.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-llama-grammar.dir/flags.make

tests/CMakeFiles/test-llama-grammar.dir/codegen:
.PHONY : tests/CMakeFiles/test-llama-grammar.dir/codegen

tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o: tests/CMakeFiles/test-llama-grammar.dir/flags.make
tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o: /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-llama-grammar.cpp
tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o: tests/CMakeFiles/test-llama-grammar.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o -MF CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o.d -o CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o -c /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-llama-grammar.cpp

tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.i"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-llama-grammar.cpp > CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.i

tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.s"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-llama-grammar.cpp -o CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.s

tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o: tests/CMakeFiles/test-llama-grammar.dir/flags.make
tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o: /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/get-model.cpp
tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o: tests/CMakeFiles/test-llama-grammar.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o -MF CMakeFiles/test-llama-grammar.dir/get-model.cpp.o.d -o CMakeFiles/test-llama-grammar.dir/get-model.cpp.o -c /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/get-model.cpp

tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test-llama-grammar.dir/get-model.cpp.i"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/get-model.cpp > CMakeFiles/test-llama-grammar.dir/get-model.cpp.i

tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test-llama-grammar.dir/get-model.cpp.s"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/get-model.cpp -o CMakeFiles/test-llama-grammar.dir/get-model.cpp.s

# Object files for target test-llama-grammar
test__llama__grammar_OBJECTS = \
"CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o" \
"CMakeFiles/test-llama-grammar.dir/get-model.cpp.o"

# External object files for target test-llama-grammar
test__llama__grammar_EXTERNAL_OBJECTS =

bin/test-llama-grammar: tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o
bin/test-llama-grammar: tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o
bin/test-llama-grammar: tests/CMakeFiles/test-llama-grammar.dir/build.make
bin/test-llama-grammar: common/libcommon.a
bin/test-llama-grammar: /usr/lib/x86_64-linux-gnu/libcurl.so
bin/test-llama-grammar: bin/libllama.so
bin/test-llama-grammar: bin/libggml.so
bin/test-llama-grammar: bin/libggml-cpu.so
bin/test-llama-grammar: bin/libggml-cuda.so
bin/test-llama-grammar: bin/libggml-base.so
bin/test-llama-grammar: /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so
bin/test-llama-grammar: tests/CMakeFiles/test-llama-grammar.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/test-llama-grammar"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-llama-grammar.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-llama-grammar.dir/build: bin/test-llama-grammar
.PHONY : tests/CMakeFiles/test-llama-grammar.dir/build

tests/CMakeFiles/test-llama-grammar.dir/clean:
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-llama-grammar.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-llama-grammar.dir/clean

tests/CMakeFiles/test-llama-grammar.dir/depend:
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests/CMakeFiles/test-llama-grammar.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test-llama-grammar.dir/depend

