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
include tests/CMakeFiles/test-quantize-stats.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-quantize-stats.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-quantize-stats.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-quantize-stats.dir/flags.make

tests/CMakeFiles/test-quantize-stats.dir/codegen:
.PHONY : tests/CMakeFiles/test-quantize-stats.dir/codegen

tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o: tests/CMakeFiles/test-quantize-stats.dir/flags.make
tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o: /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-quantize-stats.cpp
tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o: tests/CMakeFiles/test-quantize-stats.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o -MF CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o.d -o CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o -c /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-quantize-stats.cpp

tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.i"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-quantize-stats.cpp > CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.i

tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.s"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests/test-quantize-stats.cpp -o CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.s

# Object files for target test-quantize-stats
test__quantize__stats_OBJECTS = \
"CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o"

# External object files for target test-quantize-stats
test__quantize__stats_EXTERNAL_OBJECTS =

bin/test-quantize-stats: tests/CMakeFiles/test-quantize-stats.dir/test-quantize-stats.cpp.o
bin/test-quantize-stats: tests/CMakeFiles/test-quantize-stats.dir/build.make
bin/test-quantize-stats: common/libcommon.a
bin/test-quantize-stats: /usr/lib/x86_64-linux-gnu/libcurl.so
bin/test-quantize-stats: bin/libllama.so
bin/test-quantize-stats: bin/libggml.so
bin/test-quantize-stats: bin/libggml-cpu.so
bin/test-quantize-stats: bin/libggml-cuda.so
bin/test-quantize-stats: bin/libggml-base.so
bin/test-quantize-stats: /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so
bin/test-quantize-stats: tests/CMakeFiles/test-quantize-stats.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test-quantize-stats"
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-quantize-stats.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-quantize-stats.dir/build: bin/test-quantize-stats
.PHONY : tests/CMakeFiles/test-quantize-stats.dir/build

tests/CMakeFiles/test-quantize-stats.dir/clean:
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-quantize-stats.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-quantize-stats.dir/clean

tests/CMakeFiles/test-quantize-stats.dir/depend:
	cd /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/tests /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests /home/ervin/workspace/POC/llamac-lab/llamac/llamacpp/build/tests/CMakeFiles/test-quantize-stats.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test-quantize-stats.dir/depend

