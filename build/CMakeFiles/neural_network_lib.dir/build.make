# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kasinadhsarma/customneuralnetworks

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kasinadhsarma/customneuralnetworks/build

# Include any dependencies generated for this target.
include CMakeFiles/neural_network_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neural_network_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neural_network_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neural_network_lib.dir/flags.make

CMakeFiles/neural_network_lib.dir/src/Model.cpp.o: CMakeFiles/neural_network_lib.dir/flags.make
CMakeFiles/neural_network_lib.dir/src/Model.cpp.o: /home/kasinadhsarma/customneuralnetworks/src/Model.cpp
CMakeFiles/neural_network_lib.dir/src/Model.cpp.o: CMakeFiles/neural_network_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neural_network_lib.dir/src/Model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network_lib.dir/src/Model.cpp.o -MF CMakeFiles/neural_network_lib.dir/src/Model.cpp.o.d -o CMakeFiles/neural_network_lib.dir/src/Model.cpp.o -c /home/kasinadhsarma/customneuralnetworks/src/Model.cpp

CMakeFiles/neural_network_lib.dir/src/Model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network_lib.dir/src/Model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kasinadhsarma/customneuralnetworks/src/Model.cpp > CMakeFiles/neural_network_lib.dir/src/Model.cpp.i

CMakeFiles/neural_network_lib.dir/src/Model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network_lib.dir/src/Model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kasinadhsarma/customneuralnetworks/src/Model.cpp -o CMakeFiles/neural_network_lib.dir/src/Model.cpp.s

CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o: CMakeFiles/neural_network_lib.dir/flags.make
CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o: /home/kasinadhsarma/customneuralnetworks/src/ReLU.cpp
CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o: CMakeFiles/neural_network_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o -MF CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o.d -o CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o -c /home/kasinadhsarma/customneuralnetworks/src/ReLU.cpp

CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kasinadhsarma/customneuralnetworks/src/ReLU.cpp > CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.i

CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kasinadhsarma/customneuralnetworks/src/ReLU.cpp -o CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.s

CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o: CMakeFiles/neural_network_lib.dir/flags.make
CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o: /home/kasinadhsarma/customneuralnetworks/src/DenseLayer.cpp
CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o: CMakeFiles/neural_network_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o -MF CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o.d -o CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o -c /home/kasinadhsarma/customneuralnetworks/src/DenseLayer.cpp

CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kasinadhsarma/customneuralnetworks/src/DenseLayer.cpp > CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.i

CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kasinadhsarma/customneuralnetworks/src/DenseLayer.cpp -o CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.s

# Object files for target neural_network_lib
neural_network_lib_OBJECTS = \
"CMakeFiles/neural_network_lib.dir/src/Model.cpp.o" \
"CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o" \
"CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o"

# External object files for target neural_network_lib
neural_network_lib_EXTERNAL_OBJECTS =

libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/src/Model.cpp.o
libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/src/ReLU.cpp.o
libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/src/DenseLayer.cpp.o
libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/build.make
libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libneural_network_lib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/neural_network_lib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neural_network_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neural_network_lib.dir/build: libneural_network_lib.a
.PHONY : CMakeFiles/neural_network_lib.dir/build

CMakeFiles/neural_network_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neural_network_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neural_network_lib.dir/clean

CMakeFiles/neural_network_lib.dir/depend:
	cd /home/kasinadhsarma/customneuralnetworks/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kasinadhsarma/customneuralnetworks /home/kasinadhsarma/customneuralnetworks /home/kasinadhsarma/customneuralnetworks/build /home/kasinadhsarma/customneuralnetworks/build /home/kasinadhsarma/customneuralnetworks/build/CMakeFiles/neural_network_lib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/neural_network_lib.dir/depend

