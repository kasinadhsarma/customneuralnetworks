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
CMAKE_SOURCE_DIR = /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild

# Utility rule file for benchmark-populate.

# Include any custom commands dependencies for this target.
include CMakeFiles/benchmark-populate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/benchmark-populate.dir/progress.make

CMakeFiles/benchmark-populate: CMakeFiles/benchmark-populate-complete

CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-install
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-mkdir
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-download
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-patch
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-configure
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-build
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-install
CMakeFiles/benchmark-populate-complete: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-test
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'benchmark-populate'"
	/usr/bin/cmake -E make_directory /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles
	/usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles/benchmark-populate-complete
	/usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-done

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update:
.PHONY : benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-build: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No build step for 'benchmark-populate'"
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E echo_append
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-build

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-configure: benchmark-populate-prefix/tmp/benchmark-populate-cfgcmd.txt
benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-configure: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "No configure step for 'benchmark-populate'"
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E echo_append
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-configure

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-download: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-gitinfo.txt
benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-download: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'benchmark-populate'"
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps && /usr/bin/cmake -P /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/tmp/benchmark-populate-gitclone.cmake
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps && /usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-download

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-install: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No install step for 'benchmark-populate'"
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E echo_append
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-install

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'benchmark-populate'"
	/usr/bin/cmake -Dcfgdir= -P /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/tmp/benchmark-populate-mkdirs.cmake
	/usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-mkdir

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-patch: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-patch-info.txt
benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-patch: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'benchmark-populate'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-patch

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update:
.PHONY : benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-test: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No test step for 'benchmark-populate'"
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E echo_append
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-build && /usr/bin/cmake -E touch /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-test

benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update: benchmark-populate-prefix/tmp/benchmark-populate-gitupdate.cmake
benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update-info.txt
benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Performing update step for 'benchmark-populate'"
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-src && /usr/bin/cmake -Dcan_fetch=YES -P /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/benchmark-populate-prefix/tmp/benchmark-populate-gitupdate.cmake

benchmark-populate: CMakeFiles/benchmark-populate
benchmark-populate: CMakeFiles/benchmark-populate-complete
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-build
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-configure
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-download
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-install
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-mkdir
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-patch
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-test
benchmark-populate: benchmark-populate-prefix/src/benchmark-populate-stamp/benchmark-populate-update
benchmark-populate: CMakeFiles/benchmark-populate.dir/build.make
.PHONY : benchmark-populate

# Rule to build all files generated by this target.
CMakeFiles/benchmark-populate.dir/build: benchmark-populate
.PHONY : CMakeFiles/benchmark-populate.dir/build

CMakeFiles/benchmark-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/benchmark-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/benchmark-populate.dir/clean

CMakeFiles/benchmark-populate.dir/depend:
	cd /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild /home/kasinadhsarma/customneuralnetworks/build/_deps/benchmark-subbuild/CMakeFiles/benchmark-populate.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/benchmark-populate.dir/depend

