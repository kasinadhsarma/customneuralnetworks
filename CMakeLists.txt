cmake_minimum_required(VERSION 3.28)
project(customneuralnetworks)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add code coverage option
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

# Coverage settings
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0 -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()
endif()

# Enable optimizations for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include nlohmann_json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Google Benchmark
if(ENABLE_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF)
    FetchContent_MakeAvailable(benchmark)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Explicitly list source files to avoid duplicates
set(LIB_SOURCES
    src/Model.cpp
    src/ReLU.cpp
    src/DenseLayer.cpp
)

# Create main library
add_library(neural_network_lib ${LIB_SOURCES})
target_link_libraries(neural_network_lib 
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC nlohmann_json::nlohmann_json
)

# Create main executable
add_executable(${PROJECT_NAME} "main.c++")
target_link_libraries(${PROJECT_NAME} 
    PRIVATE neural_network_lib
)

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_executable(neural_network_benchmark benchmark/neural_network_benchmark.cpp)
    target_link_libraries(neural_network_benchmark
        PRIVATE neural_network_lib
        PRIVATE benchmark::benchmark
        PRIVATE benchmark::benchmark_main
    )
endif()

# Testing
enable_testing()
if(EXISTS "${PROJECT_SOURCE_DIR}/tests")
    add_subdirectory(tests)
endif()
