cmake_minimum_required(VERSION 3.28)
project(customneuralnetworks)

set(CMAKE_CXX_STANDARD 20)

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

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES 
    "src/*.cpp"
)

# Add the main file explicitly
add_executable(${PROJECT_NAME} ${SOURCES} "main.c++")

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE nlohmann_json::nlohmann_json
)

# Enable testing
enable_testing()

# Optional: Add subdirectory for tests if they exist
if(EXISTS "${PROJECT_SOURCE_DIR}/tests")
    add_subdirectory(tests)
endif()
