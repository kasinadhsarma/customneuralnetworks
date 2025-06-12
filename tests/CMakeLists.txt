enable_testing()

# Include GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(neural_network_test neural_network_test.cpp)
target_link_libraries(neural_network_test
    PRIVATE neural_network_lib
    PRIVATE GTest::gtest
    PRIVATE GTest::gtest_main
)

# Register test
add_test(NAME neural_network_test COMMAND neural_network_test)
