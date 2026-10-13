cmake_minimum_required(VERSION 3.24)

# Default to Tesla T4 (sm_75) unless the user picked an architecture.
# Must be set before project() so CUDA compiler detection sees it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build for")
endif()

project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(BUILD_TESTS "Build the GoogleTest suite" ON)
option(BUILD_BENCH "Build the benchmarks" ON)

add_library(cuda_kernels STATIC
  src/vecadd.cu
  src/reduce.cu
)
target_include_directories(cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()

if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(test_kernels tests/test_kernels.cu)
  target_link_libraries(test_kernels PRIVATE cuda_kernels GTest::gtest_main)

  include(GoogleTest)
  # PRE_TEST defers test discovery to ctest time, so building never runs the
  # binary (CI has no GPU).
  gtest_discover_tests(test_kernels DISCOVERY_MODE PRE_TEST)
endif()

if(BUILD_BENCH)
  add_executable(bench_reduce bench/bench_reduce.cu)
  target_link_libraries(bench_reduce PRIVATE cuda_kernels)
endif()
