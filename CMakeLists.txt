cmake_minimum_required(VERSION 3.20)
project(spblas)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

# Check whether compiler supports ranges and std::ranges::zip.
# If not, use range-v3 instead of standard library's ranges.

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(__cpp_lib_ranges "ranges" HAS_RANGES)
check_cxx_symbol_exists(__cpp_lib_ranges_zip "ranges" HAS_ZIP)

if (HAS_RANGES AND HAS_ZIP)
  set(USE_RANGE_V3 OFF)
  message(STATUS "Standard library supports both ranges and std::views::zip, using standard library.")
else()
  set(USE_RANGE_V3 ON)
  message(STATUS "NOTE: Standard library does not include ranges and/or std::views::zip. Using range-v3.")
endif()

if (USE_RANGE_V3)
  include(FetchContent)
  FetchContent_Declare(
    range-v3
    GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
    GIT_TAG 5300fe3)
  FetchContent_MakeAvailable(range-v3)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG mdspan-0.6.0)
FetchContent_MakeAvailable(mdspan)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.1)
FetchContent_MakeAvailable(fmt)

add_subdirectory(include)
add_subdirectory(examples)
add_subdirectory(test)
