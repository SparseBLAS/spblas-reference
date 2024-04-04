cmake_minimum_required(VERSION 3.20)
project(spblas)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

#set(CMAKE_CXX_FLAGS "-O3 -march=native")

# Get includes, which declares the `spblas` library
add_subdirectory(include)

# Download dependencies
include(FetchContent)

# mdspan
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG mdspan-0.6.0)
FetchContent_MakeAvailable(mdspan)
target_link_libraries(spblas INTERFACE mdspan)

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

# range-v3
if (USE_RANGE_V3)
  FetchContent_Declare(
    range-v3
    GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
    GIT_TAG 5300fe3)
  FetchContent_MakeAvailable(range-v3)
  target_link_libraries(spblas INTERFACE range-v3)
endif()


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Dependencies needed only for examples/test
  # Google test


  # libfmt
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1)
  FetchContent_MakeAvailable(fmt)

  add_subdirectory(examples)
  #add_subdirectory(test)
endif()
