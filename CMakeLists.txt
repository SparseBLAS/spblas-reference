cmake_minimum_required(VERSION 3.20)
project(spblas)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

# Get includes, which declares the `spblas` library
add_subdirectory(include)

# Download dependencies
include(FetchContent)

if (ENABLE_ONEMKL)
  find_package(MKL REQUIRED)
  target_link_libraries(spblas INTERFACE MKL::MKL_DPCPP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPBLAS_ENABLE_ONEMKL")
endif()

if (ENABLE_ARMPL)
  if (NOT DEFINED ENV{ARMPL_DIR})
    message(FATAL_ERROR "Environment variable ARMPL_DIR must be set when the ArmPL is enabled.")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPBLAS_ENABLE_ARMPL -I$ENV{ARMPL_DIR}/include")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L$ENV{ARMPL_DIR}/lib -larmpl -lamath -lm")
endif()

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
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # libfmt
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1)
  FetchContent_MakeAvailable(fmt)

  add_subdirectory(examples)
  add_subdirectory(test)
endif()
