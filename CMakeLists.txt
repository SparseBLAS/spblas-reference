cmake_minimum_required(VERSION 3.20)
project(spblas)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

option(ENABLE_SANITIZERS "Enable Clang sanitizers" OFF)
option(ENABLE_ROCSPARSE "Enable rocSPARSE" OFF)

# Get includes, which declares the `spblas` library
add_subdirectory(include)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.24)
  cmake_policy(SET CMP0135 NEW) # allows handling of changes to remote content
endif()

# Download dependencies
include(FetchContent)

if (ENABLE_ROCSPARSE)
  project(spblas LANGUAGES HIP)
  find_package(hip REQUIRED)
  find_package(rocsparse REQUIRED)
  find_package(rocthrust REQUIRED)
  target_link_libraries(spblas INTERFACE roc::rocsparse hip::host roc::rocthrust)
endif()

if (ENABLE_ONEMKL_SYCL)
  find_package(MKL REQUIRED)
  target_link_libraries(spblas INTERFACE MKL::MKL_SYCL) # SYCL APIs
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPBLAS_ENABLE_ONEMKL_SYCL")
endif()

if (ENABLE_ARMPL)
  if (NOT DEFINED ENV{ARMPL_DIR})
    message(FATAL_ERROR "Environment variable ARMPL_DIR must be set when the ArmPL is enabled.")
  endif()
  target_include_directories(spblas INTERFACE $ENV{ARMPL_DIR}/include)
  target_link_libraries(spblas INTERFACE $ENV{ARMPL_DIR}/lib/libarmpl.a $ENV{ARMPL_DIR}/lib/libamath.a m)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPBLAS_ENABLE_ARMPL")
endif()

# turn on/off debug logging
if (LOG_LEVEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOG_LEVEL=${LOG_LEVEL}") # SPBLAS_DEBUG | SPBLAS_WARNING | SPBLAS_TRACE | SPBLAS_INFO
endif()

# Enable sanitizers
if (ENABLE_SANITIZERS)
    set(SANITIZER_FLAGS "-fsanitize=address,undefined")
    target_compile_options(spblas INTERFACE ${SANITIZER_FLAGS} -g -O1 -fno-omit-frame-pointer)
    target_link_options(spblas INTERFACE ${SANITIZER_FLAGS})
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

install(DIRECTORY include/spblas DESTINATION include)

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
    GIT_TAG 11.1.3)
  FetchContent_MakeAvailable(fmt)

  add_subdirectory(examples)
  add_subdirectory(test)
endif()
