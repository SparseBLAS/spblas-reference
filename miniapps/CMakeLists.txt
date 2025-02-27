cmake_minimum_required(VERSION 3.24)

project(miniapps VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SPBLAS_MINIAPPS_ENABLE_CQRRPT "Enable the CQRRPT miniapp. Default ON." ON)
option(SPBLAS_MINIAPPS_ENABLE_SOLVERS "Enable the solvers miniapp. Default ON." ON)

enable_testing()

include(rl_build_options)

# Find dependencies
find_package(blaspp 2024.10.26 QUIET)
if(NOT blaspp_FOUND)
  FetchContent_Declare(
    blaspp
    GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp
    GIT_TAG v2024.10.26)
    FetchContent_MakeAvailable(blaspp)
endif()

set(blas "auto")
set(lapack "auto")

find_package(lapackpp 2024.10.26 QUIET)
if(NOT lapackpp_FOUND)
  FetchContent_Declare(
    lapackpp
    GIT_REPOSITORY https://github.com/icl-utk-edu/lapackpp
    GIT_TAG v2024.10.26)
  FetchContent_MakeAvailable(lapackpp)
endif()

# compile sources
add_subdirectory(include)

add_subdirectory(test)

# export the configuration
include(rl_config)
