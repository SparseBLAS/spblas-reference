cmake_minimum_required(VERSION 3.20)

project(miniapps)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# configure the build
enable_testing()

include(rl_build_options)
include(rl_version)

# find dependencies
find_package(lapackpp REQUIRED)

# find dependencies
find_package(blaspp REQUIRED)

# compile sources
add_subdirectory(include)

if (ENABLE_TESTING)
    add_subdirectory(test)
endif()

# export the configuration
include(rl_config)
