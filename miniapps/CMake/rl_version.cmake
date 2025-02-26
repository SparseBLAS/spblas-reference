set(tmp)
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE}
        --git-dir=${CMAKE_SOURCE_DIR}/.git describe
        --tags --match "[0-9]*.[0-9]*.[0-9]*"
        OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
if(NOT tmp)
    set(tmp "0.0.0")
endif()

set(miniapps_VERSION ${tmp} CACHE STRING "miniapps version" FORCE)

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\1" miniapps_VERSION_MAJOR ${miniapps_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\2" miniapps_VERSION_MINOR ${miniapps_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\3" miniapps_VERSION_PATCH ${miniapps_VERSION})

message(STATUS "miniapps_VERSION_MAJOR=${miniapps_VERSION_MAJOR}")
message(STATUS "miniapps_VERSION_MINOR=${miniapps_VERSION_MINOR}")
message(STATUS "miniapps_VERSION_PATCH=${miniapps_VERSION_PATCH}")
