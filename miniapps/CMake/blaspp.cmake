message(STATUS "\n\n-- miniapps checking for blaspp ... ")
find_package(blaspp REQUIRED)
message(STATUS "miniapps found blaspp ${blaspp_VERSION}\n")

# interface library for use elsewhere in the project
add_library(miniapps_blaspp INTERFACE)

target_link_libraries(miniapps_blaspp INTERFACE blaspp)

install(TARGETS miniapps_blaspp EXPORT miniapps_blaspp)

install(EXPORT miniapps_blaspp
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES)