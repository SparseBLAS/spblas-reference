message(STATUS "\n\n-- miniapps checking for lapackpp ... ")
find_package(lapackpp REQUIRED)
message(STATUS "miniapps found lapackpp ${lapackpp_VERSION}\n")

# interface library for use elsewhere in the project
add_library(miniapps_lapackpp INTERFACE)

target_link_libraries(miniapps_lapackpp INTERFACE lapackpp)

install(TARGETS miniapps_lapackpp EXPORT miniapps_lapackpp)

install(EXPORT miniapps_lapackpp
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES)
