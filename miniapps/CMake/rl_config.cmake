
configure_file(CMake/miniappsConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/miniappsConfig.cmake @ONLY)

configure_file(CMake/miniappsConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/miniappsConfigVersion.cmake @ONLY)

install(FILES
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/miniappsConfig.cmake
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/miniappsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
