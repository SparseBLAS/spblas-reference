add_executable(miniapps_solvers_tests
  test_cg.cpp
  test_bicgstab.cpp)

target_link_libraries(miniapps_solvers_tests miniapps GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(miniapps_solvers_tests)
