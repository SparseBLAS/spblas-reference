add_executable(miniapps_randnla_tests
  test_cqrrpt.cc)

include(GoogleTest)
target_link_libraries(miniapps_randnla_tests miniapps GTest::gtest_main)

gtest_discover_tests(miniapps_randnla_tests)
