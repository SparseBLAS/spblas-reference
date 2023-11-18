enable_testing()

add_executable(
  spblas-tests
  spmv_test.cpp
  spmm_test.cpp
)

target_link_libraries(spblas-tests spblas fmt mdspan GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(spblas-tests)
