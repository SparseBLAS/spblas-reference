enable_testing()

add_executable(
  spblas-tests
  spmv_test.cpp
  spmm_test.cpp
  spgemm_test.cpp
  add_test.cpp
  transpose_test.cpp
  triangular_solve_test.cpp
)

target_link_libraries(spblas-tests spblas fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(spblas-tests)
