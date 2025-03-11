enable_testing()

if (ENABLE_ROCSPARSE)
  add_executable(spblas-tests spmv_test.cpp)
else()
  add_executable(
    spblas-tests
    spmv_test.cpp
    spmm_test.cpp
    spgemm_test.cpp
    spgemm_csr_csc.cpp
    add_test.cpp
    transpose_test.cpp
    triangular_solve_test.cpp
  )
endif()

target_link_libraries(spblas-tests spblas fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(spblas-tests)
