enable_testing()

if (NOT ENABLE_ROCSPARSE)
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
elseif(ENABLE_ROCSPARSE)
  # general one is handle in another pr
  set_source_files_properties(rocsparse/spmv_test.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(rocsparse/spgemm_test.cpp PROPERTIES LANGUAGE HIP)
  add_executable(spblas-tests
                 rocsparse/spmv_test.cpp
                 rocsparse/spgemm_test.cpp)
endif()

target_link_libraries(spblas-tests spblas fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(spblas-tests)
