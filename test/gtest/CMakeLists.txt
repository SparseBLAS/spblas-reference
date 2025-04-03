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
  set_source_files_properties(rocsparse/spmv_test.cpp PROPERTIES LANGUAGE HIP)
  add_executable(spblas-tests
                 rocsparse/spmv_test.cpp)
endif()

target_link_libraries(spblas-tests spblas fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(spblas-tests)

# unify it together after cusparse
if(ENABLE_ONEMKL_SYCL)
  add_executable(spblas-gpu-tests rocsparse/spmv_test.cpp)
  target_include_directories(spblas-gpu-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(spblas-gpu-tests spblas fmt GTest::gtest_main)
  gtest_discover_tests(spblas-gpu-tests)
endif()
