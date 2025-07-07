enable_testing()

set(TEST_SOURCES)

# CPU tests
if (SPBLAS_CPU_BACKEND)
  list(APPEND TEST_SOURCES
       spmv_test.cpp
       spmm_test.cpp
       spgemm_test.cpp
       spgemm_csr_csc.cpp
       add_test.cpp
       transpose_test.cpp
       triangular_solve_test.cpp)
endif()

# GPU tests
if (SPBLAS_GPU_BACKEND)
  if (ENABLE_ROCSPARSE)
    set_source_files_properties(device/spmv_test.cpp PROPERTIES LANGUAGE HIP)
  endif()
  list(APPEND TEST_SOURCES device/spmv_test.cpp)
endif()

add_executable(spblas-tests ${TEST_SOURCES})
target_link_libraries(spblas-tests spblas fmt GTest::gtest_main)

# Backend-specific test configuration
if (ENABLE_ROCSPARSE)
  target_link_libraries(spblas-tests roc::rocthrust)
elseif (ENABLE_CUSPARSE)
  target_link_libraries(spblas-tests Thrust)
elseif (ENABLE_ONEMKL_SYCL)
  target_link_libraries(spblas-tests sycl_thrust)
endif()

include(GoogleTest)
gtest_discover_tests(spblas-tests)
