enable_testing()

set(TEST_SOURCES)

if (NOT ENABLE_ROCSPARSE)
  list(APPEND TEST_SOURCES
       spmv_test.cpp
       spmm_test.cpp
       spgemm_test.cpp
       spgemm_csr_csc.cpp
       add_test.cpp
       transpose_test.cpp
       triangular_solve_test.cpp)
endif()

if (SPBLAS_GPU_BACKEND)
  list(APPEND TEST_SOURCES
       thrust/spmv_test.cpp)

  if (ENABLE_ROCSPARSE)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE HIP)
  endif()
endif()

add_executable(spblas-tests ${TEST_SOURCES})

target_link_libraries(spblas-tests spblas fmt GTest::gtest_main)

if (ENABLE_ROCSPARSE)
  find_package(rocthrust REQUIRED)
  target_link_libraries(spblas-tests rocthrust)
endif()

if (ENABLE_ONEMKL_SYCL)
  target_link_libraries(spblas-tests sycl_thrust)
endif()

include(GoogleTest)
gtest_discover_tests(spblas-tests)
