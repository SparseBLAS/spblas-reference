enable_testing()
function(add_device_test file_list)
  add_executable(spblas-tests ${${file_list}})
  if (ENABLE_ROCSPARSE)
    set_source_files_properties(${${file_list}} PROPERTIES LANGUAGE HIP)
    target_link_libraries(spblas-tests rocthrust)
  elseif (ENABLE_CUSPARSE)
    target_link_libraries(spblas-tests Thrust)
  elseif (ENABLE_ONEMKL_SYCL)
    target_link_libraries(spblas-tests sycl_thrust)
  else()
    message(FATAL_ERROR "Device backend not found.")
  endif()
endfunction()

set(TEST_SOURCES)

if (NOT SPBLAS_GPU_BACKEND)
  list(APPEND TEST_SOURCES
       spmv_test.cpp
       spmm_test.cpp
       spgemm_test.cpp
       spgemm_csr_csc.cpp
       add_test.cpp
       transpose_test.cpp
       triangular_solve_test.cpp)
else()
   list(APPEND TEST_SOURCES device/spmv_test.cpp)
endif()

add_executable(spblas-tests ${TEST_SOURCES})

target_link_libraries(spblas-tests spblas fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(spblas-tests)
