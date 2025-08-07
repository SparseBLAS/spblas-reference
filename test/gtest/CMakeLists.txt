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
function(add_rocsparse_lang file_list)
  if (ENABLE_ROCSPARSE)
    set_source_files_properties(${${file_list}} PROPERTIES LANGUAGE HIP)
  endif()
endfunction()
# GPU tests
if (SPBLAS_GPU_BACKEND)
  if (ENABLE_ROCSPARSE)
    set(GPUTEST_SOURCES device/spmv_test.cpp device/spgemm_test.cpp device/spgemm_reuse_test.cpp device/rocsparse/spgemm_4args_test.cpp)
    add_rocsparse_lang(GPUTEST_SOURCES)
  else ()
    set(GPUTEST_SOURCES device/spmv_test.cpp)
  endif ()
  list(APPEND TEST_SOURCES ${GPUTEST_SOURCES})
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
