enable_testing()

set(TEST_SOURCES)

# CPU tests
if (SPBLAS_CPU_BACKEND)
  list(APPEND TEST_SOURCES
       mdspan_overlays.cpp
       core_types_test.cpp)

  if (SPBLAS_PAPER_API)
    list(APPEND TEST_SOURCES
         spmv_test.cpp
         spmm_test.cpp
         spgemm_test.cpp
         add_test.cpp
         transpose_test.cpp
         triangular_solve_test.cpp
         scale_test.cpp
         norm_test.cpp)
  else()
    list(APPEND TEST_SOURCES
         legacy/spmv_test.cpp
         legacy/spmm_test.cpp
         legacy/spgemm_test.cpp
         legacy/spgemm_csr_csc.cpp
         legacy/add_test.cpp
         legacy/transpose_test.cpp
         legacy/triangular_solve_test.cpp)

    if (ENABLE_ONEMKL_SYCL)
      list(APPEND TEST_SOURCES legacy/conjugate_test.cpp)
    endif()
  endif()
endif()

# GPU tests
if (SPBLAS_GPU_BACKEND)
  if (ENABLE_ROCSPARSE)
    set(GPUTEST_SOURCES device/spmv_test.cpp device/spgemm_test.cpp device/spgemm_reuse_test.cpp device/rocsparse/spgemm_4args_test.cpp)
    set_source_files_properties(${GPUTEST_SOURCES} PROPERTIES LANGUAGE HIP)
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
