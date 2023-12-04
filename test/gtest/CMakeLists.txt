enable_testing()

add_executable(
  spblas-tests
  spmv_test.cpp
  spmm_test.cpp
  spgemm_test.cpp
)

target_link_libraries(spblas-tests spblas fmt mdspan GTest::gtest_main)

if(USE_RANGE_V3)
  target_link_libraries(spblas-tests range-v3)
endif()

include(GoogleTest)
gtest_discover_tests(spblas-tests)
