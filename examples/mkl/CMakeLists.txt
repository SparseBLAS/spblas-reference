function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

if (ENABLE_ONEMKL)
  add_example(mkl_test)
endif()
