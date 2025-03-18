function(add_rocm_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

add_rocm_example(rocsparse_simple_spmv)
