function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt CUDA::cudart CUDA::cusparse)
endfunction()

add_example(simple_spmv)
add_example(simple_spmm)
add_example(simple_spgemm)
add_example(simple_spgemm_two)
