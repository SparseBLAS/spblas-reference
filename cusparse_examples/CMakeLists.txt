function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

add_example(simple_spmv)
# add_example(simple_spmm) onemkl seems to contain spmv only? or the concept already for both
add_example(simple_spgemm)
add_example(simple_spgemm_two)
