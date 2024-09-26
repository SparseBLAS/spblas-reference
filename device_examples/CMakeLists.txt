function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
  target_compile_options(${example_name} PRIVATE "-Wno-unused-result")
endfunction()

add_example(simple_spmv)
# add_example(simple_spmm) onemkl seems to contain spmv only? or the concept already for both
add_example(simple_spgemm)
add_example(simple_spgemm_two)
add_example(simple_op)
add_example(simple_conv)
add_example(simple_sddmm)
add_example(simple_trsv)
