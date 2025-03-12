function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
  target_compile_options(${example_name} PRIVATE "-Wno-unused-result")
endfunction()

add_example(simple_spmv)
