function(add_rocm_example example_name)
  set_source_files_properties(${example_name}.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

add_rocm_example(hello_world)
