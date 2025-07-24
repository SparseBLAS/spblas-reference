
function(add_sycl_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt sycl_thrust)
endfunction()

add_sycl_example(sycl_spmm)
