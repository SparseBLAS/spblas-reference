function(add_device_example example_name)
  add_executable(${example_name} ${example_name}.cpp)

  if (ENABLE_ROCSPARSE)
    find_package(rocthrust REQUIRED)
    set_source_files_properties(${example_name}.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(${example_name} rocthrust)
  # elseif (ENABLE_CUSPARSE)
    # cuSPARSE linking details will go here.
  elseif (ENABLE_ONEMKL_SYCL)
    target_link_libraries(${example_name} sycl_thrust)
  else()
    message(FATAL_ERROR "Device backend not found.")
  endif()

  target_link_libraries(${example_name} spblas fmt)
endfunction()

add_device_example(simple_spmv)
