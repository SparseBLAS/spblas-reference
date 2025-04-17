function(add_device_example example_name)
  if (ENABLE_ROCSPARSE)
    set_source_files_properties(${example_name}.cpp PROPERTIES LANGUAGE HIP)
  # elseif (ENABLE_CUSPARSE)
    # cuSPARSE linking details will go here.
  else()
    message(FATAL_ERROR "Device backend not found.")
  endif()
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

add_device_example(simple_spmv)
