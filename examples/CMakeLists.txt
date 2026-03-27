function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

# CPU examples
if (SPBLAS_CPU_BACKEND)
  add_example(simple_spmv)
  add_example(simple_spmm)
  add_example(simple_spgemm)
  add_example(simple_sptrsv)
  add_example(spmm_csc)
  add_example(matrix_opt_example)
  if (ENABLE_ONEMKL_SYCL OR SPBLAS_REFERENCE_BACKEND ) 
    # needs CPU + matrix_opt + operation_info_t to run
    add_example(sptrsv_csr) # needs triangular_solve{_inspect} to run
    add_example(spmm_csr) # needs multiply{_inspect} to run
  endif()
endif()

# GPU examples
if (SPBLAS_GPU_BACKEND)
  add_subdirectory(device)
  if (ENABLE_CUSPARSE)
    add_subdirectory(cusparse)
  endif()
  if (ENABLE_ROCSPARSE)
    add_subdirectory(rocsparse)
  endif()
endif()
