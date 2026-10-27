function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

function(add_legacy_example example_name)
  add_executable(${example_name} legacy/${example_name}.cpp)
  target_link_libraries(${example_name} spblas fmt)
endfunction()

# CPU examples
if (SPBLAS_CPU_BACKEND)
  if (SPBLAS_PAPER_API)
    # <operation>_<format>[_<feature>]; every example wraps its inputs in a
    # matrix_opt, as the SparseBLAS paper intends
    add_example(spmv_csr)
    add_example(spmm_csr)
    add_example(spmm_csc)
    add_example(spmm_csr_inspect)
    add_example(spgemm_csr)
    add_example(spgemm_csr_symbolic_numeric)
    add_example(spadd_csr)
    add_example(spadd_csr_symbolic_numeric)
    add_example(sptrsv_csr)
    add_example(sptrsv_csr_inspect)
    add_example(sort_csr_sync_properties)
  else()
    add_legacy_example(simple_spmv)
    add_legacy_example(simple_spmm)
    add_legacy_example(simple_spgemm)
    add_legacy_example(simple_sptrsv)
    add_legacy_example(spmm_csc)
    add_legacy_example(matrix_opt_example)
    if (ENABLE_ONEMKL_SYCL)
      # needs CPU + matrix_opt + operation_info_t to run
      add_legacy_example(sptrsv_csr) # needs triangular_solve{_inspect} to run
      add_legacy_example(spmm_csr) # needs multiply{_inspect} to run
    endif()
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
