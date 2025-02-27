enable_language(HIP)

add_library(amd_spblas)
set(AMD_SPARSE_SOURCE
    csr_add.hip.cpp
    filter.hip.cpp
)
target_sources(amd_spblas
    PRIVATE
        ${AMD_SPARSE_SOURCE}
)

set_source_files_properties(${AMD_SPARSE_SOURCE} PROPERTIES LANGUAGE HIP)

target_include_directories(amd_spblas
    PUBLIC
        $<BUILD_INTERFACE:${spblas_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${spblas_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${spblas_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
