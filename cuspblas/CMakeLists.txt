enable_language(CUDA)

add_library(cuspblas)
target_sources(cuspblas
    PRIVATE
        csr_add.cu
        filter.cu
)
target_compile_options(cuspblas
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

target_include_directories(cuspblas
    PUBLIC
        $<BUILD_INTERFACE:${spblas_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${spblas_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${spblas_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
