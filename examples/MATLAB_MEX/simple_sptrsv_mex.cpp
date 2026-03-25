// Includes from SparseBLAS
#include <spblas/spblas.hpp>

// Includes for MEX
#include <matrix.h>
#include <mex.h>

// General includes
#include <complex> // Support complex inputs
#include <string>  // Parse uplo input

template <typename T>
void simple_sptrsv_mex(mwIndex m, mwIndex n, mwIndex nnz, mxArray * x_out,
                     int nrhs, const mxArray *prhs[]){
  using namespace spblas;

  // Fill csc_view with:
  // - T* values
  // - mwIndex* colptr
  // - mwIndex* rowind
  // - {mwIndex m, mwIndex n} (shape)
  // - mwIndex nnz
  csr_view<T> A(static_cast<T*>(mxGetData(prhs[0])), mxGetJc(prhs[0]),
                mxGetIr(prhs[0]), {m, n}, nnz);

  csr_view<T> B;

    
  // Wrap b in a span of length n
  std::span<T> b(static_cast<T*>(mxGetData(prhs[1])), n);

  // Wrap output y in a span of length m
  std::span<T> x(static_cast<T*>(mxGetData(x_out)), m);

  bool useUpper = true;
  if (nrhs>2) {
    char uplo_in[5];
    int info = mxGetString(prhs[2], uplo_in, 6); 
    if (!strcmp(uplo_in, "lower")) {
      useUpper = false;
    } else {
      if (strcmp(uplo_in, "upper")) {
        mexErrMsgIdAndTxt("SparseBLAS_Mex:WrongUplo",
                          "3rd input must be ""upper"" or ""lower"".");
      }
    } 
  }

  // Store and apply scaling factor alpha, if provided
  T alpha = 1.0;
  if (nrhs==4) {
    alpha = static_cast<T>(mxGetScalar(prhs[2]));
  }
  auto alpha_b= scaled(alpha, b);

  // solve for x:  uplo(A) * x = alpha * b
  // triangular_solve(A, spblas::lower_triangle_t{},
  //                  spblas::explicit_diagonal_t{}, alpha_b, x);
  if (useUpper) {
    triangular_solve(A, spblas::upper_triangle_t{},
                     spblas::explicit_diagonal_t{}, alpha_b, x);
  } else {

    triangular_solve(A, spblas::lower_triangle_t{},
                     spblas::explicit_diagonal_t{}, alpha_b, x);
  }
}

// y = (alpha *) A * x
// prhs[0] = A, prhs[1] = x (optional: prhs[2] = alpha)
// plhs[0] = y
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

  // General input checking
  if (nrhs<2 && nrhs>3) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:WrongNumberOfInputs",
                      "Function needs 2 or 3 inputs.");
  }
  if (nlhs>1) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:WrongNumberOfOutputs",
                      "Function returns only 1 output.");
  }
  if (mxGetClassID(prhs[0]) != mxGetClassID(prhs[1])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:ClassMissmatch",
                      "First and second input must have matching type.");
  }
  if(!mxIsSparse(prhs[0])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:FirstInputNotSparse",
                      "First input must be sparse.");
  }
  if(mxIsSparse(prhs[1])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:SecondInputNotDense",
                      "Second input must be dense.");
  }

  // Reference SparseBLAS can handle inputs with mixed complexity,
  // however, the vendor implementations need all inputs of the same
  // complexity, hence, this example also insists on having matching 
  // complexity.
  if(mxIsComplex(prhs[0]) != mxIsComplex(prhs[1])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:ComplexityMissmatch",
                      "First and second input must have "
                      "matching complexity.");
  }

  // Gather dimensions
  mwIndex m = mxGetM(prhs[0]);
  mwIndex n = mxGetN(prhs[0]);
  mwIndex nnz = mxGetNzmax(prhs[0]);

  // Check dimensions of second input
  if ((mxGetM(prhs[1]) != n) && (mxGetN(prhs[1]) != 1)) {
     mexErrMsgIdAndTxt("SparseBLAS_Mex:InnerDimWrong",
                       "Second input must be column vector of length n, "
                       "i.e., number of columns of first input.");
  }

  // Set output size
  mwIndex dims[2] = {m, 1};

  // Calculate in complex (assume matching complexity for both inputs)
  bool isCmplx = mxIsComplex(prhs[0]);

  // Type dispatch for double, single, or logical
  if (mxIsDouble(prhs[0])) {
    if (isCmplx) {
      plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxCOMPLEX);
      simple_sptrsv_mex<std::complex<double>>(m, n, nnz, plhs[0], nrhs, prhs);
    } else {
      plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
      simple_sptrsv_mex<double>(m, n, nnz, plhs[0], nrhs, prhs);
    }
  } else if (mxIsSingle(prhs[0])) {
    if (isCmplx) {
      plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxCOMPLEX);
      simple_sptrsv_mex<std::complex<float>>(m, n, nnz, plhs[0], nrhs, prhs);
    } else {
      plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
      simple_sptrsv_mex<float>(m, n, nnz, plhs[0], nrhs, prhs);
    }
  } else {
    mxAssert(mxIsLogical(prhs[0]), "Invalid data type");
    plhs[0] = mxCreateNumericArray(2, dims, mxLOGICAL_CLASS, mxREAL);
    simple_sptrsv_mex<bool>(m, n, nnz, plhs[0], nrhs, prhs);
  }
}

// Compile from within MATLAB via:
// mex simple_spmv_mex.cpp -R2018a -I{PATH_TO_SparseBLAS_INCLUDE} 'CXXFLAGS=$CFLAGS -fobjc-arc -stdlib=libc++ -std=c++20'
//
// Add '-g' to build in Debug mode if needed (activates asserts)