// Includes from SparseBLAS
#include <spblas/spblas.hpp>

// Includes for MEX
#include <matrix.h>
#include <mex.h>

// General includes
#include <complex> // Support complex inputs

template <typename T>
void spmmDriver(mxArray* mxY, const mxArray* mxA, const mxArray* mxX,
                const mxArray* mxAlpha) {
  // Gather dimensions
  mwIndex m = mxGetM(mxA);
  mwIndex k = mxGetN(mxA);
  mwIndex n = mxGetN(mxX);

  // Fill csc_view with:
  // - T* values
  // - mwIndex* colptr
  // - mwIndex* rowind
  // - {mwIndex m, mwIndex k} (shape)
  // - mwIndex nnz
  spblas::csc_view<const T, mwIndex> A(static_cast<const T*>(mxGetData(mxA)),
                                       mxGetJc(mxA), mxGetIr(mxA), {m, k},
                                       mxGetJc(mxA)[k]);

  // Wrap X in an mdspan of size k-by-n
  spblas::mdspan_col_major<T, mwIndex> X(static_cast<T*>(mxGetData(mxX)), k, n);

  // Wrap output Y in an mdspan of size m-by-n
  spblas::mdspan_col_major<T, mwIndex> Y(static_cast<T*>(mxGetData(mxY)), m, n);

  // Store and apply scaling factor alpha, if provided and not empty
  T alpha = T(1);
  if (mxAlpha != nullptr && !mxIsEmpty(mxAlpha)) {
    // We don't use mxGetScalar as it doesn't work for complex
    alpha = *(static_cast<T*>(mxGetData(mxAlpha)));
  }
  auto alpha_A = spblas::scaled(alpha, A);

  // Y = (alpha * A) * X
  spblas::multiply(alpha_A, X, Y);
}

// Y = (alpha *) A * X
// prhs[0] = A, prhs[1] = X (optional: prhs[2] = alpha)
// plhs[0] = Y
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

  // General input checking
  if (nrhs < 2 || nrhs > 3) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:WrongNumberOfInputs",
                      "Function needs 2 or 3 inputs.");
  }
  if (nlhs > 1) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:WrongNumberOfOutputs",
                      "Function returns only 1 output.");
  }
  if (mxGetClassID(prhs[0]) != mxGetClassID(prhs[1]) ||
      ((nrhs == 3) && mxGetClassID(prhs[1]) != mxGetClassID(prhs[2]))) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:ClassMismatch",
                      "All inputs must have matching type.");
  }
  if (!mxIsDouble(prhs[0]) && !mxIsSingle(prhs[0])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:NonFloat",
                      "All inputs must be single or double.");
  }
  if(!mxIsSparse(prhs[0])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:FirstInputNotSparse",
                      "First input must be sparse.");
  }
  if (mxIsSparse(prhs[1])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:SecondInputNotDense",
                      "Second input must be dense.");
  }

  // Reference SparseBLAS can handle inputs with mixed complexity,
  // however, the vendor implementations need all inputs of the same
  // complexity, hence, this example also insists on having matching
  // complexity.
  if (mxIsComplex(prhs[0]) != mxIsComplex(prhs[1]) ||
      ((nrhs == 3) && mxIsComplex(prhs[1]) != mxIsComplex(prhs[2]))) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:ComplexityMismatch",
                      "All inputs must have matching complexity.");
  }

  // Gather dimensions
  mwIndex m = mxGetM(prhs[0]);
  mwIndex k = mxGetN(prhs[0]);
  mwIndex n = mxGetN(prhs[1]);

  // Check dimensions of second input
  if (mxGetM(prhs[1]) != k) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:InnerDimWrong",
                      "Second input must be an array with k rows, "
                      "i.e., number of columns of first input.");
  }

  // Calculate in complex (we check for matching complexity above)
  bool isCmplx = mxIsComplex(prhs[0]);

  // Type dispatch for double or single, each as real or complex flavor
  if (mxIsDouble(prhs[0])) {
    if (isCmplx) {
      plhs[0] = mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, mxCOMPLEX);
      spmmDriver<std::complex<double>>(plhs[0], prhs[0], prhs[1],
                                       nrhs == 3 ? prhs[2] : nullptr);
    } else {
      plhs[0] = mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, mxREAL);
      spmmDriver<double>(plhs[0], prhs[0], prhs[1],
                         nrhs == 3 ? prhs[2] : nullptr);
    }
  } else {
    mxAssert(mxIsSingle(prhs[0]), "Invalid data type");
    if (isCmplx) {
      plhs[0] = mxCreateNumericMatrix(m, n, mxSINGLE_CLASS, mxCOMPLEX);
      spmmDriver<std::complex<float>>(plhs[0], prhs[0], prhs[1],
                                      nrhs == 3 ? prhs[2] : nullptr);
    } else {
      plhs[0] = mxCreateNumericMatrix(m, n, mxSINGLE_CLASS, mxREAL);
      spmmDriver<float>(plhs[0], prhs[0], prhs[1],
                        nrhs == 3 ? prhs[2] : nullptr);
    }
  }
}

// Compile from within MATLAB via:
// mex simple_spmm_mex.cpp -R2018a -I{PATH_TO_SparseBLAS_INCLUDE}
// 'CXXFLAGS=$CFLAGS -std=c++20'
//
// Add '-g' to build in Debug mode if needed (activates asserts)
