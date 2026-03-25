// Includes from SparseBLAS
#include <spblas/spblas.hpp>

// Includes for MEX
#include <matrix.h>
#include <mex.h>

// General includes
#include <complex> // Support complex inputs
#include <iostream>
// C = A * B
// prhs[0] = A, prhs[1] = B
// plhs[0] = C
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
  
  // General input checking
  if (nrhs!=2) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:WrongNumberOfInputs",
                      "Function needs 2 inputs.");
  }
  if (nlhs>1) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:WrongNumberOfOutputs",
                      "Function returns only 1 output.");
  }
  if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:UnsupportedClass",
                      "Only double inputs supported.");
  }
  if(!mxIsSparse(prhs[0]) || !mxIsSparse(prhs[1])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:InputsNotSparse",
                      "All inputs must be sparse.");
  }
  

  /*
  // Reference SparseBLAS can handle inputs with mixed complexity,
  // however, the vendor implementations need all inputs of the same
  // complexity, hence, this example also insists on having matching 
  // complexity.
  */
  // For now, allow only real to avoid switch-yards.  
  if(mxIsComplex(prhs[0]) || mxIsComplex(prhs[1])) {
    mexErrMsgIdAndTxt("SparseBLAS_Mex:NonReal",
                      "Only real valued inputs supported.");
  }

  // Gather dimensions
  mwIndex m = mxGetM(prhs[0]);
  mwIndex k = mxGetN(prhs[0]);
  mwIndex n = mxGetN(prhs[1]);
  mwIndex nnzA = mxGetNzmax(prhs[0]);
  mwIndex nnzB = mxGetNzmax(prhs[1]);

  // Check dimensions of second input
  if (mxGetM(prhs[1]) != k) {
     mexErrMsgIdAndTxt("SparseBLAS_Mex:InnerDimWrong",
                       "Second input must be an array with k rows, "
                       "i.e., number of columns of first input.");
  }

  // Fill csc_view for A with:
  // - double* values
  // - mwIndex* colptr
  // - mwIndex* rowind
  // - {mwIndex m, mwIndex k} (shape)
  // - mwIndex nnzA
  spblas::csc_view<double> A(static_cast<double*>(mxGetData(prhs[0])),
                     mxGetJc(prhs[0]), mxGetIr(prhs[0]), {m, k}, nnzA);

  // Fill csc_view for B with:
  // - double* values
  // - mwIndex* colptr
  // - mwIndex* rowind
  // - {mwIndex k, mwIndex n} (shape)
  // - mwIndex nnzB
  spblas::csc_view<double> B(static_cast<double*>(mxGetData(prhs[1])),
                     mxGetJc(prhs[1]), mxGetIr(prhs[1]), {k, n}, nnzB);

  // Placeholder csc_view for C with:
  // - {mwIndex m, mwIndex n} (shape)
  spblas::csc_view<double> C(nullptr, nullptr, nullptr, {m, n}, 0);

  // C = A * B - Compute stage
  auto info = multiply_compute(A, B, C);

  // Allocate mxArray for output C and pre-computed nnzC
  mwIndex nnzC = info.result_nnz();
  plhs[0] = mxCreateSparse(m, n, nnzC, mxCOMPLEX);
  
  // Update csc_view for C with values, colptr, and rowind of plhs[0],
  // using spans to ensure correct length of the arrays
  std::span<double> C_values(static_cast<double*>(mxGetData(plhs[0])), nnzC);
  std::span<mwIndex> C_colptr(mxGetJc(plhs[0]), n+1);  
  std::span<mwIndex> C_rowind(mxGetIr(plhs[0]), nnzC);  
  C.update(C_values, C_colptr, C_rowind);
    
  // C = A * B - Fill stage
  multiply_fill(info, A, B, C);
}

// Compile from within MATLAB via:
// mex simple_spgemm_mex.cpp -R2018a -I{PATH_TO_SparseBLAS_INCLUDE} 'CXXFLAGS=$CFLAGS -fobjc-arc -stdlib=libc++ -std=c++20'
//
// Add '-g' to build in Debug mode if needed (activates asserts)
