% simple_spmv_mex - Sparse matrix times dense vector multiplication
%  simple_smpv_mex.c - example in MATLAB External Interfaces
%
%  Multiplies a (potentially scaled by a scalar alpha) sparse MxN matrix
%  with a dense Nx1 column vector and outputs a dense Mx1 column vector:
%       y = A * x  or  y = alpha * A * x
%
%  The calling syntaxes are:
% 		y = simple_smpv_mex(A, x)
% 		y = simple_smpv_mex(A, x, alpha)
%
%  The following restrictions apply:
%    * A must be sparse
%    * x must be dense column vector
%    * Number of columns in A and rows in x must match
%    * All inputs must have the same data type and complexity
%
%  This is a MEX-file for MATLAB.
