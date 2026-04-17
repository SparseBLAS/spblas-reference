% simple_spmm_mex - Sparse matrix times dense matrix multiplication
%  simple_smpm_mex.c - example in MATLAB External Interfaces
%
%  Multiplies a (potentially scaled by a scalar alpha) sparse MxK matrix
%  with a dense KxN matrix and outputs a dense MxN matrix:
%       Y = A * X  or  Y = alpha * A * X
%
%  The calling syntaxes are:
% 		Y = simple_smpv_mex(A, X)
% 		Y = simple_smpv_mex(A, X, alpha)
%
%  The following restrictions apply:
%    * A must be sparse
%    * X must be dense
%    * Number of columns in A and rows in X must match
%    * All inputs must have the same data type and complexity
%
%  This is a MEX-file for MATLAB.
