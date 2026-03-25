function build_mex_APIs(incl_path, verbose, debug)
%BUILD_MEX_APIS - Function to build all available mex APIs
%
% First input must be the path to the SparseBLAS INCLUDE folder.
% Second and third are optional logical inputs to activate VERBOSE or and
% DEBUG mode.
%
%  The calling syntaxes are:
% 		build_mex_APIs("PATH_TO_SparseBLAS_INCLUDE")
% 		build_mex_APIs("PATH_TO_SparseBLAS_INCLUDE", true)
% 		build_mex_APIs("PATH_TO_SparseBLAS_INCLUDE", false, true)

% Set default options
opts = {['-I' incl_path], "-O", "-R2018a", "CXXFLAGS=$CFLAGS -std=c++20"};

% Parse optional VERBOSE option
if nargin > 1 && verbose
    opts = [opts, "-v"];
end

% Parse optional DEBUG option
if nargin > 2 && debug
    opts = [opts, "-g"];
end

% Compile all APIs
mex("simple_spmv_mex.cpp", opts{:});
mex("simple_spmm_mex.cpp", opts{:});
end
