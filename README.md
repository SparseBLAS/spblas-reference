# Sparse BLAS Reference
This repository holds a prototype Sparse BLAS implementation.

## Matrix Wrappers
The interface is built around non-owning, lightweight matrix wrappers. They reference data,
but they do not own it, so a library can be used without copying data in and out of it.

```cpp
  using namespace spblas;

  // User owns their own data
  int m, n, nnz = ...;
  std::vector<float> values = ...;
  std::vector<int> rowptr = ...;
  std::vector<int> colind = ...;

  // They can create a wrapper
  csr_wrapper<float, int> a(values, rowptr, colind, {m, n}, nnz);

  // Every operation takes an operation_state, which can carry library data
  // between calls, and explicit scaling factors. No copies required.
  operation_state state;
  multiply(state, alpha, a, x, y);              // y = alpha * A * x
  multiply(state, alpha, a, x, beta, z, y);     // y = alpha * A * x + beta * z
```

Views such as `view::transpose(a)`, `view::conjugate(a)` and
`view::triangle(a, view::uplo::lower)` can be passed wherever a matrix is expected, and a
`matrix_opt(a)` lets the library keep optimization data for a matrix across calls.

The reference and oneMKL Inspector-Executor backends implement this API, which follows
the SparseBLAS position paper. The other backends still provide the earlier API
(`csr_view`, `scaled()`, `operation_info_t`) and are being migrated.

## Multi-Phase Operations
Multi-phase operations allow implementations to perform inspector-executor style optimizations.
In the optional inspect phase, an implementation can optimize the format of a matrix or provide other
metadata that will allow the operation to be performed more efficiently.  For operations whose
sparse output is not known in advance, such as sparse-times-sparse, the user allocates the output
once the library has computed its size, so users fully control data layout and allocation.

```cpp
  // Optional: implementations may apply arbitrary optimizations here.
  multiply_inspect(state, alpha, a, b, c);

  // Compute the size of the output C, which so far has only a shape:
  // csr_wrapper<float, int> c(m, n);
  multiply_compute(state, alpha, a, b, c);

  // Allocate C's arrays.
  std::vector<float> c_values(state.get_result_nnz());
  std::vector<int> c_rowptr(m + 1);
  std::vector<int> c_colind(state.get_result_nnz());
  c.update(c_values, c_rowptr, c_colind);

  // Fill C.
  multiply_fill(state, alpha, a, b, c);
```

A three-stage variant, `multiply_compute_symbolic` / `multiply_fill_symbolic` /
`multiply_numeric`, separates the sparsity pattern from the values so that the pattern can be
reused while the values of `a` and `b` change.

## Building
The project has a CMake build that should work out-of-the-box on most systems.

```bash
brock@slothius:~/src/spblas-reference$ cmake -B build
-- The C compiler identification is AppleClang 15.0.0.15000040
-- The CXX compiler identification is AppleClang 15.0.0.15000040
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /Library/Developer/CommandLineTools/usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /Library/Developer/CommandLineTools/usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for __cpp_lib_ranges
-- Looking for __cpp_lib_ranges - found
-- Looking for __cpp_lib_ranges_zip
-- Looking for __cpp_lib_ranges_zip - not found
-- NOTE: Standard library does not include ranges and/or std::views::zip. Using range-v3.
. . .
-- Configuring done (33.7s)
-- Generating done (0.1s)
-- Build files have been written to: /Users/bbrock/src/spblas-reference/build
brock@slothius:~/src/spblas-reference$ cd build/examples
brock@slothius:~/src/spblas-reference$ make
[ 11%] Building CXX object _deps/fmt-build/CMakeFiles/fmt.dir/src/format.cc.o
[ 22%] Building CXX object _deps/fmt-build/CMakeFiles/fmt.dir/src/os.cc.o
[ 33%] Linking CXX static library libfmt.a
[ 33%] Built target fmt
[ 44%] Building CXX object examples/CMakeFiles/spmv_csr.dir/spmv_csr.cpp.o
[ 55%] Building CXX object examples/CMakeFiles/spmm_csr.dir/spmm_csr.cpp.o
[ 66%] Building CXX object examples/CMakeFiles/spgemm_csr.dir/spgemm_csr.cpp.o
. . .
[ 77%] Linking CXX executable spmv_csr
[ 77%] Built target spmv_csr
1 warning generated.
[ 88%] Linking CXX executable spmm_csr
1 warning generated.
[100%] Linking CXX executable spgemm_csr
[100%] Built target spmm_csr
[100%] Built target spgemm_csr
brock@slothius:~/src/spblas-reference$
```

### Selecting Compiler
The project requires a compiler with C++20 and some C++23 support.  To
build with a different compiler than the system default, provide the desired
compiler to CMake using the environment variable `CXX`.

```bash
brock@slothius:~/src/spblas-reference$ CXX=g++-13 cmake -B build
```

### Compiling with a Vendor Backend
A vendor backend can be enabled by passing in an `-DENABLE_{BACKEND}=ON` switch
to `cmake`.  Currently, oneMKL (SYCL and Inspector-Executor C APIs), ArmPL, AOCL-Sparse,
rocSPARSE and cuSPARSE are the supported vendor backends.

### Compiling with oneMKL
The most straightforward way to build with oneMKL is by installing the [oneAPI
toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
You can then use the built-in `icpx` compiler to build with the oneMKL backend.

```bash
# Import oneAPI toolkit into Bash environment.
brock@slothius:~/src/spblas-reference$ source /opt/intel/oneapi/setvars.sh
# Compile with icpx and oneMKL backend.
brock@slothius:~/src/spblas-reference$ CXX=icpx cmake -B build -DENABLE_ONEMKL_SYCL=ON
```

### Compiling with oneMKL Inspector-Executor Sparse BLAS
The `ENABLE_ONEMKL_IESPBLAS` backend uses oneMKL's Inspector-Executor Sparse BLAS C API on the
CPU and works with `icpx` or GCC. It uses oneMKL's LP64 interface (`MKL_INT` is a 32-bit `int`);
matrices with 64-bit indices are supported through oneMKL's `_64` routines, which need oneMKL
2024.3 or newer. The oneMKL
threading layer is chosen to match the compiler (`intel_thread` for `icpx`, `gnu_thread` for
GCC) unless `MKL_THREADING` is set.

```bash
# Import oneAPI toolkit into Bash environment.
brock@slothius:~/src/spblas-reference$ source /opt/intel/oneapi/setvars.sh
# Compile with the oneMKL Inspector-Executor backend.
brock@slothius:~/src/spblas-reference$ CXX=icpx cmake -B build -DENABLE_ONEMKL_IESPBLAS=ON
```

### Compiling with ArmPL
In order to compile with ArmPL, the library must be installed and the
environment variable `ARMPL_DIR` set.

```bash
# Set the environemnt variable `ARMPL_DIR` to the install location of ArmPL.
brock@slothius:~/src/spblas-reference$ export ARMPL_DIR=/opt/arm/armpl_24.10_flang-new_clang_19
# Compile with the ArmPL backend.
brock@slothius:~/src/spblas-reference$ cmake -B build -DENABLE_ARMPL=ON
```

### Compiling with rocSPARSE
In order to compile with rocSPARSE, ROCm must be installed and the install
location of ROCm added to `CMAKE_PREFIX_PATH`.  Your package manager will likely
take care of this for you, but you can also manually specify the location if you
have ROCm installed in a non-standard location.

```bash
# Explicitly set the location of ROCm using `CMAKE_PREFIX_PATH`.
brock@slothius:~/src/spblas-reference$ cmake -B build -DENABLE_ROCSPARSE=ON -DCMAKE_PREFIX_PATH=/opt/rocm-6.1.2
```

### Compiling with cuSPARSE
In order to compile with cuSPARSE, CUDA must be installed and the install
location of CUDA added to `CMAKE_PREFIX_PATH`.  Your package manager will likely
take care of this for you, but you can also manually specify the location if you
have CUDA installed in a non-standard location.

```bash
# Explicitly set the location of CUDA using `CMAKE_PREFIX_PATH`.
brock@slothius:~/src/spblas-reference$ cmake -B build -DENABLE_CUSPARSE=ON -DCMAKE_PREFIX_PATH=/usr/local/cuda-12.6
```

#### Compiling with GCC on Mac OS
There is a known linking issue when compiling with GCC on recent versions of
Mac OS.  This will cause a link error inside of `ld::AtomPlacement::findAtom()`.
To workaround this error, you must use the "legacy linking" mode.  You can guide
CMake to use legacy linking mode by adding the flag `-DCMAKE_EXE_LINKER_FLAGS="-Wl,-ld_classic"`
to your CMake build.

```bash
# Setting up CMake to compile with GCC 13 with legacy linking mode (Mac OS only).
brock@slothius:~/src/spblas-reference$ CXX=g++-13 cmake -B build -DCMAKE_EXE_LINKER_FLAGS="-Wl,-ld_classic"
```

```bash
# Linking error when compiling with GCC on recent versions of Mac OS without
# enabling legacy linking.
0  0x1002ff648  __assert_rtn + 72
1  0x100233fac  ld::AtomPlacement::findAtom(unsigned char, unsigned long long, ld::AtomPlacement::AtomLoc const*&, long long&) const + 1204
2  0x100249924  ld::InputFiles::SliceParser::parseObjectFile(mach_o::Header const*) const + 15164
3  0x1002543f8  ld::InputFiles::SliceParser::parse() const + 2468
4  0x100256e30  ld::InputFiles::parseAllFiles(void (ld::AtomFile const*) block_pointer)::$_7::operator()(unsigned long, ld::FileInfo const&) const + 420
5  0x182d9b950  _dispatch_client_callout2 + 20
6  0x182daeba0  _dispatch_apply_invoke + 176
7  0x182d9b910  _dispatch_client_callout + 20
8  0x182dad3cc  _dispatch_root_queue_drain + 864
9  0x182dada04  _dispatch_worker_thread2 + 156
10  0x182f450d8  _pthread_wqthread + 228
ld: Assertion failed: (resultIndex < sectData.atoms.size()), function findAtom, file Relocations.cpp, line 1336.
collect2: error: ld returned 1 exit status
make[2]: *** [examples/spmv_csr] Error 1
make[1]: *** [examples/CMakeFiles/spmv_csr.dir/all] Error 2
make: *** [all] Error 2
```

### C++ Standard Library Features
The internal library implementation currently depends upon C++23 features
that some compilers may not support.  To work around this, the CMake build
system will automatically detect whether you are compiling with a sufficiently
new version of the C++ standard library.  If it is missing required features,
it will automatically download the [range-v3](https://github.com/ericniebler/range-v3)
library and use it instead of your standard library's implementation of ranges.

## Contributing
You're welcome to contribute to this project!  Feel free to open an issue or PR.

### Running `pre-commit` Checks
This project uses the tool `pre-commit` to run `clang-format` and a couple of other formatting
tools.  Please use them to ensure your code formatting is correct before submitting a PR.  If
you forget, you'll notice the CI failing on your PR.

```bash
brock@slothius:~/src/spblas-reference$ python3 -m venv venv
brock@slothius:~/src/spblas-reference$ source venv/bin/activate
brock@slothius:~/src/spblas-reference$ pip3 install -r requirements.txt
brock@slothius:~/src/spblas-reference$ pre-commit run --all
clang-format.............................................................Passed
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
mixed line ending........................................................Passed
check for added large files..............................................Passed
```

If you notice that none of the checks pass, that means `clang-format` or another format tool
has fixed your code formatting.  Go ahead and perform a `git add` and `git commit` to update
before pushing.

### Issues
If you have any issues building the library, please create a GitHub issue in
this repository.
