# Sparse BLAS Reference
This repository holds a prototype Sparse BLAS implementation.

## Views
The interface is currently built around views.  Views are non-owning, lightweight objects.
They reference data, but they do not own it.  Views make it easy to use a library without
having to copy data in and out of it.

```cpp
  using namespace spblas;

  // User owns their own data
  int m, n, nnz = ...;
  std::vector<float> values = ...;
  std::vector<int> rowptr = ...;
  std::vector<int> colind = ...;

  // They can create a view
  csr_view<float, int> a(values, rowptr, colind, {m, n}, nnz);

  // We can then perform an operation. No copies required.

  multiply(a, b, c);
```

## Multi-Phase Operations
Multi-phase operations allow implementations to perform inspector-executor style optimizations.
In the inspect phase, an implementation can optimize the format of a matrix or provide other
metadata that will allow the operation to be performed more efficiently.  Having an inspect
phase also allows users to fully control data layout and allocation by having them allocate more
space for the output if needed in a sparse-times-sparse operation.

```cpp
  // Call `inspect` to perform optimization phase.
  // Implementations may apply arbitrary optimizations here.
  auto info = multiply_inspect(a, b, c /*, optimization hints */);

  // If c is sparse, I may need to allocate more memory for the output.
  std::vector<float> c_values(info.result_nnz());
  std::vector<spblas::index_t> c_rowptr(info.result_shape()[0] + 1);
  std::vector<spblas::index_t> c_colind(info.result_nnz());

  c.update(c_values, c_rowptr, c_colind);

  // I can then execute the operation using `info`
  multiply_execute(info, a, b, c);
```

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
[ 44%] Building CXX object examples/CMakeFiles/simple_spmv.dir/simple_spmv.cpp.o
[ 55%] Building CXX object examples/CMakeFiles/simple_spmm.dir/simple_spmm.cpp.o
[ 66%] Building CXX object examples/CMakeFiles/simple_spgemm.dir/simple_spgemm.cpp.o
. . .
[ 77%] Linking CXX executable simple_spmv
[ 77%] Built target simple_spmv
1 warning generated.
[ 88%] Linking CXX executable simple_spmm
1 warning generated.
[100%] Linking CXX executable simple_spgemm
[100%] Built target simple_spmm
[100%] Built target simple_spgemm
brock@slothius:~/src/spblas-reference$
```

### Selecting Compiler
The project requires a compiler C++20 support and some C++23 support.  To
build with a different compiler than the default, provide the desired compiler
to CMake using the environment variable `CXX`.

```bash
brock@slothius:~/src/spblas-reference$ CXX=g++-13 cmake -B build
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
# Linking errr when compiling with GCC on recent versions of Mac OS without
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
make[2]: *** [examples/simple_spmv] Error 1
make[1]: *** [examples/CMakeFiles/simple_spmv.dir/all] Error 2
make: *** [all] Error 2
```

### C++ Standard Library Features
The internal library implementation currently depends upon some C++23 features
that some compilers may not support.  To work around this, the CMake will
automatically detect if you are compiling with a sufficiently new version of
the C++ standard library.  If it is missing required features, it will
automatically download the [range-v3](https://github.com/ericniebler/range-v3)
library and use it instead of the standard library's implementation of ranges.

### Issues
If you have any issues building the library, please create a GitHub issue in
this repository.