# tatami for TileDB matrices

![Unit tests](https://github.com/tatami-inc/tatami_tiledb/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_tiledb/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/tatami-inc/tatami_tiledb/branch/master/graph/badge.svg?token=Z189ORCLLR)](https://codecov.io/gh/tatami-inc/tatami_tiledb)

## Overview

This repository implements [**tatami**](https://github.com/tatami-inc/tatami) bindings for [TileDB](https://github.com/tiledb-inc/tiledb)-backed matrices,
allowing random access without loading the entire dataset into memory.
Any matrix stored as a 2-dimensional TileDB array (dense or sparse) can be represented as a `tatami::Matrix` and consumed by **tatami**-compatible applications.

## Quick start

**tatami_tiledb** is a header-only library, so it can be easily used by just `#include`ing the relevant source files:

```cpp
#include "tatami_tiledb/tatami_tiledb.hpp"

tatami_tiledb::TileDbDenseMatrix<double, int> dense_mat("some_dir", "attr_name");
tatami_tiledb::TileDbSparseMatrix<double, int> sparse_mat("some_dir", "attr_name");
```

If the dense/sparse nature is not known beforehand, we can use the `make_TileDbMatrix()` function to decide for us instead:

```cpp
auto some_mat = tatami_tiledb::make_TileDbMatrix("some_dir", "attr_name");
```

Advanced users can fiddle with the options, in particular the size of the tile cache.
For example, we could force the matrix to use no more than 200 MB of memory in the cache:

```cpp
tatami_tiledb::TileDbOptions opt;
opt.maximum_cache_size = 200000000;
opt.require_minimum_cache = false; // don't allow the cache to automatically expand.

auto some_mat2 = tatami_tiledb::make_TileDbMatrix("some_dir", "attr_name", opt);
```

Check out the [reference documentation](https://tatami-inc.github.io/tatami_tiledb) for more details.

## Building projects

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_tiledb
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(tatami_tiledb)
```

Then you can link to **tatami_tiledb** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe tatami_tiledb)

# For libaries
target_link_libraries(mylib INTERFACE tatami_tiledb)
```

You'll also need to link to the TileDB library yourself (version 2.15 or higher).
**tatami_tiledb** does not place any restrictions on the source of the TileDB library;
in the simplest case, we just download the latest [release](https://github.com/TileDB-Inc/TileDB/releases), unpack it in some `tiledb/` directory, and link to it as shown below.
(Proper installation instructions for TileDB can be found in their [documentation](https://docs.tiledb.com/main/how-to/installation/pre-built-packages).)
    
```cmake
find_package(TileDB PATHS tiledb)
target_link_libraries(myexe TileDB::tiledb_shared)
```

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This will also require the core [**tatami**](https://github.com/tatami-inc/tatami) library.
