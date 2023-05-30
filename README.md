# tatami for TileDB matrices

![Unit tests](https://github.com/tatami-inc/tatami_tiledb/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_tiledb/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/tatami-inc/tatami_tiledb/branch/master/graph/badge.svg?token=Z189ORCLLR)](https://codecov.io/gh/tatami-inc/tatami_tiledb)

This repository implements [**tatami**](https://github.com/tatami-inc/tatami) bindings for [TileDB](https://github.com/tiledb-inc/tiledb)-backed matrices,
allowing random access without loading the entire dataset into memory.
Any matrices stored as 2-dimensional TileDB datasets can be represented as a `tatami::Matrix`:

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
