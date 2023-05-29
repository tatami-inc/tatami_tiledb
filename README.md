# tatami for TileDB matrices

![Unit tests](https://github.com/tatami-inc/tatami_tiledb/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_tiledb/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/tatami-inc/tatami_tiledb/branch/master/graph/badge.svg?token=Z189ORCLLR)](https://codecov.io/gh/tatami-inc/tatami_tiledb)

This repository implements [**tatami**](https://github.com/tatami-inc/tatami) bindings for [TileDB](https://github.com/tiledb-inc/tiledb)-backed matrices,
allowing random access without loading the entire dataset into memory.
Matrices can be stored as 2-dimensional TileDB datasets:

```cpp
#include "tatami_tiledb/tatami_tiledb.hpp"

tatami_hdf5::TileDbDenseMatrix<double, int> dense_mat("some_file.h5", "dataset_name");
```

Check out the [reference documentation](https://tatami-inc.github.io/tatami_tiledb) for more details.
