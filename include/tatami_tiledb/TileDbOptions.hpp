#ifndef TATAMI_TILEDB_OPTIONS_HPP
#define TATAMI_TILEDB_OPTIONS_HPP

/**
 * @file TileDbOptions.hpp
 * @brief Options for TileDB extraction.
 */

namespace tatami_tiledb {

/**
 * @brief Options for TileDB extraction.
 */
struct TileDbOptions {
    /**
     * Size of the in-memory cache in bytes.
     *
     * We cache all tiles required to read a row/column during a `tatami::DenseExtractor::fetch()` or `tatami::SparseExtractor::Fetch()` call.
     * This allows us to re-use the cached tiles when adjacent rows/columns are requested, rather than re-reading them from disk.
     *
     * Larger caches improve access speed at the cost of memory usage.
     * Small values may be ignored if `require_minimum_cache` is `true`.
     */
    size_t cache_size = 100000000;

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `cache_size`.
     * This minimum is chosen to ensure that the same tiles are not read repeatedly when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;
};

}

#endif
