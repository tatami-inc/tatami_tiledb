#ifndef TATAMI_TILEDB_MAKE_TILEDBMATRIX_HPP
#define TATAMI_TILEDB_MAKE_TILEDBMATRIX_HPP

#include "TiledbDenseMatrix.hpp"
#include "TiledbSparseMatrix.hpp"
#include <memory>
#include <tiledb/tiledb.h>

/**
 * @file make_TiledbMatrix.hpp
 * @brief Make a TileDB-backed matrix.
 */

namespace tatami_tiledb {

/**
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam transpose_ Whether to transpose the on-disk data upon loading.
 * By default, this is `false`, so the first dimension corresponds to rows and the second dimension corresponds to columns.
 * If `true`, the first dimension corresponds to columns and the second dimension corresponds to rows.
 *
 * @param uri File path (or some other appropriate location) of the TileDB array.
 * @param attribute Name of the attribute containing the data of interest.
 * @param options Further options.
 *
 * Create a `tatami::Matrix` representation from a 2-dimensional dense or sparse TileDB array.
 * This function will automatically choose the most appropriate representation based on the storage format.
 * For dense on-disk arrays, a `TiledbDenseMatrix` is constructed; otherwise a `TiledbSparseMatrix` is returned.
 */
template<typename Value_, typename Index_, bool transpose_ = false>
std::shared_ptr<tatami::Matrix<Value_, Index_> > make_TiledbMatrix(std::string uri, std::string attribute, const TiledbOptions& options) {
    tiledb::Context ctx;
    tiledb::ArraySchema schema(ctx, uri);

    std::shared_ptr<tatami::Matrix<Value_, Index_> > output;
    auto atype = schema.array_type();
    if (atype == TILEDB_SPARSE) {
        output.reset(new TiledbSparseMatrix<Value_, Index_, transpose_>(schema, std::move(uri), std::move(attribute), options));
    } else if (atype == TILEDB_DENSE) {
        output.reset(new TiledbDenseMatrix<Value_, Index_, transpose_>(schema, std::move(uri), std::move(attribute), options));
    } else {
        throw std::runtime_error("unknown TileDB array type that is neither dense nor sparse");
    }

    return output; 
}

/**
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam transpose_ Whether to transpose the on-disk data upon loading.
 * By default, this is `false`, so the first dimension corresponds to rows and the second dimension corresponds to columns.
 * If `true`, the first dimension corresponds to columns and the second dimension corresponds to rows.
 *
 * @param uri File path (or some other appropriate location) of the TileDB array.
 * @param attribute Name of the attribute containing the data of interest.
 *
 * Create a `tatami::Matrix` representation from a 2-dimensional dense or sparse TileDB array.
 * Unlike its overload, this function will use the default settings in `TiledbOptions`.
 */
template<typename Value_, typename Index_, bool transpose_ = false>
std::shared_ptr<tatami::Matrix<Value_, Index_> > make_TiledbMatrix(std::string uri, std::string attribute) {
    return make_TiledbMatrix<Value_, Index_, transpose_>(std::move(uri), std::move(attribute), TiledbOptions());
}

}

#endif
