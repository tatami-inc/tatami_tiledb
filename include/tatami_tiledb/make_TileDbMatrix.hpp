#ifndef TATAMI_TILEDB_MAKE_TILEDBMATRIX_HPP
#define TATAMI_TILEDB_MAKE_TILEDBMATRIX_HPP

#include "TileDbDenseMatrix.hpp"
#include "TileDbSparseMatrix.hpp"
#include <memory>
#include <tiledb/tiledb.h>

/**
 * @file make_TileDbMatrix.hpp
 * @brief Make a TileDB-backed matrix.
 */

namespace tatami_tiledb {

/**
 * @brief Make a TileDB-backed matrix.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam transpose_ Whether to transpose the on-disk data upon loading.
 * By default, this is `false`, so the first dimension corresponds to rows and the second dimension corresponds to columns.
 * If `true`, the first dimension corresponds to columns and the second dimension corresponds to rows.
 *
 * As its name suggests, this function creates a numeric matrix stored as a 2-dimensional dense or sparse TileDB array.
 * It will automatically choose the most appropriate representation based on the storage format.
 * For dense on-disk arrays, a `TileDbDenseMatrix` is constructed; otherwise a `TileDbSparseMatrix` is returned.
 */
template<typename Value_, typename Index_, bool transpose_ = false>
std::shared_ptr<tatami::Matrix<Value_, Index_> > make_TileDbMatrix(std::string uri, std::string attribute, const TileDbOptions& options) {
    tiledb::Context ctx;
    tiledb::ArraySchema schema(ctx, uri);

    std::shared_ptr<tatami::Matrix<Value_, Index_> > output;
    auto atype = schema.array_type();
    if (atype == TILEDB_SPARSE) {
        output.reset(new TileDbSparseMatrix<Value_, Index_, transpose_>(schema, std::move(uri), std::move(attribute), options));
    } else if (atype == TILEDB_DENSE) {
        output.reset(new TileDbDenseMatrix<Value_, Index_, transpose_>(schema, std::move(uri), std::move(attribute), options));
    } else {
        throw std::runtime_error("unknown TileDB array type that is neither dense nor sparse");
    }

    return output; 
}

template<typename Value_, typename Index_, bool transpose_ = false>
std::shared_ptr<tatami::Matrix<Value_, Index_> > make_TileDbMatrix(std::string uri, std::string attribute) {
    return make_TileDbMatrix<Value_, Index_, transpose_>(std::move(uri), std::move(attribute), TileDbOptions());
}

}

#endif
