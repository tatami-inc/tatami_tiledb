#ifndef TATAMI_TILEDB_TILEDB_MATRIX_HPP
#define TATAMI_TILEDB_TILEDB_MATRIX_HPP

#include "tatami/tatami.hpp"
#include <string>

namespace tatami_tiledb {

template<bool transpose_, typename Value_, typename Index_>
class TileDbMatrix : public tatami::Matrix<Value_, Index_> {
public:
    TileDbMatrix(std::string uri, std::string attribute) : location(std::move(uri)), attr(std::move(attribute)) {{
        tiledb::Context ctx;
        tiledb::ArraySchema schema(ctx, uri);

        if (!schema.has_attribute(attr)) {
            throw std::runtime_error("no attribute '" + attr + "' is present in the TileDB array at '" + uri + "'");
        }

        sparse_internal = schema.array_type() == tiledb::TILEDB_SPARSE;

        tiledb::Domain domain = schema.domain();
        if (domain.ndim() != 2) {
            throw std::runtime_error("TileDB array should have exactly two dimensions");
        }

        // We use 'int' for the domain, just in case the domain's absolute
        // position exceeds Index_'s range, even if the actual range of the
        // domain does not.
        {
            tiledb::Dimension dim = domain.dimension(0);
            auto domain = dim.domain<int>();
            first_dim = domain.second - domain.first;
            first_tile = dim.tile_extent<int>();
        }

        {
            tiledb::Dimension dim = domain.dimension(1);
            auto domain = dim.domain<int>();
            second_dim = domain.second - domain.first;
            second_tile = dim.tile_extent<int>();
        }
    }

private:
    std::string location, attr;
    Index_ first_dim, second_dim;
    Index_ first_tile, second_tile;
    bool sparse_internal, prefer_rows_internal;

public:
    Index_ nrow() const {
        if constexpr(tranpose_) {
            return second_dim;
        } else {
            return first_dim;
        }
    }

    Index_ ncol() const {
        if constexpr(tranpose_) {
            return first_dim;
        } else {
            return second_dim;
        }
    }

    bool sparse() const {
        return sparse_internal;
    }

    double sparse_proportion() const {
        return static_cast<double>(sparse_internal);
    }

    bool prefer_rows() const {
        return prefer_rows_internal;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(prefer_rows_internal);
    }
};

}

#endif
