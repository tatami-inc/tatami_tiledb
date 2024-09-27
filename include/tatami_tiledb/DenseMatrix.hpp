#ifndef TATAMI_TILEDB_DENSE_MATRIX_HPP
#define TATAMI_TILEDB_DENSE_MATRIX_HPP

#include "tatami_chunked/tatami_chunked.hpp"
#include <tiledb/tiledb>

#include "serialize.hpp"
#include "utils.hpp"

#include <string>
#include <memory>
#include <vector>
#include <type_traits>

/**
 * @file DenseMatrix.hpp
 * @brief TileDB-backed dense matrix.
 */

namespace tatami_tiledb {

/**
 * @brief Options for dense TileDB extraction.
 */
struct DenseMatrixOptions {
    /**
     * Size of the in-memory cache in bytes.
     *
     * We cache all tiles required to read a row/column during a `tatami::DenseExtractor::fetch()` or `tatami::SparseExtractor::Fetch()` call.
     * This allows us to re-use the cached tiles when adjacent rows/columns are requested, rather than re-reading them from disk.
     *
     * Larger caches improve access speed at the cost of memory usage.
     * Small values may be ignored if `require_minimum_cache` is `true`.
     */
    size_t maximum_cache_size = 100000000;

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `maximum_cache_size`.
     * This minimum is chosen to ensure that all tiles overlapping one row (or a slice/subset thereof) can be retained in memory,
     * so that the same tiles are not repeatedly re-read from disk when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;
};

/**
 * @cond
 */
namespace DenseMatrix_internal {

typedef ::tatami_tiledb::internal::Components Components;
typedef ::tatami_tiledb::internal::VariablyTypedDimension Dimension;
typedef ::tatami_tiledb::internal::VariablyTypedVector CacheBuffer;

inline void execute_query(const Components& tdb_comp, const tiledb::Subarray& subarray, const std::string& attribute, bool row, CacheBuffer& buffer, size_t offset, size_t length) {
    tiledb::Query query(tdb_comp.ctx, tdb_comp.array);
    query.set_subarray(subarray);
    query.set_layout(row ? TILEDB_ROW_MAJOR : TILEDB_COL_MAJOR);
    buffer.set_data_buffer(query, attribute, offset, length);
    if (query.submit() != tiledb::Query::Status::COMPLETE) {
        throw std::runtime_error("failed to read dense data from TileDB");
    }
}

template<typename Index_>
void extract_block(
    const Components& tdb_comp,
    const std::string& attribute,
    bool row,
    const Dimension& target_dim,
    Index_ target_start,
    Index_ target_length,
    const Dimension& non_target_dim,
    Index_ block_start,
    Index_ block_length,
    CacheBuffer& buffer,
    size_t buffer_offset,
    size_t buffer_length)
{
    tiledb::Subarray subarray(tdb_comp.ctx, tdb_comp.array);
    int rowdex = row;
    target_dim.add_range(subarray, 1 - rowdex, target_start, target_length);
    non_target_dim.add_range(subarray, rowdex, block_start, block_length);
    execute_query(tdb_comp, subarray, attribute, row, buffer, buffer_offset, buffer_length);
}

template<typename Index_> 
void extract_indices(
    const Components& tdb_comp,
    const std::string& attribute,
    bool row,
    const Dimension& target_dim,
    Index_ target_start,
    Index_ target_length,
    const Dimension& non_target_dim,
    const std::vector<Index_>& indices,
    CacheBuffer& buffer,
    size_t buffer_offset,
    size_t buffer_length)
{
    tiledb::Subarray subarray(tdb_comp.ctx, tdb_comp.array);
    int rowdex = row;
    target_dim.add_range(subarray, 1 - rowdex, target_start, target_length);
    tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(), [&](Index_ s, Index_ l) {
        non_target_dim.add_range(subarray, rowdex, s, l);
    });
    execute_query(tdb_comp, subarray, attribute, row, buffer, buffer_offset, buffer_length);
}

/********************
 *** Core classes ***
 ********************/

template<bool oracle_, typename Index_>
class SoloCore {
public:
    SoloCore(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        [[maybe_unused]] tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, // only listed here for compatibility with the other constructors.
        const Dimension& tdb_target_dim,
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        Index_ non_target_length,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        [[maybe_unused]] const tatami_chunked::SlabCacheStats& slab_stats) :
        my_tdb_comp(tdb_comp),
        my_attribute(attribute),
        my_row(row),
        my_tdb_target_dim(tdb_target_dim),
        my_tdb_non_target_dim(tdb_non_target_dim),
        my_non_target_length(non_target_length),
        my_holding(tdb_type, non_target_length * 1),
        my_oracle(std::move(oracle))
    {}

private:
    const Components& my_tdb_comp;
    const std::string& my_attribute;

    bool my_row;
    const Dimension& my_tdb_target_dim;
    const Dimension& my_tdb_non_target_dim;

    Index_ my_non_target_length;
    CacheBuffer my_holding;

    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, size_t, bool>::type my_counter = 0;

private:
    template<typename Value_, class Extract_>
    const Value_* fetch_raw(Index_ i, Value_* buffer, Extract_ extract) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&](){
            extract(i);
        });
        my_holding.copy(0, my_non_target_length, buffer);
        return buffer;
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Value_* buffer) {
        return fetch_raw(i, buffer, [&](Index_ i0) {
            extract_block(my_tdb_comp, my_attribute, my_row, my_tdb_target_dim, i0, static_cast<Index_>(1), my_tdb_non_target_dim, block_start, my_non_target_length, my_holding, 0, my_non_target_length);
        });
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        return fetch_raw(i, buffer, [&](Index_ i0) {
            extract_indices(my_tdb_comp, my_attribute, my_row, my_tdb_target_dim, i0, static_cast<Index_>(1), my_tdb_non_target_dim, indices, my_holding, 0, indices.size());
        });
    }
};

template<typename Index_>
class MyopicCore {
public:
    MyopicCore(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, 
        const Dimension& tdb_target_dim,
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        Index_ non_target_length,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_> oracle, // for consistency with the oracular version.
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_tdb_comp(tdb_comp),
        my_attribute(attribute),
        my_row(row),
        my_target_dim_stats(std::move(target_dim_stats)),
        my_tdb_target_dim(tdb_target_dim),
        my_tdb_non_target_dim(tdb_non_target_dim),
        my_non_target_length(non_target_length),
        my_slab_size(slab_stats.slab_size_in_elements),
        my_holding(tdb_type, my_slab_size * slab_stats.max_slabs_in_cache),
        my_cache(slab_stats.max_slabs_in_cache)
    {}

private:
    const Components& my_tdb_comp;
    const std::string& my_attribute;

    bool my_row;
    tatami_chunked::ChunkDimensionStats<Index_> my_target_dim_stats;
    const Dimension& my_tdb_target_dim;
    const Dimension& my_tdb_non_target_dim;

    Index_ my_non_target_length;
    size_t my_slab_size;
    CacheBuffer my_holding;

    struct Slab {
        size_t offset;
    };
    size_t my_offset = 0;
    tatami_chunked::LruSlabCache<Index_, Slab> my_cache;

private:
    template<typename Value_, class Extract_>
    const Value_* fetch_raw(Index_ i, Value_* buffer, Extract_ extract) {
        Index_ chunk = i / my_target_dim_stats.chunk_length;
        Index_ index = i % my_target_dim_stats.chunk_length;

        const auto& info = my_cache.find(
            chunk, 
            /* create = */ [&]() -> Slab {
                Slab output;
                output.offset = my_offset;
                my_offset += my_slab_size;
                return output;
            },
            /* populate = */ [&](Index_ id, Slab& contents) -> void {
                auto curdim = tatami_chunked::get_chunk_length(my_target_dim_stats, id);
                serialize([&]() -> void {
                    extract(id * my_target_dim_stats.chunk_length, curdim, contents.offset);
                });
            }
        );

        size_t final_offset = info.offset + static_cast<size_t>(my_non_target_length) * static_cast<size_t>(index); // cast to size_t to avoid overflow
        my_holding.copy(final_offset, my_non_target_length, buffer);
        return buffer;
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Value_* buffer) {
        return fetch_raw(i, buffer, [&](Index_ start, Index_ length, size_t offset) {
            extract_block(my_tdb_comp, my_attribute, my_row, my_tdb_target_dim, start, length, my_tdb_non_target_dim, block_start, my_non_target_length, my_holding, offset, my_slab_size);
        });
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        return fetch_raw(i, buffer, [&](Index_ start, Index_ length, size_t offset) {
            extract_indices(my_tdb_comp, my_attribute, my_row, my_tdb_target_dim, start, length, my_tdb_non_target_dim, indices, my_holding, offset, my_slab_size);
        });
    }
};

template<typename Index_>
class OracularCore {
public:
    OracularCore(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        const Dimension& tdb_target_dim,
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        Index_ non_target_length,
        tatami::MaybeOracle<true, Index_> oracle, 
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_tdb_comp(tdb_comp),
        my_attribute(attribute),
        my_row(row),
        my_target_dim_stats(std::move(target_dim_stats)),
        my_tdb_target_dim(tdb_target_dim),
        my_tdb_non_target_dim(tdb_non_target_dim),
        my_non_target_length(non_target_length),
        my_slab_size(slab_stats.slab_size_in_elements),
        my_holding(tdb_type, my_slab_size * slab_stats.max_slabs_in_cache),
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache)
    {}

private:
    const Components& my_tdb_comp;
    const std::string& my_attribute;

    bool my_row;
    tatami_chunked::ChunkDimensionStats<Index_> my_target_dim_stats;
    const Dimension& my_tdb_target_dim;
    const Dimension& my_tdb_non_target_dim;

    Index_ my_non_target_length;
    size_t my_slab_size;
    CacheBuffer my_holding;

    struct Slab {
        size_t offset;
    };
    size_t my_offset = 0;
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab, true> my_cache;

private:
    template<class Function_>
    static void sort_by_field(std::vector<std::pair<Index_, Slab*> >& indices, Function_ field) {
        auto comp = [&field](const std::pair<Index_, Slab*>& l, const std::pair<Index_, Slab*>& r) -> bool {
            return field(l) < field(r);
        };
        if (!std::is_sorted(indices.begin(), indices.end(), comp)) {
            std::sort(indices.begin(), indices.end(), comp);
        }
    }

    template<typename Value_, class Configure_>
    const Value_* fetch_raw([[maybe_unused]] Index_ i, Value_* buffer, Configure_ configure) {
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_target_dim_stats.chunk_length, current % my_target_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                Slab output;
                output.offset = my_offset;
                my_offset += my_slab_size;
                return output;
            },
            /* populate = */ [&](std::vector<std::pair<Index_, Slab*> >& to_populate, std::vector<std::pair<Index_, Slab*> >& to_reuse) {
                // Defragmenting the existing chunks. We sort by offset to make 
                // sure that we're not clobbering in-use slabs during the copy().
                sort_by_field(to_reuse, [](const std::pair<Index_, Slab*>& x) -> size_t { return x.second->offset; });
                size_t running_offset = 0;
                for (auto& x : to_reuse) {
                    auto& cur_offset = x.second->offset;
                    if (cur_offset != running_offset) {
                        my_holding.shift(cur_offset, my_slab_size, running_offset);
                        cur_offset = running_offset;
                    }
                    running_offset += my_slab_size;
                }

                // Collapsing runs of consecutive ranges into a single range;
                // otherwise, making union of ranges. This allows a single TileDb call
                // to populate the contiguous memory pool that we made available after
                // defragmentation; then we just update the slab pointers to refer
                // to the slices of memory corresponding to each slab.
                sort_by_field(to_populate, [](const std::pair<Index_, Slab*>& x) -> Index_ { return x.first; });

                serialize([&]() -> void {
                    tiledb::Subarray subarray(my_tdb_comp.ctx, my_tdb_comp.array);
                    configure(subarray);

                    // Remember, the slab size is equal to the product of the chunk length and the 
                    // non-target length, so shifting the memory pool offsets by 'slab_size' will 
                    // correspond to a shift of 'chunk_length' on the target dimension. The only
                    // exception is that of the last chunk, but at that point it doesn't matter as 
                    // there's no data following the last chunk.
                    Index_ run_chunk_id = to_populate.front().first;
                    Index_ chunk_length = tatami_chunked::get_chunk_length(my_target_dim_stats, run_chunk_id);
                    Index_ run_length = chunk_length;
                    to_populate.front().second->offset = running_offset;
                    auto start_offset = running_offset;
                    running_offset += my_slab_size;

                    int dimdex = 1 - my_row;
                    for (size_t ci = 1, cend = to_populate.size(); ci < cend; ++ci) {
                        auto& current_chunk = to_populate[ci];
                        Index_ current_chunk_id = current_chunk.first;

                        if (current_chunk_id - run_chunk_id > 1) { // save the existing run of to_populate as one range, and start a new run.
                            my_tdb_target_dim.add_range(subarray, dimdex, run_chunk_id * my_target_dim_stats.chunk_length, run_length);
                            run_chunk_id = current_chunk_id;
                            run_length = 0;
                        }

                        Index_ current_length = tatami_chunked::get_chunk_length(my_target_dim_stats, current_chunk_id);
                        run_length += current_length;
                        current_chunk.second->offset = running_offset;
                        running_offset += my_slab_size;
                    }

                    my_tdb_target_dim.add_range(subarray, dimdex, run_chunk_id * my_target_dim_stats.chunk_length, run_length);
                    execute_query(my_tdb_comp, subarray, my_attribute, my_row, my_holding, start_offset, running_offset - start_offset);
                });
            }
        );

        size_t final_offset = info.first->offset + my_non_target_length * static_cast<size_t>(info.second); // cast to size_t to avoid overflow
        my_holding.copy(final_offset, my_non_target_length, buffer);
        return buffer;
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Value_* buffer) {
        return fetch_raw(i, buffer, [&](tiledb::Subarray& subarray) {
            my_tdb_non_target_dim.add_range(subarray, static_cast<int>(my_row), block_start, my_non_target_length);
        });
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        return fetch_raw(i, buffer, [&](tiledb::Subarray& subarray) {
            const int rowdex = my_row;
            tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(), [&](Index_ s, Index_ l) {
                my_tdb_non_target_dim.add_range(subarray, rowdex, s, l);
            });
        });
    }
};

template<bool solo_, bool oracle_, typename Index_>
using DenseCore = typename std::conditional<solo_, 
      SoloCore<oracle_, Index_>,
      typename std::conditional<oracle_,
          OracularCore<Index_>,
          MyopicCore<Index_>
      >::type
>::type;

/***************************
 *** Concrete subclasses ***
 ***************************/

template<bool solo_, bool oracle_, typename Value_, typename Index_>
class Full : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Full(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        const Dimension& tdb_target_dim,
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core(
            tdb_comp,
            attribute,
            row,
            std::move(target_dim_stats),
            tdb_target_dim,
            tdb_non_target_dim,
            tdb_type,
            non_target_dim,
            std::move(oracle),
            slab_stats
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, 0, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_> my_core;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_>
class Block : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Block(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        const Dimension& tdb_target_dim,
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core( 
            tdb_comp,
            attribute,
            row,
            std::move(target_dim_stats),
            tdb_target_dim,
            tdb_non_target_dim,
            tdb_type,
            block_length,
            std::move(oracle),
            slab_stats
        ),
        my_block_start(block_start)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, my_block_start, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_> my_core;
    Index_ my_block_start;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_>
class Index : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Index(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        const Dimension& tdb_target_dim,
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core(
            tdb_comp,
            attribute,
            row,
            std::move(target_dim_stats),
            tdb_target_dim,
            tdb_non_target_dim,
            tdb_type,
            indices_ptr->size(),
            std::move(oracle),
            slab_stats
        ),
        my_indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_indices(i, *my_indices_ptr, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
};

}
/**
 * @endcond
 */

/**
 * @brief TileDB-backed dense matrix.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 *
 * Numeric dense matrix stored in a 2-dimensional TileDB array.
 * Chunks of data are loaded from file as needed by `tatami::Extractor` objects, with additional caching to avoid repeated reads from disk.
 * The size of cached chunks is determined by the extent of the tiles in the TileDB array.
 *
 * The TileDB library is thread-safe so no additional work is required to use this class in parallel code.
 * Nonetheless, users can force all calls to TileDB to occur in serial by defining the `TATAMI_TILEDB_PARALLEL_LOCK` macro.
 * This should be a function-like macro that accepts a function and executes it inside a user-defined serial section.
 */
template<typename Value_, typename Index_>
class DenseMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     * @param options Further options.
     */
    DenseMatrix(const std::string& uri, std::string attribute, const DenseMatrixOptions& options) : my_attribute(std::move(attribute)) {
        serialize([&]() {
            // Serializing the deleter. 
            my_tdb_comp.reset(new DenseMatrix_internal::Components(uri), [](DenseMatrix_internal::Components* ptr) {
                serialize([&]() {
                    delete ptr;
                });
            });

            auto schema = my_tdb_comp->array.schema();
            if (schema.array_type() != TILEDB_DENSE) {
                throw std::runtime_error("TileDB array should be dense");
            }

            if (!schema.has_attribute(my_attribute)) {
                throw std::runtime_error("no attribute '" + my_attribute + "' is present in the TileDB array");
            }
            auto attr = schema.attribute(my_attribute);
            my_tdb_type = attr.type();

            my_cache_size_in_elements = options.maximum_cache_size / internal::determine_type_size(my_tdb_type);
            my_require_minimum_cache = options.require_minimum_cache;

            tiledb::Domain domain = schema.domain();
            if (domain.ndim() != 2) {
                throw std::runtime_error("TileDB array should have exactly two dimensions");
            }

            tiledb::Dimension first_dim = domain.dimension(0);
            my_tdb_first_dim.reset(first_dim);
            Index_ first_extent = my_tdb_first_dim.extent<Index_>();
            Index_ first_tile = my_tdb_first_dim.tile<Index_>();
            my_firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(first_extent, first_tile);

            tiledb::Dimension second_dim = domain.dimension(1);
            my_tdb_second_dim.reset(second_dim);
            Index_ second_extent = my_tdb_second_dim.extent<Index_>();
            Index_ second_tile = my_tdb_second_dim.tile<Index_>();
            my_seconddim_stats = tatami_chunked::ChunkDimensionStats<Index_>(second_extent, second_tile);

            // Favoring extraction on the dimension that involves pulling out fewer chunks per dimension element.
            auto tiles_per_firstdim = (second_extent / second_tile) + (second_extent % second_tile > 0);
            auto tiles_per_seconddim = (first_extent / first_tile) + (first_extent % first_tile > 0);
            my_prefer_firstdim = tiles_per_firstdim <= tiles_per_seconddim;
        });
    }

    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     */
    DenseMatrix(const std::string& uri, std::string attribute) : DenseMatrix(uri, std::move(attribute), DenseMatrixOptions()) {}

private:
    std::shared_ptr<DenseMatrix_internal::Components> my_tdb_comp;

    DenseMatrix_internal::Dimension my_tdb_first_dim, my_tdb_second_dim;
    tiledb_datatype_t my_tdb_type;

    std::string my_attribute;
    size_t my_cache_size_in_elements;
    bool my_require_minimum_cache;

    int my_first_offset, my_second_offset;
    tatami_chunked::ChunkDimensionStats<Index_> my_firstdim_stats, my_seconddim_stats;
    bool my_prefer_firstdim;

private:
    Index_ nrow_internal() const {
        return my_firstdim_stats.dimension_extent;
    }

    Index_ ncol_internal() const {
        return my_seconddim_stats.dimension_extent;
    }

public:
    Index_ nrow() const {
        return nrow_internal();
    }

    Index_ ncol() const {
        return ncol_internal();
    }

    bool is_sparse() const {
        return false;
    }

    double is_sparse_proportion() const {
        return 0;
    }

    bool prefer_rows() const {
        return my_prefer_firstdim;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(my_prefer_firstdim);
    }

    bool uses_oracle(bool) const {
        // The oracle won't necessarily be used if the cache size is non-zero,
        // but if the cache is empty, the oracle definitely _won't_ be used.
        return my_cache_size_in_elements > 0;
    }

private:
    template<bool oracle_, template<bool, bool, typename, typename> class Extractor_, typename ... Args_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(bool row, Index_ non_target_length, tatami::MaybeOracle<oracle_, Index_> oracle, Args_&& ... args) const {
        const auto& target_dim_stats = (row ? my_firstdim_stats : my_seconddim_stats);
        const auto& tdb_target_dim = (row ? my_tdb_first_dim : my_tdb_second_dim);
        const auto& tdb_non_target_dim = (row ? my_tdb_second_dim : my_tdb_first_dim);

        tatami_chunked::SlabCacheStats slab_stats(
            target_dim_stats.chunk_length,
            non_target_length,
            target_dim_stats.num_chunks,
            my_cache_size_in_elements,
            my_require_minimum_cache
        );

        if (slab_stats.max_slabs_in_cache > 0) {
            return std::make_unique<Extractor_<false, oracle_, Value_, Index_> >(
                *my_tdb_comp, 
                my_attribute,
                row,
                target_dim_stats,
                tdb_target_dim,
                tdb_non_target_dim,
                my_tdb_type,
                std::move(oracle), 
                std::forward<Args_>(args)...,
                slab_stats
            );
        } else {
            return std::make_unique<Extractor_<true, oracle_, Value_, Index_> >(
                *my_tdb_comp,
                my_attribute, 
                row,
                target_dim_stats,
                tdb_target_dim,
                tdb_non_target_dim,
                my_tdb_type,
                std::move(oracle),
                std::forward<Args_>(args)...,
                slab_stats
            );
        }
    }

    /********************
     *** Myopic dense ***
     ********************/
public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options&) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<false, DenseMatrix_internal::Full>(row, full_non_target, false, full_non_target);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options&) const {
        return populate<false, DenseMatrix_internal::Block>(row, block_length, false, block_start, block_length);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options&) const {
        auto nidx = indices_ptr->size();
        return populate<false, DenseMatrix_internal::Index>(row, nidx, false, std::move(indices_ptr));
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return std::make_unique<tatami::FullSparsifiedWrapper<false, Value_, Index_> >(dense(row, opt), full_non_target, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<false, Value_, Index_> >(dense(row, block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto ptr = dense(row, indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<false, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        const tatami::Options&) 
    const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<true, DenseMatrix_internal::Full>(row, full_non_target, std::move(oracle), full_non_target);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options&) 
    const {
        return populate<true, DenseMatrix_internal::Block>(row, block_length, std::move(oracle), block_start, block_length);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options&) 
    const {
        auto nidx = indices_ptr->size();
        return populate<true, DenseMatrix_internal::Index>(row, nidx, std::move(oracle), std::move(indices_ptr));
    }

    /***********************
     *** Oracular sparse ***
     ***********************/
public:
    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        const tatami::Options& opt) 
    const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), opt), full_non_target, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto ptr = dense(row, std::move(oracle), indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<true, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }
};

}

#endif
