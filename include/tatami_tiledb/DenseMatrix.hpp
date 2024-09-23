#ifndef TATAMI_TILEDB_DENSE_MATRIX_HPP
#define TATAMI_TILEDB_DENSE_MATRIX_HPP

#include "tatami_chunked/tatami_chunked.hpp"
#include <tiledb/tiledb>

#include "serialize.hpp"

#include <string>
#include <memory>
#include <vector>

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

// All TileDB-related members.
struct Components{
    Components(const std::string& location) : array(ctx, location, TILEDB_READ) {}
    tiledb::Context ctx;
    tiledb::Array array;
};

template<typename Index_, typename OutputValue_>
void extract_block(
    const std::string& attribute,
    bool row,
    int target_offset,
    Index_ target_start,
    Index_ target_length,
    int non_target_offset,
    Index_ block_start,
    Index_ block_length,
    OutputValue_* buffer,
    Components& comp) 
{
    int rowdex = row;

    tiledb::Subarray subarray(comp.ctx, comp.array);
    auto actual_target_start = target_offset + target_start; 
    subarray.add_range(1 - rowdex, actual_target_start, actual_target_start + target_length - 1);
    auto actual_non_target_start = non_target_offset + block_start; 
    subarray.add_range(rowdex, actual_non_target_start, actual_non_target_start + block_length - 1);

    tiledb::Query query(comp.ctx, comp.array);
    query.set_subarray(subarray)
        .set_layout(row ? TILEDB_ROW_MAJOR : TILEDB_COL_MAJOR)
        .set_data_buffer(attribute, buffer, static_cast<size_t>(target_length) * static_cast<size_t>(block_length)); // cast to avoid overflow.

    if (query.submit() != tiledb::Query::Status::COMPLETE) {
        throw std::runtime_error("failed to read dense data from TileDB");
    }
}

template<typename Index_, typename OutputValue_>
void extract_indices(
    const std::string& attribute,
    bool row,
    int target_offset,
    Index_ target_start,
    Index_ target_length,
    int non_target_offset,
    const std::vector<Index_>& indices,
    OutputValue_* buffer,
    Components& comp) 
{
    int rowdex = row;

    tiledb::Subarray subarray(comp.ctx, comp.array);
    auto actual_target_start = target_offset + target_start; 
    subarray.add_range(1 - rowdex, actual_target_start, actual_target_start + target_length - 1);

    tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(), [&](Index_ s, Index_ l) {
        auto actual_non_target_start = non_target_offset + s;
        subarray.add_range(rowdex, actual_non_target_start, actual_non_target_start + l - 1);
    });

    tiledb::Query query(comp.ctx, comp.array);
    query.set_subarray(subarray)
        .set_layout(row ? TILEDB_ROW_MAJOR : TILEDB_COL_MAJOR)
        .set_data_buffer(attribute, buffer, static_cast<size_t>(target_length) * indices.size()); // cast to avoid overflow.

    if (query.submit() != tiledb::Query::Status::COMPLETE) {
        throw std::runtime_error("failed to read dense data from TileDB");
    }
}

/********************
 *** Core classes ***
 ********************/

inline void initialize(const std::string& location, std::unique_ptr<Components>& tdbcomp) {
    serialize([&]() -> void {
        tdbcomp.reset(new Components(location));
    });
}

inline void destroy(std::unique_ptr<Components>& tdbcomp) {
    serialize([&]() -> void {
        tdbcomp.reset();
    });
}

template<bool oracle_, typename Index_>
class SoloCore {
public:
    SoloCore(
        const std::string& location,
        const std::string& attribute, 
        bool row,
        [[maybe_unused]] tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, // only listed here for compatibility with the other constructors.
        int target_offset,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        [[maybe_unused]] Index_ non_target_length, 
        int non_target_offset,
        [[maybe_unused]] const tatami_chunked::SlabCacheStats& slab_stats) :
        my_attribute(attribute),
        my_row(row),
        my_target_offset(target_offset),
        my_non_target_offset(non_target_offset),
        my_oracle(std::move(oracle))
    {
        initialize(location, my_tdbcomp);
    }

    ~SoloCore() {
        destroy(my_tdbcomp);
    }

private:
    std::unique_ptr<Components> my_tdbcomp;
    const std::string& my_attribute;
    bool my_row;

    int my_target_offset;
    int my_non_target_offset;

    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, size_t, bool>::type my_counter = 0;

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&](){
            extract_block(my_attribute, my_row, my_target_offset, i, static_cast<Index_>(1), my_non_target_offset, block_start, block_length, buffer, *my_tdbcomp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&](){
            extract_indices(my_attribute, my_row, my_target_offset, i, static_cast<Index_>(1), my_non_target_offset, indices, buffer, *my_tdbcomp);
        });
        return buffer;
    }
};

template<typename Index_, typename CachedValue_>
class MyopicCore {
public:
    MyopicCore(
        const std::string& location,
        const std::string& attribute, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, 
        int target_offset,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_> oracle, // for consistency with the oracular version.
        Index_ non_target_length, 
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_attribute(attribute),
        my_row(row),
        my_dim_stats(std::move(target_dim_stats)),
        my_target_offset(target_offset),
        my_non_target_length(non_target_length),
        my_non_target_offset(non_target_offset),
        my_factory(slab_stats), 
        my_cache(slab_stats.max_slabs_in_cache)
    {
        initialize(location, my_tdbcomp);
    }

    ~MyopicCore() {
        destroy(my_tdbcomp);
    }

private:
    std::unique_ptr<Components> my_tdbcomp;
    const std::string& my_attribute;
    bool my_row;

    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;
    int my_target_offset;
    Index_ my_non_target_length;
    int my_non_target_offset;

    tatami_chunked::DenseSlabFactory<CachedValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::LruSlabCache<Index_, Slab> my_cache;

private:
    template<typename Value_, class Extract_>
    void fetch_raw(Index_ i, Value_* buffer, Extract_ extract) {
        Index_ chunk = i / my_dim_stats.chunk_length;
        Index_ index = i % my_dim_stats.chunk_length;

        const auto& info = my_cache.find(
            chunk, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](Index_ id, Slab& contents) -> void {
                auto curdim = tatami_chunked::get_chunk_length(my_dim_stats, id);
                serialize([&]() -> void {
                    extract(id * my_dim_stats.chunk_length, curdim, contents.data);
                });
            }
        );

        auto ptr = info.data + static_cast<size_t>(my_non_target_length) * static_cast<size_t>(index); // cast to size_t to avoid overflow
        std::copy_n(ptr, my_non_target_length, buffer);
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_block(my_attribute, my_row, my_target_offset, start, length, my_non_target_offset, block_start, block_length, buf, *my_tdbcomp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_indices(my_attribute, my_row, my_target_offset, start, length, my_non_target_offset, indices, buf, *my_tdbcomp);
        });
        return buffer;
    }
};

template<typename Index_, typename CachedValue_>
struct OracularCore {
    OracularCore(
        const std::string& location,
        const std::string& attribute, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset,
        tatami::MaybeOracle<true, Index_> oracle, 
        Index_ non_target_length, 
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_attribute(attribute),
        my_row(row),
        my_dim_stats(std::move(target_dim_stats)),
        my_target_offset(target_offset),
        my_non_target_length(non_target_length),
        my_non_target_offset(non_target_offset),
        my_factory(slab_stats), 
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache)
    {
        initialize(location, my_tdbcomp);
    }

    ~OracularCore() {
        destroy(my_tdbcomp);
    }

private:
    std::unique_ptr<Components> my_tdbcomp;
    const std::string& my_attribute;
    bool my_row;

    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;
    int my_target_offset;
    Index_ my_non_target_length;
    int my_non_target_offset;

    tatami_chunked::DenseSlabFactory<CachedValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> my_cache;

public:
    template<typename Value_, class Extract_>
    void fetch_raw([[maybe_unused]] Index_ i, Value_* buffer, Extract_ extract) {
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_dim_stats.chunk_length, current % my_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](const std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                serialize([&]() -> void {
                    for (const auto& c : chunks) {
                        auto curdim = tatami_chunked::get_chunk_length(my_dim_stats, c.first);
                        extract(c.first * my_dim_stats.chunk_length, curdim, c.second->data);
                    }
                });
            }
        );

        auto ptr = info.first->data + static_cast<size_t>(my_non_target_length) * static_cast<size_t>(info.second); // cast to size_t to avoid overflow
        std::copy_n(ptr, my_non_target_length, buffer);
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_block(my_attribute, my_row, my_target_offset, start, length, my_non_target_offset, block_start, block_length, buf, *my_tdbcomp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_indices(my_attribute, my_row, my_target_offset, start, length, my_non_target_offset, indices, buf, *my_tdbcomp);
        });
        return buffer;
    }
};

template<bool solo_, bool oracle_, typename Index_, typename CachedValue_>
using DenseCore = typename std::conditional<solo_, 
      SoloCore<oracle_, Index_>,
      typename std::conditional<oracle_,
          OracularCore<Index_, CachedValue_>,
          MyopicCore<Index_, CachedValue_>
      >::type
>::type;

/***************************
 *** Concrete subclasses ***
 ***************************/

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct Full : public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Full(
        const std::string& location, 
        const std::string& attribute, 
        bool by_tdb_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim,
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core(
            location,
            attribute,
            by_tdb_row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            non_target_dim, 
            non_target_offset,
            slab_stats
        ),
        my_non_target_dim(non_target_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, 0, my_non_target_dim, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_> 
struct Block : public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Block(
        const std::string& location,
        const std::string& attribute, 
        bool by_tdb_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        int non_target_offset, 
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core( 
            location,
            attribute,
            by_tdb_row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            block_length, 
            non_target_offset,
            slab_stats
        ),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, my_block_start, my_block_length, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct Index : public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Index(
        const std::string& location,
        const std::string& attribute, 
        bool by_tdb_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        int non_target_offset, 
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core(
            location,
            attribute,
            by_tdb_row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            indices_ptr->size(), 
            non_target_offset,
            slab_stats
        ),
        my_indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_indices(i, *my_indices_ptr, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_, CachedValue_> my_core;
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
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store the values.
 *
 * Numeric dense matrix stored in a 2-dimensional TileDB array.
 * Chunks of data are loaded from file as needed by `tatami::Extractor` objects, with additional caching to avoid repeated reads from disk.
 * The size of cached chunks is determined by the extent of the tiles in the TileDB array.
 *
 * The TileDB library is thread-safe so no additional work is required to use this class in parallel code.
 * Nonetheless, users can force all calls to TileDB to occur in serial by defining the `TATAMI_TILEDB_PARALLEL_LOCK` macro.
 * This should be a function-like macro that accepts a function and executes it inside a user-defined serial section.
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_>
class DenseMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     * @param options Further options.
     */
    DenseMatrix(std::string uri, std::string attribute, const DenseMatrixOptions& options) : my_location(std::move(uri)), my_attribute(std::move(attribute)) {
        serialize([&]() -> void {
            tiledb::Context ctx;
            tiledb::ArraySchema schema(ctx, my_location);
            if (schema.array_type() != TILEDB_DENSE) {
                throw std::runtime_error("TileDB array should be dense");
            }
            initialize(schema, options);
        });
    }

    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     */
    DenseMatrix(std::string uri, std::string attribute) : DenseMatrix(std::move(uri), std::move(attribute), DenseMatrixOptions()) {}

    /**
     * @cond
     */
    DenseMatrix(tiledb::ArraySchema& schema, std::string uri, std::string attribute, const DenseMatrixOptions& options) : my_location(std::move(uri)), my_attribute(std::move(attribute)) {
        initialize(schema, options);
    }
    /**
     * @endcond
     */

private:
    void initialize(tiledb::ArraySchema& schema, const DenseMatrixOptions& options) {
        my_cache_size_in_elements = static_cast<double>(options.maximum_cache_size) / sizeof(Value_);
        my_require_minimum_cache = options.require_minimum_cache;

        if (!schema.has_attribute(my_attribute)) {
            throw std::runtime_error("no attribute '" + my_attribute + "' is present in the TileDB array at '" + my_location + "'");
        }

        tiledb::Domain domain = schema.domain();
        if (domain.ndim() != 2) {
            throw std::runtime_error("TileDB array should have exactly two dimensions");
        }

        // We use 'int' for the domain, just in case the domain's absolute
        // position exceeds Index_'s range, even if the actual range of the
        // domain does not.
        tiledb::Dimension first_dim = domain.dimension(0);
        auto first_domain = first_dim.domain<int>();
        my_first_offset = first_domain.first;
        Index_ first_extent = first_domain.second - first_domain.first + 1;
        Index_ first_tile = first_dim.tile_extent<int>();
        my_firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(first_extent, first_tile);

        tiledb::Dimension second_dim = domain.dimension(1);
        auto second_domain = second_dim.domain<int>();
        my_second_offset = second_domain.first;
        Index_ second_extent = second_domain.second - second_domain.first + 1;
        Index_ second_tile = second_dim.tile_extent<int>();
        my_seconddim_stats = tatami_chunked::ChunkDimensionStats<Index_>(second_extent, second_tile);

        // Favoring extraction on the dimension that involves pulling out fewer chunks per dimension element.
        auto tiles_per_firstdim = (second_extent / second_tile) + (second_extent % second_tile > 0);
        auto tiles_per_seconddim = (first_extent / first_tile) + (first_extent % first_tile > 0);
        my_prefer_firstdim = tiles_per_firstdim <= tiles_per_seconddim;
    }

private:
    std::string my_location, my_attribute;
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
    template<bool oracle_, template<bool, bool, typename, typename, typename> class Extractor_, typename ... Args_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(bool row, Index_ non_target_length, tatami::MaybeOracle<oracle_, Index_> oracle, Args_&& ... args) const {
        const auto& target_dim_stats = (row ? my_firstdim_stats : my_seconddim_stats);
        auto target_offset = (row ? my_first_offset : my_second_offset);
        auto non_target_offset = (row ? my_second_offset : my_first_offset);

        tatami_chunked::SlabCacheStats slab_stats(
            target_dim_stats.chunk_length,
            non_target_length,
            target_dim_stats.num_chunks,
            my_cache_size_in_elements,
            my_require_minimum_cache
        );

        if (slab_stats.max_slabs_in_cache > 0) {
            return std::make_unique<Extractor_<false, oracle_, Value_, Index_, CachedValue_> >(
                my_location, 
                my_attribute,
                row,
                target_dim_stats,
                target_offset,
                std::move(oracle), 
                std::forward<Args_>(args)...,
                non_target_offset,
                slab_stats
            );
        } else {
            return std::make_unique<Extractor_<true, oracle_, Value_, Index_, CachedValue_> >(
                my_location,
                my_attribute, 
                row,
                target_dim_stats,
                target_offset,
                std::move(oracle),
                std::forward<Args_>(args)...,
                non_target_offset,
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
