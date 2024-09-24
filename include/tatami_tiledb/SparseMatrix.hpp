#ifndef TATAMI_TILEDB_SPARSE_MATRIX_HPP
#define TATAMI_TILEDB_SPARSE_MATRIX_HPP

#include "tatami_chunked/tatami_chunked.hpp"
#include <tiledb/tiledb>

#include "serialize.hpp"

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

/**
 * @file SparseMatrix.hpp
 * @brief TileDB-backed sparse matrix.
 */

namespace tatami_tiledb {

/**
 * @brief Options for sparse TileDB extraction.
 */
struct SparseMatrixOptions {
    /**
     * Size of the in-memory cache in bytes.
     *
     * We cache all tiles required to read a row/column during a `tatami::SparseExtractor::fetch()` or `tatami::SparseExtractor::Fetch()` call.
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
namespace SparseMatrix_internal {

template<typename CachedValue_>
struct Workspace {
    Workspace(size_t slab_size, bool needs_value, bool needs_index) : 
        values(needs_value ? slab_size : 0),
        target_indices(slab_size),
        non_target_indices(needs_index ? slab_size : 0)
    {}

    std::vector<CachedValue_> values;

    // We use 'int' instead of CachedIndex_ as we don't know the
    // maximum range of the indices on-disk after we add an offset.
    std::vector<int> target_indices;
    std::vector<int> non_target_indices;
};

// All TileDB-related members, aliased here for convenience.
typedef ::tatami_tiledb::internal::Components Components;

template<typename CachedValue_, typename CachedIndex_>
using Slab = typename tatami_chunked::SparseSlabFactory<CachedValue_, CachedIndex_>::Slab;

template<bool has_value_, bool has_index_, typename CachedValue_, typename CachedIndex_>
void populate_block_cache(
    size_t result_num,
    Workspace<CachedValue_>& work,
    Slab<CachedValue_, CachedIndex_>& cache,
    int subtract_target,
    int subtract_non_target) 
{
    for (size_t r = 0; r < result_num; ++r) {
        auto i = work.target_indices[r] - subtract_target;
        auto& current = cache.number[i];
        if constexpr(has_value_) {
            cache.values[i][current] = work.values[r];
        }
        if constexpr(has_index_) {
            cache.indices[i][current] = work.non_target_indices[r] - subtract_non_target;
        }
        ++current;
    }
}

template<typename Index_, typename CachedValue_, typename CachedIndex_>
void execute_query(
    const Components& tdbcomp,
    tiledb::Subarray& subarray,
    const std::string& attribute,
    const std::string& first_dimname,
    const std::string& second_dimname,
    bool row, 
    Index_ target_length,
    int subtract_target,
    int subtract_non_target,
    Workspace<CachedValue_>& work,
    Slab<CachedValue_, CachedIndex_>& cache)
{
    tiledb::Query query(tdbcomp.ctx, tdbcomp.array);
    const auto& target_dimname = (row ? first_dimname : second_dimname);
    query.set_subarray(subarray)
        .set_layout(row ? TILEDB_ROW_MAJOR : TILEDB_COL_MAJOR)
        .set_data_buffer(target_dimname, work.target_indices);

    if (!cache.values.empty()) {
        query.set_data_buffer(attribute, work.values);
    }
    if (!cache.indices.empty()) {
        // Don't extract indices directly into the cache, because the reported
        // indices may be out of range of CachedIndex_ due to TileDB's offsets. 
        query.set_data_buffer(row ? second_dimname : first_dimname, work.non_target_indices);
    }

    if (query.submit() != tiledb::Query::Status::COMPLETE) {
        throw std::runtime_error("failed to read sparse data from TileDB");
    }

    size_t result_num = query.result_buffer_elements()[target_dimname].second;
    std::fill_n(cache.number, target_length, 0);

    // Better be in CSR (if row = true) or CSC (otherwise) order.
    if (cache.indices.empty() && cache.values.empty()) { 
        populate_block_cache<false, false, CachedValue_, CachedIndex_>(result_num, work, cache, subtract_target, subtract_non_target);
    } else if (cache.indices.empty()) {
        populate_block_cache<true, false, CachedValue_, CachedIndex_>(result_num, work, cache, subtract_target, subtract_non_target);
    } else if (cache.values.empty()) { 
        populate_block_cache<false, true, CachedValue_, CachedIndex_>(result_num, work, cache, subtract_target, subtract_non_target);
    } else {
        populate_block_cache<true, true, CachedValue_, CachedIndex_>(result_num, work, cache, subtract_target, subtract_non_target);
    }
}

template<typename Index_, typename CachedValue_, typename CachedIndex_>
void extract_block(
    const Components& tdbcomp,
    const std::string& attribute,
    const std::string& first_dimname,
    const std::string& second_dimname,
    bool row,
    Index_ target_start,
    Index_ target_length,
    int target_offset,
    Index_ block_start,
    Index_ block_length,
    int non_target_offset,
    Workspace<CachedValue_>& work,
    Slab<CachedValue_, CachedIndex_>& cache)
{
    int rowdex = row;

    tiledb::Subarray subarray(tdbcomp.ctx, tdbcomp.array);
    int actual_target_start = target_offset + target_start; 
    subarray.add_range(1 - rowdex, actual_target_start, static_cast<int>(actual_target_start + target_length - 1));

    int actual_non_target_start = non_target_offset + block_start; 
    subarray.add_range(rowdex, actual_non_target_start, static_cast<int>(actual_non_target_start + block_length - 1));

    execute_query<Index_, CachedValue_, CachedIndex_>(tdbcomp, subarray, attribute, first_dimname, second_dimname, row, target_length, actual_target_start, non_target_offset, work, cache);
}

template<typename Index_, typename CachedValue_, typename CachedIndex_>
void extract_indices(
    const Components& tdbcomp,
    const std::string& attribute,
    const std::string& first_dimname,
    const std::string& second_dimname,
    bool row,
    Index_ target_start,
    Index_ target_length,
    int target_offset,
    const std::vector<Index_>& indices,
    int non_target_offset,
    Workspace<CachedValue_>& work,
    Slab<CachedValue_, CachedIndex_>& cache)
{
    int rowdex = row;

    tiledb::Subarray subarray(tdbcomp.ctx, tdbcomp.array);
    auto actual_target_start = target_offset + target_start; 
    subarray.add_range(1 - rowdex, actual_target_start, actual_target_start + target_length - 1);

    tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(), [&](Index_ s, Index_ l) {
        auto actual_non_target_start = non_target_offset + s;
        subarray.add_range(rowdex, actual_non_target_start, actual_non_target_start + l - 1);
    });

    execute_query<Index_, CachedValue_, CachedIndex_>(tdbcomp, subarray, attribute, first_dimname, second_dimname, row, target_length, actual_target_start, non_target_offset, work, cache);
}

/********************
 *** Core classes ***
 ********************/

template<bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SoloCore {
public:
    SoloCore(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        [[maybe_unused]] tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, // only listed here for compatibility with the other constructors.
        int target_offset,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        Index_ non_target_length, 
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats,
        bool needs_value,
        bool needs_index) :
        my_tdbcomp(tdbcomp),
        my_attribute(attribute),
        my_first_dimname(first_dimname),
        my_second_dimname(second_dimname),
        my_row(row),
        my_target_offset(target_offset),
        my_non_target_offset(non_target_offset),
        my_oracle(std::move(oracle)),
        my_needs_value(needs_value),
        my_needs_index(needs_index),
        my_work(slab_stats.slab_size_in_elements, needs_value, needs_index)
    {
        // Storing pointers here, so the class had better not move. 
        my_stub_slab.number = &count;
        if (my_needs_value) {
            vbuffer.resize(non_target_length);
            my_stub_slab.values.push_back(vbuffer.data());
        }
        if (my_needs_index) {
            ibuffer.resize(non_target_length);
            my_stub_slab.indices.push_back(ibuffer.data());
        }
    }

    // Don't allow moving or copying as we have raw pointers in the members.
    SoloCore(const SoloCore&) = delete;
    SoloCore& operator=(const SoloCore&) = delete;
    SoloCore(SoloCore&&) = delete;
    SoloCore& operator=(SoloCore&&) = delete;

    ~SoloCore() = default;

private:
    const Components& my_tdbcomp;
    const std::string& my_attribute;
    const std::string& my_first_dimname;
    const std::string& my_second_dimname;
    bool my_row;

    int my_target_offset;
    int my_non_target_offset;

    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, size_t, bool>::type my_counter = 0;

    bool my_needs_value;
    bool my_needs_index;

    Workspace<CachedValue_> my_work;
    std::vector<CachedValue_> vbuffer;
    std::vector<CachedIndex_> ibuffer;
    CachedIndex_ count = 0;

    typedef typename tatami_chunked::SparseSlabFactory<CachedValue_, CachedIndex_>::Slab Slab;
    Slab my_stub_slab;

public:
    std::pair<const Slab*, Index_> fetch_block(Index_ i, Index_ block_start, Index_ block_length) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&]() {
            extract_block<Index_, CachedValue_, CachedIndex_>(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                i,
                static_cast<Index_>(1),
                my_target_offset,
                block_start,
                block_length,
                my_non_target_offset,
                my_work,
                my_stub_slab
            );
        });
        return std::make_pair(&my_stub_slab, 0);
    }

    std::pair<const Slab*, Index_> fetch_indices(Index_ i, const std::vector<Index_>& indices) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&]() {
            extract_indices<Index_, CachedValue_, CachedIndex_>(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                i,
                static_cast<Index_>(1),
                my_target_offset,
                indices,
                my_non_target_offset,
                my_work,
                my_stub_slab
            );
        });
        return std::make_pair(&my_stub_slab, 0);
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class MyopicCore {
public:
    MyopicCore(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, 
        int target_offset,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_> oracle, // for consistency with the oracular version.
        Index_ non_target_length, 
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats, 
        bool needs_value,
        bool needs_index) :
        my_tdbcomp(tdbcomp),
        my_attribute(attribute),
        my_first_dimname(first_dimname),
        my_second_dimname(second_dimname),
        my_row(row),
        my_target_dim_stats(std::move(target_dim_stats)),
        my_target_offset(target_offset),
        my_non_target_offset(non_target_offset),
        my_work(slab_stats.slab_size_in_elements, needs_value, needs_index),
        my_factory(my_target_dim_stats.dimension_extent, non_target_length, slab_stats, needs_value, needs_index), 
        my_cache(slab_stats.max_slabs_in_cache)
    {}

private:
    const Components& my_tdbcomp;
    const std::string& my_attribute;
    const std::string& my_first_dimname;
    const std::string& my_second_dimname;
    bool my_row;

    tatami_chunked::ChunkDimensionStats<Index_> my_target_dim_stats;
    int my_target_offset;
    int my_non_target_offset;

    Workspace<CachedValue_> my_work;
    tatami_chunked::SparseSlabFactory<CachedValue_, CachedIndex_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::LruSlabCache<Index_, Slab> my_cache;

private:
    template<class Extract_>
    std::pair<const Slab*, Index_> fetch_raw(Index_ i, Extract_ extract) {
        Index_ chunk = i / my_target_dim_stats.chunk_length;
        Index_ index = i % my_target_dim_stats.chunk_length;

        const auto& info = my_cache.find(
            chunk, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](Index_ id, Slab& contents) -> void {
                serialize([&]() {
                    auto curdim = tatami_chunked::get_chunk_length(my_target_dim_stats, id);
                    extract(id * my_target_dim_stats.chunk_length, curdim, contents);
                });
            }
        );

        return std::make_pair(&info, index);
    }

public:
    std::pair<const Slab*, Index_> fetch_block(Index_ i, Index_ block_start, Index_ block_length) {
        return fetch_raw(i, [&](Index_ start, Index_ len, Slab& contents) {
            extract_block<Index_, CachedValue_, CachedIndex_>(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                start,
                len,
                my_target_offset,
                block_start,
                block_length,
                my_non_target_offset,
                my_work,
                contents
            );
        });
    }

    std::pair<const Slab*, Index_> fetch_indices(Index_ i, const std::vector<Index_>& indices) {
        return fetch_raw(i, [&](Index_ start, Index_ len, Slab& contents) {
            extract_indices<Index_, CachedValue_, CachedIndex_>(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                start,
                len,
                my_target_offset,
                indices,
                my_non_target_offset,
                my_work,
                contents
            );
        });
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class OracularCore {
public:
    OracularCore(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset,
        tatami::MaybeOracle<true, Index_> oracle, 
        Index_ non_target_length, 
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats,
        bool needs_value,
        bool needs_index) :
        my_tdbcomp(tdbcomp),
        my_attribute(attribute),
        my_first_dimname(first_dimname),
        my_second_dimname(second_dimname),
        my_row(row),
        my_target_dim_stats(std::move(target_dim_stats)),
        my_target_offset(target_offset),
        my_non_target_offset(non_target_offset),
        my_work(slab_stats.slab_size_in_elements, needs_value, needs_index),
        my_factory(my_target_dim_stats.dimension_extent, non_target_length, slab_stats, needs_value, needs_index), 
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache)
    {}

private:
    const Components& my_tdbcomp;
    const std::string& my_attribute;
    const std::string& my_first_dimname;
    const std::string& my_second_dimname;
    bool my_row;

    tatami_chunked::ChunkDimensionStats<Index_> my_target_dim_stats;
    int my_target_offset;
    int my_non_target_offset;

    Workspace<CachedValue_> my_work;
    tatami_chunked::SparseSlabFactory<CachedValue_, CachedIndex_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> my_cache;

private:
    template<class Extract_>
    std::pair<const Slab*, Index_> fetch_raw([[maybe_unused]] Index_ i, Extract_ extract) {
        return my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_target_dim_stats.chunk_length, current % my_target_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](const std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                serialize([&]() -> void {
                    for (const auto& c : chunks) {
                        auto curdim = tatami_chunked::get_chunk_length(my_target_dim_stats, c.first);
                        extract(c.first * my_target_dim_stats.chunk_length, curdim, *(c.second));
                    }
                });
            }
        );
    }

public:
    std::pair<const Slab*, Index_> fetch_block(Index_ i, Index_ block_start, Index_ block_length) {
        return fetch_raw(i, [&](Index_ start, Index_ len, Slab& contents) {
            extract_block<Index_, CachedValue_, CachedIndex_>(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                start,
                len,
                my_target_offset,
                block_start,
                block_length,
                my_non_target_offset,
                my_work,
                contents
            );
        });
    }

    std::pair<const Slab*, Index_> fetch_indices(Index_ i, const std::vector<Index_>& indices) {
        return fetch_raw(i, [&](Index_ start, Index_ len, Slab& contents) {
            extract_indices<Index_, CachedValue_, CachedIndex_>(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                start,
                len,
                my_target_offset,
                indices,
                my_non_target_offset,
                my_work,
                contents
            );
        });
    }
};

template<bool solo_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using SparseCore = typename std::conditional<solo_, 
      SoloCore<oracle_, Index_, CachedValue_, CachedIndex_>,
      typename std::conditional<oracle_,
          OracularCore<Index_, CachedValue_, CachedIndex_>,
          MyopicCore<Index_, CachedValue_, CachedIndex_>
      >::type
>::type;

/*************************
 *** Sparse subclasses ***
 *************************/

template<typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_> 
tatami::SparseRange<Value_, Index_> slab_to_sparse(const Slab<CachedValue_, CachedIndex_>& slab, Index_ index, Value_* vbuffer, Index_* ibuffer) {
    tatami::SparseRange<Value_, Index_> output;
    output.number = slab.number[index];
    if (!slab.values.empty()) {
        std::copy_n(slab.values[index], output.number, vbuffer);
        output.value = vbuffer;
    }
    if (!slab.indices.empty()) {
        std::copy_n(slab.indices[index], output.number, ibuffer);
        output.index = ibuffer;
    }
    return output;
}

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SparseFull : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseFull(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim,
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats,
        bool needs_value,
        bool needs_index) :
        my_core(
            tdbcomp,
            attribute,
            first_dimname,
            second_dimname,
            row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            non_target_dim, 
            non_target_offset,
            slab_stats,
            needs_value,
            needs_index
        ),
        my_non_target_dim(non_target_dim)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto info = my_core.fetch_block(i, 0, my_non_target_dim);
        return slab_to_sparse<Value_, Index_, CachedValue_, CachedIndex_>(*(info.first), info.second, vbuffer, ibuffer);
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_> 
class SparseBlock : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseBlock(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        int non_target_offset, 
        const tatami_chunked::SlabCacheStats& slab_stats,
        bool needs_value,
        bool needs_index) :
        my_core( 
            tdbcomp,
            attribute,
            first_dimname,
            second_dimname,
            row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            block_length, 
            non_target_offset,
            slab_stats,
            needs_value,
            needs_index
        ),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto info = my_core.fetch_block(i, my_block_start, my_block_length);
        return slab_to_sparse<Value_, Index_, CachedValue_, CachedIndex_>(*(info.first), info.second, vbuffer, ibuffer);
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SparseIndex : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseIndex(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname,
        const std::string& second_dimname,
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        int non_target_offset, 
        const tatami_chunked::SlabCacheStats& slab_stats,
        bool needs_value,
        bool needs_index) :
        my_core(
            tdbcomp,
            attribute,
            first_dimname,
            second_dimname,
            row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            indices_ptr->size(), 
            non_target_offset,
            slab_stats,
            needs_value,
            needs_index
        ),
        my_indices_ptr(std::move(indices_ptr))
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto info = my_core.fetch_indices(i, *my_indices_ptr);
        return slab_to_sparse<Value_, Index_, CachedValue_, CachedIndex_>(*(info.first), info.second, vbuffer, ibuffer);
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
};

/************************
 *** Dense subclasses ***
 ************************/

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class DenseFull : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseFull(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim,
        int non_target_offset,
        const tatami_chunked::SlabCacheStats& slab_stats,
        [[maybe_unused]] bool needs_value, // for consistency with Sparse* constructors.
        [[maybe_unused]] bool needs_index) :
        my_core(
            tdbcomp,
            attribute,
            first_dimname,
            second_dimname,
            row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            non_target_dim, 
            non_target_offset,
            slab_stats,
            /* needs_value = */ true,
            /* needs_index = */ true
        ),
        my_non_target_dim(non_target_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto info = my_core.fetch_block(i, 0, my_non_target_dim);

        const auto& slab = *(info.first);
        auto vptr = slab.values[info.second];
        auto iptr = slab.indices[info.second];
        auto n = slab.number[info.second];

        std::fill_n(buffer, my_non_target_dim, 0);
        for (decltype(n) j = 0; j < n; ++j) {
            buffer[iptr[j]] = vptr[j];
        }
        return buffer;
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_> 
class DenseBlock : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseBlock(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        int non_target_offset, 
        const tatami_chunked::SlabCacheStats& slab_stats,
        [[maybe_unused]] bool needs_value, // for consistency with Sparse* constructors.
        [[maybe_unused]] bool needs_index) :
        my_core( 
            tdbcomp,
            attribute,
            first_dimname,
            second_dimname,
            row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            block_length, 
            non_target_offset,
            slab_stats,
            /* needs_value = */ true,
            /* needs_index = */ true
        ),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto info = my_core.fetch_block(i, my_block_start, my_block_length);

        const auto& slab = *(info.first);
        auto vptr = slab.values[info.second];
        auto iptr = slab.indices[info.second];
        auto n = slab.number[info.second];

        std::fill_n(buffer, my_block_length, 0);
        for (decltype(n) j = 0; j < n; ++j) {
            buffer[iptr[j] - my_block_start] = vptr[j];
        }
        return buffer;
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class DenseIndex : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseIndex(
        const Components& tdbcomp,
        const std::string& attribute, 
        const std::string& first_dimname, 
        const std::string& second_dimname, 
        bool row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        int non_target_offset, 
        const tatami_chunked::SlabCacheStats& slab_stats,
        [[maybe_unused]] bool needs_value, // for consistency with Sparse* constructors.
        [[maybe_unused]] bool needs_index) :
        my_core(
            tdbcomp,
            attribute,
            first_dimname,
            second_dimname,
            row,
            std::move(target_dim_stats),
            target_offset,
            std::move(oracle),
            indices_ptr->size(), 
            non_target_offset,
            slab_stats,
            /* needs_value = */ true,
            /* needs_index = */ true
        ),
        my_indices_ptr(std::move(indices_ptr))
    {
        const auto& indices = *my_indices_ptr;
        if (!indices.empty()) {
            auto start = indices.front();
            my_remapping.resize(indices.back() - start + 1);
            for (size_t j = 0, end = indices.size(); j < end; ++j) {
                my_remapping[indices[j] - start] = j;
            }
        }
    }

    const Value_* fetch(Index_ i, Value_* buffer) {
        const auto& indices = *my_indices_ptr;

        if (!indices.empty()) {
            auto info = my_core.fetch_indices(i, indices);

            const auto& slab = *(info.first);
            auto vptr = slab.values[info.second];
            auto iptr = slab.indices[info.second];
            auto n = slab.number[info.second];

            std::fill_n(buffer, indices.size(), 0);
            auto start = indices.front();
            for (decltype(n) j = 0; j < n; ++j) {
                buffer[my_remapping[iptr[j] - start]] = vptr[j];
            }
        }

        return buffer;
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
    std::vector<Index_> my_remapping;
};

}
/**
 * @endcond
 */

/**
 * @brief TileDB-backed sparse matrix.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store the values.
 * @tparam CachedIndex_ Type of the matrix index to store in the cache.
 * This can be set to a narrower type than `Index_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store the indices.
 *
 * Numeric sparse matrix stored in a 2-dimensional TileDB array.
 * Chunks of data are loaded from file as needed by `tatami::Extractor` objects, with additional caching to avoid repeated reads from disk.
 * The size of cached chunks is determined by the extent of the tiles in the TileDB array.
 *
 * The TileDB library is thread-safe so no additional work is required to use this class in parallel code.
 * Nonetheless, users can force all calls to TileDB to occur in serial by defining the `TATAMI_TILEDB_PARALLEL_LOCK` macro.
 * This should be a function-like macro that accepts a function and executes it inside a user-defined serial section.
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_, typename CachedIndex_ = Index_>
class SparseMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     * @param options Further options.
     */
    SparseMatrix(const std::string& uri, std::string attribute, const SparseMatrixOptions& options) : my_attribute(std::move(attribute)) {
        serialize([&]() {
            // Serializing the deleter. 
            my_tdbcomp.reset(new SparseMatrix_internal::Components(uri), [](SparseMatrix_internal::Components* ptr) {
                serialize([&]() {
                    delete ptr;
                });
            });

            auto schema = my_tdbcomp->array.schema();
            if (schema.array_type() != TILEDB_SPARSE) {
                throw std::runtime_error("TileDB array should be sparse");
            }

            my_cache_size_in_bytes = options.maximum_cache_size;
            my_require_minimum_cache = options.require_minimum_cache;

            if (!schema.has_attribute(my_attribute)) {
                throw std::runtime_error("no attribute '" + my_attribute + "' is present in the TileDB array");
            }

            tiledb::Domain domain = schema.domain();
            if (domain.ndim() != 2) {
                throw std::runtime_error("TileDB array should have exactly two dimensions");
            }

            // We use 'int' for the domain, just in case the domain's absolute
            // position exceeds Index_'s range, even if the actual range of the
            // domain does not.
            tiledb::Dimension first_dim = domain.dimension(0);
            my_first_dimname = first_dim.name();
            auto first_domain = first_dim.domain<int>();
            my_first_offset = first_domain.first;
            Index_ first_extent = first_domain.second - first_domain.first + 1;
            Index_ first_tile = first_dim.tile_extent<int>();
            my_firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(first_extent, first_tile);

            tiledb::Dimension second_dim = domain.dimension(1);
            my_second_dimname = second_dim.name();
            auto second_domain = second_dim.domain<int>();
            my_second_offset = second_domain.first;
            Index_ second_extent = second_domain.second - second_domain.first + 1;
            Index_ second_tile = second_dim.tile_extent<int>();
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
    SparseMatrix(const std::string& uri, std::string attribute) : SparseMatrix(uri, std::move(attribute), SparseMatrixOptions()) {}

private:
    std::shared_ptr<SparseMatrix_internal::Components> my_tdbcomp;

    std::string my_attribute;
    size_t my_cache_size_in_bytes;
    bool my_require_minimum_cache;

    int my_first_offset, my_second_offset;
    std::string my_first_dimname, my_second_dimname;
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
        return true;
    }

    double is_sparse_proportion() const {
        return 1;
    }

    bool prefer_rows() const {
        return my_prefer_firstdim;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(my_prefer_firstdim);
    }

    bool uses_oracle(bool) const {
        // It won't necessarily be used, but if the cache is empty,
        // the oracle definitely _won't_ be used.
        return my_cache_size_in_bytes > 0;
    }

private:
    template<
        bool oracle_,
        template<typename, typename> class Interface_, 
        template<bool, bool, typename, typename, typename, typename> class Extractor_, 
        typename ... Args_
    >
    std::unique_ptr<Interface_<Value_, Index_> > populate(
        bool row,
        Index_ non_target_length,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        const tatami::Options& opt,
        Args_&& ... args) 
    const {
        const auto& target_dim_stats = (row ? my_firstdim_stats : my_seconddim_stats);
        auto target_offset = (row ? my_first_offset : my_second_offset);
        auto non_target_offset = (row ? my_second_offset : my_first_offset);

        tatami_chunked::SlabCacheStats slab_stats(
            target_dim_stats.chunk_length,
            non_target_length,
            target_dim_stats.num_chunks,
            my_cache_size_in_bytes,
            (opt.sparse_extract_value ? sizeof(CachedValue_) : 0) + (opt.sparse_extract_index ? sizeof(CachedIndex_) : 0),
            my_require_minimum_cache
        );

        if (slab_stats.max_slabs_in_cache > 0) {
            return std::make_unique<Extractor_<false, oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                *my_tdbcomp, 
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                row,
                target_dim_stats,
                target_offset,
                std::move(oracle), 
                std::forward<Args_>(args)...,
                non_target_offset,
                slab_stats,
                opt.sparse_extract_value,
                opt.sparse_extract_index
            );
        } else {
            return std::make_unique<Extractor_<true, oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                *my_tdbcomp,
                my_attribute, 
                my_first_dimname,
                my_second_dimname,
                row,
                target_dim_stats,
                target_offset,
                std::move(oracle),
                std::forward<Args_>(args)...,
                non_target_offset,
                slab_stats,
                opt.sparse_extract_value,
                opt.sparse_extract_index
            );
        }
    }

    static tatami::Options set_extract_all(tatami::Options opt) {
        // Resetting these options so that the slab size estimates are
        // correctly estimated for dense extractors, regardless of 'opt'. 
        opt.sparse_extract_value = true;
        opt.sparse_extract_index = true;
        return opt;
    }

    /********************
     *** Myopic dense ***
     ********************/
public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options& opt) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<false, tatami::MyopicDenseExtractor, SparseMatrix_internal::DenseFull>(row, full_non_target, false, set_extract_all(opt), full_non_target);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::MyopicDenseExtractor, SparseMatrix_internal::DenseBlock>(row, block_length, false, set_extract_all(opt), block_start, block_length);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto nidx = indices_ptr->size();
        return populate<false, tatami::MyopicDenseExtractor, SparseMatrix_internal::DenseIndex>(row, nidx, false, set_extract_all(opt), std::move(indices_ptr));
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<false, tatami::MyopicSparseExtractor, SparseMatrix_internal::SparseFull>(row, full_non_target, false, opt, full_non_target);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::MyopicSparseExtractor, SparseMatrix_internal::SparseBlock>(row, block_length, false, opt, block_start, block_length);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto nidx = indices_ptr->size();
        return populate<false, tatami::MyopicSparseExtractor, SparseMatrix_internal::SparseIndex>(row, nidx, false, opt, std::move(indices_ptr));
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        const tatami::Options& opt) 
    const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<true, tatami::OracularDenseExtractor, SparseMatrix_internal::DenseFull>(row, full_non_target, std::move(oracle), set_extract_all(opt), full_non_target);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return populate<true, tatami::OracularDenseExtractor, SparseMatrix_internal::DenseBlock>(row, block_length, std::move(oracle), set_extract_all(opt), block_start, block_length);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto nidx = indices_ptr->size();
        return populate<true, tatami::OracularDenseExtractor, SparseMatrix_internal::DenseIndex>(row, nidx, std::move(oracle), set_extract_all(opt), std::move(indices_ptr));
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
        return populate<true, tatami::OracularSparseExtractor, SparseMatrix_internal::SparseFull>(row, full_non_target, std::move(oracle), opt, full_non_target);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return populate<true, tatami::OracularSparseExtractor, SparseMatrix_internal::SparseBlock>(row, block_length, std::move(oracle), opt, block_start, block_length);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto nidx = indices_ptr->size();
        return populate<true, tatami::OracularSparseExtractor, SparseMatrix_internal::SparseIndex>(row, nidx, std::move(oracle), opt, std::move(indices_ptr));
    }
};

}

#endif
