#ifndef TATAMI_TILEDB_SPARSE_MATRIX_HPP
#define TATAMI_TILEDB_SPARSE_MATRIX_HPP

#include "tatami_chunked/tatami_chunked.hpp"
#include <tiledb/tiledb>

#include "serialize.hpp"

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>

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
    std::vector<CachedValue_> values;
    
    // We use 'int' instead of CachedIndex_ as we don't know the
    // maximum range of the indices on-disk after we add an offset.
    std::vector<int> target_indices;
    std::vector<int> non_target_indices;
};

// All TileDB-related members.
struct Components{
    Components(const std::string& location) : array(ctx, location, TILEDB_READ) {}
    tiledb::Context ctx;
    tiledb::Array array;
};

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
    tiledb::Query query(comp.ctx, comp.array);
    query.set_subarray(subarray)
        .set_data_buffer(attribute, work.values);

    // Don't extract indices directly into the cache, because the reported
    // indices may be out of range of CachedIndex_ due to TileDB's offsets. 
    if (row) {
        query.set_data_buffer(first_dimname, work.target_indices)
            .set_layout(TILEDB_ROW_MAJOR);
        if (!cache.indices.empty()) {
            query.set_data_buffer(second_dimname, work.non_target_indices)
        }
    } else {
        query.set_data_buffer(second_dimname, work.target_indices)
            .set_layout(TILEDB_COL_MAJOR);
        if (!cache.indices.empty()) {
            query.set_data_buffer(first_dimname, work.non_target_indices)
        }
    }

    if (query.submit() != tiledb::Query::Status::COMPLETE) {
        throw std::runtime_error("failed to read sparse data from TileDB");
    }

    size_t result_num = query.result_buffer_elements()[attribute].second;
    std::fill_n(cache.number, target_length, 0);

    // Better be in CSR (if row = true) or CSC (otherwise) order.
    if (cache.indices.empty() && cache.values.empty()) { 
        populate_block_cache<false, false>(result_num, work, cache, subtract_target, subtract_non_target);
    } else if (cache.indices.empty()) {
        populate_block_cache<true, false>(result_num, work, cache, subtract_target, subtract_non_target);
    } else if (cache.values.empty()) { 
        populate_block_cache<false, true>(result_num, work, cache, subtract_target, subtract_non_target);
    } else {
        populate_block_cache<true, true>(result_num, work, cache, subtract_target, subtract_non_target);
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

    tiledb::Subarray subarray(comp.ctx, comp.array);
    auto actual_target_start = target_offset + target_start; 
    subarray.add_range(1 - rowdex, actual_target_start, actual_target_start + target_length - 1);

    auto actual_non_target_start = non_target_offset + block_start; 
    subarray.add_range(rowdex, actual_non_target_start, actual_non_target_start + block_length - 1);

    execute_query(tdbcomp, subarray, attribute, first_dimname, second_dimname, row, target_length, actual_target_start, non_target_offset, work, cache);
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

    tiledb::Subarray subarray(comp.ctx, comp.array);
    auto actual_target_start = target_offset + target_start; 
    subarray.add_range(1 - rowdex, actual_target_start, actual_target_start + target_length - 1);

    tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(), [&](Index_ s, Index_ l) {
        auto actual_non_target_start = non_target_offset + s;
        subarray.add_range(rowdex, actual_non_target_start, actual_non_target_start + l - 1);
    });

    execute_query(tdbcomp, subarray, attribute, first_dimname, second_dimname, row, target_length, actual_target_start, non_target_offset, work, cache);
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
        [[maybe_unused]] Index_ non_target_length, 
        int non_target_offset,
        [[maybe_unused]] const tatami_chunked::SlabCacheStats& slab_stats,
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
        my_needs_index(needs_index)
    {
        if (my_needs_value) {
            my_stub_slab.values.resize(1);
        }
        if (my_needs_index) {
            my_stub_slab.indices.resize(1);
        }
    }

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

private:
    template<typename Length_, class Extract_>
    void fetch_raw(Index_ i, Length_ non_target_length, Extract_ extract) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }

        placeholder.number = &count;
        if (my_needs_value) {
            vbuffer.resize(non_target_length);
            placeholder.value[0] = vbuffer.data();
        }
        if (my_needs_index) {
            ibuffer.resize(non_target_length);
            placeholder.index[0] = ibuffer.data();
        }

        serialize([&](){
            extract(i);
        });
    }

public:
    std::pair<const Slab*, Index_> fetch_block(Index_ i, Index_ block_start, Index_ block_length) {
        fetch_raw(i, block_length, [&](Index_ i0) {
            extract_block(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                i0,
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
        fetch_raw(i, indices.size(), [&](Index_ i0) {
            extract_indices(
                my_tdbcomp,
                my_attribute,
                my_first_dimname,
                my_second_dimname,
                my_row,
                i0,
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
    typedef decltype(my_factory)::Slab Slab;
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
            extract_block(
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
            extract_indices(
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
    tatami_chunked::SparseSlabFactory<CachedValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> my_cache;

private:
    template<class Extract_>
    std::pair<const Slab*, Index_> fetch_raw([[maybe_unused]] Index_ i, Extract_ extract) {
        return info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_target_dim_stats.chunk_length, current % my_target_dim_stats.chunk_length);
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
    }

public:
    std::pair<const Slab*, Index_> fetch_block(Index_ i, Index_ block_start, Index_ block_length) {
        return fetch_raw(i, [&](Index_ start, Index_ len, Slab& contents) {
            extract_block(
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
                contents,
                my_tdbcomp
            );
        });
    }

    std::pair<const Slab*, Index_> fetch_indices(Index_ i, const std::vector<Index_>& indices) {
        return fetch_raw(i, [&](Index_ start, Index_ len, Slab& contents) {
            extract_indices(
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
                contents,
                my_tdbcomp
            );
        });
    }
};

template<bool solo_, bool oracle_, typename Index_, typename CachedValue_>
using SparseCore = typename std::conditional<solo_, 
      SoloCore<oracle_, Index_>,
      typename std::conditional<oracle_,
          OracularCore<Index_, CachedValue_>,
          MyopicCore<Index_, CachedValue_>
      >::type
>::type;

/*************************
 *** Sparse subclasses ***
 *************************/

template<typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_> 
tatami::SparseRange<Value_, Index_> slab_to_sparse(const Slab<CachedValue_, CachedIndex_>& slab, Index_ index, Value_* vbuffer, Index_* ibuffer) {
    tatami::SparseRange<Value_, Index_> output;
    output.number = slab.number[index];
    if (!slab.value.empty()) {
        std::copy_n(slab.value[index], output.number, vbuffer);
        output.value = vbuffer;
    }
    if (!slab.index.empty()) {
        std::copy_n(slab.index[index], output.number, vbuffer);
        output.index = ibuffer;
    }
    return output;
}

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
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
        return slab_to_sparse<Value_, Index_>(*(info.first), info.second, vbuffer, ibuffer);
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_> 
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
            by_tdb_row,
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
        return slab_to_sparse<Value_, Index_>(*(info.first), info.second, vbuffer, ibuffer);
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SparseIndex : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseIndex(
        const Components& tdbcomp,
        const std::string& attribute, 
        bool by_tdb_row,
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
            by_tdb_row,
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
        return slab_to_sparse<Value_, Index_>(*(info.first), info.second, vbuffer, ibuffer);
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
};

/************************
 *** Dense subclasses ***
 ************************/

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
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
        const tatami_chunked::SlabCacheStats& slab_stats) :
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
        auto vptr = slab.value[info.second];
        auto iptr = slab.index[info.second];
        auto n = slab.number[info.second];

        std::fill_n(buffer, my_non_target_dim, 0);
        for (decltype(n) j = 0; j < n; ++j) {
            buffer[iptr[j]] = vptr[j];
        }
        return buffer;
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_> 
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
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core( 
            tdbcomp,
            attribute,
            by_tdb_row,
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
        auto vptr = slab.value[info.second];
        auto iptr = slab.index[info.second];
        auto n = slab.number[info.second];

        std::fill_n(buffer, my_block_length, 0);
        for (decltype(n) j = 0; j < n; ++j) {
            buffer[iptr[j] - block_start] = vptr[j];
        }
        return buffer;
    }

private:
    SparseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class DenseIndex : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseIndex(
        const Components& tdbcomp,
        const std::string& attribute, 
        bool by_tdb_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        int target_offset, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        int non_target_offset, 
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core(
            tdbcomp,
            attribute,
            by_tdb_row,
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
            auto vptr = slab.value[info.second];
            auto iptr = slab.index[info.second];
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
    SparseCore<solo_, oracle_, Index_, CachedValue_> my_core;
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
 * @tparam transpose_ Whether to transpose the on-disk data upon loading.
 * By default, this is `false`, so the first dimension corresponds to rows and the second dimension corresponds to columns.
 * If `true`, the first dimension corresponds to columns and the second dimension corresponds to rows.
 *
 * Numeric sparse matrix stored in a 2-dimensional TileDB array.
 * Chunks of data are loaded from file as needed by `tatami::Extractor` objects, with additional caching to avoid repeated reads from disk.
 * The size of cached chunks is determined by the extent of the tiles in the TileDB array.
 *
 * The TileDB library is thread-safe so no additional work is required to use this class in parallel code.
 * Nonetheless, users can force all calls to TileDB to occur in serial by defining the `TATAMI_TILEDB_PARALLEL_LOCK` macro.
 * This should be a function-like macro that accepts a function and executes it inside a user-defined serial section.
 */
template<typename Value_, typename Index_, bool transpose_ = false>
class TileDbSparseMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     * @param options Further options.
     */
    TileDbSparseMatrix(std::string uri, std::string attribute, const TileDbOptions& options) : location(std::move(uri)), attr(std::move(attribute)) {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        tiledb::Context ctx;
        tiledb::ArraySchema schema(ctx, location);
        if (schema.array_type() != TILEDB_SPARSE) {
            throw std::runtime_error("TileDB array should be sparse");
        }
        initialize(schema, options);

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        });
#endif
    }

    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     */
    TileDbSparseMatrix(std::string uri, std::string attribute) : TileDbSparseMatrix(std::move(uri), std::move(attribute), TileDbOptions()) {}

    /**
     * @cond
     */
    TileDbSparseMatrix(tiledb::ArraySchema& schema, std::string uri, std::string attribute, const TileDbOptions& options) : location(std::move(uri)), attr(std::move(attribute)) {
        initialize(schema, options);
    }
    /**
     * @endcond
     */

private:
    void initialize(tiledb::ArraySchema& schema, const TileDbOptions& options) {
        cache_size_in_elements = static_cast<double>(options.maximum_cache_size) / (sizeof(Value_) + sizeof(Index_));
        require_minimum_cache = options.require_minimum_cache;

        if (!schema.has_attribute(attr)) {
            throw std::runtime_error("no attribute '" + attr + "' is present in the TileDB array at '" + location + "'");
        }

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
            first_offset = domain.first;
            first_dim = domain.second - domain.first + 1;
            first_tile = dim.tile_extent<int>();
            first_dimname = dim.name();
        }

        {
            tiledb::Dimension dim = domain.dimension(1);
            auto domain = dim.domain<int>();
            second_offset = domain.first;
            second_dim = domain.second - domain.first + 1;
            second_tile = dim.tile_extent<int>();
            second_dimname = dim.name();
        }

        // Favoring extraction on the dimension that involves pulling out fewer chunks per dimension element.
        auto tiles_per_firstdim = static_cast<double>(second_dim) / second_tile;
        auto tiles_per_seconddim = static_cast<double>(first_dim) / first_tile;
        prefer_firstdim = tiles_per_firstdim <= tiles_per_seconddim;
    }

private:
    std::string location, attr;
    size_t cache_size_in_elements;
    bool require_minimum_cache;

    int first_offset, second_offset;
    std::string first_dimname, second_dimname;
    Index_ first_dim, second_dim;
    Index_ first_tile, second_tile;
    bool sparse_internal, prefer_firstdim;

    template<bool accrow_>
    Index_ get_target_dim() const {
        if constexpr(accrow_ != transpose_) {
            return first_dim;
        } else {
            return second_dim;
        }
    }

    template<bool accrow_>
    Index_ get_target_chunk_dim() const {
        if constexpr(accrow_ != transpose_) {
            return first_tile;
        } else {
            return second_tile;
        }
    }

public:
    Index_ nrow() const {
        if constexpr(transpose_) {
            return second_dim;
        } else {
            return first_dim;
        }
    }

    Index_ ncol() const {
        if constexpr(transpose_) {
            return first_dim;
        } else {
            return second_dim;
        }
    }

    bool sparse() const {
        return true;
    }

    double sparse_proportion() const {
        return 1;
    }

    bool prefer_rows() const {
        return transpose_ != prefer_firstdim;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(transpose_ != prefer_firstdim);
    }

    bool uses_oracle(bool) const {
        // It won't necessarily be used, but if the cache is empty,
        // the oracle definitely _won't_ be used.
        return cache_size_in_elements > 0;
    }

    using tatami::Matrix<Value_, Index_>::dense_row;

    using tatami::Matrix<Value_, Index_>::dense_column;

    using tatami::Matrix<Value_, Index_>::sparse_row;

    using tatami::Matrix<Value_, Index_>::sparse_column;

public:
    /*************************************************
     * Defines the TileDB workspace and chunk cache. *
     *************************************************/

    struct Slab {
        Slab() = default;
        Slab(size_t nnz, size_t ne) : values(nnz), indices(nnz), indptrs(ne + 1) {}

        std::vector<Value_> values;
        std::vector<Index_> indices;
        std::vector<size_t> indptrs;
    };

    template<bool accrow_>
    struct Workspace {
        Workspace(const TileDbSparseMatrix* parent) : array(ctx, parent->location, TILEDB_READ) {}

        void set_cache(const TileDbSparseMatrix* parent, Index_ other_dim) {
            auto chunk_dim = parent->template get_target_chunk_dim<accrow_>();
            cache_workspace = tatami_chunked::TypicalSlabCacheWorkspace<Index_, Slab>(chunk_dim, other_dim, parent->cache_size_in_elements, parent->require_minimum_cache);

            if (cache_workspace.num_slabs_in_cache == 0) {
                uncached.reset(new Slab(other_dim, 1));
                holding_coords.resize(other_dim);
            } else {
                holding_coords.resize(cache_workspace.slab_size_in_elements);
            }
        }

    public:
        // TileDB members.
        tiledb::Context ctx;
        tiledb::Array array;

        std::vector<int> holding_coords; // TODO: figure out what the maximum range of the coordinates can be.

        // Caching members.
        tatami_chunked::TypicalSlabCacheWorkspace<Index_, Slab> cache_workspace;

        std::unique_ptr<Slab> uncached;
    };

private:
    /********************************
     * Defines extraction functions *
     ********************************/

    template<bool accrow_, typename ExtractType_>
    void extract_base(Index_ primary_start, Index_ primary_end, Slab& target, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        tiledb::Subarray subarray(work.ctx, work.array);

        constexpr int dimdex = (accrow_ != transpose_);
        auto primary_offset = (dimdex == 1 ? first_offset : second_offset);
        subarray.add_range(1 - dimdex, primary_offset + primary_start, primary_offset + primary_end - 1);

        // Adding ranges along the other dimension.
        auto secondary_offset = (dimdex == 1 ? second_offset : first_offset);
        constexpr bool indexed = std::is_same<ExtractType_, std::vector<Index_> >::value;
        if constexpr(indexed) {
            tatami::process_consecutive_indices(extract_value.data(), extract_length,
                [&](Index_ s, Index_ l) {
                    auto start = secondary_offset + s;
                    subarray.add_range(dimdex, start, start + l - 1);
                }
            );
        } else {
            auto secondary_start = secondary_offset + extract_value;
            subarray.add_range(dimdex, secondary_start, secondary_start + extract_length - 1);
        }
    }

    template<bool accrow_, typename ExtractType_>
    std::pair<const Slab*, Index_> extract_without_cache(Index_ i, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        extract_base<accrow_>(i, i + 1, *(work.uncached), extract_value, extract_length, work);

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        });
#endif
        return std::pair<const Slab*, Index_>(work.uncached.get(), 0);
    }

    template<bool accrow_, typename ExtractType_>
    void extract_chunk(Index_ chunk_id, Index_ dim, Index_ chunk_dim, Slab& current_chunk, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        Index_ chunk_start = chunk_id * chunk_dim;
        Index_ chunk_end = std::min(dim, chunk_start + chunk_dim);
        extract_base<accrow_>(chunk_start, chunk_end, current_chunk, extract_value, extract_length, work);
    }

    template<bool accrow_, typename ExtractType_>
    std::pair<const Slab*, Index_> extract_with_oracle(Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        return work.cache_workspace.oracle_cache->next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / chunk_mydim, current % chunk_mydim);
            },
            /* create = */ [&]() -> Slab {
                return Slab(work.cache_workspace.slab_size_in_elements, chunk_mydim);
            },
            /* populate = */ [&](const std::vector<std::pair<Index_, Index_> >& chunks_in_need, std::vector<Slab*>& chunk_data) -> void {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

                for (const auto& c : chunks_in_need) {
                    auto& cache_target = *(chunk_data[c.second]);
                    extract_chunk<accrow_>(c.first, mydim, chunk_mydim, cache_target, extract_value, extract_length, work);
                }

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                });
#endif
            }
        );
    }

    template<bool accrow_, typename ExtractType_>
    std::pair<const Slab*, Index_> extract_without_oracle(Index_ i, Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        auto chunk = i / chunk_mydim;
        auto index = i % chunk_mydim;

        const auto& cache_target = work.cache_workspace.lru_cache->find(
            chunk,
            /* create = */ [&]() -> Slab {
                return Slab(work.cache_workspace.slab_size_in_elements, chunk_mydim);
            },
            /* populate = */ [&](Index_ id, Slab& chunk_contents) -> void {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

                extract_chunk<accrow_>(id, mydim, chunk_mydim, chunk_contents, extract_value, extract_length, work);

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                });
#endif
            }
        );

        return std::pair<const Slab*, Index_>(&cache_target, index);
    }

private:
    template<bool accrow_, typename ExtractType_>
    std::pair<const Slab*, Index_> extract(Index_ i, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        if (work.cache_workspace.num_slabs_in_cache == 0) {
            return extract_without_cache(i, extract_value, extract_length, work);
        } else {
            Index_ mydim = get_target_dim<accrow_>();
            Index_ tile_mydim = get_target_chunk_dim<accrow_>();
            if (work.cache_workspace.oracle_cache) {
                return extract_with_oracle(mydim, tile_mydim, extract_value, extract_length, work);
            } else {
                return extract_without_oracle(i, mydim, tile_mydim, extract_value, extract_length, work);
            }
        }
    }

    template<bool accrow_, typename ExtractType_>
    Value_* extract_dense(Index_ i, Value_* buffer, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        auto out = extract(i, extract_value, extract_length, work);
        const auto& values = out.first->values;
        const auto& indices = out.first->indices;
        const auto& indptrs = out.first->indptrs;

        auto idx = out.second;
        size_t start = indptrs[idx], end = indptrs[idx+1];

        constexpr bool indexed = std::is_same<ExtractType_, std::vector<Index_> >::value;
        if constexpr(indexed) {
            size_t x = start;
            for (Index_ i = 0; i < extract_length; ++i) {
                if (x < end && indices[x] == extract_value[i]) {
                    buffer[i] = values[x];
                    ++x;
                } else {
                    buffer[i] = 0;
                }
            }
        } else {
            std::fill(buffer, buffer + extract_length, static_cast<Value_>(0));
            for (size_t x = start; x < end; ++x) {
                buffer[indices[x] - extract_value] = values[x];
            }
        }

        return buffer;
    }

    template<bool accrow_, typename ExtractType_>
    tatami::SparseRange<Value_, Index_> extract_sparse(
        Index_ i, Value_* vbuffer, Index_* ibuffer, 
        const ExtractType_& extract_value, Index_ extract_length, 
        Workspace<accrow_>& work,
        bool needs_value, bool needs_index) const 
    {
        auto out = extract(i, extract_value, extract_length, work);
        const auto& values = out.first->values;
        const auto& indices = out.first->indices;
        const auto& indptrs = out.first->indptrs;

        auto idx = out.second;
        size_t start = indptrs[idx], end = indptrs[idx+1];
        tatami::SparseRange<Value_, Index_> output;
        output.number = end - start;

        if (needs_value) {
            std::copy(values.begin() + start, values.begin() + end, vbuffer);
            output.value = vbuffer;
        } else {
            output.value = NULL;
        }

        if (needs_index) {
            // No conversion of indices based on 'extract_value' is required,
            // because TileDB gives the indices to us as-is.
            std::copy(indices.begin() + start, indices.begin() + end, ibuffer);
            output.index = ibuffer;
        } else {
            output.index = NULL;
        }

        return output;
    }

private:
    /*************************************
     * Defines the extractors themselves *
     *************************************/

    template<bool accrow_, tatami::DimensionSelectionType selection_, bool sparse_>
    struct TileDbExtractor : public tatami::Extractor<selection_, sparse_, Value_, Index_> {
        TileDbExtractor(const TileDbSparseMatrix* p) : parent(p), base(parent) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                this->full_length = (accrow_ ? parent->ncol() : parent->nrow());
                base.set_cache(parent, this->full_length);
            }
        }

        TileDbExtractor(const TileDbSparseMatrix* p, Index_ start, Index_ length) : parent(p), base(parent) {
            if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                this->block_start = start;
                this->block_length = length;
                base.set_cache(parent, this->block_length);
            }
        }

        TileDbExtractor(const TileDbSparseMatrix* p, std::vector<Index_> idx) : parent(p), base(parent) {
            if constexpr(selection_ == tatami::DimensionSelectionType::INDEX) {
                this->index_length = idx.size();
                indices = std::move(idx);
                base.set_cache(parent, this->index_length);
            }
        }

    protected:
        const TileDbSparseMatrix* parent;
        Workspace<accrow_> base;
        typename std::conditional<selection_ == tatami::DimensionSelectionType::INDEX, std::vector<Index_>, bool>::type indices;

    public:
        const Index_* index_start() const {
            if constexpr(selection_ == tatami::DimensionSelectionType::INDEX) {
                return indices.data();
            } else {
                return NULL;
            }
        }

        void set_oracle(std::unique_ptr<tatami::Oracle<Index_> > o) {
            base.cache_workspace.set_oracle(std::move(o));
        }
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_>
    struct DenseExtractor : public TileDbExtractor<accrow_, selection_, false> {
        template<typename... Args_>
        DenseExtractor(const TileDbSparseMatrix* p, Args_&&... args) : 
            TileDbExtractor<accrow_, selection_, false>(p, std::forward<Args_>(args)...) {}

        const Value_* fetch(Index_ i, Value_* buffer) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                return this->parent->template extract_dense<accrow_>(i, buffer, 0, this->full_length, this->base);
            } else if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                return this->parent->template extract_dense<accrow_>(i, buffer, this->block_start, this->block_length, this->base);
            } else {
                return this->parent->template extract_dense<accrow_>(i, buffer, this->indices, this->index_length, this->base);
            }
        }
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_>
    struct SparseExtractor : public TileDbExtractor<accrow_, selection_, true> {
        template<typename... Args_>
        SparseExtractor(const TileDbSparseMatrix* p, const tatami::Options& opt, Args_&&... args) : 
            TileDbExtractor<accrow_, selection_, true>(p, std::forward<Args_>(args)...), 
            needs_index(opt.sparse_extract_index), 
            needs_value(opt.sparse_extract_value) 
        {}

        tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                return this->parent->template extract_sparse<accrow_>(i, vbuffer, ibuffer, 0, this->full_length, this->base, needs_value, needs_index);
            } else if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                return this->parent->template extract_sparse<accrow_>(i, vbuffer, ibuffer, this->block_start, this->block_length, this->base, needs_value, needs_index);
            } else {
                return this->parent->template extract_sparse<accrow_>(i, vbuffer, ibuffer, this->indices, this->index_length, this->base, needs_value, needs_index);
            }
        }

    private:
        bool needs_index;
        bool needs_value;
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_, bool sparse_, typename ... Args_>
    std::unique_ptr<tatami::Extractor<selection_, sparse_, Value_, Index_> > populate(const tatami::Options& opt, Args_&&... args) const {
        std::unique_ptr<tatami::Extractor<selection_, sparse_, Value_, Index_> > output;

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        if constexpr(sparse_) {
            output.reset(new SparseExtractor<accrow_, selection_>(this, opt, std::forward<Args_>(args)...));
        } else {
            output.reset(new DenseExtractor<accrow_, selection_>(this, std::forward<Args_>(args)...));
        }

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        });
#endif

        return output;
    }

public:
    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_row(const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::FULL, false>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_row(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::BLOCK, false>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_row(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::INDEX, false>(opt, std::move(indices));
    }

    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_column(const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::FULL, false>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_column(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::BLOCK, false>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_column(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::INDEX, false>(opt, std::move(indices));
    }

public:
    std::unique_ptr<tatami::FullSparseExtractor<Value_, Index_> > sparse_row(const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<tatami::BlockSparseExtractor<Value_, Index_> > sparse_row(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexSparseExtractor<Value_, Index_> > sparse_row(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::INDEX, true>(opt, std::move(indices));
    }

    std::unique_ptr<tatami::FullSparseExtractor<Value_, Index_> > sparse_column(const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<tatami::BlockSparseExtractor<Value_, Index_> > sparse_column(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexSparseExtractor<Value_, Index_> > sparse_column(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::INDEX, true>(opt, std::move(indices));
    }
};

}

#endif
