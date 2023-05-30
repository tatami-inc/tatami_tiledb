#ifndef TATAMI_TILEDB_DENSE_MATRIX_HPP
#define TATAMI_TILEDB_DENSE_MATRIX_HPP

#include "tatami/tatami.hpp"
#include <tiledb/tiledb>

#include <string>
#include <memory>
#include <vector>

/**
 * @file TileDbSparseMatrix.hpp
 * @brief TileDB-backed dense matrix.
 */

namespace tatami_tiledb {

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
 */
template<typename Value_, typename Index_, bool transpose_ = false>
class TileDbSparseMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     * @param cache_limit Size of the chunk cache in bytes.
     */
    TileDbSparseMatrix(std::string uri, std::string attribute, size_t cache_limit = 100000000) : 
        location(std::move(uri)), 
        attr(std::move(attribute)), 
        cache_size_in_elements(static_cast<double>(cache_limit) / (sizeof(Value_) + sizeof(Index_)))
    {
        tiledb::Context ctx;
        tiledb::ArraySchema schema(ctx, location);

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
    struct Chunk {
        Chunk() = default;
        Chunk(size_t nnz, size_t ne) : values(nnz), indices(nnz), indptrs(ne + 1) {}

        std::vector<Value_> values;
        std::vector<Index_> indices;
        std::vector<size_t> indptrs;

        void swap(Chunk& right) {
            values.swap(right.values);
            indices.swap(right.indices);
            indptrs.swap(right.indptrs);
        }
    };

    template<bool accrow_>
    using OracleCache = tatami::OracleChunkCache<Index_, Index_, Chunk>; 

    template<bool accrow_>
    using LruCache = tatami::LruChunkCache<Index_, Chunk>;

    template<bool accrow_>
    struct Workspace {
        Workspace(const TileDbSparseMatrix* parent) : array(ctx, parent->location, TILEDB_READ) {}

        void set_cache(const TileDbSparseMatrix* parent, Index_ other_dim) {
            auto chunk_dim = parent->template get_target_chunk_dim<accrow_>();
            chunk_size_in_elements = static_cast<size_t>(chunk_dim) * static_cast<size_t>(other_dim);
            num_chunks_in_cache = static_cast<double>(parent->cache_size_in_elements) / chunk_size_in_elements;

            if (num_chunks_in_cache > 0) {
                historian.reset(new LruCache<accrow_>(num_chunks_in_cache));
                holding_coords.resize(chunk_size_in_elements);
            } else {
                uncached.reset(new Chunk(other_dim, 1));
                holding_coords.resize(other_dim);
            }
        }

    public:
        // TileDB members.
        tiledb::Context ctx;
        tiledb::Array array;

        // TODO: figure out what the maximum range of the coordinates can be.
        std::vector<int> holding_coords;

    public:
        // Caching members.
        size_t chunk_size_in_elements;
        Index_ num_chunks_in_cache;

        // Uncached.
        std::unique_ptr<Chunk> uncached;

        // Cache with an oracle.
        std::unique_ptr<OracleCache<accrow_> > futurist;

        // Cache without an oracle.
        std::unique_ptr<LruCache<accrow_> > historian;
    };

private:
    template<bool accrow_, typename ExtractType_>
    void extract_base(Index_ primary_start, Index_ primary_end, Chunk& target, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
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

        tiledb::Query query(work.ctx, work.array);
        query.set_subarray(subarray)
            .set_layout(dimdex == 1 ? TILEDB_ROW_MAJOR : TILEDB_COL_MAJOR)
            .set_data_buffer(attr, target.values);

        if constexpr(dimdex == 1) {
            query.set_data_buffer(second_dimname, target.indices)
                .set_data_buffer(first_dimname, work.holding_coords);
        } else {
            query.set_data_buffer(first_dimname, target.indices)
                .set_data_buffer(second_dimname, work.holding_coords);
        }

        if (query.submit() != tiledb::Query::Status::COMPLETE) {
            throw std::runtime_error("failed to read sparse data from TileDB");
        }

        size_t result_num = query.result_buffer_elements()[attr].second;
        target.indptrs.clear();
        target.indptrs.resize(primary_end - primary_start + 1);

        if constexpr(dimdex == 1) {
            // Better be in CSR order.
            for (size_t r = 0; r < result_num; ++r) {
                target.indices[r] -= second_offset;
                Index_ i = work.holding_coords[r] - first_offset - primary_start;
                ++(target.indptrs[i + 1]);
            }
        } else {
            // Better be in CSC order.
            for (size_t r = 0; r < result_num; ++r) {
                target.indices[r] -= first_offset;
                Index_ j = work.holding_coords[r] - second_offset - primary_start;
                ++(target.indptrs[j + 1]);
            }
        }

        for (size_t i = 1; i < target.indptrs.size(); ++i) {
            target.indptrs[i] += target.indptrs[i-1];
        }
    }

    template<bool accrow_, typename ExtractType_>
    std::pair<const Chunk*, Index_> extract_without_cache(Index_ i, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
#ifndef TATAMI_TILEDB_PARALLEL_LOCK
        #pragma omp critical
        {
#else
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        extract_base<accrow_>(i, i + 1, *(work.uncached), extract_value, extract_length, work);

#ifndef TATAMI_TILEDB_PARALLEL_LOCK
        }
#else
        });
#endif
        return std::pair<const Chunk*, Index_>(work.uncached.get(), 0);
    }

    template<bool accrow_, typename ExtractType_>
    void extract_chunk(Index_ chunk_id, Index_ dim, Index_ chunk_dim, Chunk& current_chunk, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        Index_ chunk_start = chunk_id * chunk_dim;
        Index_ chunk_end = std::min(dim, chunk_start + chunk_dim);
        extract_base<accrow_>(chunk_start, chunk_end, current_chunk, extract_value, extract_length, work);
    }

    template<bool accrow_, typename ExtractType_>
    std::pair<const Chunk*, Index_> extract_with_oracle(Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        return work.futurist->next_chunk(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / chunk_mydim, current % chunk_mydim);
            },
            /* swap = */ [](Chunk& left, Chunk& right) -> void {
                left.swap(right);
            },
            /* ready = */ [](const Chunk& x) -> bool {
                return !x.indptrs.empty();
            },
            /* allocate = */ [&](Chunk& x) -> void {
                x.values.resize(work.chunk_size_in_elements);
                x.indices.resize(work.chunk_size_in_elements);
                x.indptrs.resize(chunk_mydim + 1);
            },
            /* populate = */ [&](const std::vector<std::pair<Index_, Index_> >& chunks_in_need, std::vector<Chunk>& chunk_data) -> void {
#ifndef TATAMI_TILEDB_PARALLEL_LOCK
                #pragma omp critical
                {
#else
                TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

                for (const auto& c : chunks_in_need) {
                    auto& cache_target = chunk_data[c.second];
                    extract_chunk<accrow_>(c.first, mydim, chunk_mydim, cache_target, extract_value, extract_length, work);
                }

#ifndef TATAMI_TILEDB_PARALLEL_LOCK
                }
#else
                });
#endif
            }
        );
    }

    template<bool accrow_, typename ExtractType_>
    std::pair<const Chunk*, Index_> extract_without_oracle(Index_ i, Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        auto chunk = i / chunk_mydim;
        auto index = i % chunk_mydim;

        const auto& cache_target = work.historian->find_chunk(
            chunk,
            /* create = */ [&]() -> Chunk {
                return Chunk(work.chunk_size_in_elements, chunk_mydim);
            },
            /* populate = */ [&](Index_ id, Chunk& chunk_contents) -> void {
                Index_ actual_dim;

#ifndef TATAMI_TILEDB_PARALLEL_LOCK
                #pragma omp critical
                {
#else
                TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

                extract_chunk<accrow_>(chunk, mydim, chunk_mydim, chunk_contents, extract_value, extract_length, work);

#ifndef TATAMI_TILEDB_PARALLEL_LOCK
                }
#else
                });
#endif
            }
        );

        return std::pair<const Chunk*, Index_>(&cache_target, index);
    }

private:
    template<bool accrow_, typename ExtractType_>
    std::pair<const Chunk*, Index_> extract(Index_ i, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        if (work.num_chunks_in_cache == 0) {
            return extract_without_cache(i, extract_value, extract_length, work);
        } else {
            Index_ mydim = get_target_dim<accrow_>();
            Index_ tile_mydim = get_target_chunk_dim<accrow_>();
            if (work.futurist) {
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
            if (base.num_chunks_in_cache > 0) {
                auto chunk_mydim = parent->template get_target_chunk_dim<accrow_>();
                size_t max_predictions = static_cast<size_t>(base.num_chunks_in_cache) * chunk_mydim * 2; // double the cache size, basically.
                base.futurist.reset(new OracleCache<accrow_>(std::move(o), max_predictions, base.num_chunks_in_cache));
                base.historian.reset();
                base.uncached.reset();
            }
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

#ifndef TATAMI_TILEDB_PARALLEL_LOCK
        #pragma omp critical
        {
#else
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        if constexpr(sparse_) {
            output.reset(new SparseExtractor<accrow_, selection_>(this, opt, std::forward<Args_>(args)...));
        } else {
            output.reset(new DenseExtractor<accrow_, selection_>(this, std::forward<Args_>(args)...));
        }

#ifndef TATAMI_TILEDB_PARALLEL_LOCK
        }
#else
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
