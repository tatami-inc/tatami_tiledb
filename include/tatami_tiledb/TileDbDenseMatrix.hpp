#ifndef TATAMI_TILEDB_DENSE_MATRIX_HPP
#define TATAMI_TILEDB_DENSE_MATRIX_HPP

#include "tatami_chunked/tatami_chunked.hpp"
#include <tiledb/tiledb>

#include "TileDbOptions.hpp"

#include <string>
#include <memory>
#include <vector>

/**
 * @file TileDbDenseMatrix.hpp
 * @brief TileDB-backed dense matrix.
 */

namespace tatami_tiledb {

/**
 * @brief TileDB-backed dense matrix.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam transpose_ Whether to transpose the on-disk data upon loading.
 * By default, this is `false`, so the first dimension corresponds to rows and the second dimension corresponds to columns.
 * If `true`, the first dimension corresponds to columns and the second dimension corresponds to rows.
 *
 * Numeric dense matrix stored in a 2-dimensional TileDB array.
 * Chunks of data are loaded from file as needed by `tatami::Extractor` objects, with additional caching to avoid repeated reads from disk.
 * The size of cached chunks is determined by the extent of the tiles in the TileDB array.
 *
 * The TileDB library is thread-safe so no additional work is required to use this class in parallel code.
 * Nonetheless, users can force all calls to TileDB to occur in serial by defining the `TATAMI_TILEDB_PARALLEL_LOCK` macro.
 * This should be a function-like macro that accepts a function and executes it inside a user-defined serial section.
 */
template<typename Value_, typename Index_, bool transpose_ = false>
class TileDbDenseMatrix : public tatami::VirtualDenseMatrix<Value_, Index_> {
public:
    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     * @param options Further options.
     */
    TileDbDenseMatrix(std::string uri, std::string attribute, const TileDbOptions& options) : location(std::move(uri)), attr(std::move(attribute)) {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        tiledb::Context ctx;
        tiledb::ArraySchema schema(ctx, location);
        if (schema.array_type() != TILEDB_DENSE) {
            throw std::runtime_error("TileDB array should be dense");
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
    TileDbDenseMatrix(std::string uri, std::string attribute) : TileDbDenseMatrix(std::move(uri), std::move(attribute), TileDbOptions()) {}

    /**
     * @cond
     */
    TileDbDenseMatrix(tiledb::ArraySchema& schema, std::string uri, std::string attribute, const TileDbOptions& options) : location(std::move(uri)), attr(std::move(attribute)) {
        initialize(schema, options);
    }
    /**
     * @endcond
     */

private:
    void initialize(tiledb::ArraySchema& schema, const TileDbOptions& options) {
        cache_size_in_elements = static_cast<double>(options.maximum_cache_size) / sizeof(Value_);
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
        }

        {
            tiledb::Dimension dim = domain.dimension(1);
            auto domain = dim.domain<int>();
            second_offset = domain.first;
            second_dim = domain.second - domain.first + 1;
            second_tile = dim.tile_extent<int>();
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
        return false;
    }

    double sparse_proportion() const {
        return 0;
    }

    bool prefer_rows() const {
        return transpose_ != prefer_firstdim;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(transpose_ != prefer_firstdim);
    }

    bool uses_oracle(bool) const {
        // The oracle won't necessarily be used if the cache size is non-zero,
        // but if the cache is empty, the oracle definitely _won't_ be used.
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

    typedef std::vector<Value_> Slab;

    template<bool accrow_>
    struct Workspace {
        Workspace(const TileDbDenseMatrix* parent) : array(ctx, parent->location, TILEDB_READ) {}

        void set_cache(const TileDbDenseMatrix* parent, Index_ other_dim) {
            auto chunk_dim = parent->template get_target_chunk_dim<accrow_>();
            cache_workspace = tatami_chunked::TypicalSlabCacheWorkspace<Index_, Slab>(chunk_dim, other_dim, parent->cache_size_in_elements, parent->require_minimum_cache);
        }

    public:
        // TileDB members.
        tiledb::Context ctx;
        tiledb::Array array;

        // Caching members
        tatami_chunked::TypicalSlabCacheWorkspace<Index_, Slab> cache_workspace;
    };

private:
    /********************************
     * Defines extraction functions *
     ********************************/

    template<bool accrow_, typename ExtractType_>
    void extract_base(Index_ primary_start, Index_ primary_end, Value_* target, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
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
            .set_data_buffer(attr, target, (primary_end - primary_start) * extract_length);

        if (query.submit() != tiledb::Query::Status::COMPLETE) {
            throw std::runtime_error("failed to read dense data from TileDB");
        }
    }

    template<bool accrow_, typename ExtractType_>
    const Value_* extract_without_cache(Index_ i, Value_* buffer, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        extract_base<accrow_>(i, i + 1, buffer, extract_value, extract_length, work);

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        });
#endif
        return buffer;
    }

    template<bool accrow_, typename ExtractType_>
    void extract_chunk(Index_ chunk_id, Index_ dim, Index_ chunk_dim, Value_* target, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        Index_ chunk_start = chunk_id * chunk_dim;
        Index_ chunk_end = std::min(dim, chunk_start + chunk_dim);
        extract_base<accrow_>(chunk_start, chunk_end, target, extract_value, extract_length, work);
    }

    template<bool accrow_, typename ExtractType_>
    const Value_* extract_with_oracle(Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        auto info = work.cache_workspace.oracle_cache->next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / chunk_mydim, current % chunk_mydim);
            },
            /* create = */ [&]() -> Slab {
                return Slab(work.cache_workspace.slab_size_in_elements);
            },
            /* populate = */ [&](const std::vector<std::pair<Index_, Index_> >& chunks_in_need, std::vector<Slab*>& chunk_data) -> void {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

                for (const auto& c : chunks_in_need) {
                    auto& cache_target = *(chunk_data[c.second]);
                    this->extract_chunk<accrow_>(c.first, mydim, chunk_mydim, cache_target.data(), extract_value, extract_length, work);
                }

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                });
#endif
            }
        );

        return info.first->data() + extract_length * info.second;
    }

    template<bool accrow_, typename ExtractType_>
    const Value_* extract_without_oracle(Index_ i, Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        auto chunk = i / chunk_mydim;
        auto index = i % chunk_mydim;

        const auto& cache_target = work.cache_workspace.lru_cache->find(
            chunk,
            /* create = */ [&]() -> Slab {
                return Slab(work.cache_workspace.slab_size_in_elements);
            },
            /* populate = */ [&](Index_ id, Slab& chunk_contents) -> void {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

                extract_chunk<accrow_>(id, mydim, chunk_mydim, chunk_contents.data(), extract_value, extract_length, work);

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
                });
#endif
            }
        );

        return cache_target.data() + index * extract_length;
    }

    template<bool accrow_, typename ExtractType_>
    const Value_* extract(Index_ i, Value_* buffer, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        // If there isn't any space for caching, we just extract directly.
        if (work.cache_workspace.num_slabs_in_cache == 0) {
            return extract_without_cache(i, buffer, extract_value, extract_length, work);
        }

        Index_ mydim = get_target_dim<accrow_>();
        Index_ tile_mydim = get_target_chunk_dim<accrow_>();

        const Value_* cache;
        if (work.cache_workspace.oracle_cache) {
            cache = extract_with_oracle(mydim, tile_mydim, extract_value, extract_length, work);
        } else {
            cache = extract_without_oracle(i, mydim, tile_mydim, extract_value, extract_length, work);
        }

        std::copy(cache, cache + extract_length, buffer);
        return buffer;
    }

private:
    /*************************************
     * Defines the extractors themselves *
     *************************************/

    template<bool accrow_, tatami::DimensionSelectionType selection_>
    struct Extractor : public tatami::Extractor<selection_, false, Value_, Index_> {
        Extractor(const TileDbDenseMatrix* p) : parent(p), base(parent) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                this->full_length = (accrow_ ? parent->ncol() : parent->nrow());
                base.set_cache(parent, this->full_length);
            }
        }

        Extractor(const TileDbDenseMatrix* p, Index_ start, Index_ length) : parent(p), base(parent) {
            if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                this->block_start = start;
                this->block_length = length;
                base.set_cache(parent, this->block_length);
            }
        }

        Extractor(const TileDbDenseMatrix* p, std::vector<Index_> idx) : parent(p), base(parent) {
            if constexpr(selection_ == tatami::DimensionSelectionType::INDEX) {
                this->index_length = idx.size();
                indices = std::move(idx);
                base.set_cache(parent, this->index_length);
            }
        }

    protected:
        const TileDbDenseMatrix* parent;
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

        const Value_* fetch(Index_ i, Value_* buffer) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                return parent->extract<accrow_>(i, buffer, 0, this->full_length, this->base);
            } else if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                return parent->extract<accrow_>(i, buffer, this->block_start, this->block_length, this->base);
            } else {
                return parent->extract<accrow_>(i, buffer, this->indices, this->index_length, this->base);
            }
        }

        void set_oracle(std::unique_ptr<tatami::Oracle<Index_> > o) {
            base.cache_workspace.set_oracle(std::move(o));
        }
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_, typename ... Args_>
    std::unique_ptr<tatami::Extractor<selection_, false, Value_, Index_> > populate(const tatami::Options&, Args_&&... args) const {
        std::unique_ptr<tatami::Extractor<selection_, false, Value_, Index_> > output;

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        TATAMI_TILEDB_PARALLEL_LOCK([&]() -> void {
#endif

        output.reset(new Extractor<accrow_, selection_>(this, std::forward<Args_>(args)...));

#ifdef TATAMI_TILEDB_PARALLEL_LOCK
        });
#endif

        return output;
    }

public:
    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_row(const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::FULL>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_row(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::BLOCK>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_row(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::INDEX>(opt, std::move(indices));
    }

    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_column(const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::FULL>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_column(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::BLOCK>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_column(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::INDEX>(opt, std::move(indices));
    }
};

}

#endif
