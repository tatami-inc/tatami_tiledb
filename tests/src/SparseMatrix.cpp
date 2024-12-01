#include <gtest/gtest.h>

#include "parallel.h" // include before tatami_tiledb.hpp
#include "temp_file_path.h"

#include "tatami_tiledb/tatami_tiledb.hpp"
#include "tatami_test/tatami_test.hpp"

class SparseMatrixTestCore {
public:
    static constexpr size_t NR = 200, NC = 100;

    typedef std::pair<int, int> SimulationParameters;

    inline static SimulationParameters last_params;

public:
    static auto create_combinations() {
        return ::testing::Values(
            std::pair<int, int>(NR, 1),
            std::pair<int, int>(1, NC),
            std::make_pair(7, 7), // using tile sizes that are a little odd to check for off-by-one errors.
            std::make_pair(13, 4),
            std::make_pair(5, 10)
        );
    }

protected:
    inline static std::shared_ptr<tatami::Matrix<double, int> > ref;
    inline static tatami_tiledb::SparseMatrixOptions sparse_opt;
    inline static std::string fpath, name;

    static void assemble(const SimulationParameters& params) {
        if (ref && params == last_params) {
            return;
        }
        last_params = params;

        fpath = temp_file_path("tatami-sparse-test");
        name = "stuff";

        // Creating the array.
        tiledb::Context ctx;
        tiledb::Domain domain(ctx);

        // Adding some non-trivial offsets so that life remains a bit interesting.
        int row_offset = 10;
        int col_offset = 5;
        domain.add_dimension(tiledb::Dimension::create<int>(ctx, "rows", {{ row_offset, static_cast<int>(NR) + row_offset - 1 }}, params.first));
        domain.add_dimension(tiledb::Dimension::create<int>(ctx, "cols", {{ col_offset, static_cast<int>(NC) + col_offset - 1 }}, params.second));

        tiledb::ArraySchema schema(ctx, TILEDB_SPARSE);
        schema.set_domain(domain);
        schema.add_attribute(tiledb::Attribute::create<double>(ctx, name));
        tiledb::Array::create(fpath, schema);

        // Open the array for writing and create the query.
        tiledb::Array array(ctx, fpath, TILEDB_WRITE);
        tiledb::Query query(ctx, array);
        auto contents = tatami_test::simulate_compressed_sparse<double, int>(NR, NC, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.2;
            opt.seed = 1391 + params.first * 13 + params.second * 23;
            return opt;
        }());

        std::vector<int> coords;
        for (size_t r = 0; r < NR; ++r) {
            auto start = contents.indptr[r], end = contents.indptr[r+1];
            coords.insert(coords.end(), end - start, r + row_offset); // see above for offset on the rows.
        }

        auto copy = contents.index;
        for (auto& x : copy) {
            x += col_offset; // see above for the offset on the columns.
        }

        query.set_data_buffer(name, contents.data);
        query.set_data_buffer("rows", coords);
        query.set_data_buffer("cols", copy);

        query.submit();
        query.finalize();
        array.close();

        // Don't construct a static tiledb matrix here, as the destructor doesn't get called when GoogleTest exits via _exit
        // (see https://stackoverflow.com/questions/12728535/will-global-static-variables-be-destroyed-at-program-end)
        // resulting in errors due to unjoined threads in the undestructed TileDB Context.
        sparse_opt.maximum_cache_size = static_cast<double>(NR * NC) * 0.1 * static_cast<double>(sizeof(double));
        sparse_opt.require_minimum_cache = true;

        ref.reset(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, contents.data, contents.index, contents.indptr));

        return;
    }
};

/*************************************
 *************************************/
#ifndef TATAMI_TILEDB_TEST_PARALLEL

class SparseUtilsTest : public ::testing::Test, public SparseMatrixTestCore {};

TEST_F(SparseUtilsTest, Basic) {
    assemble({ 10, 10 });
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt));

    EXPECT_EQ(mat->nrow(), NR);
    EXPECT_EQ(mat->ncol(), NC);
    EXPECT_TRUE(mat->sparse());
    EXPECT_EQ(mat->sparse_proportion(), 1);
    EXPECT_TRUE(mat->prefer_rows());
    EXPECT_EQ(mat->prefer_rows_proportion(), 1);

    {
        // First dimension is compromised, switching to the second dimension.
        assemble({ NR, 1 });
        std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt));
        EXPECT_FALSE(mat->prefer_rows());
    }

    {
        // Second dimension is compromised, but we just use the first anyway.
        assemble({ 1, NC });
        std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt));
        EXPECT_TRUE(mat->prefer_rows());
    }
}

TEST_F(SparseUtilsTest, Errors) {
    tatami_test::throws_error([&]() {
        tatami_tiledb::DenseMatrix<double, int>(fpath, name);
    }, "dense");

    tatami_test::throws_error([&]() {
        tatami_tiledb::SparseMatrix<double, int>(fpath, "foo");
    }, "foo");

    auto sfpath = temp_file_path("tatami-sparse-solo-test");
    {
        // Creating a 1-dimensional array.
        tiledb::Context ctx;
        tiledb::Domain domain(ctx);
        domain.add_dimension(tiledb::Dimension::create<int>(ctx, "solo", {{ 0, 100 }}, 10));

        tiledb::ArraySchema schema(ctx, TILEDB_SPARSE);
        schema.set_domain(domain);
        schema.add_attribute(tiledb::Attribute::create<double>(ctx, "WHEE"));
        tiledb::Array::create(sfpath, schema);
    }
    tatami_test::throws_error([&]() {
        tatami_tiledb::SparseMatrix<double, int>(sfpath, "WHEE");
    }, "two dimensions");
}

/*************************************
 *************************************/

class SparseMatrixAccessFullTest :
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions> >,
    public SparseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(SparseMatrixAccessFullTest, Basic) {
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt));
    auto opts = tatami_test::convert_test_access_options(std::get<1>(GetParam()));
    tatami_test::test_full_access(*mat, *ref, opts);
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixAccessFullTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations()
    )
);

/*************************************
 *************************************/

class SparseSlicedTest : 
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >,
    public SparseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(SparseSlicedTest, Basic) {
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt));
    auto tparam = GetParam();
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto block = std::get<2>(tparam);
    tatami_test::test_block_access(*mat, *ref, block.first, block.second, opts);
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseSlicedTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.5), 
            std::make_pair(0.25, 0.6), 
            std::make_pair(0.51, 0.4)
        )
    )
);

/*************************************
 *************************************/

class SparseIndexedTest : 
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, int> > >,
    public SparseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};
  
TEST_P(SparseIndexedTest, Basic) {
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt));
    auto tparam = GetParam();
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto index = std::get<2>(tparam);
    tatami_test::test_indexed_access(*mat, *ref, index.first, index.second, opts);
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseIndexedTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 0.2), 
            std::make_pair(0.11, 0.25),
            std::make_pair(0.4, 0.28)
        )
    )
);

/*************************************
 *************************************/

class SparseUncachedTest : 
    public ::testing::TestWithParam<tatami_test::StandardTestAccessOptions>,
    public SparseMatrixTestCore {
protected:
    void SetUp() {
        assemble({ 10, 10 });
    }
};

TEST_P(SparseUncachedTest, Basic) {
    tatami_tiledb::SparseMatrixOptions sparse_opt2;
    sparse_opt2.maximum_cache_size = 0;
    sparse_opt2.require_minimum_cache = false;
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt2));

    auto opts = tatami_test::convert_test_access_options(GetParam());
    tatami_test::test_full_access(*mat, *ref, opts);
    tatami_test::test_block_access(*mat, *ref, 0.25, 0.5, opts);
    tatami_test::test_indexed_access(*mat, *ref, 0.25, 0.25, opts);
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseUncachedTest,
    tatami_test::standard_test_access_options_combinations()
);

/*************************************
 *************************************/

class SparseMatrixMiscellaneousTest : public ::testing::Test, public SparseMatrixTestCore {
protected:
    void SetUp() {
        assemble({ 10, 10 });
    }
};

TEST_F(SparseMatrixMiscellaneousTest, DifferentType) {
    std::unique_ptr<tatami::Matrix<int, size_t> > mat(new tatami_tiledb::SparseMatrix<int, size_t>(fpath, name, sparse_opt));
    auto mext = mat->dense_row();
    std::shared_ptr<tatami::Matrix<int, size_t> > ref2 = tatami::make_DelayedCast<int, size_t>(ref);
    auto rext2 = ref2->dense_row();

    for (size_t r = 0; r < NR; ++r) {
        auto mvec = tatami_test::fetch<int, size_t>(*mext, r, NC);
        auto rvec2 = tatami_test::fetch<int, size_t>(*rext2, r, NC);
        EXPECT_EQ(mvec, rvec2);
    }
}

TEST_F(SparseMatrixMiscellaneousTest, ContextConstructor) {
    tiledb::Config cfg;
    cfg["sm.compute_concurrency_level"] = 1;
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, tiledb::Context(cfg), sparse_opt));
    auto mext = mat->dense_row();
    auto rext = ref->dense_row();

    for (int r = 0; r < static_cast<int>(NR); ++r) {
        auto mvec = tatami_test::fetch<double, int>(*mext, r, NC);
        auto rvec = tatami_test::fetch<double, int>(*rext, r, NC);
        EXPECT_EQ(mvec, rvec);
    }
}

#endif
/*************************************
 *************************************/

class SparseMatrixParallelTest : 
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, bool, bool> >,
    public SparseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }

    template<bool oracle_>
    static void compare_sums(bool row, const tatami::Matrix<double, int>* testmat, const tatami::Matrix<double, int>* refmat) {
        size_t dim = (row ? refmat->nrow() : refmat->ncol());
        size_t otherdim = (row ? refmat->ncol() : refmat->nrow());
        std::vector<double> computed(dim), expected(dim);

        tatami::parallelize([&](size_t, int start, int len) -> void {
            auto ext = [&]() {
                if constexpr(oracle_) {
                    return tatami::consecutive_extractor<false>(testmat, row, start, len);
                } else {
                    return testmat->dense(row, tatami::Options());
                }
            }();

            auto rext = [&]() {
                if constexpr(oracle_) {
                    return tatami::consecutive_extractor<false>(refmat, row, start, len);
                } else {
                    return refmat->dense(row, tatami::Options());
                }
            }();

            std::vector<double> buffer(otherdim), rbuffer(otherdim);
            for (int i = start; i < start + len; ++i) {
                auto ptr = ext->fetch(i, buffer.data());
                auto rptr = rext->fetch(i, rbuffer.data());
                computed[i] = std::accumulate(ptr, ptr + otherdim, 0.0);
                expected[i] = std::accumulate(rptr, rptr + otherdim, 0.0);
            }
        }, dim, 3); // throw it over three threads.

        EXPECT_EQ(computed, expected);
    }
};

TEST_P(SparseMatrixParallelTest, Simple) {
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::SparseMatrix<double, int>(fpath, name, sparse_opt));

    auto param = GetParam();
    bool row = std::get<1>(param);
    bool oracle = std::get<2>(param);

    if (oracle) {
        compare_sums<true>(row, mat.get(), ref.get());
    } else {
        compare_sums<false>(row, mat.get(), ref.get());
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixParallelTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);
