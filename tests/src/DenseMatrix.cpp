#include <gtest/gtest.h>

#include "parallel.h" // include before tatami_tiledb.hpp
#include "temp_file_path.h"

#include "tatami_tiledb/tatami_tiledb.hpp"
#include "tatami_test/tatami_test.hpp"

class DenseMatrixTestCore {
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
    inline static tatami_tiledb::DenseMatrixOptions dense_opt;
    inline static std::string fpath, name;

    static void assemble(const SimulationParameters& params) {
        if (ref && params == last_params) {
            return;
        }
        last_params = params;

        fpath = temp_file_path("tatami-dense-test");
        name = "stuff";

        // Creating the array.
        tiledb::Context ctx;

        // Adding some non-trivial offsets so that life remains a bit interesting.
        tiledb::Domain domain(ctx);
        domain.add_dimension(tiledb::Dimension::create<int>(ctx, "rows", {{10, NR + 10 - 1}}, params.first));
        domain.add_dimension(tiledb::Dimension::create<int>(ctx, "cols", {{5, NC + 5 - 1}}, params.second));

        tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
        schema.set_domain(domain);
        schema.add_attribute(tiledb::Attribute::create<double>(ctx, name));
        tiledb::Array::create(fpath, schema);

        // Open the array for writing and create the query.
        tiledb::Array array(ctx, fpath, TILEDB_WRITE);
        tiledb::Query query(ctx, array);
        auto values = tatami_test::simulate_vector<double>(NR * NC, [&]{
            tatami_test::SimulateVectorOptions opt;
            opt.lower = 0;
            opt.upper = 100;
            opt.seed = 19283 + params.first * 13 + params.second * 17; 
            return opt;
        }());

        query.set_layout(TILEDB_ROW_MAJOR).set_data_buffer(name, values);
        query.submit();
        query.finalize();
        array.close();

        // Don't construct a static tiledb matrix here, as the destructor doesn't get called when GoogleTest exits via _exit
        // (see https://stackoverflow.com/questions/12728535/will-global-static-variables-be-destroyed-at-program-end)
        // resulting in errors due to unjoined threads in the undestructed TileDB Context.
        dense_opt.maximum_cache_size = static_cast<double>(NR * NC) * 0.1 * static_cast<double>(sizeof(double));
        dense_opt.require_minimum_cache = true;

        ref.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, std::move(values)));

        return;
    }
};

/*************************************
 *************************************/
#ifndef TATAMI_TILEDB_TEST_PARALLEL

class DenseUtilsTest : public ::testing::Test, public DenseMatrixTestCore {};

TEST_F(DenseUtilsTest, Basic) {
    assemble({ 10, 10 });
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt));

    EXPECT_EQ(mat->nrow(), NR);
    EXPECT_EQ(mat->ncol(), NC);
    EXPECT_FALSE(mat->sparse());
    EXPECT_EQ(mat->sparse_proportion(), 0);
    EXPECT_TRUE(mat->prefer_rows());
    EXPECT_EQ(mat->prefer_rows_proportion(), 1);

    {
        // First dimension is compromised, switching to the second dimension.
        assemble({ NR, 1 });
        std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt));
        EXPECT_FALSE(mat->prefer_rows());
    }

    {
        // Second dimension is compromised, but we just use the first anyway.
        assemble({ 1, NC });
        std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt));
        EXPECT_TRUE(mat->prefer_rows());
    }
}

TEST_F(DenseUtilsTest, Errors) {
    tatami_test::throws_error([&]() {
        tatami_tiledb::SparseMatrix<double, int>(fpath, name);
    }, "sparse");

    tatami_test::throws_error([&]() {
        tatami_tiledb::DenseMatrix<double, int>(fpath, "foo");
    }, "foo");

    auto sfpath = temp_file_path("tatami-dense-solo-test");
    {
        // Creating a 1-dimensional array.
        tiledb::Context ctx;
        tiledb::Domain domain(ctx);
        domain.add_dimension(tiledb::Dimension::create<int>(ctx, "solo", {{ 0, 100 }}, 10));

        tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
        schema.set_domain(domain);
        schema.add_attribute(tiledb::Attribute::create<double>(ctx, "WHEE"));
        tiledb::Array::create(sfpath, schema);
    }
    tatami_test::throws_error([&]() {
        tatami_tiledb::DenseMatrix<double, int>(sfpath, "WHEE");
    }, "two dimensions");
}

/*************************************
 *************************************/

class DenseMatrixAccessFullTest :
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions> >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseMatrixAccessFullTest, Basic) {
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt));
    auto opts = tatami_test::convert_test_access_options(std::get<1>(GetParam()));
    tatami_test::test_full_access(*mat, *ref, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixAccessFullTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations()
    )
);

/*************************************
 *************************************/

class DenseSlicedTest : 
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseSlicedTest, Basic) {
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt));
    auto tparam = GetParam();
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto block = std::get<2>(tparam);
    tatami_test::test_block_access(*mat, *ref, block.first, block.second, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseSlicedTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
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

class DenseIndexedTest : 
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseIndexedTest, Basic) {
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt));
    auto tparam = GetParam();
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto index = std::get<2>(tparam);
    tatami_test::test_indexed_access(*mat, *ref, index.first, index.second, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseIndexedTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 0.3), 
            std::make_pair(0.11, 0.1),
            std::make_pair(0.4, 0.2)
        )
    )
);

/*************************************
 *************************************/

class DenseUncachedTest : 
    public ::testing::TestWithParam<tatami_test::StandardTestAccessOptions>,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble({ 10, 10 });
    }
};

TEST_P(DenseUncachedTest, Basic) {
    tatami_tiledb::DenseMatrixOptions dense_opt2;
    dense_opt2.maximum_cache_size = 0;
    dense_opt2.require_minimum_cache = false;
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt2));

    auto opts = tatami_test::convert_test_access_options(GetParam());
    tatami_test::test_full_access(*mat, *ref, opts);
    tatami_test::test_block_access(*mat, *ref, 0.25, 0.5, opts);
    tatami_test::test_indexed_access(*mat, *ref, 0.1, 0.2, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseUncachedTest,
    tatami_test::standard_test_access_options_combinations()
);

/*************************************
 *************************************/

class DenseMatrixMiscellaneousTest : public ::testing::Test, public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble({ 10, 10 });
    }
};

TEST_F(DenseMatrixMiscellaneousTest, DifferentType) {
    std::unique_ptr<tatami::Matrix<int, size_t> > mat(new tatami_tiledb::DenseMatrix<int, size_t>(fpath, name, dense_opt));
    auto mext = mat->dense_row();
    std::shared_ptr<tatami::Matrix<int, size_t> > ref2 = tatami::make_DelayedCast<int, size_t>(ref);
    auto rext2 = ref2->dense_row();

    for (size_t r = 0; r < NR; ++r) {
        auto mvec = tatami_test::fetch<int, size_t>(*mext, r, NC);
        auto rvec2 = tatami_test::fetch<int, size_t>(*rext2, r, NC);
        EXPECT_EQ(mvec, rvec2);
    }
}

TEST_F(DenseMatrixMiscellaneousTest, ContextConstructor) {
    tiledb::Config cfg;
    cfg["sm.compute_concurrency_level"] = 1;
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, tiledb::Context(cfg), dense_opt));
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

class DenseMatrixParallelTest : 
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, bool, bool> >,
    public DenseMatrixTestCore {
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

TEST_P(DenseMatrixParallelTest, Simple) {
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, dense_opt));

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
    DenseMatrix,
    DenseMatrixParallelTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(),
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);
