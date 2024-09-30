#include <gtest/gtest.h>
#include "parallel.h" // include before tatami_tiledb.hpp
#include "tatami_tiledb/tatami_tiledb.hpp"
#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

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
    inline static tatami_tiledb::DenseMatrixOptions opt;
    inline static std::string fpath, name;

    static void assemble(const SimulationParameters& params) {
        if (ref && params == last_params) {
            return;
        }
        last_params = params;

        fpath = tatami_test::temp_file_path("tatami-dense-test");
        tatami_test::remove_file_path(fpath);
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
        auto values = tatami_test::simulate_dense_vector<double>(NR * NC, 0, 100);

        query.set_layout(TILEDB_ROW_MAJOR).set_data_buffer(name, values);
        query.submit();
        query.finalize();
        array.close();

        // Don't construct a static tiledb matrix here, as the destructor doesn't get called when GoogleTest exits via _exit
        // (see https://stackoverflow.com/questions/12728535/will-global-static-variables-be-destroyed-at-program-end)
        // resulting in errors due to unjoined threads in the undestructed TileDB Context.
        opt.maximum_cache_size = static_cast<double>(NR * NC) * 0.1 * static_cast<double>(sizeof(double));
        opt.require_minimum_cache = true;

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
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt));

    EXPECT_EQ(mat->nrow(), NR);
    EXPECT_EQ(mat->ncol(), NC);
    EXPECT_FALSE(mat->sparse());
    EXPECT_EQ(mat->sparse_proportion(), 0);
    EXPECT_TRUE(mat->prefer_rows());
    EXPECT_EQ(mat->prefer_rows_proportion(), 1);

    {
        // First dimension is compromised, switching to the second dimension.
        assemble({ NR, 1 });
        std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt));
        EXPECT_FALSE(mat->prefer_rows());
    }

    {
        // Second dimension is compromised, but we just use the first anyway.
        assemble({ 1, NC });
        std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt));
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

    auto sfpath = tatami_test::temp_file_path("tatami-dense-solo-test");
    {
        tatami_test::remove_file_path(sfpath);

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
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters> >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseMatrixAccessFullTest, Basic) {
    auto params = tatami_test::convert_access_parameters(std::get<1>(GetParam()));
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt));
    tatami_test::test_full_access(params, mat.get(), ref.get());
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixAccessFullTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_parameter_combinations()
    )
);

/*************************************
 *************************************/

class DenseSlicedTest : 
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, double> > >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseSlicedTest, Basic) {
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt));

    auto tparam = GetParam();
    auto params = tatami_test::convert_access_parameters(std::get<1>(tparam));
    auto block = std::get<2>(tparam);
    auto len = params.use_row ? ref->ncol() : ref->nrow();
    size_t FIRST = block.first * len, LAST = block.second * len;

    tatami_test::test_block_access(params, mat.get(), ref.get(), FIRST, LAST);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseSlicedTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.5), 
            std::make_pair(0.25, 0.75), 
            std::make_pair(0.51, 1.0)
        )
    )
);

/*************************************
 *************************************/

class DenseIndexedTest : 
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, int> > >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseIndexedTest, Basic) {
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt));

    auto tparam = GetParam();
    auto params = tatami_test::convert_access_parameters(std::get<1>(tparam));
    auto index = std::get<2>(tparam);
    auto len = params.use_row ? ref->ncol() : ref->nrow();
    size_t FIRST = index.first * len, STEP = index.second;

    tatami_test::test_indexed_access(params, mat.get(), ref.get(), FIRST, STEP);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseIndexedTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 5), 
            std::make_pair(0.11, 9),
            std::make_pair(0.4, 7)
        )
    )
);

/*************************************
 *************************************/

class DenseUncachedTest : 
    public ::testing::TestWithParam<tatami_test::StandardTestAccessParameters>,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble({ 10, 10 });
    }
};

TEST_P(DenseUncachedTest, Basic) {
    tatami_tiledb::DenseMatrixOptions opt2;
    opt2.maximum_cache_size = 0;
    opt2.require_minimum_cache = false;
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt2));

    auto params = tatami_test::convert_access_parameters(GetParam());
    tatami_test::test_full_access(params, mat.get(), ref.get());

    auto len = params.use_row ? ref->ncol() : ref->nrow();
    size_t FIRST = len * 0.25, LAST = len * 0.75;
    tatami_test::test_block_access(params, mat.get(), ref.get(), FIRST, LAST);
    tatami_test::test_indexed_access(params, mat.get(), ref.get(), FIRST, 4);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseUncachedTest,
    tatami_test::standard_test_access_parameter_combinations()
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
    std::unique_ptr<tatami::Matrix<int, size_t> > mat(new tatami_tiledb::DenseMatrix<int, size_t>(fpath, name, opt));
    auto mext = mat->dense_row();
    std::shared_ptr<tatami::Matrix<int, size_t> > ref2 = tatami::make_DelayedCast<int, size_t>(ref);
    auto rext2 = ref2->dense_row();

    for (size_t r = 0; r < NR; ++r) {
        auto mvec = tatami_test::fetch(mext.get(), r, NC);
        auto rvec2 = tatami_test::fetch(rext2.get(), r, NC);
        EXPECT_EQ(mvec, rvec2);
    }
}

TEST_F(DenseMatrixMiscellaneousTest, ContextConstructor) {
    tiledb::Config cfg;
    cfg["sm.compute_concurrency_level"] = 1;
    std::unique_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, tiledb::Context(cfg), opt));
    auto mext = mat->dense_row();
    auto rext = ref->dense_row();

    for (int r = 0; r < static_cast<int>(NR); ++r) {
        auto mvec = tatami_test::fetch(mext.get(), r, NC);
        auto rvec = tatami_test::fetch(rext.get(), r, NC);
        EXPECT_EQ(mvec, rvec);
    }
}

#endif
/*************************************
 *************************************/

class DenseMatrixParallelTest : public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, bool, bool> >, public DenseMatrixTestCore {
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
    std::shared_ptr<tatami::Matrix<double, int> > mat(new tatami_tiledb::DenseMatrix<double, int>(fpath, name, opt));

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
