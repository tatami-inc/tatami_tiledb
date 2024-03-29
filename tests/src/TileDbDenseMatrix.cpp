#include <gtest/gtest.h>
#include "tatami_tiledb/tatami_tiledb.hpp"
#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

class TileDbDenseMatrixTestMethods {
public:
    static constexpr size_t NR = 200, NC = 100;

protected:
    std::vector<double> values;
    std::string fpath;
    std::string name;

    static auto custom_options() {
        tatami_tiledb::TileDbOptions custom_opt;

        // Make sure the cache size is smaller than the dataset, but not too much
        // smaller, to make sure we do some caching + evictions. Here, the cache is
        // set at 20% of the size of the entire dataset, i.e., 40 rows or 20 columns.
        custom_opt.maximum_cache_size = (NR * NC * sizeof(double)) / 5;

        return custom_opt;
    }

    static auto uncached_options() {
        tatami_tiledb::TileDbOptions opt;
        opt.require_minimum_cache = false;
        opt.maximum_cache_size = 0;
        return opt;
    }

    void dump(const std::pair<int, int>& tile_sizes) {
        fpath = tatami_test::temp_file_path("tatami-dense-test");
        tatami_test::remove_file_path(fpath);

        name = "stuff";

        // Creating the array.
        {
            tiledb::Context ctx;
            tiledb::Domain domain(ctx);

            // Adding some non-trivial offsets so that life remains a bit interesting.
            domain
                .add_dimension(tiledb::Dimension::create<int>(ctx, "rows", {{10, NR + 10 - 1}}, tile_sizes.first))
                .add_dimension(tiledb::Dimension::create<int>(ctx, "cols", {{5, NC + 5 - 1}}, tile_sizes.second));

            tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
            schema.set_domain(domain);
            schema.add_attribute(tiledb::Attribute::create<double>(ctx, name));
            tiledb::Array::create(fpath, schema);
        }

        // Open the array for writing and create the query.
        {
            tiledb::Context ctx;
            tiledb::Array array(ctx, fpath, TILEDB_WRITE);

            tiledb::Query query(ctx, array);
            values = tatami_test::simulate_dense_vector<double>(NR * NC, 0, 100);
            for (auto& v : values) {
                v = std::round(v);
            }

            query.set_layout(TILEDB_ROW_MAJOR).set_data_buffer(name, values);
            query.submit();
            query.finalize();
            array.close();
        }

        return;
    }
};

/*************************************
 *************************************/

class TileDbDenseUtilsTest : public ::testing::Test, public TileDbDenseMatrixTestMethods {};

TEST_F(TileDbDenseUtilsTest, Basic) {
    dump(std::make_pair<int, int>(10, 10));
    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name);
    EXPECT_EQ(mat.nrow(), NR);
    EXPECT_EQ(mat.ncol(), NC);
    EXPECT_FALSE(mat.sparse());
    EXPECT_EQ(mat.sparse_proportion(), 0);

    auto ptr = tatami_tiledb::make_TileDbMatrix<double, int>(fpath, name);
    EXPECT_FALSE(ptr->sparse());

    bool failed = false;
    try {
        tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name);
    } catch (std::exception& e) {
        std::string msg(e.what());
        EXPECT_TRUE(msg.find("sparse") != std::string::npos);
        failed = true;
    }
    EXPECT_TRUE(failed);
}

TEST_F(TileDbDenseUtilsTest, Preference) {
    {
        dump(std::make_pair<int, int>(10, 10));

        tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name);
        EXPECT_TRUE(mat.prefer_rows());
        EXPECT_EQ(mat.prefer_rows_proportion(), 1);

        tatami_tiledb::TileDbDenseMatrix<double, int, true> tmat(fpath, name);
        EXPECT_FALSE(tmat.prefer_rows());
        EXPECT_EQ(tmat.prefer_rows_proportion(), 0);
    }

    {
        // First dimension is compromised, switching to the second dimension.
        dump(std::make_pair<int, int>(NR, 1));

        tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name);
        EXPECT_FALSE(mat.prefer_rows());

        tatami_tiledb::TileDbDenseMatrix<double, int, true> tmat(fpath, name);
        EXPECT_TRUE(tmat.prefer_rows());
    }

    {
        // Second dimension is compromised, but we just use the first anyway.
        dump(std::make_pair<int, int>(1, NC));

        tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name);
        EXPECT_TRUE(mat.prefer_rows());

        tatami_tiledb::TileDbDenseMatrix<double, int, true> tmat(fpath, name);
        EXPECT_FALSE(tmat.prefer_rows());
    }
}

/*************************************
 *************************************/

class TileDbDenseAccessUncachedTest : public ::testing::TestWithParam<std::tuple<bool, int> >, public TileDbDenseMatrixTestMethods {};

TEST_P(TileDbDenseAccessUncachedTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    dump(std::pair<int, int>(10, 10)); // exact chunk choice doesn't matter here.

    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, uncached_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(TileDbDenseAccessUncachedTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    dump(std::pair<int, int>(10, 10)); // exact chunk choice doesn't matter here.

    tatami_tiledb::TileDbDenseMatrix<double, int, true> mat(fpath, name, uncached_options());
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(TileDbDenseAccessUncachedTest, ForcedCache) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    dump(std::pair<int, int>(10, 10)); 

    tatami_tiledb::TileDbOptions opt;
    opt.require_minimum_cache = true; // Force a minimum cache.
    opt.maximum_cache_size = 0;

    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, opt);
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

INSTANTIATE_TEST_SUITE_P(
    TileDbDenseMatrix,
    TileDbDenseAccessUncachedTest,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(1, 3)
    )
);

/*************************************
 *************************************/

class TileDbDenseAccessTest : public ::testing::TestWithParam<std::tuple<bool, int, std::pair<int, int> > >, public TileDbDenseMatrixTestMethods {};

TEST_P(TileDbDenseAccessTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    auto chunk_sizes = std::get<2>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(TileDbDenseAccessTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    auto chunk_sizes = std::get<2>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int, true> mat(fpath, name, custom_options());
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

INSTANTIATE_TEST_SUITE_P(
    TileDbDenseMatrix,
    TileDbDenseAccessTest,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(1, 3),
        ::testing::Values(
            std::make_pair(7, 17), // using tile sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11)
        )
    )
);

/*************************************
 *************************************/

class TileDbDenseAccessMiscTest : public ::testing::TestWithParam<std::tuple<std::pair<int, int> > >, public TileDbDenseMatrixTestMethods {};

TEST_P(TileDbDenseAccessMiscTest, LruReuse) {
    // Check that the LRU cache works as expected when cache elements are
    // constantly re-used in a manner that changes the last accessed element.
    auto param = GetParam();
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    {
        auto m_ext = mat.dense_row();
        auto r_ext = ref.dense_row();
        for (size_t r0 = 0; r0 < NR; ++r0) {
            auto r = (r0 % 2 ? NR - r0/2 - 1 : r0/2); // alternate between the last and first chunk.
            EXPECT_EQ(m_ext->fetch(r), r_ext->fetch(r));
        }
    }

    {
        auto m_ext = mat.dense_column();
        auto r_ext = ref.dense_column();
        for (size_t c0 = 0; c0 < NC; ++c0) {
            auto c = (c0 % 2 ? NC - c0/2 - 1 : c0/2); // alternate between the last and first.
            EXPECT_EQ(m_ext->fetch(c), r_ext->fetch(c));
        }
    }
}

TEST_P(TileDbDenseAccessMiscTest, Oracle) {
    auto param = GetParam();
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, false); // consecutive
    tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, false);

    tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, true); // randomized
    tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, true);
}

INSTANTIATE_TEST_SUITE_P(
    TileDbDenseMatrix,
    TileDbDenseAccessMiscTest,
    ::testing::Combine(
        ::testing::Values(
            std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11)
        )
    )
);

/*************************************
 *************************************/

class TileDbDenseSlicedTest : public ::testing::TestWithParam<std::tuple<bool, size_t, std::vector<double>, std::pair<int, int> > >, public TileDbDenseMatrixTestMethods {};

TEST_P(TileDbDenseSlicedTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    {
        size_t FIRST = interval_info[0] * NC, LAST = interval_info[1] * NC;
        tatami_test::test_sliced_row_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }

    {
        size_t FIRST = interval_info[0] * NR, LAST = interval_info[1] * NR;
        tatami_test::test_sliced_column_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }
}

TEST_P(TileDbDenseSlicedTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int, true> mat(fpath, name, custom_options());
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    {
        size_t FIRST = interval_info[0] * NR, LAST = interval_info[1] * NR; // NR is deliberate here, it's transposed.
        tatami_test::test_sliced_row_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }

    {
        size_t FIRST = interval_info[0] * NC, LAST = interval_info[1] * NC; // NC is deliberate here, it's transposed.
        tatami_test::test_sliced_column_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TileDbDenseMatrix,
    TileDbDenseSlicedTest,
    ::testing::Combine(
        ::testing::Values(true, false), // iterate forward or back, to test the workspace's memory.
        ::testing::Values(1, 3), // jump, to test the workspace's memory.
        ::testing::Values(
            std::vector<double>({ 0, 0.5 }),
            std::vector<double>({ 0.25, 0.75 }),
            std::vector<double>({ 0.51, 1 })
        ),
        ::testing::Values(
            std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11)
        )
    )
);

/*************************************
 *************************************/

class TileDbDenseIndexedTest : public ::testing::TestWithParam<std::tuple<bool, size_t, std::vector<double>, std::pair<int, int> > >, public TileDbDenseMatrixTestMethods {};

TEST_P(TileDbDenseIndexedTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    {
        size_t FIRST = interval_info[0] * NC, STEP = interval_info[1];
        tatami_test::test_indexed_row_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }

    {
        size_t FIRST = interval_info[0] * NR, STEP = interval_info[1];
        tatami_test::test_indexed_column_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }
}

TEST_P(TileDbDenseIndexedTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbDenseMatrix<double, int, true> mat(fpath, name, custom_options());
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    {
        size_t FIRST = interval_info[0] * NR, STEP = interval_info[1]; // NR is deliberate here, it's transposed.
        tatami_test::test_indexed_row_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }

    {
        size_t FIRST = interval_info[0] * NC, STEP = interval_info[1]; // NC is deliberate here, it's transposed.
        tatami_test::test_indexed_column_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TileDbDenseMatrix,
    TileDbDenseIndexedTest,
    ::testing::Combine(
        ::testing::Values(true, false), // iterate forward or back, to test the workspace's memory.
        ::testing::Values(1, 3), // jump, to test the workspace's memory.
        ::testing::Values(
            std::vector<double>({ 0.3, 5 }),
            std::vector<double>({ 0.11, 9 }),
            std::vector<double>({ 0.4, 7 })
        ),
        ::testing::Values(
            std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11)
        )
    )
);

/*************************************
 *************************************/

class TileDbDenseParallelTest : public ::testing::TestWithParam<int>, public TileDbDenseMatrixTestMethods {};

TEST_P(TileDbDenseParallelTest, Basic) {
    std::pair<int, int> chunk_sizes(23, 13);
    dump(chunk_sizes);

    auto mode = GetParam();
    auto opt = (mode == 0 ? uncached_options() : custom_options());
    tatami_tiledb::TileDbDenseMatrix<double, int> mat(fpath, name, opt);

    std::vector<double> test_rowsums(NR);
    tatami::parallelize([&](int, int start, int len) -> void {
        std::vector<double> buffer(NC);
        int end = start + len;

        if (mode == 2) {
            auto wrk = tatami::consecutive_extractor<true, false>(&mat, start, len);
            for (int i = start; i < end; ++i) {
                auto ptr = wrk->fetch(i, buffer.data());
                test_rowsums[i] = std::accumulate(ptr, ptr + NC, 0.0);
            }

        } else {
            auto wrk = mat.dense_row();
            for (int i = start; i < end; ++i) {
                auto ptr = wrk->fetch(i, buffer.data());
                test_rowsums[i] = std::accumulate(ptr, ptr + NC, 0.0);
            }
        }
    }, NR, 3);

    std::vector<double> test_colsums(NC);
    tatami::parallelize([&](int, int start, int len) -> void {
        std::vector<double> buffer(NR);
        int end = start + len;

        if (mode == 2) {
            auto wrk = tatami::consecutive_extractor<false, false>(&mat, start, len);
            for (int i = start; i < end; ++i) {
                auto ptr = wrk->fetch(i, buffer.data());
                test_colsums[i] = std::accumulate(ptr, ptr + NR, 0.0);
            }

        } else {
            auto wrk = mat.dense_column();
            for (int i = start; i < end; ++i) {
                auto ptr = wrk->fetch(i, buffer.data());
                test_colsums[i] = std::accumulate(ptr, ptr + NR, 0.0);
            }
        }
    }, NC, 3);

    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);
    auto rowsums = tatami::row_sums(&ref);
    EXPECT_EQ(test_rowsums, rowsums);
    auto colsums = tatami::column_sums(&ref);
    EXPECT_EQ(test_colsums, colsums);
}

INSTANTIATE_TEST_SUITE_P(
    TileDbDenseMatrix,
    TileDbDenseParallelTest,
    ::testing::Values(0, 1, 2) // 0 = uncached, 1 = LRU cache, 2 = oracle cache.
);
