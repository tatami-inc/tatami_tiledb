#include <gtest/gtest.h>
#include "tatami_tiledb/TileDbSparseMatrix.hpp"
#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

class TileDbSparseMatrixTestMethods {
public:
    static constexpr size_t NR = 200, NC = 100;

protected:
    // Make sure the cache size is smaller than the dataset, but not too much
    // smaller, to make sure we do some caching + evictions. Here, the cache is
    // set at 20% of the size of the entire dataset, i.e., 40 rows or 20 columns.
    static constexpr size_t global_cache_size = (NR * NC * 8) / 5;

    std::vector<double> values;
    std::string fpath;
    std::string name;

    tatami_test::CompressedSparseDetails<double> contents;

    void dump(const std::pair<int, int>& tile_sizes) {
        fpath = tatami_test::temp_file_path("tatami-sparse-test");
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

            tiledb::ArraySchema schema(ctx, TILEDB_SPARSE);
            schema.set_domain(domain);
            schema.add_attribute(tiledb::Attribute::create<double>(ctx, name));
            tiledb::Array::create(fpath, schema);
        }

        // Open the array for writing and create the query.
        {
            tiledb::Context ctx;
            tiledb::Array array(ctx, fpath, TILEDB_WRITE);

            tiledb::Query query(ctx, array);
            contents = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.1);

            std::vector<int> coords;
            for (size_t r = 0; r < NR; ++r) {
                auto start = contents.ptr[r], end = contents.ptr[r+1];
                coords.insert(coords.end(), end - start, r + 10); // see above for offset on the rows.
            }

            auto copy = contents.index;
            for (auto& x : copy) {
                x += 5; // see above for the offset on the columns.
            }

            query
                .set_data_buffer(name, contents.value)
                .set_data_buffer("rows", coords)
                .set_data_buffer("cols", copy);

            query.submit();
            query.finalize();
            array.close();
        }

        return;
    }
};

/*************************************
 *************************************/

class TileDbSparseUtilsTest : public ::testing::Test, public TileDbSparseMatrixTestMethods {};

TEST_F(TileDbSparseUtilsTest, Basic) {
    dump(std::make_pair<int, int>(10, 10));
    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name);
    EXPECT_EQ(mat.nrow(), NR);
    EXPECT_EQ(mat.ncol(), NC);
    EXPECT_TRUE(mat.sparse());
    EXPECT_EQ(mat.sparse_proportion(), 1);
}

TEST_F(TileDbSparseUtilsTest, Preference) {
    {
        dump(std::make_pair<int, int>(10, 10));
        tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name);
        EXPECT_TRUE(mat.prefer_rows());
        EXPECT_EQ(mat.prefer_rows_proportion(), 1);

        tatami_tiledb::TileDbSparseMatrix<double, int, true> tmat(fpath, name);
        EXPECT_FALSE(tmat.prefer_rows());
        EXPECT_EQ(tmat.prefer_rows_proportion(), 0);
    }

    {
        // First dimension is compromised, switching to the second dimension.
        dump(std::make_pair<int, int>(NR, 1));
        tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, NR * sizeof(double));
        EXPECT_FALSE(mat.prefer_rows());

        tatami_tiledb::TileDbSparseMatrix<double, int, true> tmat(fpath, name, NR * sizeof(double));
        EXPECT_TRUE(tmat.prefer_rows());
    }

    {
        // Second dimension is compromised, but we just use the first anyway.
        dump(std::make_pair<int, int>(1, NC));
        tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, NC * sizeof(double));
        EXPECT_TRUE(mat.prefer_rows());

        tatami_tiledb::TileDbSparseMatrix<double, int, true> tmat(fpath, name, NC * sizeof(double));
        EXPECT_FALSE(tmat.prefer_rows());
    }

    {
        // Both are compromised.
        dump(std::make_pair<int, int>(10, 10));
        tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, 0);
        EXPECT_TRUE(mat.prefer_rows());

        tatami_tiledb::TileDbSparseMatrix<double, int, true> tmat(fpath, name, 0);
        EXPECT_FALSE(tmat.prefer_rows());
    }
}

/*************************************
 *************************************/

class TileDbSparseAccessUncachedTest : public ::testing::TestWithParam<std::tuple<int, bool> >, public TileDbSparseMatrixTestMethods {};

TEST_P(TileDbSparseAccessUncachedTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    dump(std::pair<int, int>(10, 10)); // exact chunk choice doesn't matter here.

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, 0);
    tatami::CompressedSparseRowMatrix<double, int> ref(NR, NC, contents.value, contents.index, contents.ptr);

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(TileDbSparseAccessUncachedTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    dump(std::pair<int, int>(10, 10)); // exact chunk choice doesn't matter here.

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, 0);
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, contents.value, contents.index, contents.ptr));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

INSTANTIATE_TEST_CASE_P(
    TileDbSparseMatrix,
    TileDbSparseAccessUncachedTest,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(1, 3)
    )
);

/*************************************
 *************************************/

class TileDbSparseAccessTest : public ::testing::TestWithParam<std::tuple<bool, int, std::pair<int, int> > >, public TileDbSparseMatrixTestMethods {};

TEST_P(TileDbSparseAccessTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    auto chunk_sizes = std::get<2>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, global_cache_size);
    tatami::CompressedSparseRowMatrix<double, int> ref(NR, NC, contents.value, contents.index, contents.ptr);

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(TileDbSparseAccessTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    auto chunk_sizes = std::get<2>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int, true> mat(fpath, name, global_cache_size);
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, contents.value, contents.index, contents.ptr));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

INSTANTIATE_TEST_CASE_P(
    TileDbSparseMatrix,
    TileDbSparseAccessTest,
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

class TileDbSparseAccessMiscTest : public ::testing::TestWithParam<std::tuple<std::pair<int, int> > >, public TileDbSparseMatrixTestMethods {};

TEST_P(TileDbSparseAccessMiscTest, Apply) {
    // Putting it through its paces for correct parallelization via apply.
    auto param = GetParam();
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, global_cache_size);
    tatami::CompressedSparseRowMatrix<double, int> ref(NR, NC, contents.value, contents.index, contents.ptr);

    EXPECT_EQ(tatami::row_sums(&mat), tatami::row_sums(&ref));
    EXPECT_EQ(tatami::column_sums(&mat), tatami::column_sums(&ref));
}

TEST_P(TileDbSparseAccessMiscTest, LruReuse) {
    // Check that the LRU cache works as expected when cache elements are
    // constantly re-used in a manner that changes the last accessed element.
    auto param = GetParam();
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, global_cache_size);
    tatami::CompressedSparseRowMatrix<double, int> ref(NR, NC, contents.value, contents.index, contents.ptr);

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

TEST_P(TileDbSparseAccessMiscTest, Oracle) {
    auto param = GetParam();
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, global_cache_size);
    tatami::CompressedSparseRowMatrix<double, int> ref(NR, NC, contents.value, contents.index, contents.ptr);

    tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, false); // consecutive
    tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, false);

    tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, true); // randomized
    tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, true);
}

INSTANTIATE_TEST_CASE_P(
    TileDbSparseMatrix,
    TileDbSparseAccessMiscTest,
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

class TileDbSparseSlicedTest : public ::testing::TestWithParam<std::tuple<bool, size_t, std::vector<double>, std::pair<int, int> > >, public TileDbSparseMatrixTestMethods {};

TEST_P(TileDbSparseSlicedTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, global_cache_size);
    tatami::CompressedSparseRowMatrix<double, int> ref(NR, NC, contents.value, contents.index, contents.ptr);

    {
        size_t FIRST = interval_info[0] * NC, LAST = interval_info[1] * NC;
        tatami_test::test_sliced_row_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }

    {
        size_t FIRST = interval_info[0] * NR, LAST = interval_info[1] * NR;
        tatami_test::test_sliced_column_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }
}

TEST_P(TileDbSparseSlicedTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int, true> mat(fpath, name, global_cache_size);
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, contents.value, contents.index, contents.ptr));
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

INSTANTIATE_TEST_CASE_P(
    TileDbSparseMatrix,
    TileDbSparseSlicedTest,
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

class TileDbSparseIndexedTest : public ::testing::TestWithParam<std::tuple<bool, size_t, std::vector<double>, std::pair<int, int> > >, public TileDbSparseMatrixTestMethods {};

TEST_P(TileDbSparseIndexedTest, Basic) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int> mat(fpath, name, global_cache_size);
    tatami::CompressedSparseRowMatrix<double, int> ref(NR, NC, contents.value, contents.index, contents.ptr);

    {
        size_t FIRST = interval_info[0] * NC, STEP = interval_info[1];
        tatami_test::test_indexed_row_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }

    {
        size_t FIRST = interval_info[0] * NR, STEP = interval_info[1];
        tatami_test::test_indexed_column_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }
}

TEST_P(TileDbSparseIndexedTest, Transposed) {
    auto param = GetParam();
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_tiledb::TileDbSparseMatrix<double, int, true> mat(fpath, name, global_cache_size);
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, contents.value, contents.index, contents.ptr));
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

INSTANTIATE_TEST_CASE_P(
    TileDbSparseMatrix,
    TileDbSparseIndexedTest,
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

