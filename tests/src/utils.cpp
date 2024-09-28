#include <gtest/gtest.h>
#include "tatami_tiledb/tatami_tiledb.hpp"
#include "tatami_test/temp_file_path.hpp"
#include "tatami_test/tatami_test.hpp"

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>

/*************************************
 *************************************/

class TiledbDimensionTest : public ::testing::Test {
protected:
    tiledb::Context ctx;
    inline static std::string fpath;

    static void SetUpTestSuite() {
        fpath = tatami_test::temp_file_path("tatami-dim-test");
    }

    tiledb::Array create_array(const tiledb::Dimension& dim) {
        tiledb::Domain domain(ctx);
        domain.add_dimension(dim);
        tiledb::ArraySchema schema(ctx, TILEDB_SPARSE);
        schema.set_domain(domain);
        schema.add_attribute(tiledb::Attribute::create<double>(ctx, "WHEE"));

        std::filesystem::remove_all(fpath);
        tiledb::Array::create(fpath, schema);
        return tiledb::Array(ctx, fpath, TILEDB_READ);
    }
};

TEST_F(TiledbDimensionTest, Int32NegativeStart) {
    // Using some extremes to check for overflows that aren't hidden by promotion to int.
    int32_t lower_bound = -2147483647;
    int32_t upper_bound = 2147483647;

    auto dim = tiledb::Dimension::create<int32_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint32_t>(), 4294967295u);
    EXPECT_EQ(vdim.tile<int>(), 100);

    std::vector<int32_t> test_runs { lower_bound, 0, upper_bound };
    std::vector<uint32_t> corrected(test_runs.size());
    vdim.correct_indices(test_runs.data(), test_runs.size(), corrected.data());
    std::vector<uint32_t> expected{ 0, 2147483647u, 4294967294u };
    EXPECT_EQ(corrected, expected);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int64_t>(sarr, 0, static_cast<int64_t>(upper_bound) + 1, 100);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1); // check that addition works with a barely positive start.
        EXPECT_EQ(roundtrip[1], 100);
    }

    // Repeat this process with unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound); 
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, static_cast<uint64_t>(upper_bound) + 1, 100);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1);
        EXPECT_EQ(roundtrip[1], 100);
    }
}

TEST_F(TiledbDimensionTest, Int32ZeroStart) {
    auto upper_bound = std::numeric_limits<int32_t>::max();

    auto dim = tiledb::Dimension::create<int32_t>(ctx, "foo", {{ 0, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint32_t>(), 2147483648u);
    EXPECT_EQ(vdim.tile<int>(), 100);

    std::vector<int32_t> test_runs { 0, 100, upper_bound };
    std::vector<uint32_t> corrected(test_runs.size());
    vdim.correct_indices(test_runs.data(), test_runs.size(), corrected.data());
    std::vector<uint32_t> expected{ 0, 100u, 2147483647u };
    EXPECT_EQ(corrected, expected);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 0);
        EXPECT_EQ(roundtrip[1], 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int64_t>(sarr, 0, 200, 100);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 200); // check that addition works with a barely positive start.
        EXPECT_EQ(roundtrip[1], 299);
    }

    // Repeat this process with unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 0); 
        EXPECT_EQ(roundtrip[1], 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, 1000, 100);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1000);
        EXPECT_EQ(roundtrip[1], 1099);
    }
}

TEST_F(TiledbDimensionTest, Int32PositiveStart) {
    auto upper_bound = std::numeric_limits<int32_t>::max();

    auto dim = tiledb::Dimension::create<int32_t>(ctx, "foo", {{ 1000, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint32_t>(), 2147482648u);
    EXPECT_EQ(vdim.tile<int>(), 100);

    std::vector<int32_t> test_runs { 1000, 2000, upper_bound };
    std::vector<uint32_t> corrected(test_runs.size());
    vdim.correct_indices(test_runs.data(), test_runs.size(), corrected.data());
    std::vector<uint32_t> expected{ 0, 1000, 2147482647u };
    EXPECT_EQ(corrected, expected);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1000);
        EXPECT_EQ(roundtrip[1], 1009);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int64_t>(sarr, 0, 200, 100);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1200); // check that addition works with a barely positive start.
        EXPECT_EQ(roundtrip[1], 1299);
    }

    // Repeat this process with unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1000); 
        EXPECT_EQ(roundtrip[1], 1009);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, 1000, 100);
        auto roundtrip = sarr.range<int32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 2000);
        EXPECT_EQ(roundtrip[1], 2099);
    }
}

TEST_F(TiledbDimensionTest, Int8Simple) {
    int8_t lower_bound = -127;
    int8_t upper_bound = 127;

    auto dim = tiledb::Dimension::create<int8_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint8_t>(), 255);
    EXPECT_EQ(vdim.tile<int>(), 100);
    EXPECT_EQ(vdim.correct_index<int>(lower_bound), 0);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int8_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int64_t>(sarr, 0, static_cast<int64_t>(upper_bound) + 1, 100);
        auto roundtrip = sarr.range<int8_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1); // check that addition works with a barely positive start.
        EXPECT_EQ(roundtrip[1], 100);
    }

    // Repeat this process with unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int8_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound); 
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, static_cast<uint64_t>(upper_bound) + 1, 100);
        auto roundtrip = sarr.range<int8_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1);
        EXPECT_EQ(roundtrip[1], 100);
    }
}

TEST_F(TiledbDimensionTest, Uint8Simple) {
    uint8_t lower_bound = 0;
    uint8_t upper_bound = 254;

    auto dim = tiledb::Dimension::create<uint8_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint8_t>(), 255);
    EXPECT_EQ(vdim.tile<int>(), 100);
    EXPECT_EQ(vdim.correct_index<int>(lower_bound), 0);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<uint8_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    // Now trying unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, 10, 100);
        auto roundtrip = sarr.range<uint8_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound + 10);
        EXPECT_EQ(roundtrip[1], lower_bound + 109);
    }
}

TEST_F(TiledbDimensionTest, Int16Simple) {
    int16_t lower_bound = -32767;
    int16_t upper_bound = 32767;

    auto dim = tiledb::Dimension::create<int16_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint16_t>(), 65535);
    EXPECT_EQ(vdim.tile<int>(), 100);
    EXPECT_EQ(vdim.correct_index<int>(lower_bound), 0);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int16_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int64_t>(sarr, 0, static_cast<int64_t>(upper_bound) + 1, 100);
        auto roundtrip = sarr.range<int16_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1); // check that addition works with a barely positive start.
        EXPECT_EQ(roundtrip[1], 100);
    }

    // Repeat this process with unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int16_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound); 
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, static_cast<uint64_t>(upper_bound) + 1, 100);
        auto roundtrip = sarr.range<int16_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1);
        EXPECT_EQ(roundtrip[1], 100);
    }
}

TEST_F(TiledbDimensionTest, Uint16Simple) {
    uint16_t lower_bound = 0;
    uint16_t upper_bound = 65354;

    auto dim = tiledb::Dimension::create<uint16_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint16_t>(), 65355);
    EXPECT_EQ(vdim.tile<int>(), 100);
    EXPECT_EQ(vdim.correct_index<int>(lower_bound), 0);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<uint16_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    // Now trying unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, 10, 100);
        auto roundtrip = sarr.range<uint16_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound + 10);
        EXPECT_EQ(roundtrip[1], lower_bound + 109);
    }
}

TEST_F(TiledbDimensionTest, Uint32Simple) {
    uint32_t lower_bound = 0;
    uint32_t upper_bound = 4294967294u;

    auto dim = tiledb::Dimension::create<uint32_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint32_t>(), 4294967295u);
    EXPECT_EQ(vdim.tile<int>(), 100);
    EXPECT_EQ(vdim.correct_index<int>(lower_bound), 0);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<uint32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    // Now trying unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, 10, 100);
        auto roundtrip = sarr.range<uint32_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound + 10);
        EXPECT_EQ(roundtrip[1], lower_bound + 109);
    }
}

TEST_F(TiledbDimensionTest, Int64Simple) {
    int64_t lower_bound = -9223372036854775807l;
    int64_t upper_bound = 9223372036854775807l - 100l; // need to subtract by the tile extent to allow domain expansion.

    auto dim = tiledb::Dimension::create<int64_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint64_t>(), 18446744073709551615ul - 100ul);
    EXPECT_EQ(vdim.tile<int>(), 100);
    EXPECT_EQ(vdim.correct_index<int>(lower_bound), 0);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int64_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    // Repeat this process with unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<int64_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound); 
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, 9223372036854775808ul, 100);
        auto roundtrip = sarr.range<int64_t>(0, 0);
        EXPECT_EQ(roundtrip[0], 1);
        EXPECT_EQ(roundtrip[1], 100);
    }
}

TEST_F(TiledbDimensionTest, Uint64Simple) {
    uint64_t lower_bound = 0;
    uint64_t upper_bound = 18446744073709551614ul - 100ul; // need to subtract by the tile extent to allow domain expansion.

    auto dim = tiledb::Dimension::create<uint64_t>(ctx, "foo", {{ lower_bound, upper_bound }}, 100);
    tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
    EXPECT_EQ(vdim.extent<uint64_t>(), 18446744073709551615ul - 100ul);
    EXPECT_EQ(vdim.tile<int>(), 100);
    EXPECT_EQ(vdim.correct_index<int>(lower_bound), 0);

    auto array = create_array(dim);

    // Check for addition with signed values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<int8_t>(sarr, 0, 0, 10);
        auto roundtrip = sarr.range<uint64_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound);
        EXPECT_EQ(roundtrip[1], lower_bound + 9);
    }

    // Now trying unsigned values.
    {
        tiledb::Subarray sarr(ctx, array);
        vdim.add_range<uint64_t>(sarr, 0, 10, 100);
        auto roundtrip = sarr.range<uint64_t>(0, 0);
        EXPECT_EQ(roundtrip[0], lower_bound + 10);
        EXPECT_EQ(roundtrip[1], lower_bound + 109);
    }
}

TEST_F(TiledbDimensionTest, Unknown) {
    {
        auto dim = tiledb::Dimension::create<double>(ctx, "foo", {{ 0, 1000 }}, 100);
        tatami_test::throws_error([&]() {
            tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
        }, "unknown");
    }

    {
        auto dim = tiledb::Dimension::create<int64_t>(ctx, "foo", {{ 0, 1000 }}, 100);
        tatami_tiledb::internal::VariablyTypedDimension vdim(dim);
        tatami_test::throws_error([&]() {
            vdim.correct_index<int>(10.0); 
        }, "unsupported");
    }
}

/*************************************
 *************************************/

TEST(TypeSize, Basic) {
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_CHAR), 1);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_INT8), 1);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_UINT8), 1);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_INT16), 2);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_UINT16), 2);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_INT32), 4);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_UINT32), 4);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_INT64), 8);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_UINT64), 8);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_FLOAT32), 4);
    EXPECT_EQ(tatami_tiledb::internal::determine_type_size(TILEDB_FLOAT64), 8);

    tatami_test::throws_error([&]() {
        tatami_tiledb::internal::determine_type_size(TILEDB_BLOB);
    }, "unknown");
}

/*************************************
 *************************************/

class TiledbVectorTest : public ::testing::Test {
protected:
    tiledb::Context ctx;
    inline static std::string fpath;

    static void SetUpTestSuite() {
        fpath = tatami_test::temp_file_path("tatami-vec-test");
    }

    template<typename Type_>
    void run_basic_test(tiledb_datatype_t tdb_type) {
        {
            tiledb::Domain domain(ctx);
            domain.add_dimension(tiledb::Dimension::create<int>(ctx, "rows", {{0, 9}}, 10));
            tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
            schema.set_domain(domain);
            schema.add_attribute(tiledb::Attribute::create<Type_>(ctx, "WHEE"));

            std::filesystem::remove_all(fpath);
            tiledb::Array::create(fpath, schema);

            tiledb::Array array(ctx, fpath, TILEDB_WRITE);
            tiledb::Query query(ctx, array);

            tiledb::Subarray subarray(ctx, array);
            subarray.add_range<int>(0, 0, 9);
            query.set_subarray(subarray);

            std::vector<Type_> values(10);
            std::iota(values.begin(), values.end(), 0);

            query.set_layout(TILEDB_ROW_MAJOR).set_data_buffer("WHEE", values);
            query.submit();
            query.finalize();
            array.close();
        }

        tiledb::Array array(ctx, fpath, TILEDB_READ);
        tiledb::Query query(ctx, array);
        tiledb::Subarray subarray(ctx, array);
        subarray.add_range<int>(0, 3, 8);
        query.set_subarray(subarray);

        tatami_tiledb::internal::VariablyTypedVector vec(tdb_type, 10);
        vec.set_data_buffer(query, "WHEE", 3, 6);

        query.set_layout(TILEDB_ROW_MAJOR);
        query.submit();
        query.finalize();

        std::vector<double> test(10);
        vec.copy(5, 5, test.data() + 5);
        {
            std::vector<double> expected{ 0, 0, 0, 0, 0, 5, 6, 7, 8, 0 };
            EXPECT_EQ(expected, test);
        }

        std::fill(test.begin(), test.end(), 0);
        vec.copy(0, 5, test.data());
        {
            std::vector<double> expected{ 0, 0, 0, 3, 4, 0, 0, 0, 0, 0 };
            EXPECT_EQ(expected, test);
        }

        vec.shift(5, 3, 0);
        vec.copy(0, 10, test.data());
        {
            std::vector<double> expected{ 5, 6, 7, 3, 4, 5, 6, 7, 8, 0 };
            EXPECT_EQ(expected, test);
        }
    }

    template<typename Index_>
    void run_index_test(tiledb_datatype_t tdb_type) {
        auto dim = tiledb::Dimension::create<Index_>(ctx, "rows", {{10, 19}}, 10);

        {
            tiledb::Domain domain(ctx);
            domain.add_dimension(dim);
            tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
            schema.set_domain(domain);
            schema.add_attribute(tiledb::Attribute::create<Index_>(ctx, "WHEE"));

            std::filesystem::remove_all(fpath);
            tiledb::Array::create(fpath, schema);

            tiledb::Array array(ctx, fpath, TILEDB_WRITE);
            tiledb::Query query(ctx, array);

            tiledb::Subarray subarray(ctx, array);
            subarray.add_range<Index_>(0, 10, 19);
            query.set_subarray(subarray);

            std::vector<Index_> values { 10, 11, 11, 12, 12, 12, 13, 13, 13, 13 };
            query.set_layout(TILEDB_ROW_MAJOR).set_data_buffer("WHEE", values);
            query.submit();
            query.finalize();
            array.close();
        }

        tiledb::Array array(ctx, fpath, TILEDB_READ);
        tiledb::Query query(ctx, array);
        tiledb::Subarray subarray(ctx, array);
        subarray.add_range<Index_>(0, 10, 19);
        query.set_subarray(subarray);

        tatami_tiledb::internal::VariablyTypedVector vec(tdb_type, 10);
        vec.set_data_buffer(query, "WHEE", 0, 10);

        query.set_layout(TILEDB_ROW_MAJOR);
        query.submit();
        query.finalize();

        std::vector<double> test(10);
        vec.copy(0, 10, dim, test.data());
        std::vector<double> expected{ 0, 1, 1, 2, 2, 2, 3, 3, 3, 3 };
        EXPECT_EQ(expected, test);

        std::vector<std::pair<int, int> > counts;
        vec.compact(0, 10, dim, counts);
        std::vector<std::pair<int, int> > expected_counts { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4} };
        EXPECT_EQ(counts, expected_counts);
    }
};

TEST_F(TiledbVectorTest, Basic) {
    run_basic_test<int8_t>(TILEDB_INT8);
    run_basic_test<uint8_t>(TILEDB_UINT8);
    run_basic_test<int16_t>(TILEDB_INT16);
    run_basic_test<uint16_t>(TILEDB_UINT16);
    run_basic_test<int32_t>(TILEDB_INT32);
    run_basic_test<uint32_t>(TILEDB_UINT32);
    run_basic_test<int64_t>(TILEDB_INT64);
    run_basic_test<uint64_t>(TILEDB_UINT64);
    run_basic_test<float>(TILEDB_FLOAT32);
    run_basic_test<double>(TILEDB_FLOAT64);
}

TEST_F(TiledbVectorTest, Index) {
    run_index_test<int8_t>(TILEDB_INT8);
    run_index_test<uint8_t>(TILEDB_UINT8);
    run_index_test<int16_t>(TILEDB_INT16);
    run_index_test<uint16_t>(TILEDB_UINT16);
    run_index_test<int32_t>(TILEDB_INT32);
    run_index_test<uint32_t>(TILEDB_UINT32);
    run_index_test<int64_t>(TILEDB_INT64);
    run_index_test<uint64_t>(TILEDB_UINT64);
}
