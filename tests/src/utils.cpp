#ifndef TATAMI_TILEDB_TEST_PARALLEL
#include <gtest/gtest.h>
#include "tatami_tiledb/tatami_tiledb.hpp"
#include "tatami_test/temp_file_path.hpp"
#include <cstdint>

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

#endif
