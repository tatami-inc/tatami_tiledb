#ifndef TATAMI_TILEDB_UTILS_HPP
#define TATAMI_TILEDB_UTILS_HPP

#include <tiledb/tiledb>

#include <cstdint>
#include <stdexcept>
#include <limits>

namespace tatami_tiledb {

namespace internal {

// All TileDB-related members.
struct Components{
    Components(const std::string& location) : array(ctx, location, TILEDB_READ) {}
    tiledb::Context ctx;
    tiledb::Array array;
};

// Handling conversion from TileDB's storage type to our desired in-memory
// type - unlike HDF5, TileDB doesn't do this for us. 
class VariablyTypedVector {
public:
    VariablyTypedVector(tiledb_datatype_t type, size_t len) : my_type(type) {
        switch (my_type) {
            case TILEDB_CHAR:    my_char.resize(len); break;
            case TILEDB_INT8:    my_i8.resize(len);   break;
            case TILEDB_UINT8:   my_u8.resize(len);   break;
            case TILEDB_INT16:   my_i16.resize(len);  break;
            case TILEDB_UINT16:  my_u16.resize(len);  break;
            case TILEDB_INT32:   my_i32.resize(len);  break;
            case TILEDB_UINT32:  my_u32.resize(len);  break;
            case TILEDB_INT64:   my_i64.resize(len);  break;
            case TILEDB_UINT64:  my_u64.resize(len);  break;
            case TILEDB_FLOAT32: my_f32.resize(len);  break;
            case TILEDB_FLOAT64: my_f64.resize(len);  break;
            default: throw std::runtime_error("unknown TileDB datatype '" + std::to_string(type) + "'");
        }
    }

public:
    void set(tiledb::Query& query, const std::string& name, size_t offset, size_t len) {
        switch (my_type) {
            case TILEDB_CHAR:    query.set_data_buffer(name, my_char.data() + offset, len); break;
            case TILEDB_INT8:    query.set_data_buffer(name, my_i8.data()   + offset, len); break;
            case TILEDB_UINT8:   query.set_data_buffer(name, my_u8.data()   + offset, len); break;
            case TILEDB_INT16:   query.set_data_buffer(name, my_i16.data()  + offset, len); break;
            case TILEDB_UINT16:  query.set_data_buffer(name, my_u16.data()  + offset, len); break;
            case TILEDB_INT32:   query.set_data_buffer(name, my_i32.data()  + offset, len); break;
            case TILEDB_UINT32:  query.set_data_buffer(name, my_u32.data()  + offset, len); break;
            case TILEDB_INT64:   query.set_data_buffer(name, my_i64.data()  + offset, len); break;
            case TILEDB_UINT64:  query.set_data_buffer(name, my_u64.data()  + offset, len); break;
            case TILEDB_FLOAT32: query.set_data_buffer(name, my_f32.data()  + offset, len); break;
            case TILEDB_FLOAT64: query.set_data_buffer(name, my_f64.data()  + offset, len); break;
            default: break;
        }
    }

    template<typename Value_>
    void copy(size_t offset, size_t len, Value_* dest) const {
        switch (my_type) {
            case TILEDB_CHAR:    std::copy_n(my_char.begin() + offset, len, dest); break;
            case TILEDB_INT8:    std::copy_n(my_i8.begin()   + offset, len, dest); break;
            case TILEDB_UINT8:   std::copy_n(my_u8.begin()   + offset, len, dest); break;
            case TILEDB_INT16:   std::copy_n(my_i16.begin()  + offset, len, dest); break;
            case TILEDB_UINT16:  std::copy_n(my_u16.begin()  + offset, len, dest); break;
            case TILEDB_INT32:   std::copy_n(my_i32.begin()  + offset, len, dest); break;
            case TILEDB_UINT32:  std::copy_n(my_u32.begin()  + offset, len, dest); break;
            case TILEDB_INT64:   std::copy_n(my_i64.begin()  + offset, len, dest); break;
            case TILEDB_UINT64:  std::copy_n(my_u64.begin()  + offset, len, dest); break;
            case TILEDB_FLOAT32: std::copy_n(my_f32.begin()  + offset, len, dest); break;
            case TILEDB_FLOAT64: std::copy_n(my_f64.begin()  + offset, len, dest); break;
            default: break;
        }
    }

    void shift(size_t from, size_t len, size_t to) {
        switch (my_type) {
            case TILEDB_CHAR:    std::copy_n(my_char.begin() + from, len, my_char.begin() + to); break;
            case TILEDB_INT8:    std::copy_n(my_i8.begin()   + from, len, my_i8.begin()   + to); break;
            case TILEDB_UINT8:   std::copy_n(my_u8.begin()   + from, len, my_u8.begin()   + to); break;
            case TILEDB_INT16:   std::copy_n(my_i16.begin()  + from, len, my_i16.begin()  + to); break;
            case TILEDB_UINT16:  std::copy_n(my_u16.begin()  + from, len, my_u16.begin()  + to); break;
            case TILEDB_INT32:   std::copy_n(my_i32.begin()  + from, len, my_i32.begin()  + to); break;
            case TILEDB_UINT32:  std::copy_n(my_u32.begin()  + from, len, my_u32.begin()  + to); break;
            case TILEDB_INT64:   std::copy_n(my_i64.begin()  + from, len, my_i64.begin()  + to); break;
            case TILEDB_UINT64:  std::copy_n(my_u64.begin()  + from, len, my_u64.begin()  + to); break;
            case TILEDB_FLOAT32: std::copy_n(my_f32.begin()  + from, len, my_f32.begin()  + to); break;
            case TILEDB_FLOAT64: std::copy_n(my_f64.begin()  + from, len, my_f64.begin()  + to); break;
            default: break;
        }
    }

private:
    tiledb_datatype_t my_type = TILEDB_INT32;

    std::vector<char> my_char;
    std::vector<int8_t> my_i8;
    std::vector<uint8_t> my_u8;
    std::vector<int16_t> my_i16;
    std::vector<uint16_t> my_u16;
    std::vector<int32_t> my_i32;
    std::vector<uint32_t> my_u32;
    std::vector<int64_t> my_i64;
    std::vector<uint64_t> my_u64;
    std::vector<float> my_f32;
    std::vector<double> my_f64;
};

class VariablyTypedDimension {
public:
    VariablyTypedDimension() = default;

public:
    void reset(const tiledb::Dimension& dim) {
        my_type = dim.type();
        switch (my_type) {
            case TILEDB_INT8:    populate(dim, my_i8_start,  my_i8_end,  my_i8_tile);  break;
            case TILEDB_UINT8:   populate(dim, my_u8_start,  my_u8_end,  my_u8_tile);  break;
            case TILEDB_INT16:   populate(dim, my_i16_start, my_i16_end, my_i16_tile); break;
            case TILEDB_UINT16:  populate(dim, my_u16_start, my_u16_end, my_u16_tile); break;
            case TILEDB_INT32:   populate(dim, my_i32_start, my_i32_end, my_i32_tile); break;
            case TILEDB_UINT32:  populate(dim, my_u32_start, my_u32_end, my_u32_tile); break;
            case TILEDB_INT64:   populate(dim, my_i64_start, my_i64_end, my_i64_tile); break;
            case TILEDB_UINT64:  populate(dim, my_u64_start, my_u64_end, my_u64_tile); break;
            case TILEDB_FLOAT32: populate(dim, my_f32_start, my_f32_end, my_f32_tile); break;
            case TILEDB_FLOAT64: populate(dim, my_f64_start, my_f64_end, my_f64_tile); break;
            default: throw std::runtime_error("unknown TileDB datatype '" + std::to_string(my_type) + "'");
        }
    }

public:
    template<typename Index_>
    Index_ extent() const {
        Index_ output = 0;
        switch (my_type) {
            case TILEDB_INT8:    output = define_extent<Index_>(my_i8_start,  my_i8_end);  break;
            case TILEDB_UINT8:   output = define_extent<Index_>(my_u8_start,  my_u8_end);  break;
            case TILEDB_INT16:   output = define_extent<Index_>(my_i16_start, my_i16_end); break;
            case TILEDB_UINT16:  output = define_extent<Index_>(my_u16_start, my_u16_end); break;
            case TILEDB_INT32:   output = define_extent<Index_>(my_i32_start, my_i32_end); break;
            case TILEDB_UINT32:  output = define_extent<Index_>(my_u32_start, my_u32_end); break;
            case TILEDB_INT64:   output = define_extent<Index_>(my_i64_start, my_i64_end); break;
            case TILEDB_UINT64:  output = define_extent<Index_>(my_u64_start, my_u64_end); break;
            case TILEDB_FLOAT32: output = define_extent<Index_>(my_f32_start, my_f32_end); break;
            case TILEDB_FLOAT64: output = define_extent<Index_>(my_f64_start, my_f64_end); break;
            default: break;
        }
        return output;
    }

    template<typename Index_>
    Index_ tile() const {
        Index_ output = 0;
        switch (my_type) {
            case TILEDB_INT8:    output = my_i8_tile;  break;
            case TILEDB_UINT8:   output = my_u8_tile;  break;
            case TILEDB_INT16:   output = my_i16_tile; break;
            case TILEDB_UINT16:  output = my_u16_tile; break;
            case TILEDB_INT32:   output = my_i32_tile; break;
            case TILEDB_UINT32:  output = my_u32_tile; break;
            case TILEDB_INT64:   output = my_i64_tile; break;
            case TILEDB_UINT64:  output = my_u64_tile; break;
            case TILEDB_FLOAT32: output = my_f32_tile; break;
            case TILEDB_FLOAT64: output = my_f64_tile; break;
            default: break;
        }
        return output;
    }

    template<typename Index_>
    void add_range(tiledb::Subarray& subarray, int dim, Index_ start, Index_ length) const {
        switch (my_type) {
            case TILEDB_INT8:    add(subarray, dim, my_i8_start,  start, length);  break;
            case TILEDB_UINT8:   add(subarray, dim, my_u8_start,  start, length);  break;
            case TILEDB_INT16:   add(subarray, dim, my_i16_start, start, length); break;
            case TILEDB_UINT16:  add(subarray, dim, my_u16_start, start, length); break;
            case TILEDB_INT32:   add(subarray, dim, my_i32_start, start, length); break;
            case TILEDB_UINT32:  add(subarray, dim, my_u32_start, start, length); break;
            case TILEDB_INT64:   add(subarray, dim, my_i64_start, start, length); break;
            case TILEDB_UINT64:  add(subarray, dim, my_u64_start, start, length); break;
            case TILEDB_FLOAT32: add(subarray, dim, my_f32_start, start, length); break;
            case TILEDB_FLOAT64: add(subarray, dim, my_f64_start, start, length); break;
            default: break;
        }
    }

private:
    tiledb_datatype_t my_type = TILEDB_INT32;
    int8_t   my_i8_start,  my_i8_end,  my_i8_tile;
    uint8_t  my_u8_start,  my_u8_end,  my_u8_tile;
    int16_t  my_i16_start, my_i16_end, my_i16_tile;
    uint16_t my_u16_start, my_u16_end, my_u16_tile;
    int32_t  my_i32_start, my_i32_end, my_i32_tile;
    uint32_t my_u32_start, my_u32_end, my_u32_tile;
    int64_t  my_i64_start, my_i64_end, my_i64_tile;
    uint64_t my_u64_start, my_u64_end, my_u64_tile;
    float    my_f32_start, my_f32_end, my_f32_tile;
    double   my_f64_start, my_f64_end, my_f64_tile;

private:
    template<typename T>
    static void populate(const tiledb::Dimension& dim, T& start, T& end, T& tile) {
        auto dom = dim.domain<T>();
        start = dom.first;
        end = dom.second;
        tile = dim.tile_extent<T>();
    }

    template<typename Index_, typename T>
    Index_ define_extent(T start, T end) const {
        if constexpr(std::is_integral<T>::value && std::is_signed<T>::value) {
            if (start < 0 && end >= 0) {
                // Avoid overflow in the 'T' or its promoted equivalent. This
                // assumes that 'Index_' is actually large enough to store the
                // extent (and thus both 'end' and '-start-1'). It doesn't
                // assume that 'Index_' is signed, hence the if().
                return static_cast<Index_>(end) + static_cast<Index_>(-(start + 1)) + 2;
            }
        }

        // If start and end are the same sign, it's guaranteed not to overflow.
        return static_cast<Index_>(end - start) + 1;
    }

    template<typename T, typename Index_>
    void add(tiledb::Subarray& subarray, int dim, T domain_start, Index_ range_start, Index_ range_length) const {
        // range_length had better be positive!
        --range_length;

        if constexpr(std::is_integral<T>::value && std::is_signed<T>::value) {
            if (domain_start < 0) {
                T domain_end = safe_negative_add(domain_start, range_start + range_length);
                domain_start = safe_negative_add(domain_start, range_start);
                subarray.add_range<T>(dim, domain_start, domain_end);
                return;
            }
        }

        domain_start += range_start;
        subarray.add_range<T>(dim, domain_start, domain_start + range_length);
    }

    template<typename T, typename Index_>
    static T safe_negative_add(T l, Index_ r) {
        // The general principle here is to get us to the point where we're
        // dealing with two signed (or two unsigned) integers that can be
        // used in comparisons and arithmetic without surprises.
        if constexpr(std::is_signed<Index_>::value) {
            return l + r;
        } else {
            typename std::make_unsigned<T>::type ul = -(l + 1);
            if (ul < r) {
                return r - ul - 1;
            } else {
                return ul - r + 1;
            }
        }
    }
};

}

}

#endif
