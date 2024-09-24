#ifndef TATAMI_TILEDB_UTILS_HPP
#define TATAMI_TILEDB_UTILS_HPP

#include <tiledb/tiledb>

#include <vector>
#include <cstdint>
#include <stdexcept>

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
            case TILEDB_CHAR: my_char.resize(len); break;
            case TILEDB_INT8: my_i8.resize(len); break;
            case TILEDB_UINT8: my_u8.resize(len); break;
            case TILEDB_INT16: my_i16.resize(len); break;
            case TILEDB_UINT16: my_u16.resize(len); break;
            case TILEDB_INT32: my_i32.resize(len); break;
            case TILEDB_UINT32: my_u32.resize(len); break;
            case TILEDB_INT64: my_i64.resize(len); break;
            case TILEDB_UINT64: my_u64.resize(len); break;
            case TILEDB_FLOAT32: my_f32.resize(len); break;
            case TILEDB_FLOAT64: my_f64.resize(len); break;
            default: throw std::runtime_error("unknown TileDB datatype '" + std::to_string(type) + "'");
        }
    }

public:
    void set(tiledb::Query& query, const std::string& name) {
        switch (my_type) {
            case TILEDB_CHAR: query.set_data_buffer(name, my_char); break;
            case TILEDB_INT8: query.set_data_buffer(name, my_i8); break;
            case TILEDB_UINT8: query.set_data_buffer(name, my_u8); break;
            case TILEDB_INT16: query.set_data_buffer(name, my_i16); break;
            case TILEDB_UINT16: query.set_data_buffer(name, my_u16); break;
            case TILEDB_INT32: query.set_data_buffer(name, my_i32); break;
            case TILEDB_UINT32: query.set_data_buffer(name, my_u32); break;
            case TILEDB_INT64: query.set_data_buffer(name, my_i64); break;
            case TILEDB_UINT64: query.set_data_buffer(name, my_u64); break;
            case TILEDB_FLOAT32: query.set_data_buffer(name, my_f32); break;
            case TILEDB_FLOAT64: query.set_data_buffer(name, my_f64); break;
            default: break;
        }
    }

    template<typename Final_>
    void copy(std::vector<Final_>& dest) const {
        switch (my_type) {
            case TILEDB_CHAR: std::copy(my_char.begin(), my_char.end(), dest.begin()); break;
            case TILEDB_INT8: std::copy(my_i8.begin(), my_i8.end(), dest.begin()); break;
            case TILEDB_UINT8: std::copy(my_u8.begin(), my_u8.end(), dest.begin()); break;
            case TILEDB_INT16: std::copy(my_i16.begin(), my_i16.end(), dest.begin()); break;
            case TILEDB_UINT16: std::copy(my_u16.begin(), my_u16.end(), dest.begin()); break;
            case TILEDB_INT32: std::copy(my_i32.begin(), my_i32.end(), dest.begin()); break;
            case TILEDB_UINT32: std::copy(my_u32, my_u32.end(), dest.begin()); break;
            case TILEDB_INT64: std::copy(my_i64.begin(), my_i64.end(), dest.begin()); break;
            case TILEDB_UINT64: std::copy(my_u64.begin(), my_u64.end(), dest.begin()); break;
            case TILEDB_FLOAT32: std::copy(my_f32.begin(), my_f32.end(), dest.begin()); break;
            case TILEDB_FLOAT64: std::copy(my_f64.begin(), my_f64.end(), dest.begin()); break;
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

}

}

#endif
