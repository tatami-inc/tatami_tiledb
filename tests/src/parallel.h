#ifndef TATAMI_TILEDB_TEST_PARALLEL_HPP
#define TATAMI_TILEDB_TEST_PARALLEL_HPP

#include <mutex>

inline std::mutex& get_mutex() {
    static std::mutex mut;
    return mut;
}

template<typename Function_>
void lockerup(Function_ fun) {
    std::lock_guard lock(get_mutex());
    fun();
}

#define TATAMI_TILEDB_PARALLEL_LOCK ::lockerup

#endif
