#ifndef TATAMI_TILEDB_PROCESS_CONSECUTIVE_INDICES_HPP
#define TATAMI_TILEDB_PROCESS_CONSECUTIVE_INDICES_HPP

namespace tatami_tiledb {

template<typename Index_, class Function_>
void process_consecutive_indices(const Index_* const indices, const Index_ length, const Function_ fun) {
    if (length > 0) {
        Index_ start = indices[0], last = start + 1;
        for (Index_ i = 1; i < length; ++i) {
            if (indices[i] > last) {
                fun(start, last - start);
                start = indices[i];
            }
            last = indices[i] + 1;
        }
        fun(start, last - start);
    }
}

}

#endif
