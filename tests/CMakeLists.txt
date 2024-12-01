include(FetchContent)
FetchContent_Declare(
    tatami_test
    GIT_REPOSITORY https://github.com/tatami-inc/tatami_test
    GIT_TAG master
)
FetchContent_MakeAvailable(tatami_test)

include(CheckIncludeFiles)
check_include_files(filesystem HAVE_CXX_FS)

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

include(GoogleTest)

macro(decorate_test target)
    target_link_libraries(${target} tatami_tiledb tatami_test)

    if (NOT HAVE_CXX_FS)
        target_link_libraries(${target} stdc++fs)
    endif()

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)

    if(DO_CODE_COVERAGE)
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${target})
endmacro()

add_executable(libtest
    src/DenseMatrix.cpp
    src/SparseMatrix.cpp
    src/utils.cpp
)
decorate_test(libtest)

add_executable(partest
    src/DenseMatrix.cpp
    src/SparseMatrix.cpp
)
decorate_test(partest)
target_compile_definitions(partest PRIVATE TATAMI_TILEDB_TEST_PARALLEL=1)
