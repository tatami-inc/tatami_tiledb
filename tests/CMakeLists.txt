include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(libtest
    src/TileDbDenseMatrix.cpp
    src/TileDbSparseMatrix.cpp)

find_package(TileDB PATHS lib)

target_link_libraries(libtest 
    gtest_main
    TileDB::tiledb_shared 
    tatami_tiledb)

include(CheckIncludeFiles)
check_include_files(filesystem HAVE_CXX_FS)
if (NOT HAVE_CXX_FS)
    target_link_libraries(libtest stdc++fs)
endif()

# Checking whether to add OpenMP support. This is turned off
# by default to make it easier to debug test failures.
set(USE_OPENMP OFF CACHE BOOL "Compile with OpenMP support")
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(libtest OpenMP::OpenMP_CXX)
endif()

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

# Making the tests discoverable.
include(GoogleTest)
gtest_discover_tests(libtest)
