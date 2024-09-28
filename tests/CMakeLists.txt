include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

include(CheckIncludeFiles)
check_include_files(filesystem HAVE_CXX_FS)

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

include(GoogleTest)

macro(decorate_test target)
    target_link_libraries(${target} 
        gtest_main
        tatami_tiledb)

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
