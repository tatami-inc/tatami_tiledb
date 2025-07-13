cmake_minimum_required(VERSION 3.24)

project(tatami_tiledb
    VERSION 2.0.0
    DESCRIPTION "tatami bindings for TileDB"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(tatami_tiledb INTERFACE)
add_library(tatami::tatami_tiledb ALIAS tatami_tiledb)

option(TATAMI_TILEDB_FETCH_EXTERN "Automatically fetch tatami_tiledb's external dependencies." ON)
if(TATAMI_TILEDB_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(tatami_tatami_chunked 2.1.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.1 CONFIG REQUIRED)
endif()

target_link_libraries(tatami_tiledb INTERFACE tatami::tatami_chunked ltla::sanisizer)

option(TATAMI_TILEDB_FIND_TILEDB "Try to find and link to TileDB for tatami_tiledb." ON)
if(TATAMI_TILEDB_FIND_TILEDB)
    find_package(TileDB)
    if (TileDB_FOUND)
        target_link_libraries(tatami_tiledb INTERFACE TileDB::tiledb_shared)
    endif()
endif()

# Switch between include directories depending on whether the downstream is
# using the build directly or is using the installed package.
include(GNUInstallDirs)
target_include_directories(tatami_tiledb
    INTERFACE 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_tiledb>"
)

# Building the test-related machinery, if we are compiling this library directly.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(TATAMI_TILEDB_TESTS "Build tatami_tiledb's test suite." ON)
else()
    option(TATAMI_TILEDB_TESTS "Build tatami_tiledb's test suite." OFF)
endif()

if(TATAMI_TILEDB_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Installing for find_package.
include(CMakePackageConfigHelpers)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_tiledb)

install(TARGETS tatami_tiledb 
    EXPORT tatami_tiledbTargets)

install(EXPORT tatami_tiledbTargets
    FILE tatami_tatami_tiledbTargets.cmake
    NAMESPACE tatami::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_tiledb)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_tiledbConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_tiledb)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_tiledbConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_tiledb)
