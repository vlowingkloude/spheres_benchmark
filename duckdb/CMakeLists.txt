include(FetchContent)

cmake_policy(SET CMP0169 OLD)
set(BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHELL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        duckdb_vss_source
        GIT_REPOSITORY https://github.com/duckdb/duckdb-vss.git
        GIT_TAG 231863009fecb1bac92ad44c48ca8708b9b5cc66
        GIT_SUBMODULES ""
)
FetchContent_GetProperties(duckdb_vss_source)
if (NOT duckdb_vss_source_POPULATED)
    FetchContent_Populate(duckdb_vss_source)
endif ()

set(DUCKDB_EXTENSION_CONFIGS
        ${duckdb_vss_source_SOURCE_DIR}/extension_config.cmake
        CACHE STRING "" FORCE)

FetchContent_Declare(
        duckdb_source
        GIT_REPOSITORY https://github.com/duckdb/duckdb.git
        GIT_TAG fe871ebf45d777c97950cfe3e07a1a77a45f1503
)
FetchContent_MakeAvailable(duckdb_source)

add_library(duckdb_benchmark_adapter duckdb_benchmark.cpp)
target_include_directories(
        duckdb_benchmark_adapter
        PUBLIC ${CMAKE_SOURCE_DIR}/include
        PRIVATE
        ${duckdb_source_SOURCE_DIR}/src/include
        ${duckdb_source_SOURCE_DIR}/third_party/fmt/include
        ${duckdb_vss_source_SOURCE_DIR}/src/include
)
target_link_libraries(duckdb_benchmark_adapter PRIVATE duckdb vss_extension)
