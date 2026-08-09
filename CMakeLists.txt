cmake_minimum_required(VERSION 4.2)

project(spheres_benchmark)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_subdirectory(baseline)
add_subdirectory(faiss)
#add_subdirectory(hnswlib)
add_subdirectory(spheres)
#add_subdirectory(duckdb)
#add_subdirectory(pgvector)

add_executable(
        spheres_benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(spheres_benchmark
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
        spheres_benchmark
        PRIVATE
        spheres_benchmark_adapter
        faiss_benchmark
)

#target_link_libraries(spheres_benchmark PRIVATE baseline faiss_benchmark hnswlib_benchmark spheres_benchmark_adapter duckdb_benchmark_adapter pgvector_benchmark_adapter)
