find_package(PostgreSQL REQUIRED)

add_library(pgvector_benchmark_adapter pgvector_benchmark.cpp)
target_include_directories(
        pgvector_benchmark_adapter
        PUBLIC ${CMAKE_SOURCE_DIR}/include
        PRIVATE ${PostgreSQL_INCLUDE_DIRS}
)
target_link_libraries(pgvector_benchmark_adapter PRIVATE ${PostgreSQL_LIBRARIES})
