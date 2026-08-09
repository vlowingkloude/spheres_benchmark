include(FetchContent)

set(BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(ENABLE_BUILTIN_QWEN_EMBEDDING_MODEL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        spheresdb_source
        GIT_REPOSITORY https://github.com/vlowingkloude/SpheresDB.git
        GIT_TAG c72c13833d51d7bd609b9a0d68e0dcd4405bac0c
)
FetchContent_MakeAvailable(spheresdb_source)

add_library(spheres_benchmark_adapter spheres_benchmark.cpp)
target_include_directories(spheres_benchmark_adapter PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spheres_benchmark_adapter PRIVATE spheres)
