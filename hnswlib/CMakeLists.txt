include(FetchContent)

FetchContent_Declare(
        hnswlib_source
        GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
        GIT_TAG v0.8.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(hnswlib_source)

add_library(hnswlib_benchmark hnswlib_benchmark.cpp)
find_package(OpenMP REQUIRED)
target_include_directories(hnswlib_benchmark PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hnswlib_benchmark PRIVATE hnswlib::hnswlib OpenMP::OpenMP_CXX)
