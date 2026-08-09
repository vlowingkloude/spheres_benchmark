add_library(baseline)

target_sources(
        baseline
        PRIVATE
        baseline.cpp
)

target_include_directories(
        baseline
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)
