add_executable(
        example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(example
        PRIVATE
        spheres
        SpheresDB_build_options
)