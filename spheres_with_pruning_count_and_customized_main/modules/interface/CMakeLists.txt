add_library(spheres)

target_sources(
        spheres
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/interface.cppm
)

target_link_libraries(
        spheres
        PRIVATE
        array_view
        database
        definitions
        distance
        execution
        memorymap
        storage
        m
        SpheresDB_build_options
)

if (ENABLE_BUILTIN_QWEN_EMBEDDING_MODEL)
    target_link_libraries(spheres
            PRIVATE
            builtin_qwen_embedding
    )
endif ()