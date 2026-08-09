cmake_minimum_required(VERSION 4.2)

project(SpheresDB)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)

OPTION(ENABLE_BUILTIN_QWEN_EMBEDDING_MODEL "enable the builtin Qwen embedding model or not" OFF)
OPTION(BUILD_EXAMPLE "build the main file in example" ON)

add_subdirectory(modules/definitions)
add_subdirectory(modules/memorymap)
add_subdirectory(modules/array_view)
add_subdirectory(modules/distance)
add_subdirectory(modules/storage)
add_subdirectory(modules/execution)
add_subdirectory(modules/database)

if (ENABLE_BUILTIN_QWEN_EMBEDDING_MODEL)
    add_subdirectory(modules/builtin_qwen_embedding)
endif ()

add_subdirectory(modules/interface)

if (BUILD_EXAMPLE)
    add_subdirectory(example)
endif ()

add_library(
        SpheresDB_build_options INTERFACE
)

if (ENABLE_BUILTIN_QWEN_EMBEDDING_MODEL)
    target_compile_definitions(SpheresDB_build_options
            INTERFACE
            BUILD_QWEN=1
    )
endif ()