add_subdirectory(generated)

add_library(builtin_qwen_model)

target_sources(
        builtin_qwen_model
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/qwen_model.cppm
)

target_link_libraries(
        builtin_qwen_model
        PRIVATE
        builtin_qwen_model_generated
        memorymap
        SpheresDB_build_options
)