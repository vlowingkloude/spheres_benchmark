add_library(builtin_qwen_model_generated)

target_sources(
        builtin_qwen_model_generated
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/qwen3embedding.cppm
)

target_link_libraries(
        builtin_qwen_model_generated
        PRIVATE
        SpheresDB_build_options
)