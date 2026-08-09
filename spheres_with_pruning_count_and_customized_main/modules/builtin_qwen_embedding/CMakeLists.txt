add_library(builtin_qwen_embedding)

add_subdirectory(tokenizer)
add_subdirectory(qwen_model)

target_sources(
        builtin_qwen_embedding
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/builtin_qwen_embedding.cppm
)

target_link_libraries(
        builtin_qwen_embedding
        PRIVATE
        array_view
        builtin_qwen_tokenizer
        builtin_qwen_model
        m
)