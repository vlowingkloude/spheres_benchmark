add_library(builtin_qwen_tokenizer)

target_sources(
        builtin_qwen_tokenizer
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/builtin_qwen_tokenizer.cppm
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/generated/generated_tokenizer_merge.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/generated/generated_tokenizer_tokenid.cpp
)

target_link_libraries(
        builtin_qwen_tokenizer
        PRIVATE
        SpheresDB_build_options
)