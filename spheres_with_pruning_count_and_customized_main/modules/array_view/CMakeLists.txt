add_library(array_view)

target_sources(
        array_view
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        array_view.cppm
)

target_link_libraries(
        array_view
        PRIVATE
        SpheresDB_build_options
)