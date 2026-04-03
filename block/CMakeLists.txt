add_library(block INTERFACE)
target_include_directories(block INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(block INTERFACE project_compiler_flags)

add_executable(block_tests block_tests.cpp)
target_link_libraries(block_tests PRIVATE block Catch2::Catch2WithMain hashtable SSDLog)
catch_discover_tests(block_tests)
