add_library(zp7 INTERFACE)
target_include_directories(zp7 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(zp7 INTERFACE project_compiler_flags)

# add_executable(zp7_tests zp7_tests.cpp)

