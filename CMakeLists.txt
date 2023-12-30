cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0076 NEW)
if (${CMAKE_VERSION} VERSION_GREATER "3.24")
    cmake_policy(SET CMP0135 OLD)
endif()

project(ml_control VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/declare_options.cmake)
include(cmake/apply_options.cmake)
include(cmake/get_boost_lib.cmake)

add_subdirectory(src)

if (ENABLE_TEST)
    add_subdirectory(test)
endif()
