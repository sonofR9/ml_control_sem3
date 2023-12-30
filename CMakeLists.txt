cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0076 NEW)
if (${CMAKE_VERSION} VERSION_GREATER "3.24")
    cmake_policy(SET CMP0135 OLD)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/declare_options.cmake)
include(cmake/apply_options.cmake)
include(cmake/get_boost_lib.cmake)


project(ml_control VERSION 0.1.0)


add_executable(${PROJECT_NAME} 
               "src/main.cpp"
) 
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME} PRIVATE
                "src"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
                Boost::program_options
)

# if(ENABLE_MATPLOTLIB)
#     target_include_directories(${PROJECT_NAME} PRIVATE
#                     "/usr/local/lib/python3.10/dist-packages/numpy/core/include"
#                     "/usr/include/python3.10"
#     )
#     target_link_libraries(${PROJECT_NAME} PRIVATE 
#                         "-lpython3.10"
#     )
#     add_compile_definitions(MATPLOTLIB)
# endif()

if (ENABLE_TEST)
    add_subdirectory(test)
endif()
