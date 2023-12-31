set(TARGET_NAME ${PROJECT_NAME})

add_executable(${TARGET_NAME}) 
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_sources(${TARGET_NAME} PRIVATE
    main.cpp
)

find_boost_libs(program_options)

target_link_libraries(${TARGET_NAME} PRIVATE
    Boost::program_options
)

find_package(Qt6 COMPONENTS Widgets)

set(PRIVATE_PCH
    <vector>
    <string>
    <iostream>
)

if(Qt6_FOUND)
    qt_standard_project_setup()
    # set_target_properties(${TARGET_NAME} PROPERTIES
    #     WIN32_EXECUTABLE ON
    # )

    target_compile_definitions(${TARGET_NAME} PRIVATE BUILD_WITH_QT)

    target_link_libraries(${TARGET_NAME} PRIVATE
        Qt6::Widgets
    )

    set(PRIVATE_PCH
        ${PRIVATE_PCH}
        <vector>
        <string>
        <iostream>
    )

    message(PRIVATE_PCH: ${PRIVATE_PCH})
endif()

if (ENABLE_PCH)
    target_precompile_headers(${TARGET_NAME} PRIVATE
        ${PRIVATE_PCH}
    )
endif()
