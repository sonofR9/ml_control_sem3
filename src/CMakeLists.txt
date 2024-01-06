set(TARGET_NAME ${PROJECT_NAME})

set(SOURCES 
    global.cpp
    options.cpp
    optimize-model.cpp
    main.cpp
)

find_package(Qt6 COMPONENTS Widgets Core Charts)
if(true AND Qt6_FOUND)
    # qt_standard_project_setup()
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    set(SOURCES ${SOURCES}
        main-window.cpp

        main-window.h
    )
endif()

add_executable(${TARGET_NAME} ${SOURCES}) 


find_boost_libs(property_tree program_options serialization)

target_link_libraries(${TARGET_NAME} PRIVATE
    Boost::program_options
    Boost::property_tree
    Boost::serialization
)

set(PRIVATE_PCH
    <format>
    <vector>
    
    "optimize-model.h"
    "tensor.h"
)

if(true AND Qt6_FOUND)
    # set_target_properties(${TARGET_NAME} PROPERTIES
    #     WIN32_EXECUTABLE ON
    # )
    set_property(TARGET ${TARGET_NAME} PROPERTY WIN32_EXECUTABLE false)
    add_definitions(-DQT_NO_VERSION_TAGGING)

    target_compile_definitions(${TARGET_NAME} PRIVATE BUILD_WITH_QT)

    target_link_libraries(${TARGET_NAME} PRIVATE
        Qt6::Widgets
        Qt6::Core
        Qt6::Charts
    )

    # if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        # cmake_policy(SET CMP0111 OLD)
        # add_library(qwindows_plugin SHARED IMPORTED)
        # set_target_properties(qwindows_plugin PROPERTIES
        #     IMPORTED_LOCATION "$ENV{QT_PLUGIN_PATH}\\qwindows.dll"
        # )
        # target_link_libraries(${TARGET_NAME} PRIVATE qwindows_plugin)
    # endif()

    set(PRIVATE_PCH
        ${PRIVATE_PCH}
        <QLabel>
        <QCheckBox>
        <QComboBox>
        <QFileDialog>
        <QHBoxLayout>
        <QLabel>
        <QLineEdit>
        <QMenu>
        <QMenuBar>
        <QMessageBox>
        <QProgressBar>
        <QPushButton>
        <QSettings>
        <QTabWidget>
        <QVBoxLayout>
        <QValidator>
        <QWidget>
    )
endif()

if (ENABLE_PCH AND NOT ENABLE_CLANG_TIDY)
    target_precompile_headers(${TARGET_NAME} PRIVATE
        ${PRIVATE_PCH}
    )
endif()
