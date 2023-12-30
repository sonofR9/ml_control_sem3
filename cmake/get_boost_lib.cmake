find_package(Boost 1.84 CONFIG
    COMPONENTS program_options
)

if(${Boost_FOUND})
else()
    set(BOOST_INCLUDE_LIBRARIES program_options)
    set(BOOST_ENABLE_CMAKE ON)

    include(FetchContent)
    FetchContent_Declare(Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.7z
        USES_TERMINAL_DOWNLOAD TRUE 
        GIT_PROGRESS TRUE   
        DOWNLOAD_NO_EXTRACT FALSE
    )
    FetchContent_MakeAvailable(Boost)

    if(Boost_INCLUDE_DIRS STREQUAL "Boost_INCLUDE_DIR-NOTFOUND")
        set(Boost_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/_deps/boost-build/libs")
    endif()

    # set(Boost_LIBRARIES
    #     "Boost::program_options"
    # )
endif()
