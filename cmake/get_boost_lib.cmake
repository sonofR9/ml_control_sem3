function(find_boost_libs LIBRARIES)
  # Find Boost using find_package
  find_package(Boost 1.84 CONFIG COMPONENTS ${LIBRARIES})

  # If Boost was found, return
  if(Boost_FOUND)
    return()
  endif()

  # Otherwise, fetch Boost using FetchContent
  set(BOOST_INCLUDE_LIBRARIES ${LIBRARIES})
  set(BOOST_ENABLE_CMAKE ON)

  include(FetchContent)
  FetchContent_Declare(Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.7z
    USES_TERMINAL_DOWNLOAD TRUE
    GIT_PROGRESS TRUE
    DOWNLOAD_NO_EXTRACT FALSE
  )
  FetchContent_MakeAvailable(Boost)

  # Ensure Boost_INCLUDE_DIRS is set correctly
  if(Boost_INCLUDE_DIRS STREQUAL "Boost_INCLUDE_DIR-NOTFOUND")
    set(Boost_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/_deps/boost-build/libs")
  endif()
endfunction()
